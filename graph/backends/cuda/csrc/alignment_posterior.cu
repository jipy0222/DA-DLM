// Copyright (C) 2022 COAI @ Tsinghua University
// SPDX-License-Identifier: Apache-2.0
// Arbitrary-VJP fixed-target posterior built on alignment-partition alpha/beta.

#include <cmath>
#include <limits>
#include <tuple>

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>


#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) \
  TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")


// Implemented in alignment_partition.cu. The beta recurrence includes the
// emission at its own state, so log posterior is
// alpha + beta - emission - partition.
std::tuple<torch::Tensor, torch::Tensor> dag_loss(
    const torch::Tensor& emission_scores,
    const torch::Tensor& compact_transition_scores,
    const torch::Tensor& node_lengths,
    const torch::Tensor& target_lengths,
    bool require_gradient,
    int config);


template <
    class scalar_t,
    class InputAccessor,
    class AccumulationAccessor,
    class LengthAccessor>
__global__ void alignment_posterior_forward_kernel(
    InputAccessor posterior,
    AccumulationAccessor alpha,
    AccumulationAccessor beta,
    InputAccessor emissions,
    LengthAccessor node_lengths,
    LengthAccessor target_lengths,
    int max_target_length,
    int max_node_count) {
  const int64_t linear =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t state_count =
      static_cast<int64_t>(posterior.size(0)) *
      max_target_length * max_node_count;
  if (linear >= state_count) {
    return;
  }
  const int node = linear % max_node_count;
  const int target = (linear / max_node_count) % max_target_length;
  const int batch = linear / (max_target_length * max_node_count);
  const scalar_t negative_infinity =
      -std::numeric_limits<scalar_t>::infinity();
  const int node_length = node_lengths[batch];
  const int target_length = target_lengths[batch];
  const double partition =
      alpha[batch][target_length - 1][node_length - 1];

  if (target >= target_length || node >= node_length ||
      !isfinite(static_cast<double>(partition))) {
    posterior[batch][target][node] = negative_infinity;
    return;
  }
  const double alpha_value = alpha[batch][target][node];
  const double beta_value = beta[batch][target][node];
  const scalar_t emission = emissions[batch][target][node];
  if (!isfinite(static_cast<double>(alpha_value)) ||
      !isfinite(static_cast<double>(beta_value)) ||
      !isfinite(static_cast<double>(emission))) {
    posterior[batch][target][node] = negative_infinity;
    return;
  }
  posterior[batch][target][node] = static_cast<scalar_t>(
      alpha_value + beta_value - static_cast<double>(emission) - partition);
}


template <int BLOCK_THREADS>
__device__ double posterior_block_sum(double value, double* warp_sums) {
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  #pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    value += __shfl_down_sync(0xffffffffu, value, offset);
  }
  if (lane == 0) {
    warp_sums[warp] = value;
  }
  __syncthreads();
  if (warp == 0) {
    value = lane < (BLOCK_THREADS / 32) ? warp_sums[lane] : 0.0;
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
      value += __shfl_down_sync(0xffffffffu, value, offset);
    }
  }
  return value;
}


template <class input_t, int BLOCK_THREADS, class InputAccessor,
          class StatisticAccessor, class LengthAccessor>
__global__ void alignment_posterior_backward_initialize_kernel(
    InputAccessor grad_output,
    InputAccessor posterior,
    StatisticAccessor grad_alpha,
    StatisticAccessor grad_beta,
    StatisticAccessor grad_emissions,
    LengthAccessor node_lengths,
    LengthAccessor target_lengths,
    int max_target_length,
    int max_node_count) {
  const int batch = blockIdx.x;
  const int state_count = max_target_length * max_node_count;
  double lane_total = 0.0;
  double compensation = 0.0;
  for (int linear = threadIdx.x; linear < state_count;
       linear += BLOCK_THREADS) {
    const int target = linear / max_node_count;
    const int node = linear % max_node_count;
    const input_t posterior_value = posterior[batch][target][node];
    const double upstream = isfinite(static_cast<double>(posterior_value))
        ? static_cast<double>(grad_output[batch][target][node])
        : 0.0;
    grad_alpha[batch][target][node] = upstream;
    grad_beta[batch][target][node] = upstream;
    grad_emissions[batch][target][node] = -upstream;
    const double corrected = upstream - compensation;
    const double updated = lane_total + corrected;
    compensation = (updated - lane_total) - corrected;
    lane_total = updated;
  }
  __shared__ double warp_sums[BLOCK_THREADS / 32];
  const double total = posterior_block_sum<BLOCK_THREADS>(
      lane_total, warp_sums);
  if (threadIdx.x == 0) {
    grad_alpha[batch][target_lengths[batch] - 1]
              [node_lengths[batch] - 1] -= total;
  }
}


template <int BLOCK_THREADS, class InputAccessor,
          class StatisticAccessor, class LengthAccessor>
__global__ void alignment_posterior_backward_alpha_step_kernel(
    StatisticAccessor alpha,
    InputAccessor emissions,
    InputAccessor compact_transitions,
    LengthAccessor node_lengths,
    LengthAccessor target_lengths,
    StatisticAccessor grad_alpha,
    StatisticAccessor grad_emissions,
    StatisticAccessor grad_transitions,
    int target) {
  const int batch = blockIdx.x;
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int source = blockIdx.y * (BLOCK_THREADS / 32) + warp;
  const int node_length = node_lengths[batch];
  const int target_length = target_lengths[batch];
  // A non-zero terminal-partition adjoint can only reach states that leave
  // enough strictly increasing nodes for both the prefix and suffix.
  const int first_source = target - 1;
  const int last_source = node_length - target_length + target - 1;
  if (target >= target_length || source < first_source ||
      source > last_source) {
    return;
  }
  if (lane == 0) {
    const int node = source + 1;
    grad_emissions[batch][target][node] +=
        grad_alpha[batch][target][node];
  }
  double source_adjoint = 0.0;
  const double previous = alpha[batch][target - 1][source];
  const int last_destination = node_length - target_length + target;
  for (int destination = source + 1 + lane;
       destination <= last_destination;
       destination += 32) {
    const int offset = destination - source - 1;
    const double edge = static_cast<double>(
        compact_transitions[batch][source][offset]);
    const double emission = static_cast<double>(
        emissions[batch][target][destination]);
    const double current = alpha[batch][target][destination];
    if (!isfinite(previous) || !isfinite(edge) ||
        !isfinite(emission) || !isfinite(current)) {
      continue;
    }
    const double conditional = exp(previous + edge + emission - current);
    const double contribution =
        grad_alpha[batch][target][destination] * conditional;
    source_adjoint += contribution;
    grad_transitions[batch][source][offset] += contribution;
  }
  #pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    source_adjoint += __shfl_down_sync(
        0xffffffffu, source_adjoint, offset);
  }
  if (lane == 0) {
    grad_alpha[batch][target - 1][source] += source_adjoint;
  }
}


template <int BLOCK_THREADS, class InputAccessor,
          class StatisticAccessor, class LengthAccessor>
__global__ void alignment_posterior_backward_beta_step_kernel(
    StatisticAccessor beta,
    InputAccessor emissions,
    InputAccessor compact_transitions,
    LengthAccessor node_lengths,
    LengthAccessor target_lengths,
    StatisticAccessor grad_beta,
    StatisticAccessor grad_emissions,
    StatisticAccessor grad_transitions,
    int target) {
  const int batch = blockIdx.x;
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int destination = blockIdx.y * (BLOCK_THREADS / 32) + warp;
  const int node_length = node_lengths[batch];
  const int target_length = target_lengths[batch];
  const int first_destination = target + 1;
  const int last_destination = node_length - target_length + target + 1;
  if (target >= target_length - 1 || destination < first_destination ||
      destination > last_destination) {
    return;
  }
  if (lane == 0) {
    const int node = destination - 1;
    grad_emissions[batch][target][node] +=
        grad_beta[batch][target][node];
  }
  double destination_adjoint = 0.0;
  const double next = beta[batch][target + 1][destination];
  for (int source = target + lane; source < destination;
       source += 32) {
    const int offset = destination - source - 1;
    const double current = beta[batch][target][source];
    const double edge = static_cast<double>(
        compact_transitions[batch][source][offset]);
    const double emission = static_cast<double>(
        emissions[batch][target][source]);
    if (!isfinite(current) || !isfinite(edge) ||
        !isfinite(next) || !isfinite(emission)) {
      continue;
    }
    const double conditional = exp(edge + next + emission - current);
    const double contribution =
        grad_beta[batch][target][source] * conditional;
    destination_adjoint += contribution;
    grad_transitions[batch][source][offset] += contribution;
  }
  #pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    destination_adjoint += __shfl_down_sync(
        0xffffffffu, destination_adjoint, offset);
  }
  if (lane == 0) {
    grad_beta[batch][target + 1][destination] += destination_adjoint;
  }
}


template <class input_t, class InputAccessor, class StatisticAccessor,
          class LengthAccessor>
__global__ void alignment_posterior_backward_finalize_emissions_kernel(
    StatisticAccessor grad_alpha,
    StatisticAccessor grad_beta,
    StatisticAccessor grad_emissions_accumulation,
    LengthAccessor target_lengths,
    InputAccessor grad_emissions,
    int max_target_length,
    int max_node_count) {
  const int64_t linear =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t state_count = static_cast<int64_t>(grad_emissions.size(0)) *
      max_target_length * max_node_count;
  if (linear >= state_count) {
    return;
  }
  const int node = linear % max_node_count;
  const int target = (linear / max_node_count) % max_target_length;
  const int batch = linear / (max_target_length * max_node_count);
  double value = grad_emissions_accumulation[batch][target][node];
  if (target == 0) {
    value += grad_alpha[batch][target][node];
  }
  if (target == target_lengths[batch] - 1) {
    value += grad_beta[batch][target][node];
  }
  grad_emissions[batch][target][node] = static_cast<input_t>(value);
}


template <class input_t, class InputAccessor, class StatisticAccessor>
__global__ void alignment_posterior_backward_finalize_transitions_kernel(
    StatisticAccessor grad_transitions_accumulation,
    InputAccessor grad_transitions,
    int64_t transition_count,
    int max_node_count) {
  const int64_t linear =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (linear >= transition_count) {
    return;
  }
  const int offset = linear % (max_node_count - 1);
  const int source = (linear / (max_node_count - 1)) % max_node_count;
  const int batch = linear / (max_node_count * (max_node_count - 1));
  grad_transitions[batch][source][offset] = static_cast<input_t>(
      grad_transitions_accumulation[batch][source][offset]);
}


std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
alignment_posterior_forward(
    const torch::Tensor& emission_scores,
    const torch::Tensor& compact_transition_scores,
    const torch::Tensor& node_lengths,
    const torch::Tensor& target_lengths,
    int forward_config) {
  CHECK_CUDA(emission_scores);
  CHECK_CUDA(compact_transition_scores);
  CHECK_CUDA(node_lengths);
  CHECK_CUDA(target_lengths);
  CHECK_CONTIGUOUS(emission_scores);
  CHECK_CONTIGUOUS(compact_transition_scores);
  CHECK_CONTIGUOUS(node_lengths);
  CHECK_CONTIGUOUS(target_lengths);

  // Signed arbitrary VJPs can amplify tiny normalization errors across every
  // posterior state. Preserve the public dtype while computing sufficient
  // statistics in double for both float32 and float64 inputs.
  auto accumulation_emissions = emission_scores.to(at::kDouble);
  auto accumulation_transitions = compact_transition_scores.to(at::kDouble);
  auto alpha_and_beta = dag_loss(
      accumulation_emissions,
      accumulation_transitions,
      node_lengths,
      target_lengths,
      true,
      forward_config);
  const torch::Tensor& alpha = std::get<0>(alpha_and_beta);
  const torch::Tensor& beta = std::get<1>(alpha_and_beta);
  const c10::cuda::CUDAGuard device_guard(emission_scores.device());
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  auto posterior = at::empty_like(emission_scores);
  const int64_t state_count = emission_scores.numel();
  const int threads = 256;
  const int blocks = (state_count + threads - 1) / threads;
  AT_DISPATCH_FLOATING_TYPES(
      emission_scores.scalar_type(), "alignment_posterior_forward", [&] {
        alignment_posterior_forward_kernel<scalar_t>
            <<<blocks, threads, 0, stream>>>(
                posterior.packed_accessor64<scalar_t, 3>(),
                alpha.packed_accessor64<double, 3>(),
                beta.packed_accessor64<double, 3>(),
                emission_scores.packed_accessor64<scalar_t, 3>(),
                node_lengths.packed_accessor64<int64_t, 1>(),
                target_lengths.packed_accessor64<int64_t, 1>(),
                emission_scores.size(1),
                emission_scores.size(2));
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      });
  return std::make_tuple(posterior, alpha, beta);
}


template <int WIDTH>
std::tuple<torch::Tensor, torch::Tensor>
invoke_alignment_posterior_backward(
    cudaStream_t stream,
    const torch::Tensor& grad_output,
    const torch::Tensor& posterior,
    const torch::Tensor& alpha,
    const torch::Tensor& beta,
    const torch::Tensor& emission_scores,
    const torch::Tensor& compact_transition_scores,
    const torch::Tensor& node_lengths,
    const torch::Tensor& target_lengths) {
  // Signed VJPs can cancel over long paths, so the conditional probabilities
  // and adjoints stay float64 until the public gradients are finalized.
  auto adjoint_options = emission_scores.options().dtype(at::kDouble);
  auto grad_alpha = at::empty(emission_scores.sizes(), adjoint_options);
  auto grad_beta = at::empty(emission_scores.sizes(), adjoint_options);
  auto grad_emissions_accumulation = at::empty(
      emission_scores.sizes(), adjoint_options);
  auto grad_transitions_accumulation = at::zeros(
      compact_transition_scores.sizes(), adjoint_options);
  auto grad_emissions = at::empty_like(emission_scores);
  auto grad_compact_transitions = at::empty_like(compact_transition_scores);
  AT_DISPATCH_FLOATING_TYPES(
      emission_scores.scalar_type(), "alignment_posterior_backward", [&] {
        alignment_posterior_backward_initialize_kernel<scalar_t, WIDTH>
            <<<emission_scores.size(0), WIDTH, 0, stream>>>(
                grad_output.packed_accessor64<scalar_t, 3>(),
                posterior.packed_accessor64<scalar_t, 3>(),
                grad_alpha.packed_accessor64<double, 3>(),
                grad_beta.packed_accessor64<double, 3>(),
                grad_emissions_accumulation.packed_accessor64<double, 3>(),
                node_lengths.packed_accessor64<int64_t, 1>(),
                target_lengths.packed_accessor64<int64_t, 1>(),
                emission_scores.size(1),
                emission_scores.size(2));
        const dim3 step_grid(
            emission_scores.size(0),
            (emission_scores.size(2) + WIDTH / 32 - 1) / (WIDTH / 32));
        for (int target = emission_scores.size(1) - 1;
             target >= 1;
             --target) {
          alignment_posterior_backward_alpha_step_kernel<WIDTH>
              <<<step_grid, WIDTH, 0, stream>>>(
                  alpha.packed_accessor64<double, 3>(),
                  emission_scores.packed_accessor64<scalar_t, 3>(),
                  compact_transition_scores.packed_accessor64<scalar_t, 3>(),
                  node_lengths.packed_accessor64<int64_t, 1>(),
                  target_lengths.packed_accessor64<int64_t, 1>(),
                  grad_alpha.packed_accessor64<double, 3>(),
                  grad_emissions_accumulation.packed_accessor64<double, 3>(),
                  grad_transitions_accumulation.packed_accessor64<double, 3>(),
                  target);
        }
        for (int target = 0; target < emission_scores.size(1) - 1;
             ++target) {
          alignment_posterior_backward_beta_step_kernel<WIDTH>
              <<<step_grid, WIDTH, 0, stream>>>(
                  beta.packed_accessor64<double, 3>(),
                  emission_scores.packed_accessor64<scalar_t, 3>(),
                  compact_transition_scores.packed_accessor64<scalar_t, 3>(),
                  node_lengths.packed_accessor64<int64_t, 1>(),
                  target_lengths.packed_accessor64<int64_t, 1>(),
                  grad_beta.packed_accessor64<double, 3>(),
                  grad_emissions_accumulation.packed_accessor64<double, 3>(),
                  grad_transitions_accumulation.packed_accessor64<double, 3>(),
                  target);
        }
        const int threads = 256;
        const int64_t state_count = emission_scores.numel();
        alignment_posterior_backward_finalize_emissions_kernel<scalar_t>
            <<<(state_count + threads - 1) / threads, threads, 0, stream>>>(
                grad_alpha.packed_accessor64<double, 3>(),
                grad_beta.packed_accessor64<double, 3>(),
                grad_emissions_accumulation.packed_accessor64<double, 3>(),
                target_lengths.packed_accessor64<int64_t, 1>(),
                grad_emissions.packed_accessor64<scalar_t, 3>(),
                emission_scores.size(1),
                emission_scores.size(2));
        const int64_t transition_count =
            compact_transition_scores.numel();
        alignment_posterior_backward_finalize_transitions_kernel<scalar_t>
            <<<(transition_count + threads - 1) / threads,
               threads, 0, stream>>>(
                grad_transitions_accumulation.packed_accessor64<double, 3>(),
                grad_compact_transitions.packed_accessor64<scalar_t, 3>(),
                transition_count,
                emission_scores.size(2));
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      });
  return std::make_tuple(grad_emissions, grad_compact_transitions);
}


std::tuple<torch::Tensor, torch::Tensor> alignment_posterior_backward(
    const torch::Tensor& grad_output,
    const torch::Tensor& posterior,
    const torch::Tensor& alpha,
    const torch::Tensor& beta,
    const torch::Tensor& emission_scores,
    const torch::Tensor& compact_transition_scores,
    const torch::Tensor& node_lengths,
    const torch::Tensor& target_lengths,
    int backward_config) {
  CHECK_CUDA(grad_output);
  CHECK_CUDA(posterior);
  CHECK_CUDA(alpha);
  CHECK_CUDA(beta);
  CHECK_CUDA(emission_scores);
  CHECK_CUDA(compact_transition_scores);
  CHECK_CUDA(node_lengths);
  CHECK_CUDA(target_lengths);
  CHECK_CONTIGUOUS(grad_output);
  CHECK_CONTIGUOUS(posterior);
  CHECK_CONTIGUOUS(alpha);
  CHECK_CONTIGUOUS(beta);
  CHECK_CONTIGUOUS(emission_scores);
  CHECK_CONTIGUOUS(compact_transition_scores);
  CHECK_CONTIGUOUS(node_lengths);
  CHECK_CONTIGUOUS(target_lengths);
  TORCH_CHECK(
      grad_output.sizes() == emission_scores.sizes() &&
          posterior.sizes() == emission_scores.sizes() &&
          alpha.sizes() == emission_scores.sizes() &&
          beta.sizes() == emission_scores.sizes(),
      "posterior backward state shapes must match emissions");
  TORCH_CHECK(
      grad_output.scalar_type() == emission_scores.scalar_type() &&
          posterior.scalar_type() == emission_scores.scalar_type() &&
          compact_transition_scores.scalar_type() ==
              emission_scores.scalar_type(),
      "posterior backward scores must share a dtype");
  TORCH_CHECK(
      alpha.scalar_type() == at::kDouble && beta.scalar_type() == at::kDouble,
      "posterior alpha/beta sufficient statistics must be float64");
  TORCH_CHECK(
      backward_config >= 1 && backward_config <= 4,
      "alignment-posterior backward config must be in 1..4");
  const c10::cuda::CUDAGuard device_guard(emission_scores.device());
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  switch (backward_config) {
    case 1:
      return invoke_alignment_posterior_backward<32>(
          stream, grad_output, posterior, alpha, beta, emission_scores,
          compact_transition_scores, node_lengths, target_lengths);
    case 2:
      return invoke_alignment_posterior_backward<64>(
          stream, grad_output, posterior, alpha, beta, emission_scores,
          compact_transition_scores, node_lengths, target_lengths);
    case 3:
      return invoke_alignment_posterior_backward<128>(
          stream, grad_output, posterior, alpha, beta, emission_scores,
          compact_transition_scores, node_lengths, target_lengths);
    case 4:
      return invoke_alignment_posterior_backward<256>(
          stream, grad_output, posterior, alpha, beta, emission_scores,
          compact_transition_scores, node_lengths, target_lengths);
  }
  TORCH_CHECK(false, "unreachable alignment-posterior backward config");
}
