// Copyright (C) 2022 COAI @ Tsinghua University
// SPDX-License-Identifier: Apache-2.0
// Fixed-target FFBS sampling built on the alignment-alpha kernel.

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


// Implemented in alignment_partition.cu. With require_gradient=false this
// runs only the alpha recurrence and returns an empty beta tensor.
std::tuple<torch::Tensor, torch::Tensor> dag_loss(
    const torch::Tensor& emission_scores,
    const torch::Tensor& compact_transition_scores,
    const torch::Tensor& node_lengths,
    const torch::Tensor& target_lengths,
    bool require_gradient,
    int config);


template <
    class input_t,
    class accumulation_t,
    int WIDTH,
    class AlphaAccessor,
    class TransitionAccessor,
    class UniformAccessor,
    class NodeAccessor,
    class LengthAccessor>
__global__ void sample_alignment_backward_kernel(
    AlphaAccessor alpha,
    TransitionAccessor compact_transitions,
    LengthAccessor node_lengths,
    LengthAccessor target_lengths,
    UniformAccessor uniforms,
    NodeAccessor alignment_nodes,
    int max_target_length,
    int max_node_count) {
  const int batch = blockIdx.x;
  const int node_length = node_lengths[batch];
  const int target_length = target_lengths[batch];
  const unsigned mask = __activemask();

  for (int target = threadIdx.x;
       target < max_target_length;
       target += WIDTH) {
    alignment_nodes[batch][target] = -1;
  }
  __syncwarp(mask);

  CUDA_KERNEL_ASSERT(node_length >= 2);
  CUDA_KERNEL_ASSERT(node_length <= max_node_count);
  CUDA_KERNEL_ASSERT(target_length >= 2);
  CUDA_KERNEL_ASSERT(target_length <= max_target_length);
  CUDA_KERNEL_ASSERT(target_length <= node_length);
  if (alpha[batch][target_length - 1][node_length - 1] ==
      -std::numeric_limits<input_t>::infinity()) {
    return;
  }

  int current_node = node_length - 1;
  for (int reverse_step = 0;
       reverse_step < target_length;
       ++reverse_step) {
    const int current_target = target_length - 1 - reverse_step;
    if (threadIdx.x == 0) {
      alignment_nodes[batch][current_target] = current_node;
    }
    if (current_target == 0) {
      break;
    }

    accumulation_t lane_max =
        -std::numeric_limits<accumulation_t>::infinity();
    int lane_last_finite = -1;
    for (int source = threadIdx.x;
         source < current_node;
         source += WIDTH) {
      const int compact_offset = current_node - source - 1;
      const accumulation_t candidate =
          static_cast<accumulation_t>(
              alpha[batch][current_target - 1][source]) +
          static_cast<accumulation_t>(
              compact_transitions[batch][source][compact_offset]);
      if (candidate !=
          -std::numeric_limits<accumulation_t>::infinity()) {
        lane_last_finite = source;
      }
      lane_max = candidate > lane_max ? candidate : lane_max;
    }

    accumulation_t maximum = lane_max;
    int last_finite = lane_last_finite;
    #pragma unroll
    for (int offset = WIDTH >> 1; offset > 0; offset >>= 1) {
      const accumulation_t other_max =
          __shfl_down_sync(mask, maximum, offset, WIDTH);
      const int other_last =
          __shfl_down_sync(mask, last_finite, offset, WIDTH);
      maximum = other_max > maximum ? other_max : maximum;
      last_finite = other_last > last_finite ? other_last : last_finite;
    }
    maximum = __shfl_sync(mask, maximum, 0, WIDTH);
    last_finite = __shfl_sync(mask, last_finite, 0, WIDTH);
    CUDA_KERNEL_ASSERT(last_finite >= 0);

    accumulation_t lane_sum = accumulation_t(0);
    accumulation_t compensation = accumulation_t(0);
    for (int source = threadIdx.x;
         source < current_node;
         source += WIDTH) {
      const int compact_offset = current_node - source - 1;
      const accumulation_t candidate =
          static_cast<accumulation_t>(
              alpha[batch][current_target - 1][source]) +
          static_cast<accumulation_t>(
              compact_transitions[batch][source][compact_offset]);
      const accumulation_t value =
          candidate == -std::numeric_limits<accumulation_t>::infinity()
          ? accumulation_t(0)
          : exp(candidate - maximum);
      const accumulation_t corrected = value - compensation;
      const accumulation_t updated = lane_sum + corrected;
      compensation = (updated - lane_sum) - corrected;
      lane_sum = updated;
    }
    #pragma unroll
    for (int offset = WIDTH >> 1; offset > 0; offset >>= 1) {
      lane_sum += __shfl_down_sync(mask, lane_sum, offset, WIDTH);
    }
    const accumulation_t total =
        __shfl_sync(mask, lane_sum, 0, WIDTH);
    const accumulation_t threshold =
        static_cast<accumulation_t>(uniforms[batch][reverse_step]) * total;

    int selected = -1;
    accumulation_t cumulative_before = accumulation_t(0);
    for (int base = 0; base < current_node; base += WIDTH) {
      const int source = base + threadIdx.x;
      accumulation_t mass = accumulation_t(0);
      if (source < current_node) {
        const int compact_offset = current_node - source - 1;
        const accumulation_t candidate =
            static_cast<accumulation_t>(
                alpha[batch][current_target - 1][source]) +
            static_cast<accumulation_t>(
                compact_transitions[batch][source][compact_offset]);
        if (candidate !=
            -std::numeric_limits<accumulation_t>::infinity()) {
          mass = exp(candidate - maximum);
        }
      }

      accumulation_t inclusive = mass;
      #pragma unroll
      for (int offset = 1; offset < WIDTH; offset <<= 1) {
        const accumulation_t earlier =
            __shfl_up_sync(mask, inclusive, offset, WIDTH);
        if (threadIdx.x >= offset) {
          inclusive += earlier;
        }
      }
      const bool crosses =
          source < current_node &&
          cumulative_before + inclusive > threshold;
      const unsigned candidates = __ballot_sync(mask, crosses);
      if (candidates != 0) {
        selected = __ffs(candidates) - 1 + base;
        break;
      }
      const accumulation_t chunk_sum =
          __shfl_sync(mask, inclusive, WIDTH - 1, WIDTH);
      cumulative_before += chunk_sum;
    }

    if (selected < 0 || selected >= current_node) {
      selected = last_finite;
    }
    current_node = selected;
  }
}


template <int WIDTH>
void invoke_sample_alignment_backward(
    cudaStream_t stream,
    const torch::Tensor& alpha,
    const torch::Tensor& compact_transition_scores,
    const torch::Tensor& node_lengths,
    const torch::Tensor& target_lengths,
    const torch::Tensor& uniforms,
    torch::Tensor& alignment_nodes) {
  const int batch_size = alpha.size(0);
  const int max_target_length = alpha.size(1);
  const int max_node_count = alpha.size(2);
  AT_DISPATCH_FLOATING_TYPES(
      alpha.scalar_type(), "sample_alignment_backward", [&] {
        sample_alignment_backward_kernel<scalar_t, double, WIDTH>
            <<<batch_size, WIDTH, 0, stream>>>(
                alpha.packed_accessor64<scalar_t, 3>(),
                compact_transition_scores.packed_accessor64<scalar_t, 3>(),
                node_lengths.packed_accessor64<int64_t, 1>(),
                target_lengths.packed_accessor64<int64_t, 1>(),
                uniforms.packed_accessor64<scalar_t, 2>(),
                alignment_nodes.packed_accessor64<int64_t, 2>(),
                max_target_length,
                max_node_count);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      });
}


torch::Tensor sample_alignment(
    const torch::Tensor& emission_scores,
    const torch::Tensor& compact_transition_scores,
    const torch::Tensor& node_lengths,
    const torch::Tensor& target_lengths,
    const torch::Tensor& uniforms,
    int forward_config,
    int sampling_config) {
  CHECK_CUDA(emission_scores);
  CHECK_CUDA(compact_transition_scores);
  CHECK_CUDA(node_lengths);
  CHECK_CUDA(target_lengths);
  CHECK_CUDA(uniforms);
  CHECK_CONTIGUOUS(emission_scores);
  CHECK_CONTIGUOUS(compact_transition_scores);
  CHECK_CONTIGUOUS(node_lengths);
  CHECK_CONTIGUOUS(target_lengths);
  CHECK_CONTIGUOUS(uniforms);
  TORCH_CHECK(emission_scores.dim() == 3, "emission_scores dim must be 3");
  TORCH_CHECK(compact_transition_scores.dim() == 3,
              "compact_transition_scores dim must be 3");
  TORCH_CHECK(node_lengths.dim() == 1, "node_lengths dim must be 1");
  TORCH_CHECK(target_lengths.dim() == 1, "target_lengths dim must be 1");
  TORCH_CHECK(uniforms.dim() == 2, "uniforms dim must be 2");
  TORCH_CHECK(
      emission_scores.device() == compact_transition_scores.device() &&
          emission_scores.device() == node_lengths.device() &&
          emission_scores.device() == target_lengths.device() &&
          emission_scores.device() == uniforms.device(),
      "all inputs must share a CUDA device");
  TORCH_CHECK(
      emission_scores.scalar_type() ==
          compact_transition_scores.scalar_type() &&
          emission_scores.scalar_type() == uniforms.scalar_type(),
      "emissions, transitions, and uniforms must share a dtype");

  const int batch_size = emission_scores.size(0);
  const int max_target_length = emission_scores.size(1);
  const int max_node_count = emission_scores.size(2);
  TORCH_CHECK(
      compact_transition_scores.size(0) == batch_size &&
          compact_transition_scores.size(1) == max_node_count &&
          compact_transition_scores.size(2) == max_node_count - 1,
      "compact transition shape does not match emissions");
  TORCH_CHECK(
      uniforms.size(0) == batch_size &&
          uniforms.size(1) == max_target_length,
      "uniform shape does not match emissions");
  TORCH_CHECK(
      sampling_config >= 1 && sampling_config <= 4,
      "sample-alignment sampling config must be in 1..4");

  auto alpha_and_empty_beta = dag_loss(
      emission_scores,
      compact_transition_scores,
      node_lengths,
      target_lengths,
      false,
      forward_config);
  const torch::Tensor& alpha = std::get<0>(alpha_and_empty_beta);

  const c10::cuda::CUDAGuard device_guard(emission_scores.device());
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  auto alignment_nodes = at::empty(
      {batch_size, max_target_length},
      emission_scores.options().dtype(at::kLong));
  switch (sampling_config) {
    case 1:
      invoke_sample_alignment_backward<4>(
          stream, alpha, compact_transition_scores, node_lengths,
          target_lengths, uniforms, alignment_nodes);
      break;
    case 2:
      invoke_sample_alignment_backward<8>(
          stream, alpha, compact_transition_scores, node_lengths,
          target_lengths, uniforms, alignment_nodes);
      break;
    case 3:
      invoke_sample_alignment_backward<16>(
          stream, alpha, compact_transition_scores, node_lengths,
          target_lengths, uniforms, alignment_nodes);
      break;
    case 4:
      invoke_sample_alignment_backward<32>(
          stream, alpha, compact_transition_scores, node_lengths,
          target_lengths, uniforms, alignment_nodes);
      break;
  }
  return alignment_nodes;
}
