// Copyright (C) 2022 COAI @ Tsinghua University
// SPDX-License-Identifier: Apache-2.0
// All-path sum-product CUDA kernels.

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


template <
    class input_t,
    class accumulation_t,
    int WIDTH,
    class AlphaAccessor,
    class TransitionAccessor,
    class LengthAccessor,
    class PartitionAccessor>
__global__ void path_partition_forward_kernel(
    AlphaAccessor alpha,
    PartitionAccessor partition,
    TransitionAccessor transitions,
    LengthAccessor node_lengths,
    int node_count) {
  const int batch = blockIdx.x;
  const int node_length = node_lengths[batch];
  CUDA_KERNEL_ASSERT(node_length >= 2);
  CUDA_KERNEL_ASSERT(node_length <= node_count);

  if (threadIdx.x == 0) {
    alpha[batch][0] = accumulation_t(0);
  }
  unsigned mask = __activemask();
  __syncwarp(mask);

  for (int destination = 1; destination < node_length; ++destination) {
    accumulation_t lane_max =
        -std::numeric_limits<accumulation_t>::infinity();
    for (int source = threadIdx.x;
         source < destination;
         source += WIDTH) {
      const accumulation_t candidate = alpha[batch][source] +
          static_cast<accumulation_t>(
              transitions[batch][source][destination]);
      lane_max = candidate > lane_max ? candidate : lane_max;
    }

    accumulation_t maximum = lane_max;
    #pragma unroll
    for (int offset = WIDTH >> 1; offset > 0; offset >>= 1) {
      const accumulation_t other =
          __shfl_down_sync(mask, maximum, offset, WIDTH);
      maximum = other > maximum ? other : maximum;
    }
    maximum = __shfl_sync(mask, maximum, 0, WIDTH);

    accumulation_t lane_sum = accumulation_t(0);
    accumulation_t compensation = accumulation_t(0);
    if (maximum != -std::numeric_limits<accumulation_t>::infinity()) {
      for (int source = threadIdx.x;
           source < destination;
           source += WIDTH) {
        const accumulation_t candidate = alpha[batch][source] +
            static_cast<accumulation_t>(
                transitions[batch][source][destination]);
        const accumulation_t value = exp(candidate - maximum);
        const accumulation_t corrected = value - compensation;
        const accumulation_t updated = lane_sum + corrected;
        compensation = (updated - lane_sum) - corrected;
        lane_sum = updated;
      }
      #pragma unroll
      for (int offset = WIDTH >> 1; offset > 0; offset >>= 1) {
        lane_sum += __shfl_down_sync(mask, lane_sum, offset, WIDTH);
      }
    }

    if (threadIdx.x == 0) {
      alpha[batch][destination] =
          maximum == -std::numeric_limits<accumulation_t>::infinity()
          ? maximum
          : log(lane_sum) + maximum;
    }
    __syncwarp(mask);
  }

  if (threadIdx.x == 0) {
    partition[batch] =
        static_cast<input_t>(alpha[batch][node_length - 1]);
  }
}


template <
    class accumulation_t,
    int WIDTH,
    class BetaAccessor,
    class TransitionAccessor,
    class LengthAccessor>
__global__ void path_partition_reverse_kernel(
    BetaAccessor beta,
    TransitionAccessor transitions,
    LengthAccessor node_lengths) {
  const int batch = blockIdx.x;
  const int node_length = node_lengths[batch];
  CUDA_KERNEL_ASSERT(node_length >= 2);
  CUDA_KERNEL_ASSERT(node_length <= transitions.size(1));
  if (threadIdx.x == 0) {
    beta[batch][node_length - 1] = accumulation_t(0);
  }
  unsigned mask = __activemask();
  __syncwarp(mask);

  for (int source = node_length - 2; source >= 0; --source) {
    accumulation_t maximum =
        -std::numeric_limits<accumulation_t>::infinity();
    for (int destination = source + 1 + threadIdx.x;
         destination < node_length;
         destination += WIDTH) {
      const accumulation_t candidate =
          static_cast<accumulation_t>(
              transitions[batch][source][destination]) +
          beta[batch][destination];
      maximum = candidate > maximum ? candidate : maximum;
    }
    #pragma unroll
    for (int offset = WIDTH >> 1; offset > 0; offset >>= 1) {
      const accumulation_t other =
          __shfl_down_sync(mask, maximum, offset, WIDTH);
      maximum = other > maximum ? other : maximum;
    }
    maximum = __shfl_sync(mask, maximum, 0, WIDTH);

    accumulation_t lane_sum = accumulation_t(0);
    accumulation_t compensation = accumulation_t(0);
    if (maximum != -std::numeric_limits<accumulation_t>::infinity()) {
      for (int destination = source + 1 + threadIdx.x;
           destination < node_length;
           destination += WIDTH) {
        const accumulation_t candidate =
            static_cast<accumulation_t>(
                transitions[batch][source][destination]) +
            beta[batch][destination];
        const accumulation_t value = exp(candidate - maximum);
        const accumulation_t corrected = value - compensation;
        const accumulation_t updated = lane_sum + corrected;
        compensation = (updated - lane_sum) - corrected;
        lane_sum = updated;
      }
      #pragma unroll
      for (int offset = WIDTH >> 1; offset > 0; offset >>= 1) {
        lane_sum += __shfl_down_sync(mask, lane_sum, offset, WIDTH);
      }
    }

    if (threadIdx.x == 0) {
      beta[batch][source] =
          maximum == -std::numeric_limits<accumulation_t>::infinity()
          ? maximum
          : log(lane_sum) + maximum;
    }
    __syncwarp(mask);
  }
}


template <
    class input_t,
    class accumulation_t,
    class GradAccessor,
    class AlphaAccessor,
    class BetaAccessor,
    class TransitionAccessor,
    class LengthAccessor>
__global__ void path_partition_gradient_kernel(
    TransitionAccessor transition_gradient,
    GradAccessor grad_output,
    AlphaAccessor alpha,
    BetaAccessor beta,
    TransitionAccessor transitions,
    LengthAccessor node_lengths,
    int batch_size,
    int node_count) {
  const int64_t edge_count =
      static_cast<int64_t>(batch_size) * node_count * node_count;
  for (int64_t flat_index =
           static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       flat_index < edge_count;
       flat_index += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    const int destination = flat_index % node_count;
    const int source = (flat_index / node_count) % node_count;
    const int batch = flat_index / (node_count * node_count);
    const int node_length = node_lengths[batch];
    const accumulation_t partition = alpha[batch][node_length - 1];

    accumulation_t gradient = accumulation_t(0);
    if (source < destination && destination < node_length &&
        partition != -std::numeric_limits<accumulation_t>::infinity() &&
        alpha[batch][source] !=
            -std::numeric_limits<accumulation_t>::infinity() &&
        beta[batch][destination] !=
            -std::numeric_limits<accumulation_t>::infinity()) {
      gradient = static_cast<accumulation_t>(grad_output[batch]) * exp(
          alpha[batch][source] +
          static_cast<accumulation_t>(
              transitions[batch][source][destination]) +
          beta[batch][destination] - partition);
    }
    transition_gradient[batch][source][destination] =
        static_cast<input_t>(gradient);
  }
}


void check_path_inputs(
    const torch::Tensor& transition_scores,
    const torch::Tensor& node_lengths) {
  CHECK_CUDA(transition_scores);
  CHECK_CUDA(node_lengths);
  CHECK_CONTIGUOUS(transition_scores);
  CHECK_CONTIGUOUS(node_lengths);
  TORCH_CHECK(transition_scores.dim() == 3, "transition_scores dim must be 3");
  TORCH_CHECK(node_lengths.dim() == 1, "node_lengths dim must be 1");
  TORCH_CHECK(
      transition_scores.device() == node_lengths.device(),
      "all inputs must share a CUDA device");
  TORCH_CHECK(
      transition_scores.scalar_type() == at::kFloat ||
          transition_scores.scalar_type() == at::kDouble,
      "transition_scores must use float32 or float64");
  TORCH_CHECK(node_lengths.scalar_type() == at::kLong,
              "node_lengths must use torch.long");
  TORCH_CHECK(transition_scores.size(0) > 0, "empty batches are not supported");
  TORCH_CHECK(transition_scores.size(1) >= 2,
              "node dimension must be at least 2");
  TORCH_CHECK(
      transition_scores.size(1) == transition_scores.size(2),
      "transition node dimensions must be square");
  TORCH_CHECK(
      node_lengths.size(0) == transition_scores.size(0),
      "length batch size does not match transitions");
}


template <int WIDTH>
void invoke_path_partition_forward(
    cudaStream_t stream,
    torch::Tensor& alpha,
    torch::Tensor& partition,
    const torch::Tensor& transitions,
    const torch::Tensor& node_lengths) {
  const int batch_size = transitions.size(0);
  const int node_count = transitions.size(1);
  alpha.fill_(-std::numeric_limits<double>::infinity());
  AT_DISPATCH_FLOATING_TYPES(
      transitions.scalar_type(), "path_partition_forward", [&] {
        path_partition_forward_kernel<scalar_t, double, WIDTH>
            <<<batch_size, WIDTH, 0, stream>>>(
                alpha.packed_accessor64<double, 2>(),
                partition.packed_accessor64<scalar_t, 1>(),
                transitions.packed_accessor64<scalar_t, 3>(),
                node_lengths.packed_accessor64<int64_t, 1>(),
                node_count);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      });
}


template <int WIDTH>
void invoke_path_suffix(
    cudaStream_t stream,
    torch::Tensor& beta,
    const torch::Tensor& transitions,
    const torch::Tensor& node_lengths) {
  const int batch_size = transitions.size(0);
  beta.fill_(-std::numeric_limits<double>::infinity());
  AT_DISPATCH_FLOATING_TYPES(
      transitions.scalar_type(), "path_suffix_partition", [&] {
        path_partition_reverse_kernel<double, WIDTH>
            <<<batch_size, WIDTH, 0, stream>>>(
                beta.packed_accessor64<double, 2>(),
                transitions.packed_accessor64<scalar_t, 3>(),
                node_lengths.packed_accessor64<int64_t, 1>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      });
}


template <int WIDTH, int GRADIENT_BLOCK_SIZE>
void invoke_path_partition_backward(
    cudaStream_t stream,
    torch::Tensor& transition_gradient,
    torch::Tensor& beta,
    const torch::Tensor& grad_output,
    const torch::Tensor& alpha,
    const torch::Tensor& transitions,
    const torch::Tensor& node_lengths) {
  invoke_path_suffix<WIDTH>(stream, beta, transitions, node_lengths);
  const int batch_size = transitions.size(0);
  const int node_count = transitions.size(1);
  const int64_t edge_count =
      static_cast<int64_t>(batch_size) * node_count * node_count;
  const int gradient_blocks = static_cast<int>(
      (edge_count + GRADIENT_BLOCK_SIZE - 1) / GRADIENT_BLOCK_SIZE);
  AT_DISPATCH_FLOATING_TYPES(
      transitions.scalar_type(), "path_partition_backward", [&] {
        path_partition_gradient_kernel<scalar_t, double>
            <<<gradient_blocks, GRADIENT_BLOCK_SIZE, 0, stream>>>(
                transition_gradient.packed_accessor64<scalar_t, 3>(),
                grad_output.packed_accessor64<scalar_t, 1>(),
                alpha.packed_accessor64<double, 2>(),
                beta.packed_accessor64<double, 2>(),
                transitions.packed_accessor64<scalar_t, 3>(),
                node_lengths.packed_accessor64<int64_t, 1>(),
                batch_size,
                node_count);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      });
}


std::tuple<torch::Tensor, torch::Tensor> path_partition_forward(
    const torch::Tensor& transition_scores,
    const torch::Tensor& node_lengths,
    int config) {
  check_path_inputs(transition_scores, node_lengths);
  const c10::cuda::CUDAGuard device_guard(transition_scores.device());
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  auto alpha = at::empty(
      {transition_scores.size(0), transition_scores.size(1)},
      transition_scores.options().dtype(at::kDouble));
  auto partition = at::empty(
      {transition_scores.size(0)}, transition_scores.options());
  switch (config) {
    case 1:
      invoke_path_partition_forward<4>(
          stream, alpha, partition, transition_scores, node_lengths);
      break;
    case 2:
      invoke_path_partition_forward<8>(
          stream, alpha, partition, transition_scores, node_lengths);
      break;
    case 3:
      invoke_path_partition_forward<16>(
          stream, alpha, partition, transition_scores, node_lengths);
      break;
    case 4:
      invoke_path_partition_forward<32>(
          stream, alpha, partition, transition_scores, node_lengths);
      break;
    default:
      TORCH_CHECK(false, "path-partition forward config must be in 1..4");
  }
  return std::make_tuple(partition, alpha);
}


torch::Tensor path_suffix_partition(
    const torch::Tensor& transition_scores,
    const torch::Tensor& node_lengths,
    int config) {
  check_path_inputs(transition_scores, node_lengths);
  const c10::cuda::CUDAGuard device_guard(transition_scores.device());
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  auto beta = at::empty(
      {transition_scores.size(0), transition_scores.size(1)},
      transition_scores.options().dtype(at::kDouble));
  switch (config) {
    case 1:
      invoke_path_suffix<4>(stream, beta, transition_scores, node_lengths);
      break;
    case 2:
      invoke_path_suffix<8>(stream, beta, transition_scores, node_lengths);
      break;
    case 3:
      invoke_path_suffix<16>(stream, beta, transition_scores, node_lengths);
      break;
    case 4:
      invoke_path_suffix<32>(stream, beta, transition_scores, node_lengths);
      break;
    default:
      TORCH_CHECK(false, "path-suffix forward config must be in 1..4");
  }
  return beta.to(transition_scores.scalar_type());
}


torch::Tensor path_partition_backward(
    const torch::Tensor& grad_output,
    const torch::Tensor& alpha,
    const torch::Tensor& transition_scores,
    const torch::Tensor& node_lengths,
    int config) {
  check_path_inputs(transition_scores, node_lengths);
  CHECK_CUDA(grad_output);
  CHECK_CUDA(alpha);
  CHECK_CONTIGUOUS(grad_output);
  CHECK_CONTIGUOUS(alpha);
  TORCH_CHECK(
      grad_output.device() == transition_scores.device() &&
          alpha.device() == transition_scores.device(),
      "all backward inputs must share a CUDA device");
  TORCH_CHECK(
      grad_output.scalar_type() == transition_scores.scalar_type() &&
          alpha.scalar_type() == at::kDouble,
      "grad_output must match transitions and alpha must use float64");
  TORCH_CHECK(
      grad_output.dim() == 1 &&
          grad_output.size(0) == transition_scores.size(0),
      "grad_output shape does not match transitions");
  TORCH_CHECK(
      alpha.dim() == 2 &&
          alpha.size(0) == transition_scores.size(0) &&
          alpha.size(1) == transition_scores.size(1),
      "alpha shape does not match transitions");

  const c10::cuda::CUDAGuard device_guard(transition_scores.device());
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  auto beta = at::empty_like(alpha);
  auto transition_gradient = at::empty_like(transition_scores);
  switch (config) {
    case 1:
      invoke_path_partition_backward<4, 64>(
          stream, transition_gradient, beta, grad_output, alpha,
          transition_scores, node_lengths);
      break;
    case 2:
      invoke_path_partition_backward<8, 128>(
          stream, transition_gradient, beta, grad_output, alpha,
          transition_scores, node_lengths);
      break;
    case 3:
      invoke_path_partition_backward<16, 256>(
          stream, transition_gradient, beta, grad_output, alpha,
          transition_scores, node_lengths);
      break;
    case 4:
      invoke_path_partition_backward<32, 512>(
          stream, transition_gradient, beta, grad_output, alpha,
          transition_scores, node_lengths);
      break;
    default:
      TORCH_CHECK(false, "path-partition backward config must be in 1..4");
  }
  return transition_gradient;
}
