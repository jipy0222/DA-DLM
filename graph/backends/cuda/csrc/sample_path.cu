// Copyright (C) 2022 COAI @ Tsinghua University
// SPDX-License-Identifier: Apache-2.0
// All-path FFBS sampling built on the dense all-path alpha kernel.

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


// Implemented in path_partition.cu; shared by all-path partition and sampling.
std::tuple<torch::Tensor, torch::Tensor> path_partition_forward(
    const torch::Tensor& transition_scores,
    const torch::Tensor& node_lengths,
    int config);


template <
    class accumulation_t,
    int WIDTH,
    class AlphaAccessor,
    class TransitionAccessor,
    class UniformAccessor,
    class NodeAccessor,
    class LengthAccessor>
__global__ void sample_path_backward_kernel(
    AlphaAccessor alpha,
    TransitionAccessor transitions,
    LengthAccessor node_lengths,
    UniformAccessor uniforms,
    NodeAccessor path_nodes,
    LengthAccessor path_lengths,
    int node_count) {
  const int batch = blockIdx.x;
  const int node_length = node_lengths[batch];
  const unsigned mask = __activemask();

  for (int position = threadIdx.x;
       position < node_count;
       position += WIDTH) {
    path_nodes[batch][position] = -1;
  }
  if (threadIdx.x == 0) {
    path_lengths[batch] = 0;
  }
  __syncwarp(mask);

  CUDA_KERNEL_ASSERT(node_length >= 2);
  CUDA_KERNEL_ASSERT(node_length <= node_count);
  if (alpha[batch][node_length - 1] ==
      -std::numeric_limits<accumulation_t>::infinity()) {
    return;
  }

  int current_node = node_length - 1;
  int path_length = 0;
  for (int reverse_step = 0;
       reverse_step < node_length;
       ++reverse_step) {
    if (threadIdx.x == 0) {
      path_nodes[batch][path_length] = current_node;
      ++path_length;
    }
    path_length = __shfl_sync(mask, path_length, 0, WIDTH);
    if (current_node == 0) {
      break;
    }

    accumulation_t lane_max =
        -std::numeric_limits<accumulation_t>::infinity();
    int lane_last_finite = -1;
    for (int source = threadIdx.x;
         source < current_node;
         source += WIDTH) {
      const accumulation_t candidate = alpha[batch][source] +
          static_cast<accumulation_t>(
              transitions[batch][source][current_node]);
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
      const accumulation_t candidate = alpha[batch][source] +
          static_cast<accumulation_t>(
              transitions[batch][source][current_node]);
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
        const accumulation_t candidate = alpha[batch][source] +
            static_cast<accumulation_t>(
                transitions[batch][source][current_node]);
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

    // Clamp to the last finite category if rounding leaves the CDF below one.
    if (selected < 0 || selected >= current_node) {
      selected = last_finite;
    }
    current_node = selected;
  }

  if (threadIdx.x == 0) {
    for (int left = 0, right = path_length - 1;
         left < right;
         ++left, --right) {
      const int64_t temporary = path_nodes[batch][left];
      path_nodes[batch][left] = path_nodes[batch][right];
      path_nodes[batch][right] = temporary;
    }
    path_lengths[batch] = path_length;
  }
}


template <int WIDTH>
void invoke_sample_path_backward(
    cudaStream_t stream,
    const torch::Tensor& alpha,
    const torch::Tensor& transition_scores,
    const torch::Tensor& node_lengths,
    const torch::Tensor& uniforms,
    torch::Tensor& path_nodes,
    torch::Tensor& path_lengths) {
  const int batch_size = transition_scores.size(0);
  const int node_count = transition_scores.size(1);
  AT_DISPATCH_FLOATING_TYPES(
      transition_scores.scalar_type(), "sample_path_backward", [&] {
        sample_path_backward_kernel<double, WIDTH>
            <<<batch_size, WIDTH, 0, stream>>>(
                alpha.packed_accessor64<double, 2>(),
                transition_scores.packed_accessor64<scalar_t, 3>(),
                node_lengths.packed_accessor64<int64_t, 1>(),
                uniforms.packed_accessor64<scalar_t, 2>(),
                path_nodes.packed_accessor64<int64_t, 2>(),
                path_lengths.packed_accessor64<int64_t, 1>(),
                node_count);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      });
}


std::tuple<torch::Tensor, torch::Tensor> sample_path(
    const torch::Tensor& transition_scores,
    const torch::Tensor& node_lengths,
    const torch::Tensor& uniforms,
    int forward_config,
    int sampling_config) {
  CHECK_CUDA(transition_scores);
  CHECK_CUDA(node_lengths);
  CHECK_CUDA(uniforms);
  CHECK_CONTIGUOUS(transition_scores);
  CHECK_CONTIGUOUS(node_lengths);
  CHECK_CONTIGUOUS(uniforms);
  TORCH_CHECK(transition_scores.dim() == 3,
              "transition_scores dim must be 3");
  TORCH_CHECK(node_lengths.dim() == 1, "node_lengths dim must be 1");
  TORCH_CHECK(uniforms.dim() == 2, "uniforms dim must be 2");
  TORCH_CHECK(
      transition_scores.device() == node_lengths.device() &&
          transition_scores.device() == uniforms.device(),
      "all inputs must share a CUDA device");
  TORCH_CHECK(
      transition_scores.scalar_type() == uniforms.scalar_type(),
      "uniforms and transition_scores must share a dtype");

  const int batch_size = transition_scores.size(0);
  const int node_count = transition_scores.size(1);
  TORCH_CHECK(
      uniforms.size(0) == batch_size && uniforms.size(1) == node_count,
      "uniform shape does not match transitions");
  TORCH_CHECK(
      sampling_config >= 1 && sampling_config <= 4,
      "sample-path sampling config must be in 1..4");

  auto partition_and_alpha = path_partition_forward(
      transition_scores, node_lengths, forward_config);
  const torch::Tensor& alpha = std::get<1>(partition_and_alpha);

  const c10::cuda::CUDAGuard device_guard(transition_scores.device());
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  auto path_nodes = at::empty(
      {batch_size, node_count},
      transition_scores.options().dtype(at::kLong));
  auto path_lengths = at::empty(
      {batch_size}, transition_scores.options().dtype(at::kLong));

  switch (sampling_config) {
    case 1:
      invoke_sample_path_backward<4>(
          stream, alpha, transition_scores, node_lengths, uniforms,
          path_nodes, path_lengths);
      break;
    case 2:
      invoke_sample_path_backward<8>(
          stream, alpha, transition_scores, node_lengths, uniforms,
          path_nodes, path_lengths);
      break;
    case 3:
      invoke_sample_path_backward<16>(
          stream, alpha, transition_scores, node_lengths, uniforms,
          path_nodes, path_lengths);
      break;
    case 4:
      invoke_sample_path_backward<32>(
          stream, alpha, transition_scores, node_lengths, uniforms,
          path_nodes, path_lengths);
      break;
  }
  return std::make_tuple(path_nodes, path_lengths);
}
