// ##########################################################################
// Copyright (C) 2022 COAI @ Tsinghua University

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//         http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// ##########################################################################
// Maximum-score complete-path CUDA kernels.

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
    class scalar_t,
    int TRANS_BLOCK_SIZE,
    class ScoreAccessor,
    class TransitionAccessor,
    class TraceAccessor,
    class LengthAccessor>
__global__ void calculate_viterbi_path_kernel(
    ScoreAccessor alpha,
    TraceAccessor trace,
    ScoreAccessor node_scores,
    TransitionAccessor transition_scores,
    LengthAccessor node_lengths,
    int node_count) {
  const int batch_index = blockIdx.x;
  const int node_length = node_lengths[batch_index];

  CUDA_KERNEL_ASSERT(
      node_length >= 2 &&
      "viterbi_path: node length must be at least 2");
  CUDA_KERNEL_ASSERT(
      node_length <= node_count &&
      "viterbi_path: node length exceeds padded node dimension");

  if (threadIdx.x == 0) {
    alpha[batch_index][0] = node_scores[batch_index][0];
  }
  __syncwarp(__activemask());

  for (int destination = 1; destination < node_length; ++destination) {
    scalar_t best_score = -std::numeric_limits<scalar_t>::infinity();
    // ``node_count`` sorts after every valid predecessor and represents an
    // unreachable per-lane candidate during the lexicographic reduction.
    int best_predecessor = node_count;
    const int maximum_delta = destination;

    // The largest offset is the lowest predecessor. Strict comparison keeps
    // the lowest tie within a lane; the warp reduction resolves cross-lane
    // ties by comparing predecessor indices.
    for (
        int delta = maximum_delta - threadIdx.x;
        delta > 0;
        delta -= TRANS_BLOCK_SIZE) {
      const int predecessor = destination - delta;
      const scalar_t candidate =
          alpha[batch_index][predecessor] +
          transition_scores[batch_index][predecessor][destination];
      if (candidate > best_score) {
        best_score = candidate;
        best_predecessor = predecessor;
      }
    }

    const unsigned shuffle_mask = __activemask();
    #pragma unroll
    for (int offset = TRANS_BLOCK_SIZE >> 1; offset > 0; offset >>= 1) {
      const scalar_t candidate_score = __shfl_down_sync(
          shuffle_mask, best_score, offset, TRANS_BLOCK_SIZE);
      const int candidate_predecessor = __shfl_down_sync(
          shuffle_mask, best_predecessor, offset, TRANS_BLOCK_SIZE);
      if (candidate_score > best_score ||
          (candidate_score == best_score &&
           candidate_predecessor < best_predecessor)) {
        best_score = candidate_score;
        best_predecessor = candidate_predecessor;
      }
    }

    if (threadIdx.x == 0) {
      const scalar_t destination_score = node_scores[batch_index][destination];
      if (best_predecessor < node_count &&
          destination_score !=
              -std::numeric_limits<scalar_t>::infinity()) {
        alpha[batch_index][destination] =
            best_score + destination_score;
        trace[batch_index][destination] = best_predecessor;
      }
    }
    __syncwarp(shuffle_mask);
  }
}


template <int TRANS_BLOCK_SIZE>
void invoke_viterbi_path_forward(
    cudaStream_t stream,
    torch::Tensor& alpha,
    torch::Tensor& trace,
    const torch::Tensor& node_scores,
    const torch::Tensor& transition_scores,
    const torch::Tensor& node_lengths,
    int batch_size,
    int node_count) {
  const dim3 grid(batch_size);
  const dim3 block(TRANS_BLOCK_SIZE);
  static_assert(
      TRANS_BLOCK_SIZE <= 32,
      "TRANS_BLOCK_SIZE must not exceed the hardware warp size");

  trace.fill_(-1);
  AT_DISPATCH_FLOATING_TYPES(
      node_scores.scalar_type(), "viterbi_path_forward", [&] {
        alpha.fill_(-std::numeric_limits<scalar_t>::infinity());
        calculate_viterbi_path_kernel<
            scalar_t,
            TRANS_BLOCK_SIZE><<<grid, block, 0, stream>>>(
            alpha.packed_accessor64<scalar_t, 2>(),
            trace.packed_accessor64<int32_t, 2>(),
            node_scores.packed_accessor64<scalar_t, 2>(),
            transition_scores.packed_accessor64<scalar_t, 3>(),
            node_lengths.packed_accessor64<int64_t, 1>(),
            node_count);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      });
}


template <
    class scalar_t,
    int BLOCK_SIZE,
    class AlphaAccessor,
    class ScoreAccessor,
    class NodeAccessor,
    class TraceAccessor,
    class LengthAccessor>
__global__ void viterbi_path_backtrace_kernel(
    AlphaAccessor alpha,
    ScoreAccessor terminal_scores,
    NodeAccessor path_nodes,
    LengthAccessor path_lengths,
    TraceAccessor trace,
    LengthAccessor node_lengths,
    int batch_size) {
  const int batch_index = BLOCK_SIZE * blockIdx.x + threadIdx.x;
  if (batch_index >= batch_size) {
    return;
  }

  const int node_length = node_lengths[batch_index];
  int current_node = node_length - 1;
  const scalar_t score = alpha[batch_index][current_node];
  terminal_scores[batch_index] = score;
  if (score == -std::numeric_limits<scalar_t>::infinity()) {
    return;
  }

  int path_length = 0;
  bool reached_start = false;
  while (path_length < node_length && current_node >= 0) {
    path_nodes[batch_index][path_length] = current_node;
    ++path_length;
    if (current_node == 0) {
      reached_start = true;
      break;
    }
    current_node = trace[batch_index][current_node];
  }

  if (!reached_start) {
    terminal_scores[batch_index] =
        -std::numeric_limits<scalar_t>::infinity();
    for (int position = 0; position < path_length; ++position) {
      path_nodes[batch_index][position] = -1;
    }
    return;
  }

  for (int left = 0, right = path_length - 1;
       left < right;
       ++left, --right) {
    const int64_t temporary = path_nodes[batch_index][left];
    path_nodes[batch_index][left] = path_nodes[batch_index][right];
    path_nodes[batch_index][right] = temporary;
  }
  path_lengths[batch_index] = path_length;
}


template <int BLOCK_SIZE>
void invoke_viterbi_path_backtrace(
    cudaStream_t stream,
    const torch::Tensor& alpha,
    torch::Tensor& terminal_scores,
    torch::Tensor& path_nodes,
    torch::Tensor& path_lengths,
    const torch::Tensor& trace,
    const torch::Tensor& node_lengths,
    int batch_size) {
  const dim3 grid((batch_size - 1) / BLOCK_SIZE + 1);
  const dim3 block(BLOCK_SIZE);
  path_nodes.fill_(-1);
  path_lengths.zero_();

  AT_DISPATCH_FLOATING_TYPES(
      alpha.scalar_type(), "viterbi_path_backtrace", [&] {
        viterbi_path_backtrace_kernel<
            scalar_t,
            BLOCK_SIZE><<<grid, block, 0, stream>>>(
            alpha.packed_accessor64<scalar_t, 2>(),
            terminal_scores.packed_accessor64<scalar_t, 1>(),
            path_nodes.packed_accessor64<int64_t, 2>(),
            path_lengths.packed_accessor64<int64_t, 1>(),
            trace.packed_accessor64<int32_t, 2>(),
            node_lengths.packed_accessor64<int64_t, 1>(),
            batch_size);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      });
}


std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> viterbi_path(
    const torch::Tensor& node_scores,
    const torch::Tensor& transition_scores,
    const torch::Tensor& node_lengths,
    int config) {
  CHECK_CUDA(node_scores);
  CHECK_CUDA(transition_scores);
  CHECK_CUDA(node_lengths);
  CHECK_CONTIGUOUS(node_scores);
  CHECK_CONTIGUOUS(transition_scores);
  CHECK_CONTIGUOUS(node_lengths);
  TORCH_CHECK(node_scores.dim() == 2, "node_scores dim must be 2");
  TORCH_CHECK(
      transition_scores.dim() == 3,
      "transition_scores dim must be 3");
  TORCH_CHECK(node_lengths.dim() == 1, "node_lengths dim must be 1");
  TORCH_CHECK(
      node_scores.device() == transition_scores.device() &&
          node_scores.device() == node_lengths.device(),
      "all inputs must share a CUDA device");
  TORCH_CHECK(
      node_scores.scalar_type() ==
          transition_scores.scalar_type(),
      "score tensors must share a dtype");
  TORCH_CHECK(
      node_scores.scalar_type() == at::kFloat ||
          node_scores.scalar_type() == at::kDouble,
      "score tensors must use float32 or float64");
  TORCH_CHECK(
      node_lengths.scalar_type() == at::kLong,
      "node_lengths must use torch.long");

  const int batch_size = node_scores.size(0);
  const int node_count = node_scores.size(1);
  TORCH_CHECK(batch_size > 0, "empty batches are not supported");
  TORCH_CHECK(node_count >= 2, "node dimension must be at least 2");
  TORCH_CHECK(
      transition_scores.size(0) == batch_size &&
          transition_scores.size(1) == node_count &&
          transition_scores.size(2) == node_count,
      "transition shape does not match node scores");
  TORCH_CHECK(
      node_lengths.size(0) == batch_size,
      "length batch size does not match node scores");

  const c10::cuda::CUDAGuard device_guard(node_scores.device());
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  auto alpha = at::empty(
      {batch_size, node_count}, node_scores.options());
  auto trace = at::empty(
      {batch_size, node_count}, node_scores.options().dtype(at::kInt));
  auto terminal_scores = at::empty({batch_size}, node_scores.options());
  auto path_nodes = at::empty(
      {batch_size, node_count}, node_scores.options().dtype(at::kLong));
  auto path_lengths = at::empty(
      {batch_size}, node_scores.options().dtype(at::kLong));

  switch (config) {
    case 1:
      invoke_viterbi_path_forward<4>(
          stream, alpha, trace, node_scores, transition_scores,
          node_lengths, batch_size, node_count);
      break;
    case 2:
      invoke_viterbi_path_forward<8>(
          stream, alpha, trace, node_scores, transition_scores,
          node_lengths, batch_size, node_count);
      break;
    case 3:
      invoke_viterbi_path_forward<16>(
          stream, alpha, trace, node_scores, transition_scores,
          node_lengths, batch_size, node_count);
      break;
    case 4:
      invoke_viterbi_path_forward<32>(
          stream, alpha, trace, node_scores, transition_scores,
          node_lengths, batch_size, node_count);
      break;
    default:
      TORCH_CHECK(false, "Viterbi-path forward config must be in 1..4");
  }
  invoke_viterbi_path_backtrace<256>(
      stream, alpha, terminal_scores, path_nodes, path_lengths, trace,
      node_lengths, batch_size);
  return std::make_tuple(terminal_scores, path_nodes, path_lengths);
}
