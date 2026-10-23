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
// Fixed-target Viterbi CUDA kernels with stable lowest-predecessor ties.

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

#define BLOCK_BUCKET 16


template <
    class scalar_t,
    int TRANS_BLOCK_SIZE,
    int SEQ_BLOCK_SIZE,
    class ScoreAccessor,
    class TraceAccessor,
    class LengthAccessor>
__global__ void calculate_max_alpha_kernel(
    volatile int* bucket_queue,
    volatile int* accomplish_queue,
    ScoreAccessor alpha,
    TraceAccessor trace,
    ScoreAccessor emission_scores,
    ScoreAccessor compact_transition_scores,
    LengthAccessor node_lengths,
    LengthAccessor target_lengths,
    int batch_size,
    int node_count,
    int max_target_length,
    int compact_width,
    int segment_count) {
  const int bucket_index = blockIdx.y % BLOCK_BUCKET;
  __shared__ volatile int bucket_number;

  const bool main_thread = threadIdx.x == 0 && threadIdx.y == 0;
  if (main_thread) {
    bucket_number = atomicAdd(
        const_cast<int*>(bucket_queue) + bucket_index, 1);
  }
  __syncthreads();

  const int ticket_number = bucket_number * BLOCK_BUCKET + bucket_index;
  const int batch_index = ticket_number % batch_size;
  const int segment_index = ticket_number / batch_size;
  const int position = segment_index * SEQ_BLOCK_SIZE + threadIdx.y;
  const int node_length = node_lengths[batch_index];
  const int target_length = target_lengths[batch_index];

  CUDA_KERNEL_ASSERT(
      target_length >= 2 && node_length >= 2 &&
      "viterbi_alignment: target/node length must be at least 2");
  CUDA_KERNEL_ASSERT(
      target_length <= max_target_length &&
      "viterbi_alignment: target length exceeds padded target dimension");
  CUDA_KERNEL_ASSERT(
      node_length <= node_count &&
      "viterbi_alignment: node length exceeds padded node dimension");
  CUDA_KERNEL_ASSERT(
      node_length >= target_length &&
      "viterbi_alignment: graph has fewer nodes than target positions");

  if (position >= node_length) {
    return;
  }
  if (main_thread) {
    alpha[batch_index][0][0] = emission_scores[batch_index][0][0];
  }
  __threadfence();
  __syncthreads();
  if (main_thread) {
    atomicAdd(
        const_cast<int*>(accomplish_queue) +
            batch_index * segment_count + segment_index,
        1);
  }

  for (int target = 1; target < target_length; ++target) {
    if (position + target >= node_length) {
      return;
    }
    if (main_thread && segment_index != 0) {
      while (accomplish_queue[
                 batch_index * segment_count + segment_index - 1] < target) {
      }
    }
    __syncthreads();

    scalar_t best_score = -std::numeric_limits<scalar_t>::infinity();
    // ``node_count`` sorts after every valid predecessor and also represents
    // an unreachable per-lane candidate during the lexicographic reduction.
    int best_predecessor = node_count;
    const int current_node = position + target;
    // At target ``target - 1`` no node below ``target - 1`` can be reached.
    // These unreachable nodes can be excluded from the predecessor scan.
    const int maximum_delta = min(position + 1, compact_width);
    // Lane zero starts at the largest offset (the lowest predecessor), then
    // each lane walks toward higher predecessors.  Strict comparisons retain
    // the lowest tie within each lane; the warp reduction below resolves ties
    // between lanes lexicographically by score and predecessor index.
    for (
        int delta = maximum_delta - threadIdx.x;
        delta > 0;
        delta -= TRANS_BLOCK_SIZE) {
      const int predecessor = current_node - delta;
      const scalar_t candidate =
          alpha[batch_index][target - 1][predecessor] +
          compact_transition_scores[batch_index][predecessor][delta - 1];
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
      const scalar_t emission =
          emission_scores[batch_index][target][current_node];
      if (best_predecessor < node_count &&
          emission != -std::numeric_limits<scalar_t>::infinity()) {
        alpha[batch_index][target][current_node] = best_score + emission;
        trace[batch_index][target][current_node] = best_predecessor;
      }
    }

    __threadfence();
    __syncthreads();
    if (main_thread) {
      atomicAdd(
          const_cast<int*>(accomplish_queue) +
              batch_index * segment_count + segment_index,
          1);
    }
  }
}


template <int TRANS_BLOCK_SIZE, int SEQ_BLOCK_SIZE>
void invoke_calculate_max_alpha(
    cudaStream_t stream,
    torch::Tensor& alpha,
    torch::Tensor& trace,
    const torch::Tensor& emission_scores,
    const torch::Tensor& compact_transition_scores,
    const torch::Tensor& node_lengths,
    const torch::Tensor& target_lengths,
    int batch_size,
    int node_count,
    int max_target_length,
    int compact_width) {
  const int segment_count = (node_count - 1) / SEQ_BLOCK_SIZE + 1;
  const dim3 grid(1, segment_count * batch_size);
  const dim3 block(TRANS_BLOCK_SIZE, SEQ_BLOCK_SIZE);
  static_assert(
      TRANS_BLOCK_SIZE <= 32,
      "TRANS_BLOCK_SIZE must not exceed the hardware warp size");

  auto queues = at::zeros(
      {BLOCK_BUCKET + batch_size * segment_count},
      emission_scores.options().dtype(at::kInt));
  int* bucket_queue = queues.data_ptr<int>();
  int* accomplish_queue = bucket_queue + BLOCK_BUCKET;

  AT_DISPATCH_FLOATING_TYPES(
      emission_scores.scalar_type(), "viterbi_alignment_max_alpha", [&] {
        alpha.fill_(-std::numeric_limits<scalar_t>::infinity());
        trace.fill_(-1);
        calculate_max_alpha_kernel<
            scalar_t,
            TRANS_BLOCK_SIZE,
            SEQ_BLOCK_SIZE><<<grid, block, 0, stream>>>(
            bucket_queue,
            accomplish_queue,
            alpha.packed_accessor64<scalar_t, 3>(),
            trace.packed_accessor64<int32_t, 3>(),
            emission_scores.packed_accessor64<scalar_t, 3>(),
            compact_transition_scores.packed_accessor64<scalar_t, 3>(),
            node_lengths.packed_accessor64<int64_t, 1>(),
            target_lengths.packed_accessor64<int64_t, 1>(),
            batch_size,
            node_count,
            max_target_length,
            compact_width,
            segment_count);
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
__global__ void backtrace_kernel(
    AlphaAccessor alpha,
    ScoreAccessor terminal_scores,
    NodeAccessor alignment_nodes,
    TraceAccessor trace,
    LengthAccessor node_lengths,
    LengthAccessor target_lengths,
    int batch_size) {
  const int batch_index = BLOCK_SIZE * blockIdx.x + threadIdx.x;
  if (batch_index >= batch_size) {
    return;
  }
  const int target_length = target_lengths[batch_index];
  int current_node = node_lengths[batch_index] - 1;
  const scalar_t score =
      alpha[batch_index][target_length - 1][current_node];
  terminal_scores[batch_index] = score;
  if (score == -std::numeric_limits<scalar_t>::infinity()) {
    return;
  }

  for (int target = target_length - 1; target >= 0; --target) {
    if (current_node < 0) {
      terminal_scores[batch_index] =
          -std::numeric_limits<scalar_t>::infinity();
      for (int clear_target = 0;
           clear_target < target_length;
           ++clear_target) {
        alignment_nodes[batch_index][clear_target] = -1;
      }
      return;
    }
    alignment_nodes[batch_index][target] = current_node;
    if (target > 0) {
      current_node = trace[batch_index][target][current_node];
    }
  }
}


template <int BLOCK_SIZE>
void invoke_backtrace(
    cudaStream_t stream,
    const torch::Tensor& alpha,
    torch::Tensor& terminal_scores,
    torch::Tensor& alignment_nodes,
    const torch::Tensor& trace,
    const torch::Tensor& node_lengths,
    const torch::Tensor& target_lengths,
    int batch_size) {
  const dim3 grid((batch_size - 1) / BLOCK_SIZE + 1);
  const dim3 block(BLOCK_SIZE);
  alignment_nodes.fill_(-1);

  AT_DISPATCH_FLOATING_TYPES(
      alpha.scalar_type(), "viterbi_alignment_backtrace", [&] {
        backtrace_kernel<scalar_t, BLOCK_SIZE><<<grid, block, 0, stream>>>(
            alpha.packed_accessor64<scalar_t, 3>(),
            terminal_scores.packed_accessor64<scalar_t, 1>(),
            alignment_nodes.packed_accessor64<int64_t, 2>(),
            trace.packed_accessor64<int32_t, 3>(),
            node_lengths.packed_accessor64<int64_t, 1>(),
            target_lengths.packed_accessor64<int64_t, 1>(),
            batch_size);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      });
}


std::tuple<torch::Tensor, torch::Tensor> viterbi_alignment(
    const torch::Tensor& emission_scores,
    const torch::Tensor& compact_transition_scores,
    const torch::Tensor& node_lengths,
    const torch::Tensor& target_lengths,
    int config) {
  CHECK_CUDA(emission_scores);
  CHECK_CUDA(compact_transition_scores);
  CHECK_CUDA(node_lengths);
  CHECK_CUDA(target_lengths);
  CHECK_CONTIGUOUS(emission_scores);
  CHECK_CONTIGUOUS(compact_transition_scores);
  CHECK_CONTIGUOUS(node_lengths);
  CHECK_CONTIGUOUS(target_lengths);
  TORCH_CHECK(emission_scores.dim() == 3, "emission_scores dim must be 3");
  TORCH_CHECK(
      compact_transition_scores.dim() == 3,
      "compact_transition_scores dim must be 3");
  TORCH_CHECK(
      node_lengths.dim() == 1 && target_lengths.dim() == 1,
      "length tensors must have dim 1");
  TORCH_CHECK(
      emission_scores.device() == compact_transition_scores.device() &&
          emission_scores.device() == node_lengths.device() &&
          emission_scores.device() == target_lengths.device(),
      "all inputs must share a CUDA device");
  TORCH_CHECK(
      emission_scores.scalar_type() ==
          compact_transition_scores.scalar_type(),
      "score tensors must share a dtype");
  TORCH_CHECK(
      node_lengths.scalar_type() == at::kLong &&
          target_lengths.scalar_type() == at::kLong,
      "length tensors must use torch.long");

  const int batch_size = emission_scores.size(0);
  const int max_target_length = emission_scores.size(1);
  const int node_count = emission_scores.size(2);
  const int compact_width = compact_transition_scores.size(2);
  TORCH_CHECK(batch_size > 0, "empty batches are not supported");
  TORCH_CHECK(node_count >= 2, "node dimension must be at least 2");
  TORCH_CHECK(max_target_length >= 2, "target dimension must be at least 2");
  TORCH_CHECK(
      compact_transition_scores.size(0) == batch_size &&
          compact_transition_scores.size(1) == node_count,
      "compact transition shape does not match emissions");
  TORCH_CHECK(
      compact_width >= 1 && compact_width < node_count,
      "compact transition width must be in [1, node_count)");
  TORCH_CHECK(
      node_lengths.size(0) == batch_size &&
          target_lengths.size(0) == batch_size,
      "length batch size does not match emissions");

  const c10::cuda::CUDAGuard device_guard(emission_scores.device());
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  auto alpha = at::empty(
      {batch_size, max_target_length, node_count},
      emission_scores.options());
  auto trace = at::empty(
      {batch_size, max_target_length, node_count},
      emission_scores.options().dtype(at::kInt));
  auto terminal_scores = at::empty({batch_size}, emission_scores.options());
  auto alignment_nodes = at::empty(
      {batch_size, max_target_length},
      emission_scores.options().dtype(at::kLong));

  switch (config) {
    case 1:
      invoke_calculate_max_alpha<4, 256>(
          stream, alpha, trace, emission_scores, compact_transition_scores,
          node_lengths, target_lengths, batch_size, node_count,
          max_target_length, compact_width);
      break;
    case 2:
      invoke_calculate_max_alpha<8, 128>(
          stream, alpha, trace, emission_scores, compact_transition_scores,
          node_lengths, target_lengths, batch_size, node_count,
          max_target_length, compact_width);
      break;
    case 3:
      invoke_calculate_max_alpha<16, 64>(
          stream, alpha, trace, emission_scores, compact_transition_scores,
          node_lengths, target_lengths, batch_size, node_count,
          max_target_length, compact_width);
      break;
    case 4:
      invoke_calculate_max_alpha<32, 32>(
          stream, alpha, trace, emission_scores, compact_transition_scores,
          node_lengths, target_lengths, batch_size, node_count,
          max_target_length, compact_width);
      break;
    default:
      TORCH_CHECK(false, "Viterbi-alignment forward config must be in 1..4");
  }
  invoke_backtrace<512>(
      stream, alpha, terminal_scores, alignment_nodes, trace,
      node_lengths, target_lengths, batch_size);
  return std::make_tuple(terminal_scores, alignment_nodes);
}
