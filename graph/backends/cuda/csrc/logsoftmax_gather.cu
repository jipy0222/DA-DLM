// ##########################################################################
// Copyright (C) 2022 COAI @ Tsinghua University
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//         http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// ##########################################################################

// Block-reduction algorithm adapted from OneFlow's softmax.cuh at
// https://github.com/Oneflow-Inc/oneflow/blob/ed8d26bad3764145a24a54e7e767765aaa87d7f8/oneflow/core/cuda/softmax.cuh
// (Copyright The OneFlow Authors; Apache License 2.0).

#include <algorithm>
#include <cstdint>
#include <limits>
#include <type_traits>

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <cub/cub.cuh>
#include <cuda_runtime.h>
#include <torch/extension.h>


template <typename scalar_t>
struct LogSoftmaxGatherMax {
  __device__ __forceinline__ scalar_t operator()(
      const scalar_t& lhs, const scalar_t& rhs) const {
    return max(lhs, rhs);
  }
};

template <typename scalar_t>
struct LogSoftmaxGatherSum {
  __device__ __forceinline__ scalar_t operator()(
      const scalar_t& lhs, const scalar_t& rhs) const {
    return lhs + rhs;
  }
};


template <typename scalar_t, int THREADS, bool REQUIRE_GRADIENT>
__global__ void logsoftmax_gather_kernel(
    scalar_t* workspace,
    const int64_t* select_idx,
    scalar_t* selected,
    int64_t rows,
    int64_t node_count,
    int64_t vocab_size,
    int64_t target_count,
    int64_t index_stride_0,
    int64_t index_stride_1,
    int64_t index_stride_2) {
  using BlockReduce = cub::BlockReduce<scalar_t, THREADS>;
  __shared__ typename BlockReduce::TempStorage reduction_storage;
  __shared__ scalar_t broadcast_result;

  const int tid = threadIdx.x;
  for (int64_t row = blockIdx.x; row < rows; row += gridDim.x) {
    scalar_t* row_logits = workspace + row * vocab_size;
    scalar_t thread_max = -std::numeric_limits<scalar_t>::infinity();
    for (int64_t col = tid; col < vocab_size; col += THREADS) {
      thread_max = max(thread_max, row_logits[col]);
    }
    const scalar_t reduced_max =
        BlockReduce(reduction_storage).Reduce(
            thread_max, LogSoftmaxGatherMax<scalar_t>());
    if (tid == 0) {
      broadcast_result = reduced_max;
    }
    __syncthreads();
    const scalar_t row_max = broadcast_result;

    // The next reduction reuses reduction_storage, so every lane must have
    // finished reading the preceding reduction and its broadcast first.
    __syncthreads();
    scalar_t thread_sum = scalar_t(0);
    for (int64_t col = tid; col < vocab_size; col += THREADS) {
      thread_sum += exp(row_logits[col] - row_max);
    }
    const scalar_t reduced_sum =
        BlockReduce(reduction_storage).Reduce(
            thread_sum, LogSoftmaxGatherSum<scalar_t>());
    if (tid == 0) {
      broadcast_result = reduced_sum;
    }
    __syncthreads();
    const scalar_t row_sum = broadcast_result;
    const scalar_t log_sum = log(row_sum);

    if (target_count > 0) {
      const int64_t batch = row / node_count;
      const int64_t node = row % node_count;
      const int64_t* row_indices =
          select_idx + batch * index_stride_0 + node * index_stride_1;
      for (int64_t target = tid; target < target_count; target += THREADS) {
        const int64_t index = row_indices[target * index_stride_2];
        CUDA_KERNEL_ASSERT(
            index >= 0 && index < vocab_size &&
            "logsoftmax_gather: select_idx is out of vocabulary range");
        selected[row * target_count + target] =
            (row_logits[index] - row_max) - log_sum;
      }
    }

    if (REQUIRE_GRADIENT) {
      // Gather must finish reading the original logits before any lane
      // overwrites them with softmax probabilities for backward.
      __syncthreads();
      for (int64_t col = tid; col < vocab_size; col += THREADS) {
        row_logits[col] = exp(row_logits[col] - row_max) / row_sum;
      }
    }
    // A grid-stride block may process another row. Keep the shared reducer
    // and broadcast storage untouched until every lane has finished this row.
    __syncthreads();
  }
}


template <typename scalar_t, bool REQUIRE_GRADIENT>
void launch_logsoftmax_gather(
    torch::Tensor& workspace,
    const torch::Tensor& select_idx,
    torch::Tensor& selected,
    int64_t rows,
    int64_t node_count,
    int64_t vocab_size,
    int64_t target_count,
    int threads,
    int blocks,
    cudaStream_t stream) {
  const auto launch = [&](auto block_size) {
    constexpr int THREADS = decltype(block_size)::value;
    logsoftmax_gather_kernel<scalar_t, THREADS, REQUIRE_GRADIENT>
        <<<blocks, THREADS, 0, stream>>>(
            workspace.data_ptr<scalar_t>(),
            select_idx.data_ptr<int64_t>(),
            selected.data_ptr<scalar_t>(),
            rows,
            node_count,
            vocab_size,
            target_count,
            select_idx.stride(0),
            select_idx.stride(1),
            select_idx.stride(2));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  };

  switch (threads) {
    case 128: launch(std::integral_constant<int, 128>{}); break;
    case 256: launch(std::integral_constant<int, 256>{}); break;
    case 512: launch(std::integral_constant<int, 512>{}); break;
    case 1024: launch(std::integral_constant<int, 1024>{}); break;
  }
}


torch::Tensor logsoftmax_gather(
    torch::Tensor workspace,
    const torch::Tensor& select_idx,
    bool require_gradient,
    int threads) {
  TORCH_CHECK(workspace.is_cuda(), "workspace must be a CUDA tensor");
  TORCH_CHECK(select_idx.is_cuda(), "select_idx must be a CUDA tensor");
  TORCH_CHECK(
      workspace.device() == select_idx.device(),
      "workspace and select_idx must be on the same CUDA device");
  TORCH_CHECK(workspace.dim() == 3, "workspace must have shape [R, N, V]");
  TORCH_CHECK(select_idx.dim() == 3, "select_idx must have shape [R, N, T]");
  TORCH_CHECK(workspace.is_contiguous(), "workspace must be contiguous");
  TORCH_CHECK(
      workspace.scalar_type() == at::kFloat ||
          workspace.scalar_type() == at::kDouble,
      "workspace must have float32 or float64 dtype");
  TORCH_CHECK(select_idx.scalar_type() == at::kLong, "select_idx must be int64");
  TORCH_CHECK(
      select_idx.size(0) == workspace.size(0) &&
          select_idx.size(1) == workspace.size(1),
      "select_idx must match workspace in its first two dimensions");
  TORCH_CHECK(
      threads == 128 || threads == 256 || threads == 512 || threads == 1024,
      "logsoftmax_gather threads must be 128, 256, 512, or 1024");

  const c10::cuda::CUDAGuard device_guard(workspace.device());
  const int64_t batch_size = workspace.size(0);
  const int64_t node_count = workspace.size(1);
  const int64_t vocab_size = workspace.size(2);
  const int64_t target_count = select_idx.size(2);
  const int64_t rows = batch_size * node_count;
  TORCH_CHECK(
      vocab_size > 0 || target_count == 0 || rows == 0,
      "cannot select from an empty vocabulary");

  auto selected = torch::empty(
      {batch_size, node_count, target_count}, workspace.options());
  if (rows == 0 || vocab_size == 0 ||
      (target_count == 0 && !require_gradient)) {
    return selected;
  }

  int sm_count = 0;
  int threads_per_sm = 0;
  C10_CUDA_CHECK(cudaDeviceGetAttribute(
      &sm_count, cudaDevAttrMultiProcessorCount, workspace.get_device()));
  C10_CUDA_CHECK(cudaDeviceGetAttribute(
      &threads_per_sm, cudaDevAttrMaxThreadsPerMultiProcessor,
      workspace.get_device()));
  constexpr int64_t waves = 32;
  const int64_t resident_blocks =
      std::max<int64_t>(1, static_cast<int64_t>(sm_count) *
                              threads_per_sm / threads);
  const int blocks = static_cast<int>(std::min(rows, resident_blocks * waves));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  AT_DISPATCH_FLOATING_TYPES(
      workspace.scalar_type(), "logsoftmax_gather", [&] {
        if (require_gradient) {
          launch_logsoftmax_gather<scalar_t, true>(
              workspace, select_idx, selected, rows, node_count,
              vocab_size, target_count, threads, blocks, stream);
        } else {
          launch_logsoftmax_gather<scalar_t, false>(
              workspace, select_idx, selected, rows, node_count,
              vocab_size, target_count, threads, blocks, stream);
        }
      });
  return selected;
}
