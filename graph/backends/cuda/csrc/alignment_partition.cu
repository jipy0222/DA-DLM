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
// ###########################################################################
// Uses current CUDA streams and device guards.

#include <cmath>
#include <limits>
#include <tuple>

#include <c10/macros/Macros.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <torch/extension.h>
#include "utilities.h"

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")

#define BLOCK_BUCKET 16

// The launch dimensions below are part of the compiler's resource contract.
// Without this bound, sm_120 FP64 builds can allocate more registers than a
// 1024-thread block permits and fail with cudaErrorLaunchOutOfResources.
template<class scalar_t, int TRANS_BLOCK_SIZE, int SEQ_BLOCK_SIZE, class Accessor1, class Accessor2>
__global__ __launch_bounds__(TRANS_BLOCK_SIZE * SEQ_BLOCK_SIZE)
void calculate_alpha_kernel(
    volatile int *bucket_queue, volatile int *accomplish_queue,
    Accessor1 alpha,
    Accessor1 match_all,
    Accessor1 links,
    Accessor2 output_length,
    Accessor2 target_length,
    int bsz, int prelen, int tarlen, int translen, int n_seg)
{
    int bucket_idx = blockIdx.y % BLOCK_BUCKET;
    __shared__ volatile int bucket_no;

    bool main_thread = threadIdx.x == 0 && threadIdx.y == 0;
    if (main_thread){
        // obtain task id
        bucket_no = atomicAdd((int*)bucket_queue + bucket_idx, 1);
    }
    __syncthreads();

    int ticket_no = bucket_no * BLOCK_BUCKET + bucket_idx;
    int batch_id = ticket_no % bsz;
    int seg_id = ticket_no / bsz;
    int pos = seg_id * SEQ_BLOCK_SIZE + threadIdx.y;

    int output_len = output_length[batch_id];
    int target_len = target_length[batch_id];

    CUDA_KERNEL_ASSERT(target_len >= 2 && output_len >= 2 && "alignment partition: target/node length must be at least 2");
    CUDA_KERNEL_ASSERT(target_len <= tarlen && "alignment partition: target length exceeds padded target dimension");
    CUDA_KERNEL_ASSERT(output_len <= prelen && "alignment partition: node length exceeds padded node dimension");
    CUDA_KERNEL_ASSERT(output_len >= target_len && "alignment partition: graph has fewer nodes than target positions");

    // t = 0
    {
        if(pos >= output_len) return;
        if(main_thread){
            alpha[batch_id][0][0] = match_all[batch_id][0][0];
        }
        __threadfence();
        __syncthreads();
        if(main_thread){
            atomicAdd((int*)accomplish_queue + batch_id * n_seg + seg_id, 1);
        }
    }

    for(int t = 1; t < target_len; t++){
        if(pos + t >= output_len) return;
        if (main_thread && seg_id != 0){
            while(accomplish_queue[batch_id * n_seg + seg_id - 1] < t); // wait for previous segment to accomplish
        }
        __syncthreads();
        scalar_t maxval = -std::numeric_limits<scalar_t>::infinity();
        int nowpos = pos + t;
        int maxdelta = min(nowpos, translen);
        for(int delta = threadIdx.x + 1; delta <= maxdelta; delta += TRANS_BLOCK_SIZE){
            int lastpos = nowpos - delta;
            scalar_t nextval = alpha[batch_id][t - 1][lastpos] + links[batch_id][lastpos][delta - 1];
            if(nextval > maxval) maxval = nextval;
        }
        unsigned shfl_mask = __activemask();
        if_constexpr (TRANS_BLOCK_SIZE > 16) {scalar_t nextval = __shfl_down_sync(shfl_mask, maxval, 16, TRANS_BLOCK_SIZE); if(nextval > maxval) maxval = nextval;}
        if_constexpr (TRANS_BLOCK_SIZE > 8) {scalar_t nextval = __shfl_down_sync(shfl_mask, maxval, 8, TRANS_BLOCK_SIZE); if(nextval > maxval) maxval = nextval;}
        if_constexpr (TRANS_BLOCK_SIZE > 4) {scalar_t nextval = __shfl_down_sync(shfl_mask, maxval, 4, TRANS_BLOCK_SIZE); if(nextval > maxval) maxval = nextval;}
        if_constexpr (TRANS_BLOCK_SIZE > 2) {scalar_t nextval = __shfl_down_sync(shfl_mask, maxval, 2, TRANS_BLOCK_SIZE); if(nextval > maxval) maxval = nextval;}
        if_constexpr (TRANS_BLOCK_SIZE > 1) {scalar_t nextval = __shfl_down_sync(shfl_mask, maxval, 1, TRANS_BLOCK_SIZE); if(nextval > maxval) maxval = nextval;}
        maxval = __shfl_sync(shfl_mask, maxval, 0, TRANS_BLOCK_SIZE);
        shfl_mask = __ballot_sync(shfl_mask, !isinf(maxval));
        scalar_t res;
        if (isinf(maxval)){
            res = maxval;
        }else{
            scalar_t sumval = 0;
            for(int delta = threadIdx.x + 1; delta <= maxdelta; delta += TRANS_BLOCK_SIZE){
                int lastpos = nowpos - delta;
                sumval += exp(alpha[batch_id][t - 1][lastpos] + links[batch_id][lastpos][delta - 1] - maxval);
            }
            if_constexpr (TRANS_BLOCK_SIZE > 16) sumval += __shfl_down_sync(shfl_mask, sumval, 16, TRANS_BLOCK_SIZE);
            if_constexpr (TRANS_BLOCK_SIZE > 8) sumval += __shfl_down_sync(shfl_mask, sumval, 8, TRANS_BLOCK_SIZE);
            if_constexpr (TRANS_BLOCK_SIZE > 4) sumval += __shfl_down_sync(shfl_mask, sumval, 4, TRANS_BLOCK_SIZE);
            if_constexpr (TRANS_BLOCK_SIZE > 2) sumval += __shfl_down_sync(shfl_mask, sumval, 2, TRANS_BLOCK_SIZE);
            if_constexpr (TRANS_BLOCK_SIZE > 1) sumval += __shfl_down_sync(shfl_mask, sumval, 1, TRANS_BLOCK_SIZE);
            res = log(sumval) + maxval + match_all[batch_id][t][nowpos];
        }
        if(threadIdx.x == 0){
            alpha[batch_id][t][nowpos] = res;
        }

        __threadfence();
        __syncthreads();
        if (main_thread){
            atomicAdd((int*)accomplish_queue + batch_id * n_seg + seg_id, 1);
        }
    }
}


template<int TRANS_BLOCK_SIZE, int SEQ_BLOCK_SIZE>
void invoke_calculate_alpha(cudaStream_t stream, torch::Tensor &alpha, const torch::Tensor &match_all, const torch::Tensor &links, const torch::Tensor &output_length, const torch::Tensor &target_length, \
    int bsz, int prelen, int tarlen, int translen)
{
    int n_seg = (prelen - 1) / SEQ_BLOCK_SIZE + 1;
    dim3 dimGrid(1, n_seg * bsz);
    dim3 dimBlock(TRANS_BLOCK_SIZE, SEQ_BLOCK_SIZE);

    int *bucket_queue, *accomplish_queue;
    auto tmp_tensor = at::zeros({BLOCK_BUCKET + bsz * n_seg}, match_all.options().dtype(at::kInt));
    bucket_queue = tmp_tensor.data_ptr<int>();
    accomplish_queue = bucket_queue + BLOCK_BUCKET;
    static_assert(TRANS_BLOCK_SIZE <= 32, "TRANS_BLOCK_SIZE should be less than warp size");

    AT_DISPATCH_FLOATING_TYPES(
        match_all.scalar_type(), "invoke_calculate_alpha", [&] {
            alpha.fill_(-std::numeric_limits<scalar_t>::infinity());
            calculate_alpha_kernel<scalar_t, TRANS_BLOCK_SIZE, SEQ_BLOCK_SIZE><<<dimGrid, dimBlock, 0, stream>>>(
                bucket_queue, accomplish_queue,
                alpha.packed_accessor64<scalar_t, 3>(),
                match_all.packed_accessor64<scalar_t, 3>(),
                links.packed_accessor64<scalar_t, 3>(),
                output_length.packed_accessor64<int64_t, 1>(),
                target_length.packed_accessor64<int64_t, 1>(),
                bsz, prelen, tarlen, translen, n_seg
            );
            C10_CUDA_KERNEL_LAUNCH_CHECK();
        }
    );
}

template<class scalar_t, int TRANS_BLOCK_SIZE, int SEQ_BLOCK_SIZE, class Accessor1, class Accessor2>
__global__ __launch_bounds__(TRANS_BLOCK_SIZE * SEQ_BLOCK_SIZE)
void calculate_beta_kernel(
    volatile int *bucket_queue, volatile int *accomplish_queue,
    Accessor1 beta,
    Accessor1 match_all,
    Accessor1 links,
    Accessor2 output_length,
    Accessor2 target_length,
    int bsz, int prelen, int tarlen, int translen, int n_seg)
{
    int bucket_idx = blockIdx.y % BLOCK_BUCKET;
    __shared__ volatile int bucket_no;

    bool main_thread = threadIdx.x == 0 && threadIdx.y == 0;
    if (main_thread){
        // obtain task id
        bucket_no = atomicAdd((int*)bucket_queue + bucket_idx, 1);
    }
    __syncthreads();

    int ticket_no = bucket_no * BLOCK_BUCKET + bucket_idx;
    int batch_id = ticket_no % bsz;
    int seg_id = ticket_no / bsz;
    int pos = (n_seg - 1 - seg_id) * SEQ_BLOCK_SIZE + threadIdx.y;

    int output_len = output_length[batch_id];
    int target_len = target_length[batch_id];

    // t = target_len - 1
    {
        int nowpos = pos + target_len - 1;
        if(nowpos == output_len - 1){
            beta[batch_id][target_len - 1][nowpos] = match_all[batch_id][target_len - 1][nowpos];
        }
        __threadfence();
        __syncthreads();
        if(main_thread){
            atomicAdd((int*)accomplish_queue + batch_id * n_seg + seg_id, 1);
        }
    }

    for(int t = target_len - 2; t >= 0; t--){
        if (main_thread && seg_id != 0){
            while(accomplish_queue[batch_id * n_seg + seg_id - 1] < target_len - 1 - t); // wait for previous segment to accomplish
        }
        __syncthreads();
        int nowpos = pos + t;
        if(nowpos < output_len){
            scalar_t maxval = -std::numeric_limits<scalar_t>::infinity();
            int maxdelta = min(output_len - 1 - nowpos, translen);
            for(int delta = threadIdx.x + 1; delta <= maxdelta; delta += TRANS_BLOCK_SIZE){
                int lastpos = nowpos + delta;
                scalar_t nextval = beta[batch_id][t + 1][lastpos] + links[batch_id][nowpos][delta - 1];
                if(nextval > maxval) maxval = nextval;
            }
            unsigned shfl_mask = __activemask();
            if_constexpr (TRANS_BLOCK_SIZE > 16) {scalar_t nextval = __shfl_down_sync(shfl_mask, maxval, 16, TRANS_BLOCK_SIZE); if(nextval > maxval) maxval = nextval;}
            if_constexpr (TRANS_BLOCK_SIZE > 8) {scalar_t nextval = __shfl_down_sync(shfl_mask, maxval, 8, TRANS_BLOCK_SIZE); if(nextval > maxval) maxval = nextval;}
            if_constexpr (TRANS_BLOCK_SIZE > 4) {scalar_t nextval = __shfl_down_sync(shfl_mask, maxval, 4, TRANS_BLOCK_SIZE); if(nextval > maxval) maxval = nextval;}
            if_constexpr (TRANS_BLOCK_SIZE > 2) {scalar_t nextval = __shfl_down_sync(shfl_mask, maxval, 2, TRANS_BLOCK_SIZE); if(nextval > maxval) maxval = nextval;}
            if_constexpr (TRANS_BLOCK_SIZE > 1) {scalar_t nextval = __shfl_down_sync(shfl_mask, maxval, 1, TRANS_BLOCK_SIZE); if(nextval > maxval) maxval = nextval;}
            maxval = __shfl_sync(shfl_mask, maxval, 0, TRANS_BLOCK_SIZE);

            shfl_mask = __ballot_sync(shfl_mask, !isinf(maxval));
            scalar_t res;
            if (isinf(maxval)){
                res = maxval;
            }else{
                scalar_t sumval = 0;
                for(int delta = threadIdx.x + 1; delta <= maxdelta; delta += TRANS_BLOCK_SIZE){
                    int lastpos = nowpos + delta;
                    sumval += exp(beta[batch_id][t + 1][lastpos] + links[batch_id][nowpos][delta - 1] - maxval);
                }
                if_constexpr (TRANS_BLOCK_SIZE > 16) sumval += __shfl_down_sync(shfl_mask, sumval, 16, TRANS_BLOCK_SIZE);
                if_constexpr (TRANS_BLOCK_SIZE > 8) sumval += __shfl_down_sync(shfl_mask, sumval, 8, TRANS_BLOCK_SIZE);
                if_constexpr (TRANS_BLOCK_SIZE > 4) sumval += __shfl_down_sync(shfl_mask, sumval, 4, TRANS_BLOCK_SIZE);
                if_constexpr (TRANS_BLOCK_SIZE > 2) sumval += __shfl_down_sync(shfl_mask, sumval, 2, TRANS_BLOCK_SIZE);
                if_constexpr (TRANS_BLOCK_SIZE > 1) sumval += __shfl_down_sync(shfl_mask, sumval, 1, TRANS_BLOCK_SIZE);
                res = log(sumval) + maxval + match_all[batch_id][t][nowpos];
            }
            if(threadIdx.x == 0){
                beta[batch_id][t][nowpos] = res;
            }
        }

        __threadfence();
        __syncthreads();
        if (main_thread){
            atomicAdd((int*)accomplish_queue + batch_id * n_seg + seg_id, 1);
        }
    }
}


template<int TRANS_BLOCK_SIZE, int SEQ_BLOCK_SIZE>
void invoke_calculate_beta(cudaStream_t stream, torch::Tensor &beta, const torch::Tensor &match_all, const torch::Tensor &links, const torch::Tensor &output_length, const torch::Tensor &target_length, \
    int bsz, int prelen, int tarlen, int translen)
{
    int n_seg = (prelen - 1) / SEQ_BLOCK_SIZE + 1;
    dim3 dimGrid(1, n_seg * bsz);
    dim3 dimBlock(TRANS_BLOCK_SIZE, SEQ_BLOCK_SIZE);

    int *bucket_queue, *accomplish_queue;
    auto tmp_tensor = at::zeros({BLOCK_BUCKET + bsz * n_seg}, match_all.options().dtype(at::kInt));
    bucket_queue = tmp_tensor.data_ptr<int>();
    accomplish_queue = bucket_queue + BLOCK_BUCKET;
    static_assert(TRANS_BLOCK_SIZE <= 32, "TRANS_BLOCK_SIZE should be less than warp size");

    AT_DISPATCH_FLOATING_TYPES(
        match_all.scalar_type(), "invoke_calculate_beta", [&] {
            beta.fill_(-std::numeric_limits<scalar_t>::infinity());
            calculate_beta_kernel<scalar_t, TRANS_BLOCK_SIZE, SEQ_BLOCK_SIZE><<<dimGrid, dimBlock, 0, stream>>>(
                bucket_queue, accomplish_queue,
                beta.packed_accessor64<scalar_t, 3>(),
                match_all.packed_accessor64<scalar_t, 3>(),
                links.packed_accessor64<scalar_t, 3>(),
                output_length.packed_accessor64<int64_t, 1>(),
                target_length.packed_accessor64<int64_t, 1>(),
                bsz, prelen, tarlen, translen, n_seg
            );
            C10_CUDA_KERNEL_LAUNCH_CHECK();
        }
    );
}


std::tuple<torch::Tensor, torch::Tensor> dag_loss(const torch::Tensor &match_all, const torch::Tensor &links,
    const torch::Tensor &output_length, const torch::Tensor &target_length, bool require_gradient,
    int config)
{
    CHECK_CUDA(match_all);  // bsz * tarlen * prelen
    CHECK_CUDA(links);   // bsz * prelen * translen
    CHECK_CUDA(output_length); // bsz
    CHECK_CUDA(target_length); // bsz
    TORCH_CHECK(match_all.dim() == 3, "match_all dim != 3");
    TORCH_CHECK(links.dim() == 3, "links dim != 3");
    TORCH_CHECK(output_length.dim() == 1, "output_length dim != 1");
    TORCH_CHECK(target_length.dim() == 1, "target_length dim != 1");
    CHECK_CONTIGUOUS(match_all);
    CHECK_CONTIGUOUS(links);
    CHECK_CONTIGUOUS(output_length);
    CHECK_CONTIGUOUS(target_length);
    TORCH_CHECK(match_all.device() == links.device(), "score tensors must share a CUDA device");
    TORCH_CHECK(match_all.device() == output_length.device(), "length tensors must share the score device");
    TORCH_CHECK(match_all.device() == target_length.device(), "length tensors must share the score device");
    TORCH_CHECK(match_all.scalar_type() == links.scalar_type(), "score tensors must share a dtype");

    auto bsz = match_all.size(0);
    auto prelen = match_all.size(2);
    auto tarlen = match_all.size(1);
    auto translen = links.size(2);
    TORCH_CHECK(links.size(0) == bsz && output_length.size(0) == bsz && target_length.size(0) == bsz, "batch size not match");
    TORCH_CHECK(links.size(1) == prelen, "prelen not match");
    TORCH_CHECK(translen >= 1 && translen < prelen, "compact transition width must be in [1, prelen)");
    TORCH_CHECK(output_length.scalar_type() == at::kLong && target_length.scalar_type() == at::kLong, "length should be long");

    const c10::cuda::CUDAGuard device_guard(match_all.device());
    cudaStream_t current_stream = at::cuda::getCurrentCUDAStream();

    // calculate alpha
    auto alpha = at::empty({bsz, tarlen, prelen}, match_all.options());
    torch::Tensor beta = require_gradient
        ? at::empty({bsz, tarlen, prelen}, match_all.options())
        : at::empty({0}, match_all.options());


    switch(config){
        case 1: invoke_calculate_alpha<4, 256>(current_stream, alpha, match_all, links, output_length, target_length, bsz, prelen, tarlen, translen); break;
        case 2: invoke_calculate_alpha<8, 128>(current_stream, alpha, match_all, links, output_length, target_length, bsz, prelen, tarlen, translen); break;
        case 3: invoke_calculate_alpha<16, 64>(current_stream, alpha, match_all, links, output_length, target_length, bsz, prelen, tarlen, translen); break;
        case 4: invoke_calculate_alpha<32, 32>(current_stream, alpha, match_all, links, output_length, target_length, bsz, prelen, tarlen, translen); break;
        default: TORCH_CHECK(false, "config should be 1~4");
    }

    // calculate beta
    if(require_gradient){
        switch(config){
            case 1: invoke_calculate_beta<4, 256>(current_stream, beta, match_all, links, output_length, target_length, bsz, prelen, tarlen, translen); break;
            case 2: invoke_calculate_beta<8, 128>(current_stream, beta, match_all, links, output_length, target_length, bsz, prelen, tarlen, translen); break;
            case 3: invoke_calculate_beta<16, 64>(current_stream, beta, match_all, links, output_length, target_length, bsz, prelen, tarlen, translen); break;
            case 4: invoke_calculate_beta<32, 32>(current_stream, beta, match_all, links, output_length, target_length, bsz, prelen, tarlen, translen); break;
            default: TORCH_CHECK(false, "config should be 1~4");
        }
    }

    return std::make_tuple(alpha, beta);
}


template<class scalar_t, int TAR_BLOCK_SIZE, int PRE_BLOCK_SIZE, int SWEEP_SIZE, class Accessor1, class Accessor2, class Accessor3>
__global__ void calculate_grad_match_all_kernel(
    Accessor1 grad_match_all,
    Accessor2 grad_output,
    Accessor1 alpha,
    Accessor1 beta,
    Accessor1 match_all,
    Accessor3 output_length,
    Accessor3 target_length,
    int prelen, int tarlen)
{
    int y = blockIdx.y * TAR_BLOCK_SIZE + threadIdx.y;
    int batch_id = blockIdx.z;

    if(y >= tarlen || y >= target_length[batch_id]) return;
    #pragma unroll
    for(int i = 0; i < SWEEP_SIZE; i++){
        int x = blockIdx.x * PRE_BLOCK_SIZE * SWEEP_SIZE + threadIdx.x + PRE_BLOCK_SIZE * i;
        if(x >= prelen || x >= output_length[batch_id]) continue;
        if(isinf(match_all[batch_id][y][x]) || isinf(beta[batch_id][0][0])){
            grad_match_all[batch_id][y][x] = 0;
        }else{
            grad_match_all[batch_id][y][x] = exp(alpha[batch_id][y][x] + beta[batch_id][y][x] - match_all[batch_id][y][x] - beta[batch_id][0][0]) * grad_output[batch_id];
        }
    }
}

template<int TAR_BLOCK_SIZE, int PRE_BLOCK_SIZE, int SWEEP_SIZE>
void invoke_calculate_grad_match_all(cudaStream_t stream, torch::Tensor &grad_match_all, const torch::Tensor &grad_output,
    const torch::Tensor &alpha, const torch::Tensor &beta, const torch::Tensor &match_all,
    const torch::Tensor &output_length, const torch::Tensor &target_length,
    int bsz, int prelen, int tarlen)
{
    int n_tar = (tarlen - 1) / TAR_BLOCK_SIZE + 1;
    int n_pre = (prelen - 1) / PRE_BLOCK_SIZE / SWEEP_SIZE + 1;
    dim3 dimGrid(n_pre, n_tar, bsz);
    dim3 dimBlock(PRE_BLOCK_SIZE, TAR_BLOCK_SIZE, 1);

    AT_DISPATCH_FLOATING_TYPES(
        match_all.scalar_type(), "invoke_calculate_grad_match_all", [&] {
            calculate_grad_match_all_kernel<scalar_t, TAR_BLOCK_SIZE, PRE_BLOCK_SIZE, SWEEP_SIZE><<<dimGrid, dimBlock, 0, stream>>>(
                grad_match_all.packed_accessor64<scalar_t, 3>(),
                grad_output.packed_accessor64<scalar_t, 1>(),
                alpha.packed_accessor64<scalar_t, 3>(),
                beta.packed_accessor64<scalar_t, 3>(),
                match_all.packed_accessor64<scalar_t, 3>(),
                output_length.packed_accessor64<int64_t, 1>(),
                target_length.packed_accessor64<int64_t, 1>(),
                prelen, tarlen
            );
            C10_CUDA_KERNEL_LAUNCH_CHECK();
        }
    );
}


template<class scalar_t, int TRANS_BLOCK_SIZE, int PRE_BLOCK_SIZE, int WARP_SIZE, class Accessor1, class Accessor2, class Accessor3>
__global__ void calculate_grad_links_kernel(
    Accessor1 grad_links,
    Accessor2 grad_output,
    Accessor1 alpha,
    Accessor1 beta,
    Accessor1 links,
    Accessor3 output_length,
    Accessor3 target_length,
    int translen)
{
    int n_trans = ((translen - 1) / TRANS_BLOCK_SIZE + 1);
    int pre_block_idx = blockIdx.y / n_trans;
    int trans_block_idx = blockIdx.y % n_trans;
    int pre_thread_idx = threadIdx.y / TRANS_BLOCK_SIZE;
    int trans_thread_idx = threadIdx.y % TRANS_BLOCK_SIZE;

    int pre_id = pre_block_idx * PRE_BLOCK_SIZE + pre_thread_idx;
    int trans_id = trans_block_idx * TRANS_BLOCK_SIZE + trans_thread_idx;
    int batch_id = blockIdx.z;

    int presize = output_length[batch_id];
    int tarsize = target_length[batch_id];

    if(pre_id >= presize || trans_id >= translen) return;
    int next_pre_id = pre_id + trans_id + 1;
    if(next_pre_id >= presize || isinf(beta[batch_id][0][0])){
        if(threadIdx.x == 0) grad_links[batch_id][pre_id][trans_id] = 0;
        return;
    }

    scalar_t grad_tmp = 0;
    scalar_t extraadd = links[batch_id][pre_id][trans_id] - beta[batch_id][0][0];
    #pragma unroll(4)
    for(int i = 0; i < (tarsize - 1) / WARP_SIZE + 1; i++){
        int tar_id = threadIdx.x + WARP_SIZE * i;
        if(tar_id + 1 >= tarsize) continue;
        grad_tmp += exp(alpha[batch_id][tar_id][pre_id] + beta[batch_id][tar_id + 1][next_pre_id] + extraadd);
    }

    unsigned shfl_mask = __activemask();
    if_constexpr (WARP_SIZE > 16) grad_tmp += __shfl_down_sync(shfl_mask, grad_tmp, 16, WARP_SIZE);
    if_constexpr (WARP_SIZE > 8) grad_tmp += __shfl_down_sync(shfl_mask, grad_tmp, 8, WARP_SIZE);
    if_constexpr (WARP_SIZE > 4) grad_tmp += __shfl_down_sync(shfl_mask, grad_tmp, 4, WARP_SIZE);
    if_constexpr (WARP_SIZE > 2) grad_tmp += __shfl_down_sync(shfl_mask, grad_tmp, 2, WARP_SIZE);
    if_constexpr (WARP_SIZE > 1) grad_tmp += __shfl_down_sync(shfl_mask, grad_tmp, 1, WARP_SIZE);
    if(threadIdx.x == 0) grad_links[batch_id][pre_id][trans_id] = grad_tmp * grad_output[batch_id];
}

template<int TRANS_BLOCK_SIZE, int PRE_BLOCK_SIZE, int WARP_SIZE>
void invoke_calculate_grad_links(cudaStream_t stream, torch::Tensor &grad_links, const torch::Tensor &grad_output,
    const torch::Tensor &alpha, const torch::Tensor &beta, const torch::Tensor &links,
    const torch::Tensor &output_length, const torch::Tensor &target_length,
    int bsz, int prelen, int translen)
{
    int n_block = ((translen - 1) / TRANS_BLOCK_SIZE + 1) * ((prelen - 1) / PRE_BLOCK_SIZE + 1);
    dim3 dimGrid(1, n_block, bsz);
    dim3 dimBlock(WARP_SIZE, TRANS_BLOCK_SIZE * PRE_BLOCK_SIZE, 1);
    static_assert(WARP_SIZE <= 32, "WARP_SIZE should be less than warp size");

    AT_DISPATCH_FLOATING_TYPES(
        links.scalar_type(), "invoke_calculate_grad_links", [&] {
            calculate_grad_links_kernel<scalar_t, TRANS_BLOCK_SIZE, PRE_BLOCK_SIZE, WARP_SIZE><<<dimGrid, dimBlock, 0, stream>>>(
                grad_links.packed_accessor64<scalar_t, 3>(),
                grad_output.packed_accessor64<scalar_t, 1>(),
                alpha.packed_accessor64<scalar_t, 3>(),
                beta.packed_accessor64<scalar_t, 3>(),
                links.packed_accessor64<scalar_t, 3>(),
                output_length.packed_accessor64<int64_t, 1>(),
                target_length.packed_accessor64<int64_t, 1>(),
                translen
            );
            C10_CUDA_KERNEL_LAUNCH_CHECK();
        }
    );
}

std::tuple<torch::Tensor, torch::Tensor> dag_loss_backward(const torch::Tensor &grad_output, const torch::Tensor &alpha, const torch::Tensor &beta,
            const torch::Tensor &match_all, const torch::Tensor &links, const torch::Tensor &output_length, const torch::Tensor &target_length,
            int config1, int config2)
{
    CHECK_CUDA(grad_output);
    CHECK_CUDA(alpha);
    CHECK_CUDA(beta);
    CHECK_CUDA(match_all);
    CHECK_CUDA(links);
    CHECK_CUDA(output_length);
    CHECK_CUDA(target_length);
    CHECK_CONTIGUOUS(grad_output);
    CHECK_CONTIGUOUS(alpha);
    CHECK_CONTIGUOUS(beta);
    CHECK_CONTIGUOUS(match_all);
    CHECK_CONTIGUOUS(links);
    CHECK_CONTIGUOUS(output_length);
    CHECK_CONTIGUOUS(target_length);
    TORCH_CHECK(grad_output.dim() == 1, "grad_output dim != 1");
    TORCH_CHECK(alpha.dim() == 3 && beta.dim() == 3, "alpha/beta dim != 3");
    TORCH_CHECK(match_all.dim() == 3 && links.dim() == 3, "score tensor dim != 3");
    TORCH_CHECK(output_length.dim() == 1 && target_length.dim() == 1, "length dim != 1");
    TORCH_CHECK(match_all.device() == grad_output.device(), "saved tensors and grad_output must share a device");
    TORCH_CHECK(match_all.device() == alpha.device() && match_all.device() == beta.device(), "saved DP tensors must share a device");
    TORCH_CHECK(match_all.device() == links.device(), "saved score tensors must share a device");
    TORCH_CHECK(match_all.device() == output_length.device() && match_all.device() == target_length.device(), "saved lengths must share the score device");
    TORCH_CHECK(match_all.scalar_type() == grad_output.scalar_type(), "grad_output dtype must match scores");
    TORCH_CHECK(match_all.scalar_type() == alpha.scalar_type() && match_all.scalar_type() == beta.scalar_type(), "saved DP dtype must match scores");
    TORCH_CHECK(match_all.scalar_type() == links.scalar_type(), "saved score dtypes must match");

    auto bsz = match_all.size(0);
    auto prelen = match_all.size(2);
    auto tarlen = match_all.size(1);
    auto translen = links.size(2);
    TORCH_CHECK(grad_output.size(0) == bsz, "grad_output batch size not match");
    TORCH_CHECK(alpha.sizes() == match_all.sizes() && beta.sizes() == match_all.sizes(), "saved DP shape not match");
    TORCH_CHECK(links.size(0) == bsz && links.size(1) == prelen, "links shape not match");
    TORCH_CHECK(output_length.size(0) == bsz && target_length.size(0) == bsz, "length batch size not match");
    TORCH_CHECK(translen >= 1 && translen < prelen, "compact transition width must be in [1, prelen)");
    TORCH_CHECK(output_length.scalar_type() == at::kLong && target_length.scalar_type() == at::kLong, "length should be long");

    auto grad_match_all = at::zeros({bsz, tarlen, prelen}, match_all.options());
    torch::Tensor grad_links = at::zeros({bsz, prelen, translen}, match_all.options());

    const c10::cuda::CUDAGuard device_guard(match_all.device());
    cudaStream_t current_stream = at::cuda::getCurrentCUDAStream();

    // calculate grad_match_all
    switch(config1){
        case 1: invoke_calculate_grad_match_all<4, 256, 4>(current_stream, grad_match_all, grad_output, alpha, beta, match_all, output_length, target_length, bsz, prelen, tarlen); break;
        case 2: invoke_calculate_grad_match_all<4, 128, 4>(current_stream, grad_match_all, grad_output, alpha, beta, match_all, output_length, target_length, bsz, prelen, tarlen); break;
        default: TORCH_CHECK(false, "config1 should be 1~2");
    }


    // calculate grad_links
    switch(config2){
        case 1: invoke_calculate_grad_links<8, 32, 4>(current_stream, grad_links, grad_output, alpha, beta, links, output_length, target_length, bsz, prelen, translen); break;
        case 2: invoke_calculate_grad_links<4, 64, 4>(current_stream, grad_links, grad_output, alpha, beta, links, output_length, target_length, bsz, prelen, translen); break;
        case 3: invoke_calculate_grad_links<4, 32, 4>(current_stream, grad_links, grad_output, alpha, beta, links, output_length, target_length, bsz, prelen, translen); break;
        default: TORCH_CHECK(false, "config2 should be 1~3");
    }

    return std::make_tuple(grad_match_all, grad_links);
}
