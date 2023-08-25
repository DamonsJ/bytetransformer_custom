// Copyright 2023 Bytedance Ltd. and/or its affiliates.
/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */
#include <mma.h>

#include "bytetransformer/include/attention.h"
#include "bytetransformer/include/reduce.h"
#include "bytetransformer/include/common.h"

namespace bytetransformer {
#define SKEW_HALF 8  // offset for avoding bank conflict
#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

template <const int max_seq_len, const int size_per_head>
__global__
    // __launch_bounds__(512,4)//THREADS_PER_BLOCK
    void
    wmma_attention_kernel_16(const half2 *qkv, const half2 *qkv_bias, const __half *attention_mask,
                             __half *attention_output, const int seq_len, const float scale) {
#if __CUDA_ARCH__ >= 700
  using namespace nvcuda;
  __shared__ __half s_kv[max_seq_len][size_per_head + SKEW_HALF];
  __shared__ __half s_query[max_seq_len][size_per_head + SKEW_HALF];
  __shared__ __half s_logits[max_seq_len][max_seq_len + SKEW_HALF];

  const int warpNums = (blockDim.x >> 5);
  const int warpId = (threadIdx.x >> 5);
  const int warp_tid = getLaneId();
  const int half_hidden_dim = gridDim.x * (size_per_head / 2);
  const int batch_seq_offset = blockIdx.y * seq_len;
  const int from_size = max_seq_len / 16;
  const int to_size = max_seq_len / 16;

  const int quart_warpId = threadIdx.x >> 3;
  const int quart_warp_tid = threadIdx.x & 0x7;
  const int quart_thread_offset = blockIdx.x * (size_per_head / 2) + quart_warp_tid;

  // loading Query & Key
  half2 q_bias = __ldg(&qkv_bias[quart_thread_offset]);
  half2 k_bias = __ldg(&qkv_bias[quart_thread_offset + half_hidden_dim]);
  for (int seq_id = quart_warpId; seq_id < seq_len; seq_id += warpNums * 4) {
    int pos = (batch_seq_offset + seq_id) * (half_hidden_dim * 3) + quart_thread_offset;
    int offset = seq_id * (size_per_head + SKEW_HALF) + (quart_warp_tid << 1);
    *(__half2 *)(*s_query + offset) = __hadd2(__ldg(&qkv[pos]), q_bias);
    *(__half2 *)(*s_kv + offset) = __hadd2(__ldg(&qkv[pos + half_hidden_dim]), k_bias);
  }

  __syncthreads();

  if (warpId < from_size * to_size) {
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> Q_mat;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> K_mat;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half> QK_mat;
    wmma::fill_fragment(QK_mat, 0.0f);
    const int warp_from_offset = (warpId / to_size) << 4;
    const int warp_to_offset = (warpId % to_size) << 4;

#pragma unroll
    for (int k = 0; k < 1; k++) {
      wmma::load_matrix_sync(Q_mat, s_query[warp_from_offset] + k * WMMA_K,
                             size_per_head + SKEW_HALF);
      wmma::load_matrix_sync(K_mat, s_kv[warp_to_offset] + k * WMMA_K, size_per_head + SKEW_HALF);
      wmma::mma_sync(QK_mat, Q_mat, K_mat, QK_mat);
    }
    wmma::store_matrix_sync(s_logits[warp_from_offset] + warp_to_offset, QK_mat,
                            max_seq_len + SKEW_HALF, wmma::mem_row_major);
  }

  __syncthreads();

  // softmax
  for (int from_id = warpId; from_id < seq_len; from_id += warpNums) {
    float max_val = -1e20f;

    const int n = (max_seq_len + 31) / 32;
    float logits[n];
    int to_id[n];

#pragma unroll
    for (int i = 0; i < n; i++) {
      to_id[i] = warp_tid + (i << 5);
      logits[i] = -1e20f;

      if (to_id[i] < seq_len) {
        float mask =
            (float)__ldg(&attention_mask[(batch_seq_offset + from_id) * seq_len + to_id[i]]);
        mask = (1.0f - mask) * (-10000.0f);

        logits[i] = (float)(s_logits[from_id][to_id[i]]) * scale + mask;
      }
      max_val = max(max_val, logits[i]);
    }
    max_val = warpReduceMax(max_val);

    float sum_val = 0.0f;
#pragma unroll
    for (int i = 0; i < n; i++) {
      logits[i] = __expf(logits[i] - max_val);
      sum_val += logits[i];
    }
    sum_val = warpReduceSum(sum_val) + 1e-6f;

#pragma unroll
    for (int i = 0; i < n; i++)
      if (to_id[i] < max_seq_len)
        s_logits[from_id][to_id[i]] = (__half)__fdividef(logits[i], sum_val);
  }

  // loading Value
  half2 v_bias = __ldg(&qkv_bias[quart_thread_offset + half_hidden_dim * 2]);
  for (int seq_id = quart_warpId; seq_id < seq_len; seq_id += warpNums * 4) {
    int pos = (batch_seq_offset + seq_id) * (half_hidden_dim * 3) + quart_thread_offset;
    ((__half2 *)(s_kv[seq_id]))[quart_warp_tid] =
        __hadd2(__ldg(&qkv[pos + half_hidden_dim * 2]), v_bias);
  }

  // K dim clear 0
  for (int seq_id = seq_len + quart_warpId; seq_id < max_seq_len; seq_id += warpNums * 4)
    ((float *)(s_kv[seq_id]))[quart_warp_tid] = 0.0f;
  __syncthreads();

  //* V
  if (warpId < from_size) {
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> Logits_mat;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> V_mat;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half> QKV_mat;
    wmma::fill_fragment(QKV_mat, 0.0f);
    const int warp_from_offset = (warpId) << 4;
    const int warp_to_offset = 0;

#pragma unroll
    for (int k = 0; k < to_size; k++) {
      wmma::load_matrix_sync(Logits_mat, s_logits[warp_from_offset] + k * WMMA_K,
                             max_seq_len + SKEW_HALF);
      wmma::load_matrix_sync(V_mat, s_kv[k * WMMA_K] + warp_to_offset, size_per_head + SKEW_HALF);
      wmma::mma_sync(QKV_mat, Logits_mat, V_mat, QKV_mat);
    }
    wmma::store_matrix_sync(s_query[warp_from_offset] + warp_to_offset, QKV_mat,
                            size_per_head + SKEW_HALF, wmma::mem_row_major);
  }
  __syncthreads();

  for (int from_id = quart_warpId; from_id < seq_len; from_id += warpNums * 4) {
    int pos = (batch_seq_offset + from_id) * half_hidden_dim + quart_thread_offset;
    ((__half2 *)(attention_output))[pos] = ((__half2 *)(s_query[from_id]))[quart_warp_tid];
  }
#endif
}

template <const int max_seq_len, const int size_per_head>
__global__
    // __launch_bounds__(512,4)//THREADS_PER_BLOCK
    void
    wmma_attention_kernel(const half2 *qkv, const half2 *qkv_bias, const __half *attention_mask,
                          __half *attention_output, const int seq_len, const float scale) {
#if __CUDA_ARCH__ >= 700
  using namespace nvcuda;
  __shared__ __half s_kv[max_seq_len][size_per_head + SKEW_HALF];
  __shared__ __half s_query[max_seq_len][size_per_head + SKEW_HALF];
  __shared__ __half s_logits[max_seq_len][max_seq_len + SKEW_HALF];

  const int warpNums = (blockDim.x >> 5);
  const int warpId = (threadIdx.x >> 5);
  const int warp_tid = getLaneId();
  const int half_hidden_dim = gridDim.x * (size_per_head / 2);
  const int thread_offset = blockIdx.x * (size_per_head / 2) + warp_tid;
  const int batch_seq_offset = blockIdx.y * seq_len;
  const int from_size = max_seq_len / 16;
  const int to_size = max_seq_len / 16;
  // 抛的线程是(max_seq_len / 16) * max((max_seq_len / 16), (size_per_head / 16))
  // 但是QK乘完之后的矩阵是SxS,所以 to_size是max_seq_len / 16
  // 也就是按照结果的行列排布的
  // loading Query & Key
  half2 q_bias = __ldg(&qkv_bias[thread_offset]);
  half2 k_bias = __ldg(&qkv_bias[thread_offset + half_hidden_dim]);
  for (int seq_id = warpId; seq_id < seq_len; seq_id += warpNums) {
    int pos = (batch_seq_offset + seq_id) * (half_hidden_dim * 3) + thread_offset;
    int offset = seq_id * (size_per_head + SKEW_HALF) + (warp_tid << 1);
    *(__half2 *)(*s_query + offset) = __hadd2(__ldg(&qkv[pos]), q_bias);
    *(__half2 *)(*s_kv + offset) = __hadd2(__ldg(&qkv[pos + half_hidden_dim]), k_bias);
  }
  __syncthreads();
  // 结果矩阵是SxS的，且分成16*16的块
  // 这个地方实际抛的线程数是比from_size * to_size要大的
  // 实际抛的是(max_seq_len / 16) * max((max_seq_len / 16), (size_per_head / 16))
  // size_per_head 是定死的64
  if (warpId < from_size * to_size) {
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> Q_mat;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> K_mat;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half> QK_mat;
    wmma::fill_fragment(QK_mat, 0.0f);
    // 把warp分成矩阵的形式
    // (0,0) (0,1), ...., (0, to_size)
    // (1,0) (1,1), ...., (1, to_size)
    // 。。。
    // (from_size,0) (from_size,1), ...., (from_size, to_size)
    const int warp_from_offset = (warpId / to_size) << 4;
    const int warp_to_offset = (warpId % to_size) << 4;

#pragma unroll
    // 这个4的意思是 size_per_head / 16 ,size_per_head 是定死的64 所以是4
    for (int k = 0; k < 4; k++) {
      // 这个地方把一个Sxsize_per_head的矩阵分成了16*16的矩阵块， 然后WMMA计算一个16*16的矩阵块
      // 需要注意的是K是列主序，所以K的转置矩阵不需要内存上的变化
      wmma::load_matrix_sync(Q_mat, s_query[warp_from_offset] + k * WMMA_K,
                             size_per_head + SKEW_HALF);  // size_per_head + SKEW_HALF这个是stride
      wmma::load_matrix_sync(K_mat, s_kv[warp_to_offset] + k * WMMA_K, size_per_head + SKEW_HALF);
      wmma::mma_sync(QK_mat, Q_mat, K_mat, QK_mat);
    }
    wmma::store_matrix_sync(s_logits[warp_from_offset] + warp_to_offset, QK_mat,
                            max_seq_len + SKEW_HALF, wmma::mem_row_major);
  }
  __syncthreads();

  // softmax
  half2 v_bias = __ldg(&qkv_bias[thread_offset + half_hidden_dim * 2]);
  for (int from_id = warpId; from_id < seq_len; from_id += warpNums) {
    const int n = (max_seq_len + 31) / 32;
    float logits[n];
    int to_id[n];

    float max_val = -1e20f;
#pragma unroll
    for (int i = 0; i < n; i++) {
      to_id[i] = warp_tid + (i << 5);
      logits[i] = -1e20f;

      if (to_id[i] < seq_len) {
        float mask =
            (float)__ldg(&attention_mask[(batch_seq_offset + from_id) * seq_len + to_id[i]]);
        mask = (1.0f - mask) * (-10000.0f);
        logits[i] = (float)(s_logits[from_id][to_id[i]]) * scale + mask;
      }
      max_val = max(max_val, logits[i]);
    }
    max_val = warpReduceMax(max_val);

    float sum_val = 0.0f;
#pragma unroll
    for (int i = 0; i < n; i++) {
      logits[i] = __expf(logits[i] - max_val);
      sum_val += logits[i];
    }
    sum_val = warpReduceSum(sum_val) + 1e-6f;

#pragma unroll
    for (int i = 0; i < n; i++)
      if (to_id[i] < max_seq_len)
        s_logits[from_id][to_id[i]] = (__half)__fdividef(logits[i], sum_val);

    // loading Value
    int pos = (batch_seq_offset + from_id) * (half_hidden_dim * 3) + thread_offset;
    ((__half2 *)(s_kv[from_id]))[warp_tid] =
        __hadd2(__ldg(&qkv[pos + half_hidden_dim * 2]), v_bias);
  }

  // K dim clear 0
  // 这个地方是为了把seq_len和max_seq_len之间的数据清0
  // 因为输入的是max_seq_len 的长度，可能比seq_len要大
  for (int seq_id = seq_len + warpId; seq_id < max_seq_len; seq_id += warpNums)
    ((float *)(s_kv[seq_id]))[warp_tid] = 0.0f;
  __syncthreads();

  //* V
  // 这个地方应该是 from_size * (size_per_head / 16) size_per_head是64
  // 因为这个地方的矩阵乘法是SxS 和 Sxsize_per_head 相乘
  // 结果是Sxsize_per_head size_per_head是64 所以是from_size*4
  if (warpId < (from_size << 2)) {
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> Logits_mat;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> V_mat;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half> QKV_mat;
    wmma::fill_fragment(QKV_mat, 0.0f);
    // 这个地方除4也是一样的道理： (size_per_head / 16) size_per_head是64
    const int warp_from_offset = (warpId >> 2) << 4;
    const int warp_to_offset = (warpId & 0x3) * WMMA_K;

#pragma unroll
    for (int k = 0; k < to_size; k++) {
      // 这个地方循环的是行 也就是 SxS中的一行，和 Sxsize_per_head的一列相乘
      wmma::load_matrix_sync(Logits_mat, s_logits[warp_from_offset] + k * WMMA_K,
                             max_seq_len + SKEW_HALF);
      wmma::load_matrix_sync(V_mat, s_kv[k * WMMA_K] + warp_to_offset, size_per_head + SKEW_HALF);
      wmma::mma_sync(QKV_mat, Logits_mat, V_mat, QKV_mat);
    }
    wmma::store_matrix_sync(s_query[warp_from_offset] + warp_to_offset, QKV_mat,
                            size_per_head + SKEW_HALF, wmma::mem_row_major);
  }
  __syncthreads();

  for (int from_id = warpId; from_id < seq_len; from_id += warpNums) {
    int pos = (batch_seq_offset + from_id) * half_hidden_dim + thread_offset;
    ((__half2 *)(attention_output))[pos] = ((__half2 *)(s_query[from_id]))[warp_tid];
  }
#endif
}

template <const int max_seq_len, const int size_per_head>
__global__
    // __launch_bounds__(512,4)//THREADS_PER_BLOCK
    void
    wmma_attention_rm_kernel(const half2 *qkv, const half2 *qkv_bias, const __half *attention_mask,
                             __half *attention_output, const float scale, const int *batch_idx) {
#if __CUDA_ARCH__ >= 700
  using namespace nvcuda;
  __shared__ __half s_kv[max_seq_len][size_per_head + SKEW_HALF];
  __shared__ __half s_query[max_seq_len][size_per_head + SKEW_HALF];
  __shared__ __half s_logits[max_seq_len][max_seq_len + SKEW_HALF];

  const int warpNums = (blockDim.x >> 5);
  const int warpId = (threadIdx.x >> 5);
  const int warp_tid = getLaneId();
  const int half_hidden_dim = gridDim.x * (size_per_head / 2);
  const int thread_offset = blockIdx.x * (size_per_head / 2) + warp_tid;
  const int batch_offset = __ldg(&batch_idx[blockIdx.y]);
  const int batch_seq_len = __ldg(&batch_idx[blockIdx.y + 1]) - batch_offset;
  const int from_size = (batch_seq_len + 15) >> 4;
  const int to_size = (batch_seq_len + 15) >> 4;

  // loading Query & Key
  half2 q_bias = __ldg(&qkv_bias[thread_offset]);
  half2 k_bias = __ldg(&qkv_bias[thread_offset + half_hidden_dim]);
  for (int seq_id = warpId; seq_id < batch_seq_len; seq_id += warpNums) {
    int pos = (batch_offset + seq_id) * (half_hidden_dim * 3) + thread_offset;
    int offset = seq_id * (size_per_head + SKEW_HALF) + (warp_tid << 1);
    *(__half2 *)(*s_query + offset) = __hadd2(__ldg(&qkv[pos]), q_bias);
    *(__half2 *)(*s_kv + offset) = __hadd2(__ldg(&qkv[pos + half_hidden_dim]), k_bias);
  }
  __syncthreads();

  if (warpId < from_size * to_size) {
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> Q_mat;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> K_mat;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half> QK_mat;
    wmma::fill_fragment(QK_mat, 0.0f);
    const int warp_from_offset = (warpId / to_size) << 4;
    const int warp_to_offset = (warpId % to_size) << 4;

#pragma unroll
    for (int k = 0; k < 4; k++) {
      wmma::load_matrix_sync(Q_mat, s_query[warp_from_offset] + k * WMMA_K,
                             size_per_head + SKEW_HALF);
      wmma::load_matrix_sync(K_mat, s_kv[warp_to_offset] + k * WMMA_K, size_per_head + SKEW_HALF);
      wmma::mma_sync(QK_mat, Q_mat, K_mat, QK_mat);
    }
    wmma::store_matrix_sync(s_logits[warp_from_offset] + warp_to_offset, QK_mat,
                            max_seq_len + SKEW_HALF, wmma::mem_row_major);
  }
  __syncthreads();

  // softmax
  half2 v_bias = __ldg(&qkv_bias[thread_offset + half_hidden_dim * 2]);
  for (int from_id = warpId; from_id < batch_seq_len; from_id += warpNums) {
    const int n = (max_seq_len + 31) / 32;
    float logits[n];
    int to_id[n];

    float max_val = -1e20f;
#pragma unroll
    for (int i = 0; i < n; i++) {
      to_id[i] = warp_tid + (i << 5);
      logits[i] = to_id[i] < batch_seq_len ? (float)(s_logits[from_id][to_id[i]]) * scale : -1e20f;
      max_val = max(max_val, logits[i]);
    }
    max_val = warpReduceMax(max_val);

    float sum_val = 0.0f;
#pragma unroll
    for (int i = 0; i < n; i++) {
      logits[i] = __expf(logits[i] - max_val);
      sum_val += logits[i];
    }
    sum_val = warpReduceSum(sum_val) + 1e-6f;

#pragma unroll
    for (int i = 0; i < n; i++)
      if (to_id[i] < max_seq_len)
        s_logits[from_id][to_id[i]] = (__half)__fdividef(logits[i], sum_val);

    // loading Value
    int pos = (batch_offset + from_id) * (half_hidden_dim * 3) + thread_offset;
    ((__half2 *)(s_kv[from_id]))[warp_tid] =
        __hadd2(__ldg(&qkv[pos + half_hidden_dim * 2]), v_bias);
  }

  // K dim clear 0
  for (int seq_id = batch_seq_len + warpId; seq_id < max_seq_len; seq_id += warpNums)
    ((float *)(s_kv[seq_id]))[warp_tid] = 0.0f;
  __syncthreads();

  //* V
  if (warpId < (from_size << 2)) {
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> Logits_mat;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> V_mat;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half> QKV_mat;
    wmma::fill_fragment(QKV_mat, 0.0f);
    const int warp_from_offset = (warpId >> 2) << 4;
    const int warp_to_offset = (warpId & 0x3) * WMMA_K;

#pragma unroll
    for (int k = 0; k < to_size; k++) {
      wmma::load_matrix_sync(Logits_mat, s_logits[warp_from_offset] + k * WMMA_K,
                             max_seq_len + SKEW_HALF);
      wmma::load_matrix_sync(V_mat, s_kv[k * WMMA_K] + warp_to_offset, size_per_head + SKEW_HALF);
      wmma::mma_sync(QKV_mat, Logits_mat, V_mat, QKV_mat);
    }
    wmma::store_matrix_sync(s_query[warp_from_offset] + warp_to_offset, QKV_mat,
                            size_per_head + SKEW_HALF, wmma::mem_row_major);
  }
  __syncthreads();

  for (int from_id = warpId; from_id < batch_seq_len; from_id += warpNums) {
    int pos = (batch_offset + from_id) * half_hidden_dim + thread_offset;
    ((__half2 *)(attention_output))[pos] = ((__half2 *)(s_query[from_id]))[warp_tid];
  }
#endif
}

#define WMMA_ATTENTION_16(SEQ_LEN, SIZE_PER_HEAD)                                           \
  wmma_attention_kernel_16<SEQ_LEN, SIZE_PER_HEAD><<<grid, block, 0, infer_param.stream>>>( \
      qkv_ptr, qkv_bias_ptr, (__half *)atten_mask, (__half *)attention_output, seq_len, scale)

#define WMMA_ATTENTION(SEQ_LEN, SIZE_PER_HEAD)                                           \
  wmma_attention_kernel<SEQ_LEN, SIZE_PER_HEAD><<<grid, block, 0, infer_param.stream>>>( \
      qkv_ptr, qkv_bias_ptr, (__half *)atten_mask, (__half *)attention_output, seq_len, scale)

#define WMMA_ATTENTION_RM(SEQ_LEN, SIZE_PER_HEAD)                                           \
  wmma_attention_rm_kernel<SEQ_LEN, SIZE_PER_HEAD><<<grid, block, 0, infer_param.stream>>>( \
      qkv_ptr, qkv_bias_ptr, (__half *)atten_mask, (__half *)attention_output, scale,       \
      et_param.batch_idx)

template <OperationType OpType>
void Attention<OpType>::fused_infer(AttentionInferParam infer_param) {
  const DataType_ *atten_mask = infer_param.atten_mask;
  DataType_ *attention_output = infer_param.attention_output;
  const int batch_size = infer_param.batch_size;
  const int seq_len = infer_param.seq_len;

  dim3 grid(head_num_, batch_size), block;

  if (OpType == OperationType::HALF) {
    const half2 *qkv_ptr = (const half2 *)infer_param.qkv;
    const half2 *qkv_bias_ptr = (const half2 *)param_.attr_bias_QKV;
    float scale = (1.0f / sqrt(size_per_head_ * 1.0f)) / param_.tao;

    block.x = 32 * ((seq_len + 15) / 16) * max(((seq_len + 15) / 16), size_per_head_ / 16);
    if (size_per_head_ == 64) {
      if (seq_len <= 16)
        WMMA_ATTENTION(16, 64);
      else if (seq_len <= 32)
        WMMA_ATTENTION(32, 64);
      else if (seq_len <= 48)
        WMMA_ATTENTION(48, 64);
      else if (seq_len <= 64)
        WMMA_ATTENTION(64, 64);
      else if (seq_len <= 80)
        WMMA_ATTENTION(80, 64);
    } else if (size_per_head_ == 16) {
      if (seq_len <= 48)
        WMMA_ATTENTION_16(48, 16);
    }
  }
}

template <OperationType OpType>
void Attention<OpType>::fused_rm_infer(AttentionInferParam infer_param) {
  const DataType_ *atten_mask = infer_param.atten_mask;
  DataType_ *attention_output = infer_param.attention_output;
  const int batch_size = infer_param.batch_size;
  const int seq_len = infer_param.seq_len;
  ET_Param et_param = infer_param.et_param;

  dim3 grid(head_num_, batch_size), block;

  if (OpType == OperationType::HALF) {
    const half2 *qkv_ptr = (const half2 *)infer_param.qkv;
    const half2 *qkv_bias_ptr = (const half2 *)param_.attr_bias_QKV;
    float scale = 1.0f / sqrt(size_per_head_ * 1.0f) / param_.tao;

    block.x = 32 * ((seq_len + 15) / 16) * max(((seq_len + 15) / 16), size_per_head_ / 16);
    if (size_per_head_ == 64) {
      if (seq_len <= 16)
        WMMA_ATTENTION_RM(16, 64);
      else if (seq_len <= 32)
        WMMA_ATTENTION_RM(32, 64);
      else if (seq_len <= 48)
        WMMA_ATTENTION_RM(48, 64);
      else if (seq_len <= 64)
        WMMA_ATTENTION_RM(64, 64);
      else if (seq_len <= 80)
        WMMA_ATTENTION_RM(80, 64);
    }
  }
}

template void Attention<OperationType::FP32>::fused_infer(AttentionInferParam infer_param);
template void Attention<OperationType::HALF>::fused_infer(AttentionInferParam infer_param);
template void Attention<OperationType::FP32>::fused_rm_infer(AttentionInferParam infer_param);
template void Attention<OperationType::HALF>::fused_rm_infer(AttentionInferParam infer_param);
}  // namespace bytetransformer
