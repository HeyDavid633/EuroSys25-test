/**********
 * 2024.2.26 Mon.
 * syncfree_attention.cu:
 * 专注于提升下三角的性能
 *  Attention:
 *   Q(B, H, S, W) @ K^T(B, H, W, S) -> mask -> (B, H, S, S) -softmax-> (B, H, S, S)
 *    (B, H, S, S) @ V(B, H, S, W) -> (B, H, S, W) -trans-> (B, S, H, W) -merge-> (B, S, D)
 *
 */

#include <torch/extension.h>
#include <torch/types.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
#include <iostream>

#ifndef WARP_SIZE
#define WARP_SIZE 32
#endif

__global__ void syncfree_triangle_attention_kernel(
    const float *q, const float *k, const float *v, float *result,
    const int stride_0, const int stride_1, const int stride_2, const int stride_3,
    const int batch_size, const int seq_len, const int head_num, const int head_size)
{
    int row_idx = blockIdx.x;
    int batch = blockIdx.y;
    int head_channel = threadIdx.x;
    int head = threadIdx.y + blockIdx.z * head_num / 3;

    if (batch >= batch_size || head >= head_num || row_idx >= seq_len || head_channel >= WARP_SIZE)
        return;

    float sum_exp_score = 0.0f;
    float score_0 = 0.0f;
    float score_1 = 0.0f;

    float lane_score, exp_score;
    float lane_score2, exp_score2;
    float lane_score3, exp_score3;
    float lane_score4, exp_score4;

    int offset_res = head_num * seq_len * head_size * batch + head_num * head_size * row_idx + head_size * head;
    int offset_common = stride_0 * batch + stride_1 * head;
    int offset_q, offset_k;
    int offset_k2;
    int offset_k3, offset_k4;

    offset_q = offset_common + stride_2 * row_idx;

    for (int col = 0; col <= row_idx;)
    {
        // Q * K^T
        if (col + 3 <= row_idx)
        {
            offset_k = offset_common + stride_2 * col;
            lane_score = q[offset_q + stride_3 * head_channel] * k[offset_k + stride_3 * head_channel];
            lane_score += q[offset_q + stride_3 * (head_channel + WARP_SIZE)] * k[offset_k + stride_3 * (head_channel + WARP_SIZE)];

            offset_k2 = offset_common + stride_2 * (col + 1);
            lane_score2 = q[offset_q + stride_3 * head_channel] * k[offset_k2 + stride_3 * head_channel];
            lane_score2 += q[offset_q + stride_3 * (head_channel + WARP_SIZE)] * k[offset_k2 + stride_3 * (head_channel + WARP_SIZE)];

            offset_k3 = offset_common + stride_2 * (col + 2);
            lane_score3 = q[offset_q + stride_3 * head_channel] * k[offset_k3 + stride_3 * head_channel];
            lane_score3 += q[offset_q + stride_3 * (head_channel + WARP_SIZE)] * k[offset_k3 + stride_3 * (head_channel + WARP_SIZE)];

            offset_k4 = offset_common + stride_2 * (col + 3);
            lane_score4 = q[offset_q + stride_3 * head_channel] * k[offset_k4 + stride_3 * head_channel];
            lane_score4 += q[offset_q + stride_3 * (head_channel + WARP_SIZE)] * k[offset_k4 + stride_3 * (head_channel + WARP_SIZE)];
            __syncwarp();

            // Softmax : reduction by warp-level
            for (int i = WARP_SIZE / 2; i > 0; i >>= 1)
            {
                lane_score += __shfl_xor_sync(0xffffffff, lane_score, i, WARP_SIZE);
                lane_score2 += __shfl_xor_sync(0xffffffff, lane_score2, i, WARP_SIZE);
                lane_score3 += __shfl_xor_sync(0xffffffff, lane_score3, i, WARP_SIZE);
                lane_score4 += __shfl_xor_sync(0xffffffff, lane_score4, i, WARP_SIZE);
            }

            if (head_channel == 0)
            {
                exp_score = __expf(__fdividef(lane_score, sqrtf(static_cast<float>(head_size))));
                exp_score2 = __expf(__fdividef(lane_score2, sqrtf(static_cast<float>(head_size))));
                exp_score3 = __expf(__fdividef(lane_score3, sqrtf(static_cast<float>(head_size))));
                exp_score4 = __expf(__fdividef(lane_score4, sqrtf(static_cast<float>(head_size))));
                sum_exp_score += (exp_score + exp_score2 + exp_score3 + exp_score4);
            }
            exp_score = __shfl_sync(0xffffffff, exp_score, 0);
            exp_score2 = __shfl_sync(0xffffffff, exp_score2, 0);
            exp_score3 = __shfl_sync(0xffffffff, exp_score3, 0);
            exp_score4 = __shfl_sync(0xffffffff, exp_score4, 0);

            // * V
            score_0 += (exp_score * v[offset_common + stride_2 * col + stride_3 * head_channel] +
                        exp_score2 * v[offset_common + stride_2 * (col + 1) + stride_3 * head_channel] +
                        exp_score3 * v[offset_common + stride_2 * (col + 2) + stride_3 * head_channel] +
                        exp_score4 * v[offset_common + stride_2 * (col + 3) + stride_3 * head_channel]);
            score_1 += (exp_score * v[offset_common + stride_2 * col + stride_3 * (head_channel + WARP_SIZE)] +
                        exp_score2 * v[offset_common + stride_2 * (col + 1) + stride_3 * (head_channel + WARP_SIZE)] +
                        exp_score3 * v[offset_common + stride_2 * (col + 2) + stride_3 * (head_channel + WARP_SIZE)] +
                        exp_score4 * v[offset_common + stride_2 * (col + 3) + stride_3 * (head_channel + WARP_SIZE)]);

            col = col + 4;
        }

        else
        {
            offset_k = offset_common + stride_2 * col;
            lane_score = q[offset_q + stride_3 * head_channel] * k[offset_k + stride_3 * head_channel];
            lane_score += q[offset_q + stride_3 * (head_channel + WARP_SIZE)] * k[offset_k + stride_3 * (head_channel + WARP_SIZE)];

            // Softmax : reduction by warp-level
            for (int i = WARP_SIZE / 2; i > 0; i >>= 1)
                lane_score += __shfl_xor_sync(0xffffffff, lane_score, i, WARP_SIZE);

            if (head_channel == 0)
            {
                exp_score = __expf(__fdividef(lane_score, sqrtf(static_cast<float>(head_size))));
                sum_exp_score += exp_score;
            }
            exp_score = __shfl_sync(0xffffffff, exp_score, 0);

            //* V
            score_0 += exp_score * v[offset_common + stride_2 * col + stride_3 * head_channel];
            score_1 += exp_score * v[offset_common + stride_2 * col + stride_3 * (head_channel + WARP_SIZE)];

            col++;
        }
    }
    sum_exp_score = __shfl_sync(0xffffffff, sum_exp_score, 0);

    result[offset_res + head_channel] = __fdividef(score_0, sum_exp_score);
    result[offset_res + head_channel + WARP_SIZE] = __fdividef(score_1, sum_exp_score);
}

__global__ void syncfree_strided_attention_kernel(
    const float *q, const float *k, const float *v, float *result,
    const int stride_0, const int stride_1, const int stride_2, const int stride_3, const int strided_step,
    const int batch_size, const int seq_len, const int head_num, const int head_size)
{
    int row_idx = blockIdx.x;
    int batch = blockIdx.y;
    int head_channel = threadIdx.x;
    int head = threadIdx.y + blockIdx.z * head_num / 3;

    if (batch >= batch_size || head >= head_num || row_idx >= seq_len || head_channel >= WARP_SIZE)
        return;

    float sum_exp_score = 0.0f;
    float score_0 = 0.0f;
    float score_1 = 0.0f;

    float lane_score;
    float exp_score;

    int offset_res = head_num * seq_len * head_size * batch + head_num * head_size * row_idx + head_size * head;
    int offset_common = stride_0 * batch + stride_1 * head;
    int offset_q, offset_k;

    // (B, H, S, W) @ (B, H, W, S) -> (B, H, S, S) -softmax-> (B, H, S, S)
    offset_q = offset_common + stride_2 * row_idx;

    for (int col = 0; col <= row_idx; ++col)
    {
        if ((row_idx - col) % strided_step == 0 || col > row_idx - strided_step)
        {
            offset_k = offset_common + stride_2 * col;
            lane_score = q[offset_q + stride_3 * head_channel] * k[offset_k + stride_3 * head_channel];
            lane_score += q[offset_q + stride_3 * (head_channel + WARP_SIZE)] * k[offset_k + stride_3 * (head_channel + WARP_SIZE)];

            for (int i = WARP_SIZE / 2; i > 0; i >>= 1)
                lane_score += __shfl_xor_sync(0xffffffff, lane_score, i, WARP_SIZE);

            if (head_channel == 0)
            {
                exp_score = expf(lane_score / sqrtf(static_cast<float>(head_size)));
                sum_exp_score += exp_score;
            }
            exp_score = __shfl_sync(0xffffffff, exp_score, 0);

            //(B, H, S, S) @ (B, H, S, W) -> (B, H, S, W) -trans-> (B, S, H, W) -merge-> (B, S, D)
            score_0 += exp_score * v[offset_common + stride_2 * col + stride_3 * head_channel];
            score_1 += exp_score * v[offset_common + stride_2 * col + stride_3 * (head_channel + WARP_SIZE)];
        }
    }
    sum_exp_score = __shfl_sync(0xffffffff, sum_exp_score, 0);

    result[offset_res + head_channel] = __fdividef(score_0, sum_exp_score);
    result[offset_res + head_channel + WARP_SIZE] = __fdividef(score_1, sum_exp_score);
}

__global__ void syncfree_fixed_attention_kernel(
    const float *q, const float *k, const float *v, float *result,
    const int stride_0, const int stride_1, const int stride_2, const int stride_3, const int fixed_step,
    const int batch_size, const int seq_len, const int head_num, const int head_size)
{
    int row_idx = blockIdx.x;
    int batch = blockIdx.y;
    int head_channel = threadIdx.x;
    int head = threadIdx.y + blockIdx.z * head_num / 3;

    if (batch >= batch_size || head >= head_num || row_idx >= seq_len || head_channel >= WARP_SIZE)
        return;

    float sum_exp_score = 0.0f;
    float score_0 = 0.0f;
    float score_1 = 0.0f;

    float lane_score;
    float exp_score;

    int offset_res = head_num * seq_len * head_size * batch + head_num * head_size * row_idx + head_size * head;
    int offset_common = stride_0 * batch + stride_1 * head;
    int offset_q, offset_k;

    // (B, H, S, W) @ (B, H, W, S) -> (B, H, S, S) -softmax-> (B, H, S, S)
    offset_q = offset_common + stride_2 * row_idx;

    for (int col = 0; col <= row_idx; ++col)
    {
        if (col % fixed_step == fixed_step - 1 || col > row_idx + (col % fixed_step) - fixed_step)
        {
            offset_k = offset_common + stride_2 * col;
            lane_score = q[offset_q + stride_3 * head_channel] * k[offset_k + stride_3 * head_channel];
            lane_score += q[offset_q + stride_3 * (head_channel + WARP_SIZE)] * k[offset_k + stride_3 * (head_channel + WARP_SIZE)];

            for (int i = WARP_SIZE / 2; i > 0; i >>= 1)
                lane_score += __shfl_xor_sync(0xffffffff, lane_score, i, WARP_SIZE);

            if (head_channel == 0)
            {
                exp_score = expf(lane_score / sqrtf(static_cast<float>(head_size)));
                sum_exp_score += exp_score;
            }
            exp_score = __shfl_sync(0xffffffff, exp_score, 0);

            //(B, H, S, S) @ (B, H, S, W) -> (B, H, S, W) -trans-> (B, S, H, W) -merge-> (B, S, D)
            score_0 += exp_score * v[offset_common + stride_2 * col + stride_3 * head_channel];
            score_1 += exp_score * v[offset_common + stride_2 * col + stride_3 * (head_channel + WARP_SIZE)];
        }
    }
    sum_exp_score = __shfl_sync(0xffffffff, sum_exp_score, 0);

    result[offset_res + head_channel] = __fdividef(score_0, sum_exp_score);
    result[offset_res + head_channel + WARP_SIZE] = __fdividef(score_1, sum_exp_score);
}

torch::Tensor run_syncfree_triangle_attention(torch::Tensor q, torch::Tensor k, torch::Tensor v)
{
    const auto batch_size = q.size(0);
    const auto head_num = q.size(1);
    const auto seq_len = q.size(2);
    const auto head_size = q.size(3);

    const auto stride_0 = q.stride(0);
    const auto stride_1 = q.stride(1);
    const auto stride_2 = q.stride(2);
    const auto stride_3 = q.stride(3);

    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(q.device());
    auto result = torch::empty({batch_size, seq_len, head_num * head_size}, options);

    dim3 blockSizeDim(head_size / 2, head_num / 3); // 32 * 4 = 128
    dim3 gridSizeDim(seq_len, batch_size, 3);       // eg: 256 * 1/8/16 * 3 总会是64的倍数

    syncfree_triangle_attention_kernel<<<gridSizeDim, blockSizeDim>>>(
        q.data_ptr<float>(), k.data_ptr<float>(), v.data_ptr<float>(), result.data_ptr<float>(),
        stride_0, stride_1, stride_2, stride_3,
        batch_size, seq_len, head_num, head_size);

    return result;
}

torch::Tensor run_syncfree_strided_attention(torch::Tensor q, torch::Tensor k, torch::Tensor v)
{
    const auto batch_size = q.size(0);
    const auto head_num = q.size(1);
    const auto seq_len = q.size(2);
    const auto head_size = q.size(3);

    const auto stride_0 = q.stride(0);
    const auto stride_1 = q.stride(1);
    const auto stride_2 = q.stride(2);
    const auto stride_3 = q.stride(3);

    const int stride_step = sqrt(seq_len);

    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(q.device());
    auto result = torch::empty({batch_size, seq_len, head_num * head_size}, options);

    dim3 blockSizeDim(head_size / 2, head_num / 3);
    dim3 gridSizeDim(seq_len, batch_size, 3);

    syncfree_strided_attention_kernel<<<gridSizeDim, blockSizeDim>>>(
        q.data_ptr<float>(), k.data_ptr<float>(), v.data_ptr<float>(), result.data_ptr<float>(),
        stride_0, stride_1, stride_2, stride_3, stride_step,
        batch_size, seq_len, head_num, head_size);

    return result;
}

torch::Tensor run_syncfree_fixed_attention(torch::Tensor q, torch::Tensor k, torch::Tensor v)
{
    const auto batch_size = q.size(0);
    const auto head_num = q.size(1);
    const auto seq_len = q.size(2);
    const auto head_size = q.size(3);

    const auto stride_0 = q.stride(0);
    const auto stride_1 = q.stride(1);
    const auto stride_2 = q.stride(2);
    const auto stride_3 = q.stride(3);

    const int fixed_step = sqrt(seq_len);

    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(q.device());
    auto result = torch::empty({batch_size, seq_len, head_num * head_size}, options);

    dim3 blockSizeDim(head_size / 2, head_num / 3);
    dim3 gridSizeDim(seq_len, batch_size, 3);

    syncfree_fixed_attention_kernel<<<gridSizeDim, blockSizeDim>>>(
        q.data_ptr<float>(), k.data_ptr<float>(), v.data_ptr<float>(), result.data_ptr<float>(),
        stride_0, stride_1, stride_2, stride_3, fixed_step,
        batch_size, seq_len, head_num, head_size);

    return result;
}