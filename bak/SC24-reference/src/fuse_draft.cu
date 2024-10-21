#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <iostream>

__global__ void fused_attention_kernel1(
    const float* q, const float* k, const float* v, const float* mask, float* result,
    int batch_size, int seq_len, int head_num, int head_size) {

    // 计算当前线程所处理的元素索引
    int batch = blockIdx.z;
    int head = blockIdx.x * blockDim.x + threadIdx.x;
    int seq = blockIdx.y * blockDim.y + threadIdx.y;

    // 确保索引不越界
    if (head >= head_num || seq >= seq_len) return;

    // 这里简化了矩阵乘法和Softmax操作的实现
    // 实际实现应该包括：
    // 1. 对Q和K进行矩阵乘法
    // 2. 应用掩码和Softmax
    // 3. 与V进行矩阵乘法

    // 示例：计算Q和K的点积
    float score = 0.0f;
    for (int i = 0; i < head_size; ++i) {
        score += q[batch * head_num * seq_len * head_size + head * seq_len * head_size + seq * head_size + i] *
                 k[batch * head_num * seq_len * head_size + head * seq_len * head_size + seq * head_size + i];
    }

    // 示例：假设的Softmax和乘以V的操作
    // 这里只是示例代码，实际实现应该更复杂
    float softmax_result = exp(score); 
    for (int i = 0; i < head_size; ++i) {
        result[batch * seq_len * head_num * head_size + seq * head_num * head_size + head * head_size + i] = softmax_result *
                                                       v[batch * head_num * seq_len * head_size + head * seq_len * head_size + seq * head_size + i];
    }
}

__global__ void fused_attention_kernel2(
    const float* q, const float* k, const float* v, const float* mask, float* result,
    int batch_size, int seq_len, int head_num, int head_size) {

    // 确定当前线程的工作索引
    int head = blockIdx.x * blockDim.x + threadIdx.x;
    int seq1 = blockIdx.y * blockDim.y + threadIdx.y;
    int seq2 = threadIdx.z;

    // 确保索引不越界
    if (head >= head_num || seq1 >= seq_len || seq2 >= seq_len) return;

    // 计算Q和K的点积
    float score = 0.0f;
    for (int i = 0; i < head_size; ++i) {
        score += q[head * seq_len * head_size + seq1 * head_size + i] *
                 k[head * seq_len * head_size + seq2 * head_size + i];
    }

    // 应用掩码
    score -= mask[seq1 * seq_len + seq2] * 10000.0f;

    // 写入中间结果到共享内存或全局内存，以便进行Softmax计算
    // 注意：这里需要考虑同步和内存访问模式，以优化性能

    // 在这里，每个线程块负责一个seq1的Softmax计算
    // 执行Softmax计算
    // 注意：这里需要一个合适的策略来计算Softmax，确保数值稳定性

    // 读取Softmax结果
    // 注意：这里需要考虑同步和内存访问模式，以优化性能

    // 计算Softmax结果与V的矩阵乘法
    float output = 0.0f;
    for (int i = 0; i < head_size; ++i) {
        output += softmax_result * v[head * seq_len * head_size + seq2 * head_size + i];
    }

    // 写入最终结果
    result[head * seq_len * head_size + seq1 * head_size + threadIdx.z] = output;
}


__global__ void fused_attention_kernel3(
    const float* q, const float* k, const float* v, const float* mask, float* result,
    int batch_size, int seq_len, int head_num, int head_size) {

    // 确定当前线程的工作索引
    int batch = blockIdx.z;
    int head = blockIdx.x * blockDim.x + threadIdx.x;
    int seq1 = blockIdx.y * blockDim.y + threadIdx.y;
    int seq2 = threadIdx.z;

    // 确保索引不越界
    if (batch >= batch_size || head >= head_num || seq1 >= seq_len || seq2 >= seq_len) return;

    // 计算Q和K的点积
    float score = 0.0f;
    for (int i = 0; i < head_size; ++i) {
        score += q[batch * head_num * seq_len * head_size + head * seq_len * head_size + seq1 * head_size + i] *
                 k[batch * head_num * seq_len * head_size + head * seq_len * head_size + seq2 * head_size + i];
    }

    // 应用掩码
    score -= mask[batch * seq_len * seq_len + seq1 * seq_len + seq2] * 10000.0f;

    // 执行Softmax计算
    // 注意：这里简化了Softmax的实现。在实际应用中，您需要一个更精确和数值稳定的实现。
    float softmax_result = exp(score); 

    // 计算Softmax结果与V的矩阵乘法
    float output = 0.0f;
    for (int i = 0; i < head_size; ++i) {
        output += softmax_result * v[batch * head_num * seq_len * head_size + head * seq_len * head_size + seq2 * head_size + i];
    }

    // 写入最终结果
    result[batch * seq_len * head_num * head_size + seq1 * head_num * head_size + head * head_size + seq2] = output;
}


//进入时 q.k.v的维度为：(batch_size, head_num, seq_len, head_size)
__global__ void fused_attention_kernel4(
    const float* q, const float* k, const float* v, const float* mask, float* result,
    int batch_size, int seq_len, int head_num, int head_size) {

    // 确定当前线程的工作索引
    int batch = blockIdx.z;
    int head = blockIdx.x * blockDim.x + threadIdx.x;
    int seq1 = blockIdx.y * blockDim.y + threadIdx.y;
    int seq2 = threadIdx.z;

    // 确保索引不越界
    if (batch >= batch_size || head >= head_num || seq1 >= seq_len || seq2 >= seq_len) return;

    // 计算Q和K的点积  (B, H, S, W) @ (B, H, W, S) -> (B, H, S, S) 
    float score = 0.0f;
    for (int i = 0; i < head_size; ++i) {
        score += q[batch * head_num * seq_len * head_size + head * seq_len * head_size + seq1 * head_size + i] *
                 k[batch * head_num * seq_len * head_size + head * seq_len * head_size + seq2 * head_size + i];
    }

    // 应用掩码
    score -= mask[batch * seq_len * seq_len + seq1 * seq_len + seq2] * 10000.0f;

    // 执行Softmax计算  (B, H, S, S) -softmax-> (B, H, S, S)
    // 注意：这里简化了Softmax的实现。在实际应用中，您需要一个更精确和数值稳定的实现。
    float softmax_result = exp(score);    //我的softmax

    // 计算Softmax结果与V的矩阵乘法
    float output = 0.0f;
    for (int i = 0; i < head_size; ++i) {
        output += softmax_result * v[batch * head_num * seq_len * head_size + head * seq_len * head_size + seq2 * head_size + i];
    }

    // 写入最终结果
    result[batch * seq_len * head_num * head_size + seq1 * head_num * head_size + head * head_size + seq2] = output;
}

//进入时 q.k.v的维度为：(batch_size, head_num, seq_len, head_size)即(B, H, S, W)
__global__ void fused_attention_kernel5(
    const float* q, const float* k, const float* v, const float* mask, float* result,
    int batch_size, int seq_len, int head_num, int head_size) {

    int batch = blockIdx.z;
    int head = blockIdx.x * blockDim.x + threadIdx.x;
    int seq1 = blockIdx.y * blockDim.y + threadIdx.y;
    int seq2 = threadIdx.z;
    if (batch >= batch_size || head >= head_num || seq1 >= seq_len || seq2 >= seq_len) return;

    //计算出关于 batch 和 head 的索引偏移
    int batch_offset = batch * head_num * seq_len * head_size;
    int head_offset = head * seq_len * head_size;

    // 计算Q和K的点积  (B, H, S, W) @ (B, H, W, S) -> (B, H, S, S) 
    float score = 0.0f;
    for (int i = 0; i < head_size; ++i) {
        score += q[batch_offset + head_offset + seq1 * head_size + i] *
                 k[batch_offset + head_offset + seq2 * head_size + i];
    }

    // 应用掩码
    score -= mask[batch * seq_len * seq_len + seq1 * seq_len + seq2] * 10000.0f;

    // 执行Softmax计算  (B, H, S, S) -softmax-> (B, H, S, S)
    float softmax_result = exp(score);    //我的softmax

    // 计算Softmax结果与V的矩阵乘法
    float output = 0.0f;
    for (int i = 0; i < head_size; ++i) {
        output += softmax_result * v[batch_offset + head_offset + seq2 * head_size + i];
    }

    // 写入最终结果
    result[batch_offset + seq1 * head_num * head_size + head * head_size + seq2] = output;
}


//【完整实现 留存】softmax实现 + 共享内存  ------  有内存覆盖的问题
__global__ void fused_attention_kernel6(
    const float* q, const float* k, const float* v, const float* mask, float* result,
    int batch_size, int seq_len, int head_num, int head_size) {

    int batch = blockIdx.z;
    int head = blockIdx.x * blockDim.x + threadIdx.x;
    int seq1 = blockIdx.y * blockDim.y + threadIdx.y;
    int seq2 = threadIdx.z;
    if (batch >= batch_size || head >= head_num || seq1 >= seq_len || seq2 >= seq_len) return;

    int batch_offset = batch * head_num * seq_len * head_size;
    int head_offset = head * seq_len * head_size;

    // 计算Q和K的点积，并应用掩码
    float score = 0.0f;
    for (int i = 0; i < head_size; ++i) {
        score += q[batch_offset + head_offset + seq1 * head_size + i] *
                 k[batch_offset + head_offset + seq2 * head_size + i];
    }
    score = score / sqrtf(static_cast<float>(head_size));
    score -= mask[batch * seq_len * seq_len + seq1 * seq_len + seq2] * 10000.0f;

    // 共享内存，存储每一行的最大值和指数和
    extern __shared__ float shared_mem[];
    int shared_index = threadIdx.y * blockDim.z + threadIdx.z;
    // 存储每个线程的score到共享内存
    shared_mem[shared_index] = score;
    __syncthreads();

    // 找出每一行的最大值
    float max_score = shared_mem[shared_index];
    for (int i = 0; i < blockDim.z; i++) {
        max_score = max(max_score, shared_mem[threadIdx.y * blockDim.z + i]);
    }
    __syncthreads();

    // 计算指数并归一化
    float exp_score = exp(score - max_score);
    shared_mem[shared_index] = exp_score;
    __syncthreads();

    // 累加这一行的所有指数和
    float sum_exp_scores = 0.0f;
    for (int i = 0; i < blockDim.z; i++) {
        sum_exp_scores += shared_mem[threadIdx.y * blockDim.z + i];
    }
    __syncthreads();

    float softmax_result = exp_score / sum_exp_scores;

    // 计算Softmax结果与V的矩阵乘法
    float output = 0.0f;
    for (int i = 0; i < head_size; ++i) {
        output += softmax_result * v[batch_offset + head_offset + seq2 * head_size + i];
    }

    // 写入最终结果
    result[batch_offset + seq1 * head_num * head_size + head * head_size + seq2] = output;
}


//存档【QK乘积正确】softmax实现 + 共享内存  ------  解决有内存覆盖 / 单个head\batch_size
__global__ void fused_attention_kernel(
    const float* q, const float* k, const float* v, 
    const float* mask, float* result,
    int batch_size, int seq_len, int head_num, int head_size) {

    int batch = blockIdx.z;
    int head = threadIdx.z;
    int seq1 = blockIdx.x * blockDim.x + threadIdx.x;
    int seq2 = blockIdx.y * blockDim.y + threadIdx.y;

    // 上面的对应调用方式
    // dim3 blockSizeDim(8, 8, head_num);
    // dim3 gridSizeDim((seq_len + blockSizeDim.x - 1) / blockSizeDim.x, 
    //              (seq_len + blockSizeDim.y - 1) / blockSizeDim.y, batch_size);
    
    if (batch >= batch_size || head >= head_num || seq1 >= seq_len || seq2 >= seq_len) return;

    int batch_offset = batch * head_num * seq_len * head_size;
    int head_offset = head * seq_len * head_size;
    //注意内存管理 需要stride 来使 q跨过kv line1的片段来取q line2
    int stride_offset = batch_size * head_num * head_size*2; 
    

    // 计算Q和K的点积，并应用掩码
    float score = 0.0f;
    for (int i = 0; i < head_size; ++i) {
        
        // printf("q(B:%d, H%d, %d, %d):%10f * k (B:%d, H%d, %d, %d):%10f\n", 
        //     batch, head, seq1, i, q[seq1 * stride_offset + batch_offset + head_offset + seq1 * head_size + i], 
        //     batch, head, i, seq2, k[seq2 * stride_offset + batch_offset + head_offset + seq2 * head_size + i]);
        
        score += q[seq1 * stride_offset + batch_offset + head_offset + seq1 * head_size + i] *
                 k[seq2 * stride_offset + batch_offset + head_offset + seq2 * head_size + i];
    }
    score = score / sqrtf(static_cast<float>(head_size));
    
    // printf("batch_size:%d, seq_len:%d, head_num:%d, head_size:%d\n", batch_size, seq_len, head_num, head_size);
    // printf("batch: %d , head: %d , seq1: %d , seq2: %d, socre: %.4f , sqrtf : %f\n", 
    //                 batch, head, seq1, seq2, score, sqrtf(static_cast<float>(head_size)));

    result[batch * head_num * seq_len * seq_len 
            + head * seq_len * seq_len 
            + seq1 * seq_len + seq2] = score;

    // score -= mask[batch * seq_len * seq_len + seq1 * seq_len + seq2] * 10000.0f;

    // // 共享内存，存储每一行的最大值和指数和
    // extern __shared__ float shared_mem[];
    // int shared_index = threadIdx.y * blockDim.z + threadIdx.z;
    // // 存储每个线程的score到共享内存
    // shared_mem[shared_index] = score;
    // __syncthreads();

    // // 找出每一行的最大值
    // float max_score = shared_mem[shared_index];
    // for (int i = 0; i < blockDim.z; i++) {
    //     max_score = max(max_score, shared_mem[threadIdx.y * blockDim.z + i]);
    // }
    // __syncthreads();

    // // 计算指数并归一化
    // float exp_score = exp(score - max_score);
    // shared_mem[shared_index] = exp_score;
    // __syncthreads();

    // // 累加这一行的所有指数和
    // float sum_exp_scores = 0.0f;
    // for (int i = 0; i < blockDim.z; i++) {
    //     sum_exp_scores += shared_mem[threadIdx.y * blockDim.z + i];
    // }
    // __syncthreads();

    // float softmax_result = exp_score / sum_exp_scores;

    // // 计算Softmax结果与V的矩阵乘法
    // float output = 0.0f;
    // for (int i = 0; i < head_size; ++i) {
    //     output += softmax_result * v[batch_offset + head_offset + seq2 * head_size + i];
    // }

    // // 写入最终结果
    // result[batch_offset + seq1 * head_num * head_size + head * head_size + seq2] = output;
}


// 【QK乘积正确】清爽版 --- 单个head、batch_size， softmax相关索引没修正
__global__ void fused_attention_kernel(
    const float* q, const float* k, const float* v, 
    const float* mask, float* result,
    int batch_size, int seq_len, int head_num, int head_size) {

    int batch = blockIdx.z;
    int head = threadIdx.z;
    int seq1 = blockIdx.x * blockDim.x + threadIdx.x;
    int seq2 = blockIdx.y * blockDim.y + threadIdx.y;

    if (batch >= batch_size || head >= head_num || seq1 >= seq_len || seq2 >= seq_len) return;

    int batch_offset = batch * head_num * seq_len * head_size;
    int head_offset = head * seq_len * head_size;
    //注意内存管理 需要stride 来使 q跨过kv line1的片段来取q line2
    int stride_offset = batch_size * head_num * head_size*2; 
    
    //计算Q K^T的点积
    float score = 0.0f;
    for (int i = 0; i < head_size; ++i) { 
        score += q[seq1 * stride_offset + batch_offset + head_offset + seq1 * head_size + i] *
                 k[seq2 * stride_offset + batch_offset + head_offset + seq2 * head_size + i];
    }
    score = score / sqrtf(static_cast<float>(head_size));
    

    score -= mask[batch * seq_len * seq_len + seq1 * seq_len + seq2] * 10000.0f;

    // 共享内存，存储每一行的最大值和指数和
    extern __shared__ float shared_mem[];
    int shared_index = threadIdx.y * blockDim.z + threadIdx.z;
    // 存储每个线程的score到共享内存
    shared_mem[shared_index] = score;
    __syncthreads();

    // 找出每一行的最大值
    float max_score = shared_mem[shared_index];
    for (int i = 0; i < blockDim.z; i++) {
        max_score = max(max_score, shared_mem[threadIdx.y * blockDim.z + i]);
    }
    __syncthreads();

    // 计算指数并归一化
    float exp_score = exp(score - max_score);
    shared_mem[shared_index] = exp_score;
    __syncthreads();

    // 累加这一行的所有指数和
    float sum_exp_scores = 0.0f;
    for (int i = 0; i < blockDim.z; i++) {
        sum_exp_scores += shared_mem[threadIdx.y * blockDim.z + i];
    }
    __syncthreads();

    float softmax_result = exp_score / sum_exp_scores;

    // 计算Softmax结果与V的矩阵乘法
    float output = 0.0f;
    for (int i = 0; i < head_size; ++i) {
        output += softmax_result * v[batch_offset + head_offset + seq2 * head_size + i];
    }

    // 写入最终结果
    result[batch_offset + seq1 * head_num * head_size + head * head_size + seq2] = output;
}

// 【QK乘积正确】softmax索引修正
__global__ void fused_attention_kernel(
    const float* q, const float* k, const float* v, const float* mask, float* result,
    int batch_size, int seq_len, int head_num, int head_size) {

    int batch = blockIdx.z;
    int head = threadIdx.z;
    int seq1 = blockIdx.x * blockDim.x + threadIdx.x;
    int seq2 = blockIdx.y * blockDim.y + threadIdx.y;

    if (batch >= batch_size || head >= head_num || seq1 >= seq_len || seq2 >= seq_len) return;

    int batch_offset = batch * head_num * seq_len * head_size;
    int head_offset = head * seq_len * head_size;
    int stride_offset = batch_size * head_num * head_size * 2; 

    // 计算Q和K的点积
    float score = 0.0f;
    for (int i = 0; i < head_size; ++i) {
        score += q[seq1 * stride_offset + batch_offset + head_offset + i] *
                 k[seq2 * stride_offset + batch_offset + head_offset + i];
    }
    score = score / sqrtf(static_cast<float>(head_size));
    score -= mask[batch * seq_len * seq_len + seq1 * seq_len + seq2] * 10000.0f;

    // Softmax calculations
    extern __shared__ float shared_data[];
    float exp_score = expf(score);
    shared_data[threadIdx.y * blockDim.x + threadIdx.x] = exp_score;
    __syncthreads();

    float sum_exp = 0.0f;
    if (threadIdx.x == 0) {
        for (int i = 0; i < blockDim.x; i++) {
            sum_exp += shared_data[threadIdx.y * blockDim.x + i];
        }
    }
    __syncthreads();

    float softmax_result = exp_score / sum_exp;

    // 计算Softmax结果与V的矩阵乘法
    float output = 0.0f;
    for (int i = 0; i < head_size; ++i) {
        output += softmax_result * v[seq1 * stride_offset + batch_offset + head_offset + i];
    }

    // 写入最终结果
    result[batch * seq_len * head_num * head_size + seq1 * head_num * head_size + head] = output;
}


//2024.1.23 QK 到mask 修正版本
// when seq <= 16   此时对于共享内存更好操作；包含了softmax
// head_size = 12
__global__ void shortseq_fused_attention_kernel(
    const float* q, const float* k, const float* mask, float* result,
    int batch_size, int seq_len, int head_num, int head_size) {

    int batch = blockIdx.y;
    int head = blockIdx.x;
    int seq1 = threadIdx.x;
    int seq2 = threadIdx.y;

    if (batch >= batch_size || head >= head_num || seq1 >= seq_len || seq2 >= seq_len) return;

    int batch_offset = batch * head_num * seq_len * head_size;
    int head_offset = head * seq_len * head_size;
    int stride_offset = batch_size * head_num * head_size * 2; 

    // 计算Q和K的点积
    float score = 0.0f;
    for (int i = 0; i < head_size; ++i) {
        score += q[seq1 * stride_offset + batch_offset + head_offset + seq1 * head_size + i] *
                 k[seq2 * stride_offset + batch_offset + head_offset + seq2 * head_size + i];
    }
    score = score / sqrtf(static_cast<float>(head_size));
    score -= (1.0 - mask[batch * seq_len * seq_len + seq1 * seq_len + seq2]) * 10000.0f;

    // 开始 Softmax 
    // seq<32 这个blcok中可以共享一个 shared mem以存储(seq_len,seq_len)的exp结果
    extern __shared__ float shared_data[]; 
    extern __shared__ float shared_sum_exp[];

    float exp_score = expf(score);
    float sum_exp = 0.0;

    shared_data[seq1 * seq_len + seq2] = exp_score;
    __syncthreads();

    if (seq2 == 0) {
        for (int i = 0; i < seq_len; i++) {
            sum_exp += shared_data[seq1 * seq_len + i];
        }
        shared_sum_exp[seq1] = sum_exp;
    }
    __syncthreads();

    float softmax_result = exp_score / shared_sum_exp[seq1];

    result[batch * head_num * seq_len * seq_len + head * seq_len * seq_len + seq1 * seq_len + seq2] = softmax_result;
}

// when seq > 16  包含了softmax；但因为不在一个block中 性能差
__global__ void longseq_fused_attention_kernel(
    const float* q, const float* k, const float* mask, float* result, float* softmax_exp, float* softmax_sum,
    int batch_size, int seq_len, int head_num, int head_size) {

    int batch = blockIdx.z;
    int head = threadIdx.z;
    int seq1 = blockIdx.x * blockDim.x + threadIdx.x;
    int seq2 = blockIdx.y * blockDim.y + threadIdx.y;

    if (batch >= batch_size || head >= head_num || seq1 >= seq_len || seq2 >= seq_len) return;

    int batch_offset = batch * head_num * seq_len * head_size;
    int head_offset = head * seq_len * head_size;
    int stride_offset = batch_size * head_num * head_size * 2; 
    int result_offset = batch * head_num * seq_len * seq_len + head * seq_len * seq_len;

    // 计算Q和K的点积
    float score = 0.0f;
    for (int i = 0; i < head_size; ++i) {
        score += q[seq1 * stride_offset + batch_offset + head_offset + seq1 * head_size + i] *
                 k[seq2 * stride_offset + batch_offset + head_offset + seq2 * head_size + i];
    }
    score = score / sqrtf(static_cast<float>(head_size));
    score -= (1.0 - mask[batch * seq_len * seq_len + seq1 * seq_len + seq2]) * 10000.0f;


    // Softmax 
    float exp_score = expf(score);
    float sum_exp = 0.0;

    softmax_exp[result_offset + seq1 * seq_len + seq2] = exp_score;
    __syncthreads();

    if (seq2 == 0) {
        for (int i = 0; i < seq_len; i++) {
            sum_exp += softmax_exp[result_offset + seq1 * seq_len + i];
        }
        softmax_sum[batch * head_num * seq_len + head * seq_len + seq1] = sum_exp;
    }
    __syncthreads();//不在一个block中不能同步！

    float softmax_result = exp_score / softmax_sum[batch * head_num * seq_len + head * seq_len + seq1];
    
    if(softmax_result < 0){
        printf("exp_score: %.4f, softmax_sum[%d]: %.4f, softmax_result: %.4f\n",
            exp_score, batch * head_num * seq_len + head * seq_len + seq1, 
            softmax_sum[batch * head_num * seq_len + head * seq_len + seq1], softmax_result);
    }

    result[result_offset + seq1 * seq_len + seq2] = softmax_result;
}

//通过 offet + stride实现了
//1.batch_size 可以不等于1，在1 ，8 ，16 都可以成功   2.head_num 可以不等于1， 在测试时一般设置为 12
// when seq <= 16   此时对于共享内存更好操作
__global__ void shortseq_fused_attention_kernel(
    const float* q, const float* k, const float* mask, float* result,
    int offset_q, int offset_k, int stride_0, int stride_1, int stride_2, int stride_3,
    int batch_size, int seq_len, int head_num, int head_size) {

    int batch = blockIdx.y;
    int head = blockIdx.x;
    int seq1 = threadIdx.x;
    int seq2 = threadIdx.y;

    if (batch >= batch_size || head >= head_num || seq1 >= seq_len || seq2 >= seq_len) return;

    int batch_offset = batch * head_num * seq_len * head_size;
    int head_offset = head * seq_len * head_size;
    int stride_offset = batch_size * head_num * head_size * 2; 

    //     dim3 blockSizeDim(seq_len, seq_len);
    //     dim3 gridSizeDim(head_num, batch_size);

    //     shortseq_fused_attention_kernel<<<gridSizeDim, blockSizeDim>>>(
    //         q.data_ptr<float>(), k.data_ptr<float>(), 
    //         mask.data_ptr<float>(), result.data_ptr<float>(),
    //         offset_q, offset_k, stride_0, stride_1, stride_2, stride_3,
    //         batch_size, seq_len, head_num, head_size);

    // 计算Q和K的点积 q与k的维度 (B, H, S, W) 
    float score = 0.0f;
    for (int i = 0; i < head_size; ++i) {
        // score += q[seq1 * stride_offset + batch_offset + head_offset + seq1 * head_size + i] *
        //          k[seq2 * stride_offset + batch_offset + head_offset + seq2 * head_size + i];

        //q,k首字母 本身自带offest
        printf("Golden q(B:%d, H%d,  q[%d] %d, %d):%.4f   *   k(B:%d, H%d, k[%d] %d, %d):%.4f\nNew    q(B:%d, H%d,  q[%d] %d, %d):%.4f   *   k(B:%d, H%d, k[%d] %d, %d):%.4f\n\n", 
            batch, head, seq1 * stride_offset + batch_offset + head_offset + seq1 * head_size + i, seq1, i,
            q[seq1 * stride_offset + batch_offset + head_offset + seq1 * head_size + i], 
            batch, head, seq2 * stride_offset + batch_offset + head_offset + seq2 * head_size + i, i, seq2,
            k[seq2 * stride_offset + batch_offset + head_offset + seq2 * head_size + i],
            batch, head, stride_0 * batch + stride_1 * head + stride_2 * seq1 + stride_3 * i, seq1, i, 
            q[stride_0 * batch + stride_1 * head + stride_2 * seq1 + stride_3 * i], 
            batch, head, stride_0 * batch + stride_1 * head + stride_2 * seq2 + stride_3 * i, i, seq2, 
            k[stride_0 * batch + stride_1 * head + stride_2 * seq2 + stride_3 * i]);

        score += q[stride_0 * batch + stride_1 * head + stride_2 * seq1 + stride_3 * i ] *
                 k[stride_0 * batch + stride_1 * head + stride_2 * seq2 + stride_3 * i ];
    }
    score = score / sqrtf(static_cast<float>(head_size));
    score -= (1.0 - mask[batch * seq_len * seq_len + seq1 * seq_len + seq2]) * 10000.0f;

    // 开始 Softmax 
    // seq<32 这个blcok中可以共享一个 shared mem以存储(seq_len,seq_len)的exp结果
    extern __shared__ float shared_data[]; 
    extern __shared__ float shared_sum_exp[];

    float exp_score = expf(score);
    float sum_exp = 0.0;

    shared_data[seq1 * seq_len + seq2] = exp_score;
    __syncthreads();

    if (seq2 == 0) {
        for (int i = 0; i < seq_len; i++) {
            sum_exp += shared_data[seq1 * seq_len + i];
        }
        shared_sum_exp[seq1] = sum_exp;
    }
    __syncthreads();

    float softmax_result = exp_score / shared_sum_exp[seq1];

    result[batch * head_num * seq_len * seq_len + head * seq_len * seq_len + seq1 * seq_len + seq2] = softmax_result;
}


// 此时q.k.v的维度为: (batch_size, head_num, seq_len, head_size)
torch::Tensor run_my_fused_attention(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor mask)
{
    const auto batch_size = q.size(0);
    const auto head_num = q.size(1);
    const auto seq_len = q.size(2);
    const auto head_size = q.size(3);

    // std::cout << "in my Func:\n  q.device():" << q.device() << std::endl;

    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(q.device());
    auto result = torch::empty({batch_size, seq_len, head_num*head_size}, options);

    dim3 blockSizeDim(8, 8, 8);
    dim3 gridSizeDim((head_num + blockSizeDim.x - 1) / blockSizeDim.x, 
                 (seq_len + blockSizeDim.y - 1) / blockSizeDim.y, 
                 batch_size);

    //one Fused attention kernel
    fused_attention_kernel<<<gridSizeDim, blockSizeDim>>>(
        q.data_ptr<float>(), k.data_ptr<float>(), v.data_ptr<float>(), mask.data_ptr<float>(), result.data_ptr<float>(),
        batch_size, seq_len, head_num, head_size);
    
    cudaDeviceSynchronize(); 
    cudaError_t cuda_error = cudaGetLastError();
    if (cuda_error != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(cuda_error) << std::endl;
    }
    return result;
}


// 无论 seq_len的尺寸，但总是8的倍数; 一次处理一整个行
__global__ void unified_fused_attention_kernel(
    const float* q, const float* k, const float* mask, float* result,
    int stride_0, int stride_1, int stride_2, int stride_3,
    const int batch_size, const int seq_len, const int head_num, const int head_size) {

    int batch = blockIdx.y;
    int head = blockIdx.x;
    int row_idx = threadIdx.x;

    if (batch >= batch_size || head >= head_num || row_idx >= seq_len) return;

    float exp_score = 0.0;
    float sum_exp_score = 0.0;
    float storage_exp_score[1024];

    // 计算Q和K的点积 q与k的维度 (B, H, S, W)   这里可以循环展开
    float score = 0.0f;
    for (int col = 0; col<seq_len; ++col) {
        score = 0.0;
        for(int i = 0; i < head_size; ++i){
            score += q[stride_0 * batch + stride_1 * head + stride_2 * row_idx + stride_3 * i ] *
                 k[stride_0 * batch + stride_1 * head + stride_2 * col + stride_3 * i ];
        }   
        score = score / sqrtf(static_cast<float>(head_size));
        score -= (1.0 - mask[batch * seq_len * seq_len + row_idx * seq_len + col]) * 10000.0f;

        exp_score = expf(score);
        storage_exp_score[col] = exp_score;

        sum_exp_score += exp_score;
    }

    for (int col = 0; col<seq_len; ++col){
        float softmax_result = storage_exp_score[col] / sum_exp_score;
        result[batch * head_num * seq_len * seq_len + head * seq_len * seq_len + row_idx * seq_len + col] = softmax_result;
    }
}

/****
 * 解决了  seq_len 16 -- 1024(总是8的倍数) 都统一为一个kernel 计算正确
 * 可以变化任意的 batch_size / head_num
 * *****/
torch::Tensor run_fused_attention0125(torch::Tensor q, torch::Tensor k, torch::Tensor mask)
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
    auto result = torch::empty({batch_size, head_num, seq_len, seq_len}, options);

    dim3 blockSizeDim(seq_len);
    dim3 gridSizeDim(head_num, batch_size);

    unified_fused_attention_kernel<<<gridSizeDim, blockSizeDim>>>(
        q.data_ptr<float>(), k.data_ptr<float>(), 
        mask.data_ptr<float>(), result.data_ptr<float>(),
        stride_0, stride_1, stride_2, stride_3,
        batch_size, seq_len, head_num, head_size);

    cudaDeviceSynchronize(); 
    cudaError_t cuda_error = cudaGetLastError();
    if (cuda_error != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(cuda_error) << std::endl;
    }

    return result;
}




torch::Tensor run_fused_sparse_strided_attention(torch::Tensor q, torch::Tensor k, torch::Tensor v)
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
    auto result = torch::empty({batch_size, seq_len, head_num*head_size}, options);

    dim3 blockSizeDim(seq_len);
    dim3 gridSizeDim(head_num, batch_size);

    unified_fused_attention_kernel<<<gridSizeDim, blockSizeDim>>>(
        q.data_ptr<float>(), k.data_ptr<float>(), v.data_ptr<float>(), 
        mask.data_ptr<float>(), result.data_ptr<float>(),
        stride_0, stride_1, stride_2, stride_3,
        batch_size, seq_len, head_num, head_size);


    return result;
}

torch::Tensor run_fused_sparse_fixed_attention(torch::Tensor q, torch::Tensor k, torch::Tensor v)
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
    auto result = torch::empty({batch_size, seq_len, head_num*head_size}, options);

    dim3 blockSizeDim(seq_len);
    dim3 gridSizeDim(head_num, batch_size);

    unified_fused_attention_kernel<<<gridSizeDim, blockSizeDim>>>(
        q.data_ptr<float>(), k.data_ptr<float>(), v.data_ptr<float>(), 
        mask.data_ptr<float>(), result.data_ptr<float>(),
        stride_0, stride_1, stride_2, stride_3,
        batch_size, seq_len, head_num, head_size);


    return result;
}