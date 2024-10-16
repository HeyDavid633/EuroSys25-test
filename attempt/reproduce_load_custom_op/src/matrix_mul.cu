#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <iostream>

//方阵
__global__ void matrixMultiply(const float* A, const float* B, float* C, int size)
{
    int row = threadIdx.y + blockIdx.y * blockDim.y;
    int col = threadIdx.x + blockIdx.x * blockDim.x;

    float sum = 0.0f;
    for (int k = 0; k < size; ++k) {
        sum += A[row * size + k] * B[k * size + col];
    
    C[row * size + col] = sum;
    }
}

//处理非方阵
__global__ void matrixMultiply2(const float* A, const float* B, float* C, int d1, int d2, int d22)
{
    int row = threadIdx.y + blockIdx.y * blockDim.y;
    int col = threadIdx.x + blockIdx.x * blockDim.x;

    if (row < d1 && col < d22) {
        float sum = 0.0f;
        for (int k = 0; k < d2; ++k) {
            sum += A[row * d2 + k] * B[k * d22 + col];
        }
        C[row * d22 + col] = sum;
    }
}

//bmm()替换 --- (b, n, m)*(b, m, p) --- 有batch_size个 (n*m)(m*p) 的矩阵乘
__global__ void batchMatrixMultiply(const float* A, const float* B, float* C, int batch_size, int n, int m, int p)
{
    int row = threadIdx.y + blockIdx.y * blockDim.y;
    int col = threadIdx.x + blockIdx.x * blockDim.x;
    int batch_idx = blockIdx.z;

    if (row < n && col < p && batch_idx < batch_size) {
        float sum = 0.0f;
        for (int k = 0; k < m; ++k) {
            sum += A[(batch_idx * n + row) * m + k] * B[(batch_idx * m + k) * p + col];
        }
        C[(batch_idx * n + row) * p + col] = sum;
    }
}


torch::Tensor run_matrix_mul(torch::Tensor tensor1, torch::Tensor tensor2)
{
    const auto d1 = tensor1.size(0);
    const auto d2 = tensor1.size(1);    
    std::cout << "\nTensor1 size: " << d1 << "*" << d2 << std::endl;
    const auto d21 = tensor2.size(0);
    const auto d22 = tensor2.size(1);
    std::cout << "\nTensor2 size: " << d21 << "*" << d22 << std::endl;

    tensor1 = tensor1.cuda();
    tensor2 = tensor2.cuda();

    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(tensor1.device());
    auto result = torch::empty({d1, d2}, options);

    dim3 blockSizeDim(d1, d2, 1);
    dim3 gridSizeDim((d2 + blockSizeDim.x - 1) / blockSizeDim.x, (d1 + blockSizeDim.y - 1) / blockSizeDim.y, 1);

    //  注意 从调用函数 进入到kernel的时候，数据类型有强制转换
    matrixMultiply<<<gridSizeDim, blockSizeDim>>>(tensor1.data_ptr<float>(), tensor2.data_ptr<float>(), result.data_ptr<float>(), d1);
    cudaDeviceSynchronize();  // 添加同步
    cudaError_t cuda_error = cudaGetLastError();
    if (cuda_error != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(cuda_error) << std::endl;
    }

    return {result};
}

torch::Tensor run_matrix_mul2(torch::Tensor tensor1, torch::Tensor tensor2)
{
    const auto d1 = tensor1.size(0);
    const auto d2 = tensor1.size(1);
    std::cout << "\nTensor1 size: " << d1 << "*" << d2 << std::endl;
    const auto d21 = tensor2.size(0);
    const auto d22 = tensor2.size(1);
    std::cout << "\nTensor2 size: " << d21 << "*" << d22 << std::endl;

    tensor1 = tensor1.cuda();
    tensor2 = tensor2.cuda();

    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(tensor1.device());
    auto result = torch::empty({d1, d22}, options);

    dim3 blockSizeDim(16, 16, 1);  // 适当设置线程块大小
    dim3 gridSizeDim((d22 + blockSizeDim.x - 1) / blockSizeDim.x, (d1 + blockSizeDim.y - 1) / blockSizeDim.y, 1);

    matrixMultiply2<<<gridSizeDim, blockSizeDim>>>(tensor1.data_ptr<float>(), tensor2.data_ptr<float>(), result.data_ptr<float>(), d1, d2, d22);
    cudaDeviceSynchronize();  // 添加同步
    cudaError_t cuda_error = cudaGetLastError();
    if (cuda_error != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(cuda_error) << std::endl;
    }

    return {result};
}

torch::Tensor run_batchMatrixMul(torch::Tensor tensor1, torch::Tensor tensor2)
{
    const auto batch_size = tensor1.size(0);
    const auto n = tensor1.size(1);
    const auto m = tensor1.size(2);
    const auto p = tensor2.size(2);
    // std::cout << "\nTensor1 size: " << batch_size << "*" << n << "*" << m <<std::endl;

    tensor1 = tensor1.cuda();
    tensor2 = tensor2.cuda();

    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(tensor1.device());
    auto result = torch::empty({batch_size, n, p}, options);

    dim3 blockSizeDim(16, 16, 1);  // 适当设置线程块大小
    dim3 gridSizeDim((p + blockSizeDim.x - 1) / blockSizeDim.x, (n + blockSizeDim.y - 1) / blockSizeDim.y, batch_size);

    batchMatrixMultiply<<<gridSizeDim, blockSizeDim>>>(tensor1.data_ptr<float>(), tensor2.data_ptr<float>(), result.data_ptr<float>(), batch_size, n, m, p);
    cudaDeviceSynchronize();  // 添加同步
    cudaError_t cuda_error = cudaGetLastError();
    if (cuda_error != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(cuda_error) << std::endl;
    }

    return result;
}

