#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <iostream>

__global__ void tensor_add_kernel(int d1, int d2, const float *t1, const float *t2, float *result)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < d1 && col < d2) {
        int idx = row * d2 + col;
        result[idx] = t1[idx] + t2[idx];
    }
}

torch::Tensor tensor_add(torch::Tensor tensor1, torch::Tensor tensor2)
{
    const auto d1 = tensor1.size(0);
    const auto d2 = tensor1.size(1);
    std::cout << "\n Tensor size: " << d1 << "*" << d2 << std::endl;

    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(tensor1.device());
    auto result = torch::empty({d1, d2}, options);

    // dim3 blockSizeDim(16, 16, 1);  // 调整线程块大小
    dim3 blockSizeDim(1, 1, 1);
    dim3 gridSizeDim((d2 + blockSizeDim.x - 1) / blockSizeDim.x, (d1 + blockSizeDim.y - 1) / blockSizeDim.y, 1);

    tensor_add_kernel<<<gridSizeDim, blockSizeDim>>>(d1, d2, tensor1.data_ptr<float>(), tensor2.data_ptr<float>(), result.data_ptr<float>());

    return {result};
}

