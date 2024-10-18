#include <torch/extension.h>
#include <torch/serialize/tensor.h>
#include <ATen/ATen.h>

#include "helper.h"
#include <iostream>
#include <cuda_fp16.h>

#define CHECK_CUDA(x) \
    TORCH_CHECK(x.type().is_cuda(), #x, " must be a CUDA tensor ")
#define CHECK_CONTIGUOUS(x) \
    TORCH_CHECK(x.is_contiguous(), #x, " must be contiguous ")
#define CHECK_INPUT(x) \
    CHECK_CUDA(x);     \
    CHECK_CONTIGUOUS(x)

// CUDA 函数的入口
// void launcher_CutlassGemmNN_float(const float *mat_A, const float *mat_B, float *mat_C, int M, int N, int K, float alpha, float beta);

// // 这样封装使得 在cpp文件中 才会出现 at::Tensor, 在.cu文件中完全就是cuda计算的函数，看不出来张量的痕迹
// void cutlass_00_basic_gemm_gpu(at::Tensor mat_A, at::Tensor mat_B, at::Tensor mat_C, float alpha, float beta)
// {
//     const int M = mat_A.size(0);
//     const int K = mat_A.size(1);
//     const int N = mat_B.size(1);

//     // std::cout << "M:" << M << "  K:" << K << "  N:" << N << std::endl;

//     launcher_CutlassGemmNN_float(mat_A.data_ptr<float>(), mat_B.data_ptr<float>(), mat_C.data_ptr<float>(), M, N, K, alpha, beta);
// }


// Declare the function from the .cu file
void launcher_CutlassGemmNN_half(const __half* mat_A, const __half* mat_B, __half* mat_C, int M, int N, int K, float alpha, float beta);

// Wrapper function exposed to Python, using ATen Tensor interface
void cutlass_00_basic_gemm_gpu(at::Tensor mat_A, at::Tensor mat_B, at::Tensor mat_C, float alpha, float beta)
{
    const int M = mat_A.size(0);
    const int K = mat_A.size(1);
    const int N = mat_B.size(1);

    // Ensure tensors are contiguous and on the CUDA device
    mat_A = mat_A.contiguous();
    mat_B = mat_B.contiguous();
    mat_C = mat_C.contiguous();

    // Pass data pointers of ATen tensors, ensuring we use at::Half for half-precision tensors
    launcher_CutlassGemmNN_half(
        reinterpret_cast<const __half*>(mat_A.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(mat_B.data_ptr<at::Half>()),
        reinterpret_cast<__half*>(mat_C.data_ptr<at::Half>()),
        M, N, K, alpha, beta
    );
}

// Binding for Python with Pybind11
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("forward", &cutlass_00_basic_gemm_gpu, "my cutlass link demo 00_gemm (CUDA)");
}
