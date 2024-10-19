#include <torch/extension.h>
#include <torch/serialize/tensor.h>

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

// void launcher_batched_gemm_float(const int batch_count, const float *mat_A, const float *mat_B, float *mat_C, const int M, const int N, const int K, const float alpha, const float beta);

// void cutlass_05_batched_gemm_gpu(at::Tensor mat_A, at::Tensor mat_B, at::Tensor mat_C, const float alpha, const float beta)
// {
//     const int batch_count = mat_A.size(0);
//     const int M = mat_A.size(1);
//     const int K = mat_A.size(2);
//     const int N = mat_B.size(2);

//     // std::cout << "batch_cnt:" << batch_count <<"  M:" << M << "  K:" << K << "  N:" << N << std::endl;

//     launcher_batched_gemm_float(batch_count, mat_A.data_ptr<float>(), mat_B.data_ptr<float>(), mat_C.data_ptr<float>(), M, N, K, alpha, beta);
// }


void launcher_batched_gemm_half(const int batch_count, const __half* mat_A, const __half* mat_B, __half* mat_C, int M, int N, int K, float alpha, float beta);


void cutlass_05_batched_gemm_gpu(at::Tensor mat_A, at::Tensor mat_B, at::Tensor mat_C, const float alpha, const float beta)
{
    const int batch_count = mat_A.size(0);
    const int M = mat_A.size(1);
    const int K = mat_A.size(2);
    const int N = mat_B.size(2);

    mat_A = mat_A.contiguous();
    mat_B = mat_B.contiguous();
    mat_C = mat_C.contiguous();

    launcher_batched_gemm_half(
        batch_count,
        reinterpret_cast<const __half*>(mat_A.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(mat_B.data_ptr<at::Half>()),
        reinterpret_cast<__half*>(mat_C.data_ptr<at::Half>()),
        M, N, K, alpha, beta
    );
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("forward", &cutlass_05_batched_gemm_gpu, "my cutlass link demo 05_batched_gemm (CUDA)");
}