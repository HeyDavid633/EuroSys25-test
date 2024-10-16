#include <torch/extension.h>
#include <torch/serialize/tensor.h>

#include "helper.h"
#include <iostream>

#define CHECK_CUDA(x) \
    TORCH_CHECK(x.type().is_cuda(), #x, " must be a CUDA tensor ")
#define CHECK_CONTIGUOUS(x) \
    TORCH_CHECK(x.is_contiguous(), #x, " must be contiguous ")
#define CHECK_INPUT(x) \
    CHECK_CUDA(x);     \
    CHECK_CONTIGUOUS(x)


void launcher_gemm_softmax_float(const int batch_count, const float* mat_A, const float* mat_B, float* mat_C, float *mat_D, const int M, const int N, const int K, const float alpha, const float beta);


// 改函数的名称 前缀需要（最好）和文件名保持一致
void cutlass_35_gemm_softmax_gpu(at::Tensor mat_A, at::Tensor mat_B, at::Tensor mat_C, at::Tensor mat_D, const float alpha, const float beta)
{
    const int batch_count = mat_A.size(0);
    const int M = mat_A.size(1);
    const int K = mat_A.size(2);
    const int N = mat_B.size(2);

    // std::cout << "batch_cnt:" << batch_count <<"  M:" << M << "  K:" << K << "  N:" << N << std::endl;

    launcher_gemm_softmax_float(batch_count, mat_A.data_ptr<float>(), mat_B.data_ptr<float>(), mat_C.data_ptr<float>(), mat_D.data_ptr<float>(), M, N, K, alpha, beta);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("forward", &cutlass_35_gemm_softmax_gpu, "my cutlass link demo 35_gemm_softmax (CUDA)");
}