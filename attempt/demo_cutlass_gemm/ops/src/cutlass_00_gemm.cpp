#include <torch/extension.h>
#include <torch/serialize/tensor.h>

#include "helper.h"
#include <iostream>

#define CHECK_CUDA(x) \
  TORCH_CHECK(x.type().is_cuda(), #x, " must be a CUDAtensor ")
#define CHECK_CONTIGUOUS(x) \
  TORCH_CHECK(x.is_contiguous(), #x, " must be contiguous ")
#define CHECK_INPUT(x) \
  CHECK_CUDA(x);       \
  CHECK_CONTIGUOUS(x)


cudaError_t TestCutlassGemm(int M, int N, int K, float alpha, float beta);


void cutlass_00_gemm_gpu()
{
    int problem[3] = {128, 128, 128};
    float scalars[2] = {1, 0};

    cudaError_t result = TestCutlassGemm(
        problem[0], // GEMM M dimension
        problem[1], // GEMM N dimension
        problem[2], // GEMM K dimension
        scalars[0], // alpha
        scalars[1]  // beta
    );

    if (result == cudaSuccess)
    {
        std::cout << "Passed." << std::endl;
    }
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &cutlass_00_gemm_gpu, "my cutlass link demo 00_gemm (CUDA)");
}