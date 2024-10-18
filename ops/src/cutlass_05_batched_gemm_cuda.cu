
#include <iostream>
#include <vector>

#include "cutlass/cutlass.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/gemm/device/gemm_array.h"
#include "cutlass/gemm/device/gemm_batched.h"

#pragma warning(disable : 4503)

cudaError_t cutlass_strided_batched_sgemm(
    int m,
    int n,
    int k,
    float alpha,
    float const *A,
    int lda,
    long long int batch_stride_A,
    float const *B,
    int ldb,
    long long int batch_stride_B,
    float *C,
    int ldc,
    long long int batch_stride_C,
    float beta,
    int batch_count)
{

    using Gemm = cutlass::gemm::device::GemmBatched<
        float, cutlass::layout::RowMajor,
        float, cutlass::layout::RowMajor,
        float, cutlass::layout::RowMajor>;

    Gemm gemm_op;

    cutlass::Status status = gemm_op({{m, n, k},
                                      {A, lda},
                                      batch_stride_A,
                                      {B, ldb},
                                      batch_stride_B,
                                      {C, ldc},
                                      batch_stride_C,
                                      {C, ldc},
                                      batch_stride_C,
                                      {alpha, beta},
                                      batch_count});

    if (status != cutlass::Status::kSuccess)
    {
        return cudaErrorUnknown;
    }

    return cudaSuccess;
}


void launcher_batched_gemm_float(const int batch_count, const float* mat_A, const float* mat_B, float* mat_C, const int M, const int N, const int K, const float alpha, const float beta)
{
    // A, B are non-transpose, column major
    int const lda = K;
    int const ldb = N;
    int const ldc = N;

    // int const count_A = batch_count * M * lda;
    // int const count_B = batch_count * K * ldb;
    // int const count_C = batch_count * M * ldc;

    // the memory is batched along M dimension for A, K dimension for B, and M dimension for C
    long long int batch_stride_A = static_cast<long long int>(M) * static_cast<long long int>(lda);
    long long int batch_stride_B = static_cast<long long int>(K) * static_cast<long long int>(ldb);
    long long int batch_stride_C = static_cast<long long int>(M) * static_cast<long long int>(ldc);

    cutlass_strided_batched_sgemm(
        M, N, K, alpha,
        mat_A, lda, batch_stride_A, mat_B, ldb, batch_stride_B, mat_C, ldc, batch_stride_C, 
        beta, batch_count);
}
