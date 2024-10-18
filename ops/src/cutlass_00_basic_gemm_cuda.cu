// 10.14 cutlass: gemm

#include <iostream>
#include <sstream>
#include <vector>

#include "helper.h"

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/numeric_types.h"

// cudaError_t CutlassSgemmNN_float(
//     int M,
//     int N,
//     int K,
//     float alpha,
//     float const *A,
//     int lda,
//     float const *B,
//     int ldb,
//     float beta,
//     float *C,
//     int ldc)
// {
//     // using ColumnMajor = cutlass::layout::ColumnMajor;
//     using RowMajor = cutlass::layout::RowMajor;
//     using CutlassGemm = cutlass::gemm::device::Gemm<float,        // Data-type of A matrix
//                                                     RowMajor,  // Layout of A matrix
//                                                     float,        // Data-type of B matrix
//                                                     RowMajor,  // Layout of B matrix
//                                                     float,        // Data-type of C matrix
//                                                     RowMajor>; // Layout of C matrix

//     CutlassGemm gemm_operator;

//     CutlassGemm::Arguments args({M, N, K},      // Gemm Problem dimensions
//                                 {A, lda},       // Tensor-ref for source matrix A
//                                 {B, ldb},       // Tensor-ref for source matrix B
//                                 {C, ldc},       // Tensor-ref for source matrix C
//                                 {C, ldc},       // Tensor-ref for destination matrix D (may be different memory than source C matrix)
//                                 {alpha, beta}); // Scalars used in the Epilogue

//     cutlass::Status status = gemm_operator(args);

//     if (status != cutlass::Status::kSuccess)
//     {
//         return cudaErrorUnknown;
//     }

//     // Return success, if no errors were encountered.
//     return cudaSuccess;
// }

// /// Allocate several matrices in GPU device memory and call a single-precision
// /// CUTLASS GEMM kernel.
// void launcher_CutlassGemmNN_float(const float* mat_A, const float* mat_B, float* mat_C, int M, int N, int K, float alpha, float beta)
// {
//     // Compute leading dimensions for each matrix. 
//     // 如果以行主序的话，则leading dim应该是列维度值
//     int lda = K;
//     int ldb = N;
//     int ldc = N;

//     CutlassSgemmNN_float(M, N, K, alpha, mat_A, lda, mat_B, ldb, beta, mat_C, ldc);
// }



cudaError_t CutlassSgemmNN_half(
    int M,
    int N,
    int K,
    cutlass::half_t alpha,
    cutlass::half_t const *A,
    int lda,
    cutlass::half_t const *B,
    int ldb,
    cutlass::half_t beta,
    cutlass::half_t *C,
    int ldc)
{
    using RowMajor = cutlass::layout::RowMajor;
    using CutlassGemm = cutlass::gemm::device::Gemm<cutlass::half_t,  // Data-type of A matrix
                                                    RowMajor,         // Layout of A matrix
                                                    cutlass::half_t,  // Data-type of B matrix
                                                    RowMajor,         // Layout of B matrix
                                                    cutlass::half_t,  // Data-type of C matrix
                                                    RowMajor>;        // Layout of C matrix

    CutlassGemm gemm_operator;

    CutlassGemm::Arguments args({M, N, K},         // Gemm Problem dimensions
                                {A, lda},          // Tensor-ref for source matrix A
                                {B, ldb},          // Tensor-ref for source matrix B
                                {C, ldc},          // Tensor-ref for source matrix C
                                {C, ldc},          // Tensor-ref for destination matrix D (may be different memory than source C matrix)
                                {alpha, beta});    // Scalars used in the Epilogue

    cutlass::Status status = gemm_operator(args);

    if (status != cutlass::Status::kSuccess)
    {
        return cudaErrorUnknown;
    }

    // Return success, if no errors were encountered.
    return cudaSuccess;
}


void launcher_CutlassGemmNN_half(const __half* mat_A, const __half* mat_B, __half* mat_C, int M, int N, int K, float alpha, float beta)
{
    // Compute leading dimensions for each matrix.
    int lda = K;
    int ldb = N;
    int ldc = N;

    // Convert alpha and beta to cutlass::half_t (CUTLASS's half precision type)
    cutlass::half_t alpha_cutlass = cutlass::half_t(alpha);
    cutlass::half_t beta_cutlass = cutlass::half_t(beta);

    // Cast input matrices from __half* (CUDA type) to cutlass::half_t* (CUTLASS type)
    const cutlass::half_t* mat_A_cutlass = reinterpret_cast<const cutlass::half_t*>(mat_A);
    const cutlass::half_t* mat_B_cutlass = reinterpret_cast<const cutlass::half_t*>(mat_B);
    cutlass::half_t* mat_C_cutlass = reinterpret_cast<cutlass::half_t*>(mat_C);

    // Call the CUTLASS GEMM function for half precision
    CutlassSgemmNN_half(M, N, K, alpha_cutlass, mat_A_cutlass, lda, mat_B_cutlass, ldb, beta_cutlass, mat_C_cutlass, ldc);
}