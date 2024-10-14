// 10.14 cutlass: 00_basic_gemm_1 
//
// 无脑链进去成功后，希望带有传入传出参数的过程

#include <iostream>
#include <sstream>
#include <vector>

#include "helper.h"
#include "cutlass/gemm/device/gemm.h"


/// Define a CUTLASS GEMM template and launch a GEMM kernel.
cudaError_t CutlassSgemmNN_float(
    int M,
    int N,
    int K,
    float alpha,
    float const *A,
    int lda,
    float const *B,
    int ldb,
    float beta,
    float *C,
    int ldc)
{

    // Define type definition for single-precision CUTLASS GEMM with column-major
    // input matrices and 128x128x8 threadblock tile size (chosen by default).
    //
    // To keep the interface manageable, several helpers are defined for plausible compositions
    // including the following example for single-precision GEMM. Typical values are used as
    // default template arguments. See `cutlass/gemm/device/default_gemm_configuration.h` for more details.
    //
    // To view the full gemm device API interface, see `cutlass/gemm/device/gemm.h`

    // using ColumnMajor = cutlass::layout::ColumnMajor;
    using RowMajor = cutlass::layout::RowMajor;

    using CutlassGemm = cutlass::gemm::device::Gemm<float,        // Data-type of A matrix
                                                    RowMajor,  // Layout of A matrix
                                                    float,        // Data-type of B matrix
                                                    RowMajor,  // Layout of B matrix
                                                    float,        // Data-type of C matrix
                                                    RowMajor>; // Layout of C matrix

    // Define a CUTLASS GEMM type
    CutlassGemm gemm_operator;

    // Construct the CUTLASS GEMM arguments object.
    //
    // One of CUTLASS's design patterns is to define gemm argument objects that are constructible
    // in host code and passed to kernels by value. These may include pointers, strides, scalars,
    // and other arguments needed by Gemm and its components.
    //
    // The benefits of this pattern are (1.) a structured, composable strategy for passing host-constructible
    // arguments to kernels and (2.) minimized initialization overhead on kernel entry.
    //
    CutlassGemm::Arguments args({M, N, K},      // Gemm Problem dimensions
                                {A, lda},       // Tensor-ref for source matrix A
                                {B, ldb},       // Tensor-ref for source matrix B
                                {C, ldc},       // Tensor-ref for source matrix C
                                {C, ldc},       // Tensor-ref for destination matrix D (may be different memory than source C matrix)
                                {alpha, beta}); // Scalars used in the Epilogue

    //
    // Launch the CUTLASS GEMM kernel.
    //

    cutlass::Status status = gemm_operator(args);

    //
    // Return a cudaError_t if the CUTLASS GEMM operator returned an error code.
    //

    if (status != cutlass::Status::kSuccess)
    {
        return cudaErrorUnknown;
    }

    // Return success, if no errors were encountered.
    return cudaSuccess;
}



/// Allocate several matrices in GPU device memory and call a single-precision
/// CUTLASS GEMM kernel.
void CutlassGemmNN_float_launcher(const float* mat_A, const float* mat_B, float* mat_C, int M, int N, int K, float alpha, float beta)
{
    // Compute leading dimensions for each matrix. 
    // 如果以行主序的话，则leading dim应该是列维度值
    int lda = K;
    int ldb = N;
    int ldc = N;

    CutlassSgemmNN_float(M, N, K, alpha, mat_A, lda, mat_B, ldb, beta, mat_C, ldc);
}
