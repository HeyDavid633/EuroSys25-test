#include <cmath>
#include <iostream>
#include <vector>
#include <limits>

#include "cutlass/cutlass.h"
#include "cutlass/arch/memory.h"
#include "cutlass/arch/memory_sm75.h"
#include "cutlass/gemm/device/gemm_complex.h"
#include "cutlass/numeric_types.h"
#include "cutlass/numeric_size.h" // cutlass::bits_to_bytes
#include "cutlass/util/command_line.h"
#include "cutlass/util/host_tensor.h"

#include "cutlass/util/reference/host/gemm_complex.h"
#include "cutlass/util/reference/device/gemm_complex.h"
#include "cutlass/util/reference/host/tensor_reduce.h"
#include "cutlass/util/reference/host/tensor_compare.h"
#include "cutlass/util/reference/host/tensor_norm.h"
#include "cutlass/util/reference/host/tensor_copy.h"
#include "cutlass/util/reference/device/tensor_fill.h"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "cutlass/util/reference/host/error_metrics.h"
#include "cutlass/util/tensor_view_io.h"

#include "cutlass/layout/matrix.h"
#include "cutlass/epilogue/thread/linear_combination.h"

#include "gemm_with_softmax.h"

void cutlass_gemm_softmax_float(
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

    using ThreadblockShape = cutlass::gemm::GemmShape<128, 128, 32>;
    using WarpShape = cutlass::gemm::GemmShape<64, 64, 32>;
    using InstructionShape = cutlass::gemm::GemmShape<16, 8, 16>;
    using OperatorClass = cutlass::arch::OpClassTensorOp;
    using ArchTag = cutlass::arch::Sm80;
    using ApplyShape = cutlass::MatrixShape<1, 1024>;
    using ElementC = float;
    using ElementCompute = float;

    static int const kStages = 3;

    using EpilogueFunctorOp = cutlass::epilogue::thread::LinearCombination<
        ElementC,
        128 / cutlass::sizeof_bits<ElementC>::value,
        ElementCompute,
        ElementCompute>;

    using GemmSoftmax = cutlass::GemmSoftmax<
        float, cutlass::layout::RowMajor,
        float, cutlass::layout::RowMajor,
        float,
        float,
        OperatorClass,
        ArchTag,
        ThreadblockShape,
        WarpShape,
        InstructionShape,
        EpilogueFunctorOp,
        kStages,
        ApplyShape>;

    int64_t ldn = m;
    int64_t lds = ldn;
    int64_t total_elements_A_per_batch = m * k;
    int64_t total_elements_B_per_batch = k * n;
    int64_t total_elements_C_per_batch = m * n;
    int64_t total_elements_D_per_batch = m * n;
    int block_num = (n + GemmSoftmax::ThreadblockShape::kN - 1) / GemmSoftmax::ThreadblockShape::kN;
    int64_t total_elements_partial_norm_per_batch = block_num * m;
    int64_t total_elements_partial_norm = total_elements_partial_norm_per_batch * batch_count;

    // 有一些中间处理的 矩阵 还是需要定义然后随机给初始值
    using ElementSoftmax = float;
    cutlass::DeviceAllocation<float> block_Softmax;
    cutlass::DeviceAllocation<float> block_Norm;
    cutlass::DeviceAllocation<float> block_Sum;
    block_Norm.reset(total_elements_partial_norm);
    block_Sum.reset(total_elements_partial_norm);
    block_Softmax.reset(total_elements_C_per_batch * batch_count);
    cutlass::reference::device::BlockFillRandomUniform(
        block_Softmax.get(), total_elements_C_per_batch * batch_count, 1, ElementSoftmax(5), ElementSoftmax(-5), 0);

    GemmSoftmax::Arguments args(
        {m, n, k},
        batch_count,
        {A, lda}, // 此处 模仿05的实现写成{A, lda}但报错，但类型限制到了float16？
        {B, ldb},
        {C, ldc},
        {C, ldc},
        {alpha, beta},
        {block_Norm.get(), ldn},
        {block_Sum.get(), lds},
        {block_Softmax.get(), ldc},
        total_elements_A_per_batch,
        total_elements_B_per_batch,
        total_elements_C_per_batch,
        total_elements_D_per_batch,
        total_elements_partial_norm_per_batch,
        total_elements_partial_norm_per_batch,
        total_elements_D_per_batch);

    GemmSoftmax gemm_softmax;

    gemm_softmax.initialize(args);

    gemm_softmax();
}

void launcher_gemm_softmax_float(const int batch_count, const float *A, const float *B, float *C, const int M, const int N, const int K, const float alpha, const float beta)
{
    // A, B are non-transpose, column major
    int const lda = K;
    int const ldb = N;
    int const ldc = N;

    int const count_A = batch_count * M * lda;
    int const count_B = batch_count * K * ldb;
    int const count_C = batch_count * M * ldc;

    // the memory is batched along M dimension for A, K dimension for B, and M dimension for C
    long long int batch_stride_A = static_cast<long long int>(M) * static_cast<long long int>(lda);
    long long int batch_stride_B = static_cast<long long int>(K) * static_cast<long long int>(ldb);
    long long int batch_stride_C = static_cast<long long int>(M) * static_cast<long long int>(ldc);

    cutlass_gemm_softmax_float(
        M, N, K, alpha,
        A, lda, batch_stride_A, B, ldb, batch_stride_B, C, ldc, batch_stride_C,
        beta, batch_count);
}
