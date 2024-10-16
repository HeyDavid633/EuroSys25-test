#include <cmath>
#include <iostream>
#include <vector>
#include <limits>

#include "cutlass/cutlass.h"
#include "cutlass/arch/memory.h"
#include "cutlass/arch/memory_sm75.h"
#include "cutlass/gemm/device/gemm_complex.h"
#include "cutlass/numeric_types.h"
#include "cutlass/numeric_size.h"// cutlass::bits_to_bytes
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


/// Returns true if the environment and Toolkit support this
bool supported(bool verbose = true)
{

    // Ampere Tensor Core operations exposed with mma.sync and ldmatrix are first available
    // in CUDA 11.0.
    //
    // CUTLASS must be compiled with CUDA 11.0 Toolkit to run these examples.
    if (!(__CUDACC_VER_MAJOR__ >= 11))
    {
        if (verbose)
        {
            std::cerr << "Ampere Tensor Core operations must be compiled with CUDA 11.0 Toolkit or later." << std::endl;
        }
        return false;
    }

    cudaDeviceProp props;

    cudaError_t error = cudaGetDeviceProperties(&props, 0);
    if (error != cudaSuccess)
    {
        if (verbose)
        {
            std::cerr << "cudaGetDeviceProperties() returned an error: " << cudaGetErrorString(error) << std::endl;
        }
        return false;
    }

    if (!((props.major * 10 + props.minor) >= 80))
    {
        if (verbose)
        {
            std::cerr << "Ampere Tensor Core operations must be run on a machine with compute capability at least 80."
                        << std::endl;
        }
        return false;
    }

    return true;
}


void cutlass_gemm_softmax_float(const int batch_count, float const *A, float const *B, float *C, float *D, const int m = 1024, const int n = 1024, const int k = 1024, const float alpha = 1.0, const float beta = 0.0)
{
    // using ElementA = cutlass::half_t;
    // using ElementB = cutlass::half_t;
    // using ElementC = cutlass::half_t;
    using ElementA = float;
    using ElementB = float;
    using ElementC = float;
    using ElementCompute = float;
    using ElementD = ElementC;
    using ElementSoftmax = ElementC;

    using LayoutA = cutlass::layout::RowMajor;
    using LayoutB = cutlass::layout::ColumnMajor;

    using ThreadblockShape = cutlass::gemm::GemmShape<128, 128, 32>;
    using WarpShape = cutlass::gemm::GemmShape<64, 64, 32>;
    using InstructionShape = cutlass::gemm::GemmShape<16, 8, 16>;

    using OperatorClass = cutlass::arch::OpClassTensorOp;
    using ArchTag = cutlass::arch::Sm80;

    // ApplyShape impacts the final Softmax performance a lot.
    // Set ApplyShape::kColumn to be the next multiple of 32 number that is after
    // (gemm_N / alignment).
    // Set ApplyShape::kRow to max(1, 128 / ApplyShape::kColumn).
    using ApplyShape = cutlass::MatrixShape<1, 1024>;

    static int const kStages = 3;

    /// Linear scaling operator
    using EpilogueFunctorOp = cutlass::epilogue::thread::LinearCombination<
        ElementC,
        128 / cutlass::sizeof_bits<ElementC>::value,
        ElementCompute,
        ElementCompute>;

    using GemmSoftmax = cutlass::GemmSoftmax<
        ElementA, LayoutA,
        ElementB, LayoutB,
        ElementC,
        ElementCompute,
        OperatorClass,
        ArchTag,
        ThreadblockShape,
        WarpShape,
        InstructionShape,
        EpilogueFunctorOp,
        kStages,
        ApplyShape>;

    using ElementNorm = float;
    using ElementSum = float;
    using LayoutC = cutlass::layout::RowMajor;

    cutlass::DeviceAllocation<ElementSoftmax> block_Softmax;
    cutlass::DeviceAllocation<ElementNorm> block_Norm;
    cutlass::DeviceAllocation<ElementSum> block_Sum;

    int block_num = (n + 128 - 1) / 128; //int block_num = (options.problem_size.n() + GemmSoftmax::ThreadblockShape::kN - 1) / GemmSoftmax::ThreadblockShape::kN;

    cutlass::gemm::GemmCoord problem = {m, n, k};

    int64_t lda = LayoutA::packed({problem.m(), problem.k()}).stride(0);
    int64_t ldb = LayoutB::packed({problem.k(), problem.n()}).stride(0);
    int64_t ldc = LayoutC::packed({problem.m(), problem.n()}).stride(0);

    // fixed rowmajor for norm and sum
    int64_t ldn = problem.m();
    int64_t lds = ldn;

    int64_t total_elements_A_per_batch = problem.m() * problem.k();
    int64_t total_elements_B_per_batch = problem.k() * problem.n();
    int64_t total_elements_C_per_batch = problem.m() * problem.n();
    int64_t total_elements_D_per_batch = problem.m() * problem.n();
    int64_t total_elements_partial_norm_per_batch = block_num * problem.m();

    int64_t total_elements_A = total_elements_A_per_batch * batch_count;
    int64_t total_elements_B = total_elements_B_per_batch * batch_count;
    int64_t total_elements_C = total_elements_C_per_batch * batch_count;
    int64_t total_elements_D = total_elements_D_per_batch * batch_count;
    int64_t total_elements_partial_norm = total_elements_partial_norm_per_batch * batch_count;

    block_Norm.reset(total_elements_partial_norm);
    block_Sum.reset(total_elements_partial_norm);
    block_Softmax.reset(total_elements_D);
    cutlass::reference::device::BlockFillRandomUniform(
            block_Softmax.get(), total_elements_D, 1, ElementSoftmax(5), ElementSoftmax(-5), 0);


    GemmSoftmax::Arguments args(
        {m, n, k},
        batch_count,
        {A, lda},
        {B, ldb},
        {C, ldc},
        {D, ldc},
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

    //launch
    gemm_softmax();
};


void launcher_gemm_softmax_float(const int batch_count, const float* mat_A, const float* mat_B, float* mat_C, float *mat_D, const int M, const int N, const int K, const float alpha, const float beta)
{
    if (!supported()) exit(0);

    cutlass_gemm_softmax_float(batch_count, mat_A, mat_B, mat_C, mat_D, M, N, K, alpha, beta);

}