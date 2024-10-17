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

struct Options
{
    cutlass::gemm::GemmCoord problem_size;
    int batch_count;
    unsigned seed;
    float alpha;
    float beta;
    float const *A;
    float const *B;
    float *C;


    Options() : problem_size({16, 24, 64}),
                batch_count(16),
                seed(2022),
                alpha(1),
                beta(0)
    {
    }

    // Set parameters directly
    void get_para(int batch_count, float const *A, float const *B, float *C, int m, int n, int k, float alpha, float beta)
    {
        this->batch_count = batch_count;
        problem_size = cutlass::gemm::GemmCoord(m, n, k);
        this->alpha = alpha;
        this->beta = beta;
        this->A = A;
        this->B = B;
        this->C = C;
    }

    // Returns true if the environment and Toolkit support this
    bool supported(bool verbose = true) const
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
};

struct Testbed
{
    // using ElementA = float;
    // using ElementB = float;
    // using ElementC = float; // 不可以用float的类型？--- 否则出错

    using ElementA = cutlass::half_t;
    using ElementB = cutlass::half_t;
    using ElementC = cutlass::half_t;
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

    using ElementNorm = typename GemmSoftmax::ElementNorm;
    using ElementSum = typename GemmSoftmax::ElementSum;
    using LayoutC = typename GemmSoftmax::LayoutC;

    Options const &options;

    cutlass::DeviceAllocation<ElementA> block_A;
    cutlass::DeviceAllocation<ElementB> block_B;
    cutlass::DeviceAllocation<ElementC> block_C;
    cutlass::DeviceAllocation<ElementD> block_D;
    cutlass::DeviceAllocation<ElementSoftmax> block_Softmax;
    cutlass::DeviceAllocation<ElementNorm> block_Norm;
    cutlass::DeviceAllocation<ElementSum> block_Sum;

    int block_num = (options.problem_size.n() + GemmSoftmax::ThreadblockShape::kN - 1) / GemmSoftmax::ThreadblockShape::kN;

    cutlass::gemm::GemmCoord problem = options.problem_size;

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

    int64_t total_elements_A = total_elements_A_per_batch * options.batch_count;
    int64_t total_elements_B = total_elements_B_per_batch * options.batch_count;
    int64_t total_elements_C = total_elements_C_per_batch * options.batch_count;
    int64_t total_elements_D = total_elements_D_per_batch * options.batch_count;
    int64_t total_elements_partial_norm = total_elements_partial_norm_per_batch * options.batch_count;

    Testbed(Options const &options_) : options(options_) {}


    /// Random initialization
    void initialize()
    {
        block_A.reset(total_elements_A);
        block_B.reset(total_elements_B);
        block_C.reset(total_elements_C);
        block_D.reset(total_elements_D);
        cutlass::reference::device::BlockFillRandomUniform(
            block_A.get(), total_elements_A, options.seed, ElementA(5), ElementA(-5), 0);

        cutlass::reference::device::BlockFillRandomUniform(
            block_B.get(), total_elements_B, options.seed + 1, ElementB(5), ElementB(-5), 0);

        cutlass::reference::device::BlockFillRandomUniform(
            block_C.get(), total_elements_C, options.seed + 2, ElementC(5), ElementC(-5), 0);

        cutlass::reference::device::BlockFillRandomUniform(
            block_D.get(), total_elements_D, options.seed + 3, ElementD(5), ElementD(-5), 0);

        block_Norm.reset(total_elements_partial_norm);
        block_Sum.reset(total_elements_partial_norm);
        block_Softmax.reset(total_elements_D);
        cutlass::reference::device::BlockFillRandomUniform(
            block_Softmax.get(), total_elements_D, options.seed + 3, ElementSoftmax(5), ElementSoftmax(-5), 0);
    }

    void execute_device_kernel()
    {
        GemmSoftmax::Arguments args(
            options.problem_size,
            options.batch_count,
            {block_A.get(), lda},   //此处 模仿05的实现写成{A, lda}但报错，但类型限制到了float16
            {block_B.get(), ldb},
            {block_C.get(), ldc},
            {block_D.get(), ldc},
            {ElementCompute(options.alpha),
             ElementCompute(options.beta)},
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

        // Launch ready
        GemmSoftmax gemm_softmax;

        // Initialize
        gemm_softmax.initialize(args);

        // Run
        gemm_softmax();

        std::cout << "call success !" << std::endl;
    }
};

void launcher_gemm_softmax_float(const int batch_count, const float *A, const float *B, float *C, const int M, const int N, const int K, const float alpha, const float beta)
{
    // Options parsing
    Options options;
    options.get_para(batch_count, A, B, C, M, N, K, alpha, beta);

    if (!options.supported())
        exit(0);

    // init this func
    Testbed testbed(options);

    // run
    testbed.initialize();
    testbed.execute_device_kernel();
}
