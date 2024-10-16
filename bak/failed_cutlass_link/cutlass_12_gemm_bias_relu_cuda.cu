// 2024.10.15
// 
// 两个问题：
// 1. 实际上在Transformer中做的所有运算都是 bmm 或者都是带有 批量 的，但此处的以及后面的实现都只是二维、非批量的
// 2. 此处的编译的问题：对于 line135 Gemm::Arguments arguments{ ... } 参数始终不正确，M N K的值在传入时才能确定
//
#include <algorithm>
#include <iostream>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/epilogue/thread/linear_combination_relu.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/reference/device/gemm.h"
#include "cutlass/util/reference/host/tensor_compare.h"
#include "cutlass/util/reference/host/tensor_copy.h"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "cutlass/util/tensor_view_io.h"
#include "helper.h"

// The code section below describes datatype for input, output matrices and computation between
// elements in input matrices.
using ElementAccumulator = float;                  // <- data type of accumulator
using ElementComputeEpilogue = ElementAccumulator; // <- data type of epilogue operations
using ElementInputA = float;             // <- data type of elements in input matrix A
using ElementInputB = float;             // <- data type of elements in input matrix B
using ElementOutput = float;                       // <- data type of elements in output matrix D
// using ElementInputA = cutlass::half_t;             // <- data type of elements in input matrix A
// using ElementInputB = cutlass::half_t;             // <- data type of elements in input matrix B

using LayoutInputA = cutlass::layout::RowMajor;
using LayoutInputB = cutlass::layout::RowMajor;
using LayoutOutput = cutlass::layout::RowMajor;

// This code section describes whether you want to use tensor cores or regular SIMT cores on GPU SM
using MMAOp = cutlass::arch::OpClassTensorOp;

// This code section describes CUDA SM architecture number
using SmArch = cutlass::arch::Sm75;  // 这一块意味着换了机器还要手动换？

// This code section describes the tile size a thread block will compute
using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<128, 128, 32>; // <- threadblock tile M = 128, N = 128, K = 32
// This code section describes tile size a warp will compute
using ShapeMMAWarp = cutlass::gemm::GemmShape<64, 64, 32>; // <- warp tile M = 64, N = 64, K = 32
// This code section describes the size of MMA op
using ShapeMMAOp = cutlass::gemm::GemmShape<16, 8, 8>; // <- MMA Op tile M = 16, N = 8, K = 8

// This code section describes how threadblocks are scheduled on GPU
using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;

// 对于我的代码样例中为gelu，但Transformer原文用的是relu
// https://github.com/NVIDIA/cutlass/blob/cutlass-3.5.0/include/cutlass/epilogue/thread/linear_combination_gelu.h
// 原例中epilogue是LinearCombinationRelu.  
using EpilogueOp = cutlass::epilogue::thread::LinearCombinationRelu<
    ElementOutput,                                        // <- data type of output matrix
    128 / cutlass::sizeof_bits<ElementOutput>::value,     // <- this is the number of elements per
                                                          // vectorized memory access. For half
                                                          // precision, it's 8 elements. This becomes
                                                          // the vector width of math instructions in
                                                          // epilogue too
    ElementAccumulator,                                   // <- data type of accumulator
    ElementComputeEpilogue,                               // <- data type for alpha in linear combination function
    cutlass::epilogue::thread::ScaleType::NoBetaScaling>; // <- alpha x C + bias

// Number of pipelines you want to use
constexpr int NumStages = 2;


using Gemm = cutlass::gemm::device::Gemm<ElementInputA,
                                         LayoutInputA,
                                         ElementInputB,
                                         LayoutInputB,
                                         ElementOutput,
                                         LayoutOutput,
                                         ElementAccumulator,
                                         MMAOp,
                                         SmArch,
                                         ShapeMMAThreadBlock,
                                         ShapeMMAWarp,
                                         ShapeMMAOp,
                                         EpilogueOp,
                                         SwizzleThreadBlock,
                                         NumStages>;

int launcher_gemm_bias_relu(const float *mat_A, const float *mat_B, const float *mat_C, float *mat_D, const int M, const int N, const int K)
{
    bool notSupported = false;
    // Turing Tensor Core operations exposed with mma.sync are first available in CUDA 10.2.
    //
    // CUTLASS must be compiled with CUDA 10.1 Toolkit to run these examples.
    if (!(__CUDACC_VER_MAJOR__ > 10 || (__CUDACC_VER_MAJOR__ == 10 && __CUDACC_VER_MINOR__ >= 2)))
    {
        std::cerr << "Turing Tensor Core operations must be compiled with CUDA 10.2 Toolkit or later." << std::endl;
        notSupported = true;
    }
    cudaDeviceProp props;

    cudaError_t error = cudaGetDeviceProperties(&props, 0);
    if (error != cudaSuccess)
    {
        std::cerr << "cudaGetDeviceProperties() returned an error: " << cudaGetErrorString(error) << std::endl;
        return -1;
    }

    if (!(props.major * 10 + props.minor >= 75))
    {
        std::cerr << "Turing Tensor Ops must be run on a machine with compute capability at least 75."
                  << std::endl;
        notSupported = true;
    }

    if (notSupported)
    {
        std::cout << "Damie! not support this GPU Arch" << std::endl;
        // Returning zero so this test passes on older Toolkits. Its actions are no-op.
        return 0;
    }


    const int length_m = M;
    const int length_n = N;
    const int length_k = K;
    // Create a tuple of problem size for matrix multiplication
    cutlass::gemm::GemmCoord problem_size(length_m, length_n, length_k);


    // Initialize alpha for dot product computation
    ElementComputeEpilogue alpha = ElementComputeEpilogue(1);

    // Split K dimension into 1 partitions
    int split_k_slices = 1;

    // Create a tuple of gemm kernel arguments. This is later passed as arguments to launch
    // instantiated CUTLASS kernel
    typename Gemm::Arguments arguments{
        problem_size,          // <- problem size of matrix multiplication
        mat_A, // <- reference to matrix A on device
        mat_B, // <- reference to matrix B on device

        mat_C,
        mat_D, // <- reference to matrix D on device
        {alpha},               // <- alpha
        split_k_slices};       // <- k-dimension split factor


    // Using the arguments, query for extra workspace required for matrix multiplication computation
    size_t workspace_size = Gemm::get_workspace_size(arguments);

    // Allocate workspace memory
    cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

    // Instantiate CUTLASS kernel depending on templates
    Gemm gemm_op;

    // Check the problem size is supported or not
    cutlass::Status status = gemm_op.can_implement(arguments);
    CUTLASS_CHECK(status);

    // Initialize CUTLASS kernel with arguments and workspace pointer
    status = gemm_op.initialize(arguments, workspace.get());
    CUTLASS_CHECK(status);

    // Launch initialized CUTLASS kernel
    status = gemm_op();
    CUTLASS_CHECK(status);

    return 0;
}