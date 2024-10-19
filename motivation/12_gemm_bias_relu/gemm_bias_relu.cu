/***************************************************************************************************
 * 2024.10.19
 *
 * 使用不同序列长度的 gemm_bias_relu 精度需要全部修改为 half
 * 以A100上的性能为准
 * as my motivation 2
 **************************************************************************************************/

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
using ElementInputA = cutlass::half_t;             // <- data type of elements in input matrix A
using ElementInputB = cutlass::half_t;             // <- data type of elements in input matrix B
using ElementOutput = float;                       // <- data type of elements in output matrix D

using LayoutInputA = cutlass::layout::ColumnMajor;
using LayoutInputB = cutlass::layout::ColumnMajor;
using LayoutOutput = cutlass::layout::ColumnMajor;

// This code section describes whether you want to use tensor cores or regular SIMT cores on GPU SM
using MMAOp = cutlass::arch::OpClassTensorOp;

// This code section describes CUDA SM architecture number
using SmArch = cutlass::arch::Sm80;

// This code section describes the tile size a thread block will compute
using ShapeMMAThreadBlock =
    cutlass::gemm::GemmShape<128, 128, 32>; // <- threadblock tile M = 128, N = 128, K = 32
// This code section describes tile size a warp will compute
using ShapeMMAWarp = cutlass::gemm::GemmShape<64, 64, 32>; // <- warp tile M = 64, N = 64, K = 32
// This code section describes the size of MMA op
using ShapeMMAOp = cutlass::gemm::GemmShape<16, 8, 8>; // <- MMA Op tile M = 16, N = 8, K = 8

// This code section describes how threadblocks are scheduled on GPU
using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;

// Define the epilogue operation as LinearCombinationRelu. This is approximately equal to
//
//    d_ij = max(0, alpha * sum_k(a_ik * b_kj) + c_ij )
//
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

int run()
{

    const int length_m = 32768;
    const int length_n = 2048;
    const int length_k = 512;

    // Create a tuple of problem size for matrix multiplication
    cutlass::gemm::GemmCoord problem_size(length_m, length_n, length_k);

    // Initialize tensors using CUTLASS helper functions
    cutlass::HostTensor<ElementInputA, LayoutInputA> tensor_a(
        problem_size.mk()); // <- Create matrix A with dimensions M x K
    cutlass::HostTensor<ElementInputB, LayoutInputB> tensor_b(
        problem_size.kn()); // <- Create matrix B with dimensions K x N

    cutlass::HostTensor<ElementOutput, LayoutOutput> tensor_c_bias(
        {problem_size.m(), 1}); // <- Create matrix C with dimensions M x 1

    cutlass::HostTensor<ElementOutput, LayoutOutput> tensor_d(
        problem_size.mn()); // <- Create matrix D with dimensions M x N used to store output from
                            // CUTLASS kernel
    cutlass::HostTensor<ElementOutput, LayoutOutput> tensor_ref_d(
        problem_size.mn()); // <- Create matrix D with dimensions M x N used to store output from
                            // reference kernel

    // Fill input and output matrices on host using CUTLASS helper functions
    cutlass::reference::host::TensorFillRandomUniform(
        tensor_a.host_view(),
        1,
        ElementInputA(4),
        ElementInputA(-4),
        0); // <- Fill matrix A on host with uniform-distribution random data
    cutlass::reference::host::TensorFillRandomUniform(
        tensor_b.host_view(),
        1,
        ElementInputB(4),
        ElementInputB(-4),
        0); // <- Fill matrix B on host with uniform-distribution random data
    cutlass::reference::host::TensorFillRandomUniform(
        tensor_c_bias.host_view(),
        1,
        ElementOutput(4),
        ElementOutput(-4),
        0); // <- Fill matrix C on host with uniform-distribution random data
    cutlass::reference::host::TensorFill(
        tensor_d.host_view()); // <- fill matrix D on host with zeros
    cutlass::reference::host::TensorFill(
        tensor_ref_d.host_view()); // <- fill matrix D for reference on host with zeros

    // Copy data from host to GPU
    tensor_a.sync_device();
    tensor_b.sync_device();
    tensor_c_bias.sync_device();
    tensor_d.sync_device();
    tensor_ref_d.sync_device();

    // Initialize alpha for dot product computation
    ElementComputeEpilogue alpha = ElementComputeEpilogue(1);

    // Split K dimension into 1 partitions
    int split_k_slices = 1;

    cudaError_t result;
    cudaEvent_t events[2];
    int const kIterations = 20;

    for (cudaEvent_t &evt : events)
    {
        result = cudaEventCreate(&evt);
        if (result != cudaSuccess)
        {
            std::cerr << "cudaEventCreate failed with error " << cudaGetErrorString(result) << std::endl;
            return false;
        }
    }

    result = cudaEventRecord(events[0]);
    for (int iter = 0; iter < kIterations; ++iter)
    {
        typename Gemm::Arguments arguments{
            problem_size,          // <- problem size of matrix multiplication
            tensor_a.device_ref(), // <- reference to matrix A on device
            tensor_b.device_ref(), // <- reference to matrix B on device

            {tensor_c_bias.device_data(), 0}, // <- the C matrix is treated as the bias vector. We can enable the GEMM
                                              //    to project away the N dimension by setting the stride to zero.

            tensor_d.device_ref(), // <- reference to matrix D on device
            {alpha},               // <- alpha
            split_k_slices};       // <- k-dimension split factor

        Gemm gemm_op;

        gemm_op();
    }
    result = cudaEventRecord(events[1]);
    result = cudaDeviceSynchronize();
    float elapsed_ms = 0;
    result = cudaEventElapsedTime(&elapsed_ms, events[0], events[1]);

    int64_t flops = int64_t(problem_size.m()) * int64_t(problem_size.n()) * int64_t(problem_size.k()) * 2;

    double gflops_per_second = double(flops) * kIterations / double(elapsed_ms / 1000.0f) / double(1.0e9);

    std::cout << "    GFLOPs: " << gflops_per_second << "  GFLOPs" << std::endl;

    //
    // Create instantiation for device reference gemm kernel
    //

    cutlass::reference::device::Gemm<ElementInputA,
                                     LayoutInputA,
                                     ElementInputB,
                                     LayoutInputB,
                                     ElementOutput,
                                     LayoutOutput,
                                     ElementComputeEpilogue,
                                     ElementComputeEpilogue>
        gemm_device_reference;

    // Launch device reference to compute strictly the product A * B
    gemm_device_reference(
        problem_size,
        alpha,
        tensor_a.device_ref(),
        tensor_b.device_ref(),
        0,
        tensor_ref_d.device_ref());

    // Wait for kernels to finish
    cudaDeviceSynchronize();

    // Copy output data from CUTLASS and reference kernel to host for comparison
    tensor_d.sync_host();
    tensor_ref_d.sync_host();

    // Compute bias + relu in host code
    for (int i = 0; i < problem_size.m(); ++i)
    {
        for (int j = 0; j < problem_size.n(); ++j)
        {
            tensor_ref_d.at({i, j}) = std::max(
                ElementOutput(0),
                ElementOutput(tensor_ref_d.at({i, j}) + tensor_c_bias.at({i, 0})));
        }
    }

    // Check if output from CUTLASS kernel and reference kernel are equal or not
    std::cout << (cutlass::reference::host::TensorEquals(tensor_d.host_view(),
                                                         tensor_ref_d.host_view())
                      ? "Passed"
                      : "Failed")
              << std::endl;

    // CUTLASS_CHECK(status);
    return 0;
}

int main()
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
        // Returning zero so this test passes on older Toolkits. Its actions are no-op.
        return 0;
    }

    return run();
}
