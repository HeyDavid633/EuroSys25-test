#include <torch/extension.h>
#include <torch/serialize/tensor.h>
#include <ATen/ATen.h>

#include <iostream>
#include <cuda_fp16.h>


void launcher_syncfree_fixed_attn(const __half* q, const __half* k, const __half* v, __half* result,
    const int stride_0, const int stride_1, const int stride_2, const int stride_3, const int fixed_step,
    const int batch_size, const int seq_len, const int head_num, const int head_size);

void syncfree_fixed_attn_gpu(at::Tensor q, at::Tensor k, at::Tensor v, at::Tensor result)
{
    const auto batch_size = q.size(0);
    const auto head_num = q.size(1);
    const auto seq_len = q.size(2);
    const auto head_size = q.size(3);

    const auto stride_0 = q.stride(0);
    const auto stride_1 = q.stride(1);
    const auto stride_2 = q.stride(2);
    const auto stride_3 = q.stride(3);

    int stride_step = sqrt(seq_len);

    launcher_syncfree_fixed_attn(
        reinterpret_cast<const __half*>(q.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(k.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(v.data_ptr<at::Half>()),
        reinterpret_cast< __half*>(result.data_ptr<at::Half>()),
        stride_0, stride_1, stride_2, stride_3, stride_step,
        batch_size, seq_len, head_num, head_size);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.doc() = "My_fused_attention: Test for EuroSys25 ";
    m.def("forward", &syncfree_fixed_attn_gpu, "launcher_syncfree_fixed_attn"); 
} 