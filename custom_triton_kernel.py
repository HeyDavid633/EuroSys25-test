# 9.15 
# Torch中使用自定义的triton算子
# 体现 triton.autotune 与 torch.compile 可以配合使用

import torch
from torch.utils._triton import has_triton


# if not has_triton():
#     print("Skipping because triton is not supported on this device.")
# else:
#     import triton
#     from triton import language as tl
    
    
#     # 一个纯triton的kernel，没有troch的元素
#     @triton.jit
#     def add_kernel(
#         in_ptr0,
#         in_ptr1,
#         out_ptr,
#         n_elements,
#         BLOCK_SIZE: "tl.constexpr",
#     ):
#         pid = tl.program_id(axis=0)
#         block_start = pid * BLOCK_SIZE
#         offsets = block_start + tl.arange(0, BLOCK_SIZE)
#         mask = offsets < n_elements
#         x = tl.load(in_ptr0 + offsets, mask=mask)
#         y = tl.load(in_ptr1 + offsets, mask=mask)
#         output = x + y
#         tl.store(out_ptr + offsets, output, mask=mask)

#     # torch的函数，调用了刚才的triton的kernel
#     @torch.compile(fullgraph=True)
#     def add_fn(x, y):
#         output = torch.zeros_like(x)
#         n_elements = output.numel()
#         grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
#         add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=4)
#         return output

#     x = torch.randn(4, device="cuda")
#     y = torch.randn(4, device="cuda")
#     out = add_fn(x, y)
#     print(f"Vector addition of\nX:\t{x}\nY:\t{y}\nis equal to\n{out}")


# triton 的自调优
if not has_triton():
    print("Skipping because triton is not supported on this device.")
else:
    import triton
    from triton import language as tl

    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_SIZE": 4}, num_stages=3, num_warps=8),
            triton.Config({"BLOCK_SIZE": 4}, num_stages=4, num_warps=4),
            triton.Config({"BLOCK_SIZE": 2}, num_stages=3, num_warps=8),
            triton.Config({"BLOCK_SIZE": 2}, num_stages=4, num_warps=4),
        ],
        key=[],
    )
    @triton.jit
    def add_kernel_autotuned(
        in_ptr0,
        in_ptr1,
        out_ptr,
        n_elements,
        BLOCK_SIZE: "tl.constexpr",
    ):
        pid = tl.program_id(axis=0)
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x = tl.load(in_ptr0 + offsets, mask=mask)
        y = tl.load(in_ptr1 + offsets, mask=mask)
        output = x + y
        tl.store(out_ptr + offsets, output, mask=mask)

    @torch.compile(fullgraph=True)
    def add_fn(x, y):
        output = torch.zeros_like(x)
        n_elements = output.numel()
        grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
        add_kernel_autotuned[grid](x, y, output, n_elements) # 这里的  BLOCK_SIZE 相比于之前，没有明确指定了，而靠自调优
        return output 

    x = torch.randn(4, device="cuda")
    y = torch.randn(4, device="cuda")
    out = add_fn(x, y)
    print(f"Vector addition of\nX:\t{x}\nY:\t{y}\nis equal to\n{out}")