
from ctypes import c_void_p, c_long
import torch
import math
import random
import os
import tempfile
from math import inf, nan
from torch._inductor.hooks import run_intermediate_hooks
from torch._inductor.utils import maybe_profile
from torch._inductor.codegen.memory_planning import _align as align

from torch import device, empty_strided
from torch._inductor.codecache import AsyncCompile
from torch._inductor.select_algorithm import extern_kernels
from torch._inductor.codegen.multi_kernel import MultiKernelCall

aten = torch.ops.aten
inductor_ops = torch.ops.inductor
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cpu = torch._C._dynamo.guards._empty_strided_cpu
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
alloc_from_pool = torch.ops.inductor._alloc_from_pool
reinterpret_tensor = torch.ops.inductor._reinterpret_tensor
async_compile = AsyncCompile()


# kernel path: /tmp/torchinductor_root/dx/cdx5icr4ro34nil6ka7drx7gidnt2isjfzpzm3ipiiexmx362ygf.py
# Source Nodes: [matmul], Original ATen: [aten.clone]
# matmul => clone
triton_poi_fused_clone_0 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(
    size_hints=[2097152], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp16', 1: '*fp16', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_0', 'mutated_arg_names': [], 'no_x_dim': False, 'backend_hash': '6ca8ffb1afdc2eeb7437fed1322ce1180c58e0acee9082166140e084f2a8edff'},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2097152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 32
    x1 = (xindex // 32) % 256
    x2 = (xindex // 8192) % 16
    x3 = (xindex // 131072)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (32*x2) + (1536*x1) + (393216*x3)), None).to(tl.float32)
    tl.store(out_ptr0 + (x4), tmp0, None)
''', device_str='cuda')

import triton
import triton.language as tl
from torch._inductor.triton_heuristics import grid, split_scan_grid, start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_raw_stream


# kernel path: /tmp/torchinductor_root/bf/cbflyrz7ai3gtwwrx2xih2hnt3b5jsskihfy7l25ku6mtfsflj6t.py
# Source Nodes: [matmul], Original ATen: [aten.clone]
# matmul => clone_1
triton_poi_fused_clone_1 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(
    size_hints=[8192, 256], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp16', 1: '*fp16', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_1', 'mutated_arg_names': [], 'no_x_dim': False, 'backend_hash': '6ca8ffb1afdc2eeb7437fed1322ce1180c58e0acee9082166140e084f2a8edff'},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 8192
    xnumel = 256
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 512
    y1 = (yindex // 512)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (1536*x2) + (393216*y1)), xmask, eviction_policy='evict_last').to(tl.float32)
    tl.store(out_ptr0 + (x2 + (256*y3)), tmp0, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_root/2f/c2ftmn7mfr4gf3cocgew7z6azla23fjcqk2dpstedftdiaeym5js.py
# Source Nodes: [probs, scores], Original ATen: [aten._softmax, aten.div]
# probs => amax, convert_element_type_2, convert_element_type_3, div_1, exp, sub, sum_1
# scores => div
triton_per_fused__softmax_div_2 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@persistent_reduction(
    size_hints=[65536, 256],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp16', 1: '*fp16', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax_div_2', 'mutated_arg_names': [], 'no_x_dim': True, 'backend_hash': '6ca8ffb1afdc2eeb7437fed1322ce1180c58e0acee9082166140e084f2a8edff'}
)
@triton.jit
def triton_(in_ptr0, out_ptr2, xnumel, rnumel):
    xnumel = 65536
    XBLOCK: tl.constexpr = 1
    rnumel = 256
    RBLOCK: tl.constexpr = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    roffset = 0
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (256*x0)), rmask, other=0.0).to(tl.float32)
    tmp1 = 5.656854249492381
    tmp2 = tmp0 / tmp1
    tmp3 = tmp2.to(tl.float32)
    tmp4 = tl.broadcast_to(tmp3, [RBLOCK])
    tmp6 = tl.where(rmask, tmp4, float("-inf"))
    tmp7 = triton_helpers.promote_to_tensor(triton_helpers.max2(tmp6, 0))
    tmp8 = tmp3 - tmp7
    tmp9 = tl.exp(tmp8)
    tmp10 = tl.broadcast_to(tmp9, [RBLOCK])
    tmp12 = tl.where(rmask, tmp10, 0)
    tmp13 = triton_helpers.promote_to_tensor(tl.sum(tmp12, 0))
    tmp14 = tmp9 / tmp13
    tmp15 = tmp14.to(tl.float32)
    tl.store(out_ptr2 + (r1 + (256*x0)), tmp15, rmask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1 = args
    args.clear()
    assert_size_stride(arg0_1, (16, 16, 256, 32), (393216, 32, 1536, 1))
    assert_size_stride(arg1_1, (16, 16, 256, 32), (393216, 32, 1536, 1))
    assert_size_stride(arg2_1, (16, 16, 256, 32), (393216, 32, 1536, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((16, 16, 256, 32), (131072, 8192, 32, 1), torch.float16)
        # Source Nodes: [matmul], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_0.run(arg1_1, buf0, 2097152, grid=grid(2097152), stream=stream0)
        del arg1_1
        buf1 = empty_strided_cuda((16, 16, 32, 256), (131072, 8192, 256, 1), torch.float16)
        # Source Nodes: [matmul], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(arg0_1, buf1, 8192, 256, grid=grid(8192, 256), stream=stream0)
        del arg0_1
        buf2 = empty_strided_cuda((256, 256, 256), (65536, 256, 1), torch.float16)
        # Source Nodes: [matmul], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf0, (256, 256, 32), (8192, 32, 1), 0), reinterpret_tensor(buf1, (256, 32, 256), (8192, 256, 1), 0), out=buf2)
        buf5 = empty_strided_cuda((16, 16, 256, 256), (1048576, 65536, 256, 1), torch.float16)
        # Source Nodes: [probs, scores], Original ATen: [aten._softmax, aten.div]
        triton_per_fused__softmax_div_2.run(buf2, buf5, 65536, 256, grid=grid(65536), stream=stream0)
        del buf2
        buf6 = reinterpret_tensor(buf1, (16, 16, 256, 32), (131072, 8192, 32, 1), 0); del buf1  # reuse
        # Source Nodes: [h], Original ATen: [aten.clone]
        triton_poi_fused_clone_0.run(arg2_1, buf6, 2097152, grid=grid(2097152), stream=stream0)
        del arg2_1
        buf7 = reinterpret_tensor(buf0, (256, 256, 32), (8192, 32, 1), 0); del buf0  # reuse
        # Source Nodes: [h], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf5, (256, 256, 256), (65536, 256, 1), 0), reinterpret_tensor(buf6, (256, 256, 32), (8192, 32, 1), 0), out=buf7)
        del buf5
        del buf6
    return (reinterpret_tensor(buf7, (16, 16, 256, 32), (131072, 8192, 32, 1), 0), )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((16, 16, 256, 32), (393216, 32, 1536, 1), device='cuda:0', dtype=torch.float16)
    arg1_1 = rand_strided((16, 16, 256, 32), (393216, 32, 1536, 1), device='cuda:0', dtype=torch.float16)
    arg2_1 = rand_strided((16, 16, 256, 32), (393216, 32, 1536, 1), device='cuda:0', dtype=torch.float16)
    fn = lambda: call([arg0_1, arg1_1, arg2_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
