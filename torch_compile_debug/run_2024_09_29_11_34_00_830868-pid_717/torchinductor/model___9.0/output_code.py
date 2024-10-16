
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


# kernel path: /tmp/torchinductor_root/jj/cjjwgmw3vjwju535gpi5s54mld3tzbmarkznittsv2zzpsslymgm.py
# Source Nodes: [], Original ATen: []

triton_poi_fused_0 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp16', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_0', 'mutated_arg_names': [], 'no_x_dim': False, 'backend_hash': '6ca8ffb1afdc2eeb7437fed1322ce1180c58e0acee9082166140e084f2a8edff'},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2097152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 512
    x1 = (xindex // 512)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (1536*x1)), None).to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x2), tmp2, None)
''', device_str='cuda')

import triton
import triton.language as tl
from torch._inductor.triton_heuristics import grid, split_scan_grid, start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_raw_stream


# kernel path: /tmp/torchinductor_root/h2/ch2tpgaoggo6tyfm7gqfbquvac35xcahsdcezdof4ddng2dxz3eh.py
# Source Nodes: [], Original ATen: []

triton_poi_fused_1 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp16', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_1', 'mutated_arg_names': [], 'no_x_dim': False, 'backend_hash': '6ca8ffb1afdc2eeb7437fed1322ce1180c58e0acee9082166140e084f2a8edff'},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2097152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 512
    x1 = (xindex // 512)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (512 + x0 + (1536*x1)), None).to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (512 + x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x2), tmp2, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_root/mh/cmhp2m6dsrrzx3cqh6ehz2qongy327zx6dncivvn4ozni3vm3mx4.py
# Source Nodes: [], Original ATen: []

triton_poi_fused_2 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp16', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_2', 'mutated_arg_names': [], 'no_x_dim': False, 'backend_hash': '6ca8ffb1afdc2eeb7437fed1322ce1180c58e0acee9082166140e084f2a8edff'},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2097152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 512
    x1 = (xindex // 512)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (1024 + x0 + (1536*x1)), None).to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (1024 + x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x2), tmp2, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_root/n2/cn23ttuihu54uavdnlz2jm4ywwlraq7yvqhvpmicqcya67aiichx.py
# Source Nodes: [hidden_states_2, hidden_states_3, residual], Original ATen: [aten.add, aten.native_layer_norm]
# hidden_states_2 => add_1
# hidden_states_3 => add_2
# residual => add_3, add_4, convert_element_type_10, convert_element_type_11, mul, mul_1, rsqrt, sub_1, var_mean
triton_per_fused_add_native_layer_norm_3 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@persistent_reduction(
    size_hints=[4096, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp16', 3: '*fp16', 4: '*fp16', 5: '*fp16', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_3', 'mutated_arg_names': [], 'no_x_dim': True, 'backend_hash': '6ca8ffb1afdc2eeb7437fed1322ce1180c58e0acee9082166140e084f2a8edff'}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, xnumel, rnumel):
    xnumel = 4096
    XBLOCK: tl.constexpr = 1
    rnumel = 512
    RBLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    roffset = 0
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (512*x0)), rmask, other=0.0).to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp3 = tl.load(in_ptr2 + (r1 + (512*x0)), rmask, other=0.0).to(tl.float32)
    tmp29 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp32 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp5 = tmp4.to(tl.float32)
    tmp6 = tl.broadcast_to(tmp5, [RBLOCK])
    tmp8 = tl.where(rmask, tmp6, 0)
    tmp9 = tl.broadcast_to(tmp6, [RBLOCK])
    tmp11 = tl.where(rmask, tmp9, 0)
    tmp12 = triton_helpers.promote_to_tensor(tl.sum(tmp11, 0))
    tmp13 = tl.full([1], 512, tl.int32)
    tmp14 = tmp13.to(tl.float32)
    tmp15 = tmp12 / tmp14
    tmp16 = tmp6 - tmp15
    tmp17 = tmp16 * tmp16
    tmp18 = tl.broadcast_to(tmp17, [RBLOCK])
    tmp20 = tl.where(rmask, tmp18, 0)
    tmp21 = triton_helpers.promote_to_tensor(tl.sum(tmp20, 0))
    tmp22 = tmp5 - tmp15
    tmp23 = 512.0
    tmp24 = tmp21 / tmp23
    tmp25 = 1e-05
    tmp26 = tmp24 + tmp25
    tmp27 = tl.math.rsqrt(tmp26)
    tmp28 = tmp22 * tmp27
    tmp30 = tmp29.to(tl.float32)
    tmp31 = tmp28 * tmp30
    tmp33 = tmp32.to(tl.float32)
    tmp34 = tmp31 + tmp33
    tmp35 = tmp34.to(tl.float32)
    tl.store(out_ptr2 + (r1 + (512*x0)), tmp35, rmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_root/eo/ceomnvom4u5753jkq4gvw56rg6pkwp42ikld7abpqvs2ayrjsp3k.py
# Source Nodes: [hidden_states_5, hidden_states_6], Original ATen: [aten.add, aten.gelu]
# hidden_states_5 => add_5
# hidden_states_6 => add_6, convert_element_type_14, convert_element_type_15, erf, mul_2, mul_3, mul_4
triton_poi_fused_add_gelu_4 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(
    size_hints=[8388608], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp16', 1: '*fp16', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_gelu_4', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'backend_hash': '6ca8ffb1afdc2eeb7437fed1322ce1180c58e0acee9082166140e084f2a8edff'},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8388608
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 2048
    tmp0 = tl.load(in_out_ptr0 + (x2), None).to(tl.float32)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp2 = tmp0 + tmp1
    tmp3 = tmp2.to(tl.float32)
    tmp4 = 0.5
    tmp5 = tmp3 * tmp4
    tmp6 = 0.7071067811865476
    tmp7 = tmp3 * tmp6
    tmp8 = tl.math.erf(tmp7)
    tmp9 = 1.0
    tmp10 = tmp8 + tmp9
    tmp11 = tmp5 * tmp10
    tmp12 = tmp11.to(tl.float32)
    tl.store(in_out_ptr0 + (x2), tmp12, None)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1 = args
    args.clear()
    assert_size_stride(arg0_1, (16, 256, 512), (131072, 512, 1))
    assert_size_stride(arg1_1, (512, 1536), (1536, 1))
    assert_size_stride(arg2_1, (1536, ), (1, ))
    assert_size_stride(arg3_1, (512, 512), (512, 1))
    assert_size_stride(arg4_1, (512, ), (1, ))
    assert_size_stride(arg5_1, (512, ), (1, ))
    assert_size_stride(arg6_1, (512, ), (1, ))
    assert_size_stride(arg7_1, (512, 2048), (2048, 1))
    assert_size_stride(arg8_1, (2048, ), (1, ))
    assert_size_stride(arg9_1, (2048, 512), (512, 1))
    assert_size_stride(arg10_1, (512, ), (1, ))
    assert_size_stride(arg11_1, (512, ), (1, ))
    assert_size_stride(arg12_1, (512, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((4096, 1536), (1536, 1), torch.float16)
        # Source Nodes: [matmul], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(arg0_1, (4096, 512), (512, 1), 0), arg1_1, out=buf0)
        del arg1_1
        buf1 = empty_strided_cuda((16, 16, 256, 32), (131072, 32, 512, 1), torch.float16)
        # Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_0.run(buf0, arg2_1, buf1, 2097152, grid=grid(2097152), stream=stream0)
        buf2 = empty_strided_cuda((16, 16, 256, 32), (131072, 32, 512, 1), torch.float16)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf0, arg2_1, buf2, 2097152, grid=grid(2097152), stream=stream0)
        buf3 = empty_strided_cuda((16, 16, 256, 32), (131072, 32, 512, 1), torch.float16)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf0, arg2_1, buf3, 2097152, grid=grid(2097152), stream=stream0)
        del arg2_1
        del buf0
        # Source Nodes: [], Original ATen: []
        buf4 = aten._scaled_dot_product_flash_attention.default(buf1, buf2, buf3, scale=0.17677669529663687)
        del buf1
        buf5 = buf4[0]
        del buf4
        buf10 = reinterpret_tensor(buf3, (4096, 512), (512, 1), 0); del buf3  # reuse
        # Source Nodes: [matmul_3], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf5, (4096, 512), (512, 1), 0), arg3_1, out=buf10)
        del arg3_1
        buf14 = reinterpret_tensor(buf5, (16, 256, 512), (131072, 512, 1), 0); del buf5  # reuse
        # Source Nodes: [hidden_states_2, hidden_states_3, residual], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_3.run(buf10, arg4_1, arg0_1, arg5_1, arg6_1, buf14, 4096, 512, grid=grid(4096), stream=stream0)
        del arg0_1
        del arg4_1
        del arg5_1
        del arg6_1
        buf15 = empty_strided_cuda((4096, 2048), (2048, 1), torch.float16)
        # Source Nodes: [matmul_4], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf14, (4096, 512), (512, 1), 0), arg7_1, out=buf15)
        del arg7_1
        buf16 = reinterpret_tensor(buf15, (16, 256, 2048), (524288, 2048, 1), 0); del buf15  # reuse
        # Source Nodes: [hidden_states_5, hidden_states_6], Original ATen: [aten.add, aten.gelu]
        triton_poi_fused_add_gelu_4.run(buf16, arg8_1, 8388608, grid=grid(8388608), stream=stream0)
        del arg8_1
        buf17 = buf10; del buf10  # reuse
        # Source Nodes: [matmul_5], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf16, (4096, 2048), (2048, 1), 0), arg9_1, out=buf17)
        del arg9_1
        del buf16
        buf21 = reinterpret_tensor(buf2, (16, 256, 512), (131072, 512, 1), 0); del buf2  # reuse
        # Source Nodes: [hidden_states_7, hidden_states_8, hidden_states_9], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_3.run(buf17, arg10_1, buf14, arg11_1, arg12_1, buf21, 4096, 512, grid=grid(4096), stream=stream0)
        del arg10_1
        del arg11_1
        del arg12_1
        del buf14
        del buf17
    return (buf21, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((16, 256, 512), (131072, 512, 1), device='cuda:0', dtype=torch.float16)
    arg1_1 = rand_strided((512, 1536), (1536, 1), device='cuda:0', dtype=torch.float16)
    arg2_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg3_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float16)
    arg4_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg5_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg6_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg7_1 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float16)
    arg8_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg9_1 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float16)
    arg10_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg11_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg12_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float16)
    fn = lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
