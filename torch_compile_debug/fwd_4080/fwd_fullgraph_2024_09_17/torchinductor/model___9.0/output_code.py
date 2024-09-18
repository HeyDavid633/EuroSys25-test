
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
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1 = args
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
    assert_size_stride(arg13_1, (512, 1536), (1536, 1))
    assert_size_stride(arg14_1, (1536, ), (1, ))
    assert_size_stride(arg15_1, (512, 512), (512, 1))
    assert_size_stride(arg16_1, (512, ), (1, ))
    assert_size_stride(arg17_1, (512, ), (1, ))
    assert_size_stride(arg18_1, (512, ), (1, ))
    assert_size_stride(arg19_1, (512, 2048), (2048, 1))
    assert_size_stride(arg20_1, (2048, ), (1, ))
    assert_size_stride(arg21_1, (2048, 512), (512, 1))
    assert_size_stride(arg22_1, (512, ), (1, ))
    assert_size_stride(arg23_1, (512, ), (1, ))
    assert_size_stride(arg24_1, (512, ), (1, ))
    assert_size_stride(arg25_1, (512, 1536), (1536, 1))
    assert_size_stride(arg26_1, (1536, ), (1, ))
    assert_size_stride(arg27_1, (512, 512), (512, 1))
    assert_size_stride(arg28_1, (512, ), (1, ))
    assert_size_stride(arg29_1, (512, ), (1, ))
    assert_size_stride(arg30_1, (512, ), (1, ))
    assert_size_stride(arg31_1, (512, 2048), (2048, 1))
    assert_size_stride(arg32_1, (2048, ), (1, ))
    assert_size_stride(arg33_1, (2048, 512), (512, 1))
    assert_size_stride(arg34_1, (512, ), (1, ))
    assert_size_stride(arg35_1, (512, ), (1, ))
    assert_size_stride(arg36_1, (512, ), (1, ))
    assert_size_stride(arg37_1, (512, 1536), (1536, 1))
    assert_size_stride(arg38_1, (1536, ), (1, ))
    assert_size_stride(arg39_1, (512, 512), (512, 1))
    assert_size_stride(arg40_1, (512, ), (1, ))
    assert_size_stride(arg41_1, (512, ), (1, ))
    assert_size_stride(arg42_1, (512, ), (1, ))
    assert_size_stride(arg43_1, (512, 2048), (2048, 1))
    assert_size_stride(arg44_1, (2048, ), (1, ))
    assert_size_stride(arg45_1, (2048, 512), (512, 1))
    assert_size_stride(arg46_1, (512, ), (1, ))
    assert_size_stride(arg47_1, (512, ), (1, ))
    assert_size_stride(arg48_1, (512, ), (1, ))
    assert_size_stride(arg49_1, (512, 1536), (1536, 1))
    assert_size_stride(arg50_1, (1536, ), (1, ))
    assert_size_stride(arg51_1, (512, 512), (512, 1))
    assert_size_stride(arg52_1, (512, ), (1, ))
    assert_size_stride(arg53_1, (512, ), (1, ))
    assert_size_stride(arg54_1, (512, ), (1, ))
    assert_size_stride(arg55_1, (512, 2048), (2048, 1))
    assert_size_stride(arg56_1, (2048, ), (1, ))
    assert_size_stride(arg57_1, (2048, 512), (512, 1))
    assert_size_stride(arg58_1, (512, ), (1, ))
    assert_size_stride(arg59_1, (512, ), (1, ))
    assert_size_stride(arg60_1, (512, ), (1, ))
    assert_size_stride(arg61_1, (512, 1536), (1536, 1))
    assert_size_stride(arg62_1, (1536, ), (1, ))
    assert_size_stride(arg63_1, (512, 512), (512, 1))
    assert_size_stride(arg64_1, (512, ), (1, ))
    assert_size_stride(arg65_1, (512, ), (1, ))
    assert_size_stride(arg66_1, (512, ), (1, ))
    assert_size_stride(arg67_1, (512, 2048), (2048, 1))
    assert_size_stride(arg68_1, (2048, ), (1, ))
    assert_size_stride(arg69_1, (2048, 512), (512, 1))
    assert_size_stride(arg70_1, (512, ), (1, ))
    assert_size_stride(arg71_1, (512, ), (1, ))
    assert_size_stride(arg72_1, (512, ), (1, ))
    assert_size_stride(arg73_1, (512, 1536), (1536, 1))
    assert_size_stride(arg74_1, (1536, ), (1, ))
    assert_size_stride(arg75_1, (512, 512), (512, 1))
    assert_size_stride(arg76_1, (512, ), (1, ))
    assert_size_stride(arg77_1, (512, ), (1, ))
    assert_size_stride(arg78_1, (512, ), (1, ))
    assert_size_stride(arg79_1, (512, 2048), (2048, 1))
    assert_size_stride(arg80_1, (2048, ), (1, ))
    assert_size_stride(arg81_1, (2048, 512), (512, 1))
    assert_size_stride(arg82_1, (512, ), (1, ))
    assert_size_stride(arg83_1, (512, ), (1, ))
    assert_size_stride(arg84_1, (512, ), (1, ))
    assert_size_stride(arg85_1, (512, 1536), (1536, 1))
    assert_size_stride(arg86_1, (1536, ), (1, ))
    assert_size_stride(arg87_1, (512, 512), (512, 1))
    assert_size_stride(arg88_1, (512, ), (1, ))
    assert_size_stride(arg89_1, (512, ), (1, ))
    assert_size_stride(arg90_1, (512, ), (1, ))
    assert_size_stride(arg91_1, (512, 2048), (2048, 1))
    assert_size_stride(arg92_1, (2048, ), (1, ))
    assert_size_stride(arg93_1, (2048, 512), (512, 1))
    assert_size_stride(arg94_1, (512, ), (1, ))
    assert_size_stride(arg95_1, (512, ), (1, ))
    assert_size_stride(arg96_1, (512, ), (1, ))
    assert_size_stride(arg97_1, (512, 1536), (1536, 1))
    assert_size_stride(arg98_1, (1536, ), (1, ))
    assert_size_stride(arg99_1, (512, 512), (512, 1))
    assert_size_stride(arg100_1, (512, ), (1, ))
    assert_size_stride(arg101_1, (512, ), (1, ))
    assert_size_stride(arg102_1, (512, ), (1, ))
    assert_size_stride(arg103_1, (512, 2048), (2048, 1))
    assert_size_stride(arg104_1, (2048, ), (1, ))
    assert_size_stride(arg105_1, (2048, 512), (512, 1))
    assert_size_stride(arg106_1, (512, ), (1, ))
    assert_size_stride(arg107_1, (512, ), (1, ))
    assert_size_stride(arg108_1, (512, ), (1, ))
    assert_size_stride(arg109_1, (512, 1536), (1536, 1))
    assert_size_stride(arg110_1, (1536, ), (1, ))
    assert_size_stride(arg111_1, (512, 512), (512, 1))
    assert_size_stride(arg112_1, (512, ), (1, ))
    assert_size_stride(arg113_1, (512, ), (1, ))
    assert_size_stride(arg114_1, (512, ), (1, ))
    assert_size_stride(arg115_1, (512, 2048), (2048, 1))
    assert_size_stride(arg116_1, (2048, ), (1, ))
    assert_size_stride(arg117_1, (2048, 512), (512, 1))
    assert_size_stride(arg118_1, (512, ), (1, ))
    assert_size_stride(arg119_1, (512, ), (1, ))
    assert_size_stride(arg120_1, (512, ), (1, ))
    assert_size_stride(arg121_1, (512, 1536), (1536, 1))
    assert_size_stride(arg122_1, (1536, ), (1, ))
    assert_size_stride(arg123_1, (512, 512), (512, 1))
    assert_size_stride(arg124_1, (512, ), (1, ))
    assert_size_stride(arg125_1, (512, ), (1, ))
    assert_size_stride(arg126_1, (512, ), (1, ))
    assert_size_stride(arg127_1, (512, 2048), (2048, 1))
    assert_size_stride(arg128_1, (2048, ), (1, ))
    assert_size_stride(arg129_1, (2048, 512), (512, 1))
    assert_size_stride(arg130_1, (512, ), (1, ))
    assert_size_stride(arg131_1, (512, ), (1, ))
    assert_size_stride(arg132_1, (512, ), (1, ))
    assert_size_stride(arg133_1, (512, 1536), (1536, 1))
    assert_size_stride(arg134_1, (1536, ), (1, ))
    assert_size_stride(arg135_1, (512, 512), (512, 1))
    assert_size_stride(arg136_1, (512, ), (1, ))
    assert_size_stride(arg137_1, (512, ), (1, ))
    assert_size_stride(arg138_1, (512, ), (1, ))
    assert_size_stride(arg139_1, (512, 2048), (2048, 1))
    assert_size_stride(arg140_1, (2048, ), (1, ))
    assert_size_stride(arg141_1, (2048, 512), (512, 1))
    assert_size_stride(arg142_1, (512, ), (1, ))
    assert_size_stride(arg143_1, (512, ), (1, ))
    assert_size_stride(arg144_1, (512, ), (1, ))
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
        # Source Nodes: [], Original ATen: []
        buf4 = aten._scaled_dot_product_flash_attention.default(buf1, buf2, buf3, scale=0.17677669529663687)
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
        buf21 = reinterpret_tensor(buf2, (16, 256, 512), (131072, 512, 1), 0); del buf2  # reuse
        # Source Nodes: [hidden_states_7, hidden_states_8, input_tensor_1], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_3.run(buf17, arg10_1, buf14, arg11_1, arg12_1, buf21, 4096, 512, grid=grid(4096), stream=stream0)
        del arg10_1
        del arg11_1
        del arg12_1
        buf22 = buf0; del buf0  # reuse
        # Source Nodes: [matmul_6], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf21, (4096, 512), (512, 1), 0), arg13_1, out=buf22)
        del arg13_1
        buf23 = reinterpret_tensor(buf17, (16, 16, 256, 32), (131072, 32, 512, 1), 0); del buf17  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_0.run(buf22, arg14_1, buf23, 2097152, grid=grid(2097152), stream=stream0)
        buf24 = reinterpret_tensor(buf14, (16, 16, 256, 32), (131072, 32, 512, 1), 0); del buf14  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf22, arg14_1, buf24, 2097152, grid=grid(2097152), stream=stream0)
        buf25 = buf1; del buf1  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf22, arg14_1, buf25, 2097152, grid=grid(2097152), stream=stream0)
        del arg14_1
        # Source Nodes: [], Original ATen: []
        buf26 = aten._scaled_dot_product_flash_attention.default(buf23, buf24, buf25, scale=0.17677669529663687)
        buf27 = buf26[0]
        del buf26
        buf32 = reinterpret_tensor(buf25, (4096, 512), (512, 1), 0); del buf25  # reuse
        # Source Nodes: [matmul_9], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf27, (4096, 512), (512, 1), 0), arg15_1, out=buf32)
        del arg15_1
        buf36 = reinterpret_tensor(buf27, (16, 256, 512), (131072, 512, 1), 0); del buf27  # reuse
        # Source Nodes: [hidden_states_11, hidden_states_12, residual_1], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_3.run(buf32, arg16_1, buf21, arg17_1, arg18_1, buf36, 4096, 512, grid=grid(4096), stream=stream0)
        del arg16_1
        del arg17_1
        del arg18_1
        buf37 = reinterpret_tensor(buf16, (4096, 2048), (2048, 1), 0); del buf16  # reuse
        # Source Nodes: [matmul_10], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf36, (4096, 512), (512, 1), 0), arg19_1, out=buf37)
        del arg19_1
        buf38 = reinterpret_tensor(buf37, (16, 256, 2048), (524288, 2048, 1), 0); del buf37  # reuse
        # Source Nodes: [hidden_states_14, hidden_states_15], Original ATen: [aten.add, aten.gelu]
        triton_poi_fused_add_gelu_4.run(buf38, arg20_1, 8388608, grid=grid(8388608), stream=stream0)
        del arg20_1
        buf39 = buf32; del buf32  # reuse
        # Source Nodes: [matmul_11], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf38, (4096, 2048), (2048, 1), 0), arg21_1, out=buf39)
        del arg21_1
        buf43 = reinterpret_tensor(buf24, (16, 256, 512), (131072, 512, 1), 0); del buf24  # reuse
        # Source Nodes: [hidden_states_16, hidden_states_17, input_tensor_2], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_3.run(buf39, arg22_1, buf36, arg23_1, arg24_1, buf43, 4096, 512, grid=grid(4096), stream=stream0)
        del arg22_1
        del arg23_1
        del arg24_1
        buf44 = buf22; del buf22  # reuse
        # Source Nodes: [matmul_12], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf43, (4096, 512), (512, 1), 0), arg25_1, out=buf44)
        del arg25_1
        buf45 = reinterpret_tensor(buf39, (16, 16, 256, 32), (131072, 32, 512, 1), 0); del buf39  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_0.run(buf44, arg26_1, buf45, 2097152, grid=grid(2097152), stream=stream0)
        buf46 = reinterpret_tensor(buf36, (16, 16, 256, 32), (131072, 32, 512, 1), 0); del buf36  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf44, arg26_1, buf46, 2097152, grid=grid(2097152), stream=stream0)
        buf47 = buf23; del buf23  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf44, arg26_1, buf47, 2097152, grid=grid(2097152), stream=stream0)
        del arg26_1
        # Source Nodes: [], Original ATen: []
        buf48 = aten._scaled_dot_product_flash_attention.default(buf45, buf46, buf47, scale=0.17677669529663687)
        buf49 = buf48[0]
        del buf48
        buf54 = reinterpret_tensor(buf47, (4096, 512), (512, 1), 0); del buf47  # reuse
        # Source Nodes: [matmul_15], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf49, (4096, 512), (512, 1), 0), arg27_1, out=buf54)
        del arg27_1
        buf58 = reinterpret_tensor(buf49, (16, 256, 512), (131072, 512, 1), 0); del buf49  # reuse
        # Source Nodes: [hidden_states_20, hidden_states_21, residual_2], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_3.run(buf54, arg28_1, buf43, arg29_1, arg30_1, buf58, 4096, 512, grid=grid(4096), stream=stream0)
        del arg28_1
        del arg29_1
        del arg30_1
        buf59 = reinterpret_tensor(buf38, (4096, 2048), (2048, 1), 0); del buf38  # reuse
        # Source Nodes: [matmul_16], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf58, (4096, 512), (512, 1), 0), arg31_1, out=buf59)
        del arg31_1
        buf60 = reinterpret_tensor(buf59, (16, 256, 2048), (524288, 2048, 1), 0); del buf59  # reuse
        # Source Nodes: [hidden_states_23, hidden_states_24], Original ATen: [aten.add, aten.gelu]
        triton_poi_fused_add_gelu_4.run(buf60, arg32_1, 8388608, grid=grid(8388608), stream=stream0)
        del arg32_1
        buf61 = buf54; del buf54  # reuse
        # Source Nodes: [matmul_17], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf60, (4096, 2048), (2048, 1), 0), arg33_1, out=buf61)
        del arg33_1
        buf65 = reinterpret_tensor(buf46, (16, 256, 512), (131072, 512, 1), 0); del buf46  # reuse
        # Source Nodes: [hidden_states_25, hidden_states_26, input_tensor_3], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_3.run(buf61, arg34_1, buf58, arg35_1, arg36_1, buf65, 4096, 512, grid=grid(4096), stream=stream0)
        del arg34_1
        del arg35_1
        del arg36_1
        buf66 = buf44; del buf44  # reuse
        # Source Nodes: [matmul_18], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf65, (4096, 512), (512, 1), 0), arg37_1, out=buf66)
        del arg37_1
        buf67 = reinterpret_tensor(buf61, (16, 16, 256, 32), (131072, 32, 512, 1), 0); del buf61  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_0.run(buf66, arg38_1, buf67, 2097152, grid=grid(2097152), stream=stream0)
        buf68 = reinterpret_tensor(buf58, (16, 16, 256, 32), (131072, 32, 512, 1), 0); del buf58  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf66, arg38_1, buf68, 2097152, grid=grid(2097152), stream=stream0)
        buf69 = buf45; del buf45  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf66, arg38_1, buf69, 2097152, grid=grid(2097152), stream=stream0)
        del arg38_1
        # Source Nodes: [], Original ATen: []
        buf70 = aten._scaled_dot_product_flash_attention.default(buf67, buf68, buf69, scale=0.17677669529663687)
        buf71 = buf70[0]
        del buf70
        buf76 = reinterpret_tensor(buf69, (4096, 512), (512, 1), 0); del buf69  # reuse
        # Source Nodes: [matmul_21], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf71, (4096, 512), (512, 1), 0), arg39_1, out=buf76)
        del arg39_1
        buf80 = reinterpret_tensor(buf71, (16, 256, 512), (131072, 512, 1), 0); del buf71  # reuse
        # Source Nodes: [hidden_states_29, hidden_states_30, residual_3], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_3.run(buf76, arg40_1, buf65, arg41_1, arg42_1, buf80, 4096, 512, grid=grid(4096), stream=stream0)
        del arg40_1
        del arg41_1
        del arg42_1
        buf81 = reinterpret_tensor(buf60, (4096, 2048), (2048, 1), 0); del buf60  # reuse
        # Source Nodes: [matmul_22], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf80, (4096, 512), (512, 1), 0), arg43_1, out=buf81)
        del arg43_1
        buf82 = reinterpret_tensor(buf81, (16, 256, 2048), (524288, 2048, 1), 0); del buf81  # reuse
        # Source Nodes: [hidden_states_32, hidden_states_33], Original ATen: [aten.add, aten.gelu]
        triton_poi_fused_add_gelu_4.run(buf82, arg44_1, 8388608, grid=grid(8388608), stream=stream0)
        del arg44_1
        buf83 = buf76; del buf76  # reuse
        # Source Nodes: [matmul_23], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf82, (4096, 2048), (2048, 1), 0), arg45_1, out=buf83)
        del arg45_1
        buf87 = reinterpret_tensor(buf68, (16, 256, 512), (131072, 512, 1), 0); del buf68  # reuse
        # Source Nodes: [hidden_states_34, hidden_states_35, input_tensor_4], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_3.run(buf83, arg46_1, buf80, arg47_1, arg48_1, buf87, 4096, 512, grid=grid(4096), stream=stream0)
        del arg46_1
        del arg47_1
        del arg48_1
        buf88 = buf66; del buf66  # reuse
        # Source Nodes: [matmul_24], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf87, (4096, 512), (512, 1), 0), arg49_1, out=buf88)
        del arg49_1
        buf89 = reinterpret_tensor(buf83, (16, 16, 256, 32), (131072, 32, 512, 1), 0); del buf83  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_0.run(buf88, arg50_1, buf89, 2097152, grid=grid(2097152), stream=stream0)
        buf90 = reinterpret_tensor(buf80, (16, 16, 256, 32), (131072, 32, 512, 1), 0); del buf80  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf88, arg50_1, buf90, 2097152, grid=grid(2097152), stream=stream0)
        buf91 = buf67; del buf67  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf88, arg50_1, buf91, 2097152, grid=grid(2097152), stream=stream0)
        del arg50_1
        # Source Nodes: [], Original ATen: []
        buf92 = aten._scaled_dot_product_flash_attention.default(buf89, buf90, buf91, scale=0.17677669529663687)
        buf93 = buf92[0]
        del buf92
        buf98 = reinterpret_tensor(buf91, (4096, 512), (512, 1), 0); del buf91  # reuse
        # Source Nodes: [matmul_27], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf93, (4096, 512), (512, 1), 0), arg51_1, out=buf98)
        del arg51_1
        buf102 = reinterpret_tensor(buf93, (16, 256, 512), (131072, 512, 1), 0); del buf93  # reuse
        # Source Nodes: [hidden_states_38, hidden_states_39, residual_4], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_3.run(buf98, arg52_1, buf87, arg53_1, arg54_1, buf102, 4096, 512, grid=grid(4096), stream=stream0)
        del arg52_1
        del arg53_1
        del arg54_1
        buf103 = reinterpret_tensor(buf82, (4096, 2048), (2048, 1), 0); del buf82  # reuse
        # Source Nodes: [matmul_28], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf102, (4096, 512), (512, 1), 0), arg55_1, out=buf103)
        del arg55_1
        buf104 = reinterpret_tensor(buf103, (16, 256, 2048), (524288, 2048, 1), 0); del buf103  # reuse
        # Source Nodes: [hidden_states_41, hidden_states_42], Original ATen: [aten.add, aten.gelu]
        triton_poi_fused_add_gelu_4.run(buf104, arg56_1, 8388608, grid=grid(8388608), stream=stream0)
        del arg56_1
        buf105 = buf98; del buf98  # reuse
        # Source Nodes: [matmul_29], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf104, (4096, 2048), (2048, 1), 0), arg57_1, out=buf105)
        del arg57_1
        buf109 = reinterpret_tensor(buf90, (16, 256, 512), (131072, 512, 1), 0); del buf90  # reuse
        # Source Nodes: [hidden_states_43, hidden_states_44, input_tensor_5], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_3.run(buf105, arg58_1, buf102, arg59_1, arg60_1, buf109, 4096, 512, grid=grid(4096), stream=stream0)
        del arg58_1
        del arg59_1
        del arg60_1
        buf110 = buf88; del buf88  # reuse
        # Source Nodes: [matmul_30], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf109, (4096, 512), (512, 1), 0), arg61_1, out=buf110)
        del arg61_1
        buf111 = reinterpret_tensor(buf105, (16, 16, 256, 32), (131072, 32, 512, 1), 0); del buf105  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_0.run(buf110, arg62_1, buf111, 2097152, grid=grid(2097152), stream=stream0)
        buf112 = reinterpret_tensor(buf102, (16, 16, 256, 32), (131072, 32, 512, 1), 0); del buf102  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf110, arg62_1, buf112, 2097152, grid=grid(2097152), stream=stream0)
        buf113 = buf89; del buf89  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf110, arg62_1, buf113, 2097152, grid=grid(2097152), stream=stream0)
        del arg62_1
        # Source Nodes: [], Original ATen: []
        buf114 = aten._scaled_dot_product_flash_attention.default(buf111, buf112, buf113, scale=0.17677669529663687)
        buf115 = buf114[0]
        del buf114
        buf120 = reinterpret_tensor(buf113, (4096, 512), (512, 1), 0); del buf113  # reuse
        # Source Nodes: [matmul_33], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf115, (4096, 512), (512, 1), 0), arg63_1, out=buf120)
        del arg63_1
        buf124 = reinterpret_tensor(buf115, (16, 256, 512), (131072, 512, 1), 0); del buf115  # reuse
        # Source Nodes: [hidden_states_47, hidden_states_48, residual_5], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_3.run(buf120, arg64_1, buf109, arg65_1, arg66_1, buf124, 4096, 512, grid=grid(4096), stream=stream0)
        del arg64_1
        del arg65_1
        del arg66_1
        buf125 = reinterpret_tensor(buf104, (4096, 2048), (2048, 1), 0); del buf104  # reuse
        # Source Nodes: [matmul_34], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf124, (4096, 512), (512, 1), 0), arg67_1, out=buf125)
        del arg67_1
        buf126 = reinterpret_tensor(buf125, (16, 256, 2048), (524288, 2048, 1), 0); del buf125  # reuse
        # Source Nodes: [hidden_states_50, hidden_states_51], Original ATen: [aten.add, aten.gelu]
        triton_poi_fused_add_gelu_4.run(buf126, arg68_1, 8388608, grid=grid(8388608), stream=stream0)
        del arg68_1
        buf127 = buf120; del buf120  # reuse
        # Source Nodes: [matmul_35], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf126, (4096, 2048), (2048, 1), 0), arg69_1, out=buf127)
        del arg69_1
        buf131 = reinterpret_tensor(buf112, (16, 256, 512), (131072, 512, 1), 0); del buf112  # reuse
        # Source Nodes: [hidden_states_52, hidden_states_53, input_tensor_6], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_3.run(buf127, arg70_1, buf124, arg71_1, arg72_1, buf131, 4096, 512, grid=grid(4096), stream=stream0)
        del arg70_1
        del arg71_1
        del arg72_1
        buf132 = buf110; del buf110  # reuse
        # Source Nodes: [matmul_36], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf131, (4096, 512), (512, 1), 0), arg73_1, out=buf132)
        del arg73_1
        buf133 = reinterpret_tensor(buf127, (16, 16, 256, 32), (131072, 32, 512, 1), 0); del buf127  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_0.run(buf132, arg74_1, buf133, 2097152, grid=grid(2097152), stream=stream0)
        buf134 = reinterpret_tensor(buf124, (16, 16, 256, 32), (131072, 32, 512, 1), 0); del buf124  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf132, arg74_1, buf134, 2097152, grid=grid(2097152), stream=stream0)
        buf135 = buf111; del buf111  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf132, arg74_1, buf135, 2097152, grid=grid(2097152), stream=stream0)
        del arg74_1
        # Source Nodes: [], Original ATen: []
        buf136 = aten._scaled_dot_product_flash_attention.default(buf133, buf134, buf135, scale=0.17677669529663687)
        buf137 = buf136[0]
        del buf136
        buf142 = reinterpret_tensor(buf135, (4096, 512), (512, 1), 0); del buf135  # reuse
        # Source Nodes: [matmul_39], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf137, (4096, 512), (512, 1), 0), arg75_1, out=buf142)
        del arg75_1
        buf146 = reinterpret_tensor(buf137, (16, 256, 512), (131072, 512, 1), 0); del buf137  # reuse
        # Source Nodes: [hidden_states_56, hidden_states_57, residual_6], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_3.run(buf142, arg76_1, buf131, arg77_1, arg78_1, buf146, 4096, 512, grid=grid(4096), stream=stream0)
        del arg76_1
        del arg77_1
        del arg78_1
        buf147 = reinterpret_tensor(buf126, (4096, 2048), (2048, 1), 0); del buf126  # reuse
        # Source Nodes: [matmul_40], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf146, (4096, 512), (512, 1), 0), arg79_1, out=buf147)
        del arg79_1
        buf148 = reinterpret_tensor(buf147, (16, 256, 2048), (524288, 2048, 1), 0); del buf147  # reuse
        # Source Nodes: [hidden_states_59, hidden_states_60], Original ATen: [aten.add, aten.gelu]
        triton_poi_fused_add_gelu_4.run(buf148, arg80_1, 8388608, grid=grid(8388608), stream=stream0)
        del arg80_1
        buf149 = buf142; del buf142  # reuse
        # Source Nodes: [matmul_41], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf148, (4096, 2048), (2048, 1), 0), arg81_1, out=buf149)
        del arg81_1
        buf153 = reinterpret_tensor(buf134, (16, 256, 512), (131072, 512, 1), 0); del buf134  # reuse
        # Source Nodes: [hidden_states_61, hidden_states_62, input_tensor_7], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_3.run(buf149, arg82_1, buf146, arg83_1, arg84_1, buf153, 4096, 512, grid=grid(4096), stream=stream0)
        del arg82_1
        del arg83_1
        del arg84_1
        buf154 = buf132; del buf132  # reuse
        # Source Nodes: [matmul_42], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf153, (4096, 512), (512, 1), 0), arg85_1, out=buf154)
        del arg85_1
        buf155 = reinterpret_tensor(buf149, (16, 16, 256, 32), (131072, 32, 512, 1), 0); del buf149  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_0.run(buf154, arg86_1, buf155, 2097152, grid=grid(2097152), stream=stream0)
        buf156 = reinterpret_tensor(buf146, (16, 16, 256, 32), (131072, 32, 512, 1), 0); del buf146  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf154, arg86_1, buf156, 2097152, grid=grid(2097152), stream=stream0)
        buf157 = buf133; del buf133  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf154, arg86_1, buf157, 2097152, grid=grid(2097152), stream=stream0)
        del arg86_1
        # Source Nodes: [], Original ATen: []
        buf158 = aten._scaled_dot_product_flash_attention.default(buf155, buf156, buf157, scale=0.17677669529663687)
        buf159 = buf158[0]
        del buf158
        buf164 = reinterpret_tensor(buf157, (4096, 512), (512, 1), 0); del buf157  # reuse
        # Source Nodes: [matmul_45], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf159, (4096, 512), (512, 1), 0), arg87_1, out=buf164)
        del arg87_1
        buf168 = reinterpret_tensor(buf159, (16, 256, 512), (131072, 512, 1), 0); del buf159  # reuse
        # Source Nodes: [hidden_states_65, hidden_states_66, residual_7], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_3.run(buf164, arg88_1, buf153, arg89_1, arg90_1, buf168, 4096, 512, grid=grid(4096), stream=stream0)
        del arg88_1
        del arg89_1
        del arg90_1
        buf169 = reinterpret_tensor(buf148, (4096, 2048), (2048, 1), 0); del buf148  # reuse
        # Source Nodes: [matmul_46], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf168, (4096, 512), (512, 1), 0), arg91_1, out=buf169)
        del arg91_1
        buf170 = reinterpret_tensor(buf169, (16, 256, 2048), (524288, 2048, 1), 0); del buf169  # reuse
        # Source Nodes: [hidden_states_68, hidden_states_69], Original ATen: [aten.add, aten.gelu]
        triton_poi_fused_add_gelu_4.run(buf170, arg92_1, 8388608, grid=grid(8388608), stream=stream0)
        del arg92_1
        buf171 = buf164; del buf164  # reuse
        # Source Nodes: [matmul_47], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf170, (4096, 2048), (2048, 1), 0), arg93_1, out=buf171)
        del arg93_1
        buf175 = reinterpret_tensor(buf156, (16, 256, 512), (131072, 512, 1), 0); del buf156  # reuse
        # Source Nodes: [hidden_states_70, hidden_states_71, input_tensor_8], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_3.run(buf171, arg94_1, buf168, arg95_1, arg96_1, buf175, 4096, 512, grid=grid(4096), stream=stream0)
        del arg94_1
        del arg95_1
        del arg96_1
        buf176 = buf154; del buf154  # reuse
        # Source Nodes: [matmul_48], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf175, (4096, 512), (512, 1), 0), arg97_1, out=buf176)
        del arg97_1
        buf177 = reinterpret_tensor(buf171, (16, 16, 256, 32), (131072, 32, 512, 1), 0); del buf171  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_0.run(buf176, arg98_1, buf177, 2097152, grid=grid(2097152), stream=stream0)
        buf178 = reinterpret_tensor(buf168, (16, 16, 256, 32), (131072, 32, 512, 1), 0); del buf168  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf176, arg98_1, buf178, 2097152, grid=grid(2097152), stream=stream0)
        buf179 = buf155; del buf155  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf176, arg98_1, buf179, 2097152, grid=grid(2097152), stream=stream0)
        del arg98_1
        # Source Nodes: [], Original ATen: []
        buf180 = aten._scaled_dot_product_flash_attention.default(buf177, buf178, buf179, scale=0.17677669529663687)
        buf181 = buf180[0]
        del buf180
        buf186 = reinterpret_tensor(buf179, (4096, 512), (512, 1), 0); del buf179  # reuse
        # Source Nodes: [matmul_51], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf181, (4096, 512), (512, 1), 0), arg99_1, out=buf186)
        del arg99_1
        buf190 = reinterpret_tensor(buf181, (16, 256, 512), (131072, 512, 1), 0); del buf181  # reuse
        # Source Nodes: [hidden_states_74, hidden_states_75, residual_8], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_3.run(buf186, arg100_1, buf175, arg101_1, arg102_1, buf190, 4096, 512, grid=grid(4096), stream=stream0)
        del arg100_1
        del arg101_1
        del arg102_1
        buf191 = reinterpret_tensor(buf170, (4096, 2048), (2048, 1), 0); del buf170  # reuse
        # Source Nodes: [matmul_52], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf190, (4096, 512), (512, 1), 0), arg103_1, out=buf191)
        del arg103_1
        buf192 = reinterpret_tensor(buf191, (16, 256, 2048), (524288, 2048, 1), 0); del buf191  # reuse
        # Source Nodes: [hidden_states_77, hidden_states_78], Original ATen: [aten.add, aten.gelu]
        triton_poi_fused_add_gelu_4.run(buf192, arg104_1, 8388608, grid=grid(8388608), stream=stream0)
        del arg104_1
        buf193 = buf186; del buf186  # reuse
        # Source Nodes: [matmul_53], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf192, (4096, 2048), (2048, 1), 0), arg105_1, out=buf193)
        del arg105_1
        buf197 = reinterpret_tensor(buf178, (16, 256, 512), (131072, 512, 1), 0); del buf178  # reuse
        # Source Nodes: [hidden_states_79, hidden_states_80, input_tensor_9], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_3.run(buf193, arg106_1, buf190, arg107_1, arg108_1, buf197, 4096, 512, grid=grid(4096), stream=stream0)
        del arg106_1
        del arg107_1
        del arg108_1
        buf198 = buf176; del buf176  # reuse
        # Source Nodes: [matmul_54], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf197, (4096, 512), (512, 1), 0), arg109_1, out=buf198)
        del arg109_1
        buf199 = reinterpret_tensor(buf193, (16, 16, 256, 32), (131072, 32, 512, 1), 0); del buf193  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_0.run(buf198, arg110_1, buf199, 2097152, grid=grid(2097152), stream=stream0)
        buf200 = reinterpret_tensor(buf190, (16, 16, 256, 32), (131072, 32, 512, 1), 0); del buf190  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf198, arg110_1, buf200, 2097152, grid=grid(2097152), stream=stream0)
        buf201 = buf177; del buf177  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf198, arg110_1, buf201, 2097152, grid=grid(2097152), stream=stream0)
        del arg110_1
        # Source Nodes: [], Original ATen: []
        buf202 = aten._scaled_dot_product_flash_attention.default(buf199, buf200, buf201, scale=0.17677669529663687)
        buf203 = buf202[0]
        del buf202
        buf208 = reinterpret_tensor(buf201, (4096, 512), (512, 1), 0); del buf201  # reuse
        # Source Nodes: [matmul_57], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf203, (4096, 512), (512, 1), 0), arg111_1, out=buf208)
        del arg111_1
        buf212 = reinterpret_tensor(buf203, (16, 256, 512), (131072, 512, 1), 0); del buf203  # reuse
        # Source Nodes: [hidden_states_83, hidden_states_84, residual_9], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_3.run(buf208, arg112_1, buf197, arg113_1, arg114_1, buf212, 4096, 512, grid=grid(4096), stream=stream0)
        del arg112_1
        del arg113_1
        del arg114_1
        buf213 = reinterpret_tensor(buf192, (4096, 2048), (2048, 1), 0); del buf192  # reuse
        # Source Nodes: [matmul_58], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf212, (4096, 512), (512, 1), 0), arg115_1, out=buf213)
        del arg115_1
        buf214 = reinterpret_tensor(buf213, (16, 256, 2048), (524288, 2048, 1), 0); del buf213  # reuse
        # Source Nodes: [hidden_states_86, hidden_states_87], Original ATen: [aten.add, aten.gelu]
        triton_poi_fused_add_gelu_4.run(buf214, arg116_1, 8388608, grid=grid(8388608), stream=stream0)
        del arg116_1
        buf215 = buf208; del buf208  # reuse
        # Source Nodes: [matmul_59], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf214, (4096, 2048), (2048, 1), 0), arg117_1, out=buf215)
        del arg117_1
        buf219 = reinterpret_tensor(buf200, (16, 256, 512), (131072, 512, 1), 0); del buf200  # reuse
        # Source Nodes: [hidden_states_88, hidden_states_89, input_tensor_10], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_3.run(buf215, arg118_1, buf212, arg119_1, arg120_1, buf219, 4096, 512, grid=grid(4096), stream=stream0)
        del arg118_1
        del arg119_1
        del arg120_1
        buf220 = buf198; del buf198  # reuse
        # Source Nodes: [matmul_60], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf219, (4096, 512), (512, 1), 0), arg121_1, out=buf220)
        del arg121_1
        buf221 = reinterpret_tensor(buf215, (16, 16, 256, 32), (131072, 32, 512, 1), 0); del buf215  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_0.run(buf220, arg122_1, buf221, 2097152, grid=grid(2097152), stream=stream0)
        buf222 = reinterpret_tensor(buf212, (16, 16, 256, 32), (131072, 32, 512, 1), 0); del buf212  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf220, arg122_1, buf222, 2097152, grid=grid(2097152), stream=stream0)
        buf223 = buf199; del buf199  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf220, arg122_1, buf223, 2097152, grid=grid(2097152), stream=stream0)
        del arg122_1
        # Source Nodes: [], Original ATen: []
        buf224 = aten._scaled_dot_product_flash_attention.default(buf221, buf222, buf223, scale=0.17677669529663687)
        buf225 = buf224[0]
        del buf224
        buf230 = reinterpret_tensor(buf223, (4096, 512), (512, 1), 0); del buf223  # reuse
        # Source Nodes: [matmul_63], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf225, (4096, 512), (512, 1), 0), arg123_1, out=buf230)
        del arg123_1
        buf234 = reinterpret_tensor(buf225, (16, 256, 512), (131072, 512, 1), 0); del buf225  # reuse
        # Source Nodes: [hidden_states_92, hidden_states_93, residual_10], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_3.run(buf230, arg124_1, buf219, arg125_1, arg126_1, buf234, 4096, 512, grid=grid(4096), stream=stream0)
        del arg124_1
        del arg125_1
        del arg126_1
        buf235 = reinterpret_tensor(buf214, (4096, 2048), (2048, 1), 0); del buf214  # reuse
        # Source Nodes: [matmul_64], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf234, (4096, 512), (512, 1), 0), arg127_1, out=buf235)
        del arg127_1
        buf236 = reinterpret_tensor(buf235, (16, 256, 2048), (524288, 2048, 1), 0); del buf235  # reuse
        # Source Nodes: [hidden_states_95, hidden_states_96], Original ATen: [aten.add, aten.gelu]
        triton_poi_fused_add_gelu_4.run(buf236, arg128_1, 8388608, grid=grid(8388608), stream=stream0)
        del arg128_1
        buf237 = buf230; del buf230  # reuse
        # Source Nodes: [matmul_65], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf236, (4096, 2048), (2048, 1), 0), arg129_1, out=buf237)
        del arg129_1
        buf241 = reinterpret_tensor(buf222, (16, 256, 512), (131072, 512, 1), 0); del buf222  # reuse
        # Source Nodes: [hidden_states_97, hidden_states_98, input_tensor_11], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_3.run(buf237, arg130_1, buf234, arg131_1, arg132_1, buf241, 4096, 512, grid=grid(4096), stream=stream0)
        del arg130_1
        del arg131_1
        del arg132_1
        buf242 = buf220; del buf220  # reuse
        # Source Nodes: [matmul_66], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf241, (4096, 512), (512, 1), 0), arg133_1, out=buf242)
        del arg133_1
        buf243 = reinterpret_tensor(buf237, (16, 16, 256, 32), (131072, 32, 512, 1), 0); del buf237  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_0.run(buf242, arg134_1, buf243, 2097152, grid=grid(2097152), stream=stream0)
        buf244 = reinterpret_tensor(buf234, (16, 16, 256, 32), (131072, 32, 512, 1), 0); del buf234  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf242, arg134_1, buf244, 2097152, grid=grid(2097152), stream=stream0)
        buf245 = buf221; del buf221  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf242, arg134_1, buf245, 2097152, grid=grid(2097152), stream=stream0)
        del arg134_1
        del buf242
        # Source Nodes: [], Original ATen: []
        buf246 = aten._scaled_dot_product_flash_attention.default(buf243, buf244, buf245, scale=0.17677669529663687)
        del buf243
        buf247 = buf246[0]
        del buf246
        buf252 = reinterpret_tensor(buf245, (4096, 512), (512, 1), 0); del buf245  # reuse
        # Source Nodes: [matmul_69], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf247, (4096, 512), (512, 1), 0), arg135_1, out=buf252)
        del arg135_1
        buf256 = reinterpret_tensor(buf247, (16, 256, 512), (131072, 512, 1), 0); del buf247  # reuse
        # Source Nodes: [hidden_states_101, hidden_states_102, residual_11], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_3.run(buf252, arg136_1, buf241, arg137_1, arg138_1, buf256, 4096, 512, grid=grid(4096), stream=stream0)
        del arg136_1
        del arg137_1
        del arg138_1
        buf257 = reinterpret_tensor(buf236, (4096, 2048), (2048, 1), 0); del buf236  # reuse
        # Source Nodes: [matmul_70], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf256, (4096, 512), (512, 1), 0), arg139_1, out=buf257)
        del arg139_1
        buf258 = reinterpret_tensor(buf257, (16, 256, 2048), (524288, 2048, 1), 0); del buf257  # reuse
        # Source Nodes: [hidden_states_104, hidden_states_105], Original ATen: [aten.add, aten.gelu]
        triton_poi_fused_add_gelu_4.run(buf258, arg140_1, 8388608, grid=grid(8388608), stream=stream0)
        del arg140_1
        buf259 = buf252; del buf252  # reuse
        # Source Nodes: [matmul_71], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf258, (4096, 2048), (2048, 1), 0), arg141_1, out=buf259)
        del arg141_1
        del buf258
        buf263 = reinterpret_tensor(buf244, (16, 256, 512), (131072, 512, 1), 0); del buf244  # reuse
        # Source Nodes: [hidden_states_106, hidden_states_107, hidden_states_108], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_3.run(buf259, arg142_1, buf256, arg143_1, arg144_1, buf263, 4096, 512, grid=grid(4096), stream=stream0)
        del arg142_1
        del arg143_1
        del arg144_1
        del buf256
        del buf259
    return (buf21, buf43, buf65, buf87, buf109, buf131, buf153, buf175, buf197, buf219, buf241, buf263, )


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
    arg13_1 = rand_strided((512, 1536), (1536, 1), device='cuda:0', dtype=torch.float16)
    arg14_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg15_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float16)
    arg16_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg17_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg18_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg19_1 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float16)
    arg20_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg21_1 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float16)
    arg22_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg23_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg24_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg25_1 = rand_strided((512, 1536), (1536, 1), device='cuda:0', dtype=torch.float16)
    arg26_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg27_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float16)
    arg28_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg29_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg30_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg31_1 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float16)
    arg32_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg33_1 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float16)
    arg34_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg35_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg36_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg37_1 = rand_strided((512, 1536), (1536, 1), device='cuda:0', dtype=torch.float16)
    arg38_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg39_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float16)
    arg40_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg41_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg42_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg43_1 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float16)
    arg44_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg45_1 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float16)
    arg46_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg47_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg48_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg49_1 = rand_strided((512, 1536), (1536, 1), device='cuda:0', dtype=torch.float16)
    arg50_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg51_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float16)
    arg52_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg53_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg54_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg55_1 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float16)
    arg56_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg57_1 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float16)
    arg58_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg59_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg60_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg61_1 = rand_strided((512, 1536), (1536, 1), device='cuda:0', dtype=torch.float16)
    arg62_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg63_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float16)
    arg64_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg65_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg66_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg67_1 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float16)
    arg68_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg69_1 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float16)
    arg70_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg71_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg72_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg73_1 = rand_strided((512, 1536), (1536, 1), device='cuda:0', dtype=torch.float16)
    arg74_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg75_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float16)
    arg76_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg77_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg78_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg79_1 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float16)
    arg80_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg81_1 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float16)
    arg82_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg83_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg84_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg85_1 = rand_strided((512, 1536), (1536, 1), device='cuda:0', dtype=torch.float16)
    arg86_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg87_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float16)
    arg88_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg89_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg90_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg91_1 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float16)
    arg92_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg93_1 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float16)
    arg94_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg95_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg96_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg97_1 = rand_strided((512, 1536), (1536, 1), device='cuda:0', dtype=torch.float16)
    arg98_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg99_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float16)
    arg100_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg101_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg102_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg103_1 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float16)
    arg104_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg105_1 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float16)
    arg106_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg107_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg108_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg109_1 = rand_strided((512, 1536), (1536, 1), device='cuda:0', dtype=torch.float16)
    arg110_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg111_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float16)
    arg112_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg113_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg114_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg115_1 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float16)
    arg116_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg117_1 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float16)
    arg118_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg119_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg120_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg121_1 = rand_strided((512, 1536), (1536, 1), device='cuda:0', dtype=torch.float16)
    arg122_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg123_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float16)
    arg124_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg125_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg126_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg127_1 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float16)
    arg128_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg129_1 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float16)
    arg130_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg131_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg132_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg133_1 = rand_strided((512, 1536), (1536, 1), device='cuda:0', dtype=torch.float16)
    arg134_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg135_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float16)
    arg136_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg137_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg138_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg139_1 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float16)
    arg140_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg141_1 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float16)
    arg142_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg143_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg144_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float16)
    fn = lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
