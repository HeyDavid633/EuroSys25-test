
# AOT ID: ['0_inference']
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
from torch._inductor.async_compile import AsyncCompile
from torch._inductor.select_algorithm import extern_kernels
from torch._inductor.codegen.multi_kernel import MultiKernelCall

aten = torch.ops.aten
inductor_ops = torch.ops.inductor
_quantized = torch.ops._quantized
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cpu = torch._C._dynamo.guards._empty_strided_cpu
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor
alloc_from_pool = torch.ops.inductor._alloc_from_pool
async_compile = AsyncCompile()


# kernel path: /tmp/torchinductor_root/as/cask52izqs6yt536gfumemfwch2x62gslfe2e7bt3fjlhdgbq44v.py
# Source Nodes: [], Original ATen: []

triton_poi_fused_0 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[4194304], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp16', 3: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=108), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_0', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'EE52B3E22FE0E1CBEBB76214115F53FF3B61A525BEEA2EA79A4DAA2457974610', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4194304
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
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
from torch._inductor.runtime.triton_heuristics import grid, split_scan_grid, grid_combo_kernels, start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_raw_stream


# kernel path: /tmp/torchinductor_root/wf/cwfsxx2gbqxgyw3opmz5kw6iqntewqsxgt5bwqsdtgngsnejiz5y.py
# Source Nodes: [], Original ATen: []

triton_poi_fused_1 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[4194304], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp16', 3: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=108), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_1', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'EE52B3E22FE0E1CBEBB76214115F53FF3B61A525BEEA2EA79A4DAA2457974610', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4194304
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex % 512
    x1 = (xindex // 512)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (512 + x0 + (1536*x1)), None).to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (512 + x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x2), tmp2, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_root/yi/cyiyyslgwojro5tq2wmdjfgkzhjjk4ksbct7edn5morycarm6tko.py
# Source Nodes: [], Original ATen: []

triton_poi_fused_2 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[4194304], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp16', 3: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=108), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_2', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'EE52B3E22FE0E1CBEBB76214115F53FF3B61A525BEEA2EA79A4DAA2457974610', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4194304
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex % 512
    x1 = (xindex // 512)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (1024 + x0 + (1536*x1)), None).to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (1024 + x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x2), tmp2, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_root/6w/c6wbojp3pv7lw6zsqp7amz3kks2klgjefick4ygd45mhjjekyfzc.py
# Source Nodes: [hidden_states_1, hidden_states_2, hidden_states_3], Original ATen: [aten.add, aten.native_layer_norm]
# hidden_states_1 => add_1
# hidden_states_2 => add_2
# hidden_states_3 => add_3, add_4, convert_element_type_10, convert_element_type_11, mul, mul_1, rsqrt, sub_1, var_mean
triton_per_fused_add_native_layer_norm_3 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.persistent_reduction(
    size_hints=[8192, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp16', 3: '*fp16', 4: '*fp16', 5: '*fp16', 6: 'i32', 7: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=108), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_3', 'mutated_arg_names': [], 'no_x_dim': True, 'num_load': 5, 'num_reduction': 4, 'backend_hash': 'EE52B3E22FE0E1CBEBB76214115F53FF3B61A525BEEA2EA79A4DAA2457974610', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, xnumel, rnumel):
    xnumel = 8192
    XBLOCK: tl.constexpr = 1
    rnumel = 512
    RBLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = tl.full([RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[:]
    roffset = 0
    rmask = tl.full([RBLOCK], True, tl.int1)
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (512*x0)), None).to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (r1), None, eviction_policy='evict_last').to(tl.float32)
    tmp3 = tl.load(in_ptr2 + (r1 + (512*x0)), None).to(tl.float32)
    tmp26 = tl.load(in_ptr3 + (r1), None, eviction_policy='evict_last').to(tl.float32)
    tmp29 = tl.load(in_ptr4 + (r1), None, eviction_policy='evict_last').to(tl.float32)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp5 = tmp4.to(tl.float32)
    tmp6 = tl.broadcast_to(tmp5, [RBLOCK])
    tmp8 = tl.broadcast_to(tmp6, [RBLOCK])
    tmp10 = triton_helpers.promote_to_tensor(tl.sum(tmp8, 0))
    tmp11 = tl.full([1], 512, tl.int32)
    tmp12 = tmp11.to(tl.float32)
    tmp13 = tmp10 / tmp12
    tmp14 = tmp6 - tmp13
    tmp15 = tmp14 * tmp14
    tmp16 = tl.broadcast_to(tmp15, [RBLOCK])
    tmp18 = triton_helpers.promote_to_tensor(tl.sum(tmp16, 0))
    tmp19 = tmp5 - tmp13
    tmp20 = 512.0
    tmp21 = tmp18 / tmp20
    tmp22 = 1e-05
    tmp23 = tmp21 + tmp22
    tmp24 = libdevice.rsqrt(tmp23)
    tmp25 = tmp19 * tmp24
    tmp27 = tmp26.to(tl.float32)
    tmp28 = tmp25 * tmp27
    tmp30 = tmp29.to(tl.float32)
    tmp31 = tmp28 + tmp30
    tmp32 = tmp31.to(tl.float32)
    tl.store(out_ptr2 + (r1 + (512*x0)), tmp32, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_root/xv/cxvde63sevpzlq2ubojop72nqi3d22dein53hghy3z7gptbhh4kj.py
# Source Nodes: [hidden_states_4, hidden_states_5], Original ATen: [aten.add, aten.gelu]
# hidden_states_4 => add_5
# hidden_states_5 => add_6, convert_element_type_14, convert_element_type_15, erf, mul_2, mul_3, mul_4
triton_poi_fused_add_gelu_4 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[16777216], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp16', 1: '*fp16', 2: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=108), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_gelu_4', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'EE52B3E22FE0E1CBEBB76214115F53FF3B61A525BEEA2EA79A4DAA2457974610', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16777216
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
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
    tmp8 = libdevice.erf(tmp7)
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
    assert_size_stride(arg0_1, (8, 1024, 512), (524288, 512, 1))
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
        buf0 = empty_strided_cuda((8192, 1536), (1536, 1), torch.float16)
        # Source Nodes: [matmul], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(arg0_1, (8192, 512), (512, 1), 0), arg1_1, out=buf0)
        del arg1_1
        buf1 = empty_strided_cuda((8, 16, 1024, 32), (524288, 32, 512, 1), torch.float16)
        # Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_0.run(buf0, arg2_1, buf1, 4194304, grid=grid(4194304), stream=stream0)
        buf2 = empty_strided_cuda((8, 16, 1024, 32), (524288, 32, 512, 1), torch.float16)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf0, arg2_1, buf2, 4194304, grid=grid(4194304), stream=stream0)
        buf3 = empty_strided_cuda((8, 16, 1024, 32), (524288, 32, 512, 1), torch.float16)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf0, arg2_1, buf3, 4194304, grid=grid(4194304), stream=stream0)
        del arg2_1
        del buf0
        # Source Nodes: [], Original ATen: []
        buf4 = torch.ops.aten._scaled_dot_product_flash_attention.default(buf1, buf2, buf3, scale=0.17677669529663687)
        del buf1
        buf5 = buf4[0]
        del buf4
        buf10 = reinterpret_tensor(buf3, (8192, 512), (512, 1), 0); del buf3  # reuse
        # Source Nodes: [matmul_3], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf5, (8192, 512), (512, 1), 0), arg3_1, out=buf10)
        del arg3_1
        buf14 = reinterpret_tensor(buf5, (8, 1024, 512), (524288, 512, 1), 0); del buf5  # reuse
        # Source Nodes: [hidden_states_1, hidden_states_2, hidden_states_3], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_3.run(buf10, arg4_1, arg0_1, arg6_1, arg5_1, buf14, 8192, 512, grid=grid(8192), stream=stream0)
        del arg0_1
        del arg4_1
        del arg5_1
        del arg6_1
        buf15 = empty_strided_cuda((8192, 2048), (2048, 1), torch.float16)
        # Source Nodes: [matmul_4], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf14, (8192, 512), (512, 1), 0), arg7_1, out=buf15)
        del arg7_1
        buf16 = reinterpret_tensor(buf15, (8, 1024, 2048), (2097152, 2048, 1), 0); del buf15  # reuse
        # Source Nodes: [hidden_states_4, hidden_states_5], Original ATen: [aten.add, aten.gelu]
        triton_poi_fused_add_gelu_4.run(buf16, arg8_1, 16777216, grid=grid(16777216), stream=stream0)
        del arg8_1
        buf17 = buf10; del buf10  # reuse
        # Source Nodes: [matmul_5], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf16, (8192, 2048), (2048, 1), 0), arg9_1, out=buf17)
        del arg9_1
        del buf16
        buf21 = reinterpret_tensor(buf2, (8, 1024, 512), (524288, 512, 1), 0); del buf2  # reuse
        # Source Nodes: [hidden_states_6, hidden_states_7, hidden_states_8], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_3.run(buf17, arg10_1, buf14, arg12_1, arg11_1, buf21, 8192, 512, grid=grid(8192), stream=stream0)
        del arg10_1
        del arg11_1
        del arg12_1
        del buf14
        del buf17
    return (buf21, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((8, 1024, 512), (524288, 512, 1), device='cuda:0', dtype=torch.float16)
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
