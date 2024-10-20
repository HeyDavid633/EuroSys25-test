
import torch
from torch import tensor, device
import torch.fx as fx
from torch._dynamo.testing import rand_strided
from math import inf
import torch._inductor.inductor_prims

import torch._dynamo.config
import torch._inductor.config
import torch._functorch.config
import torch.fx.experimental._config

torch._inductor.config.trace.enabled = True
torch._functorch.config.unlift_effect_tokens = True



isolate_fails_code_str = None



# torch version: 2.5.0a0+b465a5843b.nv24.09
# torch cuda version: 12.6
# torch git version: Unknown


# CUDA Info: 
# nvcc: NVIDIA (R) Cuda compiler driver 
# Copyright (c) 2005-2024 NVIDIA Corporation 
# Built on Wed_Aug_14_10:10:22_PDT_2024 
# Cuda compilation tools, release 12.6, V12.6.68 
# Build cuda_12.6.r12.6/compiler.34714021_0 

# GPU Hardware Info: 
# NVIDIA A100-PCIE-40GB : 1 


from torch.nn import *
class Repro(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    
    
    def forward(self, arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1):
        view = torch.ops.aten.view.default(arg0_1, [8192, 512])
        mm = torch.ops.aten.mm.default(view, arg1_1);  view = arg1_1 = None
        view_1 = torch.ops.aten.view.default(mm, [8, 1024, 1536]);  mm = None
        add = torch.ops.aten.add.Tensor(view_1, arg2_1);  view_1 = arg2_1 = None
        split = torch.ops.aten.split.Tensor(add, 512, -1);  add = None
        getitem = split[0]
        getitem_1 = split[1]
        getitem_2 = split[2];  split = None
        view_2 = torch.ops.aten.view.default(getitem, [8, 1024, 16, 32]);  getitem = None
        permute = torch.ops.aten.permute.default(view_2, [0, 2, 1, 3]);  view_2 = None
        view_3 = torch.ops.aten.view.default(getitem_1, [8, 1024, 16, 32]);  getitem_1 = None
        permute_1 = torch.ops.aten.permute.default(view_3, [0, 2, 1, 3]);  view_3 = None
        view_4 = torch.ops.aten.view.default(getitem_2, [8, 1024, 16, 32]);  getitem_2 = None
        permute_2 = torch.ops.aten.permute.default(view_4, [0, 2, 1, 3]);  view_4 = None
        permute_3 = torch.ops.aten.permute.default(permute_1, [0, 1, 3, 2]);  permute_1 = None
        expand = torch.ops.aten.expand.default(permute, [8, 16, 1024, 32]);  permute = None
        clone = torch.ops.aten.clone.default(expand, memory_format = torch.contiguous_format);  expand = None
        view_5 = torch.ops.aten.view.default(clone, [128, 1024, 32]);  clone = None
        expand_1 = torch.ops.aten.expand.default(permute_3, [8, 16, 32, 1024]);  permute_3 = None
        clone_1 = torch.ops.aten.clone.default(expand_1, memory_format = torch.contiguous_format);  expand_1 = None
        view_6 = torch.ops.aten.view.default(clone_1, [128, 32, 1024]);  clone_1 = None
        bmm = torch.ops.aten.bmm.default(view_5, view_6);  view_5 = view_6 = None
        view_7 = torch.ops.aten.view.default(bmm, [8, 16, 1024, 1024]);  bmm = None
        div = torch.ops.aten.div.Tensor(view_7, 5.656854249492381);  view_7 = None
        unsqueeze = torch.ops.aten.unsqueeze.default(arg3_1, 1);  arg3_1 = None
        sub = torch.ops.aten.sub.Tensor(1.0, unsqueeze);  unsqueeze = None
        mul = torch.ops.aten.mul.Tensor(sub, 10000.0);  sub = None
        sub_1 = torch.ops.aten.sub.Tensor(div, mul);  div = mul = None
        convert_element_type_default = torch.ops.prims.convert_element_type.default(sub_1, torch.float32);  sub_1 = None
        amax = torch.ops.aten.amax.default(convert_element_type_default, [-1], True)
        sub_2 = torch.ops.aten.sub.Tensor(convert_element_type_default, amax);  convert_element_type_default = amax = None
        exp = torch.ops.aten.exp.default(sub_2);  sub_2 = None
        sum_1 = torch.ops.aten.sum.dim_IntList(exp, [-1], True)
        div_1 = torch.ops.aten.div.Tensor(exp, sum_1);  exp = sum_1 = None
        convert_element_type_6 = torch.ops.prims.convert_element_type.default(div_1, torch.float16);  div_1 = None
        expand_2 = torch.ops.aten.expand.default(convert_element_type_6, [8, 16, 1024, 1024]);  convert_element_type_6 = None
        view_8 = torch.ops.aten.view.default(expand_2, [128, 1024, 1024]);  expand_2 = None
        expand_3 = torch.ops.aten.expand.default(permute_2, [8, 16, 1024, 32]);  permute_2 = None
        clone_2 = torch.ops.aten.clone.default(expand_3, memory_format = torch.contiguous_format);  expand_3 = None
        view_9 = torch.ops.aten.view.default(clone_2, [128, 1024, 32]);  clone_2 = None
        bmm_1 = torch.ops.aten.bmm.default(view_8, view_9);  view_8 = view_9 = None
        view_10 = torch.ops.aten.view.default(bmm_1, [8, 16, 1024, 32]);  bmm_1 = None
        permute_4 = torch.ops.aten.permute.default(view_10, [0, 2, 1, 3]);  view_10 = None
        clone_3 = torch.ops.aten.clone.default(permute_4, memory_format = torch.contiguous_format);  permute_4 = None
        view_11 = torch.ops.aten.view.default(clone_3, [8, 1024, 512]);  clone_3 = None
        view_12 = torch.ops.aten.view.default(view_11, [8192, 512]);  view_11 = None
        mm_1 = torch.ops.aten.mm.default(view_12, arg4_1);  view_12 = arg4_1 = None
        view_13 = torch.ops.aten.view.default(mm_1, [8, 1024, 512]);  mm_1 = None
        add_1 = torch.ops.aten.add.Tensor(view_13, arg5_1);  view_13 = arg5_1 = None
        add_2 = torch.ops.aten.add.Tensor(add_1, arg0_1);  add_1 = arg0_1 = None
        convert_element_type_11 = torch.ops.prims.convert_element_type.default(add_2, torch.float32)
        var_mean = torch.ops.aten.var_mean.correction(convert_element_type_11, [2], correction = 0, keepdim = True);  convert_element_type_11 = None
        getitem_3 = var_mean[0]
        getitem_4 = var_mean[1];  var_mean = None
        add_3 = torch.ops.aten.add.Tensor(getitem_3, 1e-05);  getitem_3 = None
        rsqrt = torch.ops.aten.rsqrt.default(add_3);  add_3 = None
        sub_3 = torch.ops.aten.sub.Tensor(add_2, getitem_4);  add_2 = getitem_4 = None
        mul_1 = torch.ops.aten.mul.Tensor(sub_3, rsqrt);  sub_3 = rsqrt = None
        mul_2 = torch.ops.aten.mul.Tensor(mul_1, arg7_1);  mul_1 = arg7_1 = None
        add_4 = torch.ops.aten.add.Tensor(mul_2, arg6_1);  mul_2 = arg6_1 = None
        convert_element_type_12 = torch.ops.prims.convert_element_type.default(add_4, torch.float16);  add_4 = None
        view_14 = torch.ops.aten.view.default(convert_element_type_12, [8192, 512])
        mm_2 = torch.ops.aten.mm.default(view_14, arg8_1);  view_14 = arg8_1 = None
        view_15 = torch.ops.aten.view.default(mm_2, [8, 1024, 2048]);  mm_2 = None
        add_5 = torch.ops.aten.add.Tensor(view_15, arg9_1);  view_15 = arg9_1 = None
        convert_element_type_15 = torch.ops.prims.convert_element_type.default(add_5, torch.float32);  add_5 = None
        mul_3 = torch.ops.aten.mul.Tensor(convert_element_type_15, 0.5)
        mul_4 = torch.ops.aten.mul.Tensor(convert_element_type_15, 0.7071067811865476);  convert_element_type_15 = None
        erf = torch.ops.aten.erf.default(mul_4);  mul_4 = None
        add_6 = torch.ops.aten.add.Tensor(erf, 1);  erf = None
        mul_5 = torch.ops.aten.mul.Tensor(mul_3, add_6);  mul_3 = add_6 = None
        convert_element_type_16 = torch.ops.prims.convert_element_type.default(mul_5, torch.float16);  mul_5 = None
        view_16 = torch.ops.aten.view.default(convert_element_type_16, [8192, 2048]);  convert_element_type_16 = None
        mm_3 = torch.ops.aten.mm.default(view_16, arg10_1);  view_16 = arg10_1 = None
        view_17 = torch.ops.aten.view.default(mm_3, [8, 1024, 512]);  mm_3 = None
        add_7 = torch.ops.aten.add.Tensor(view_17, arg11_1);  view_17 = arg11_1 = None
        add_8 = torch.ops.aten.add.Tensor(add_7, convert_element_type_12);  add_7 = convert_element_type_12 = None
        convert_element_type_19 = torch.ops.prims.convert_element_type.default(add_8, torch.float32)
        var_mean_1 = torch.ops.aten.var_mean.correction(convert_element_type_19, [2], correction = 0, keepdim = True);  convert_element_type_19 = None
        getitem_5 = var_mean_1[0]
        getitem_6 = var_mean_1[1];  var_mean_1 = None
        add_9 = torch.ops.aten.add.Tensor(getitem_5, 1e-05);  getitem_5 = None
        rsqrt_1 = torch.ops.aten.rsqrt.default(add_9);  add_9 = None
        sub_4 = torch.ops.aten.sub.Tensor(add_8, getitem_6);  add_8 = getitem_6 = None
        mul_6 = torch.ops.aten.mul.Tensor(sub_4, rsqrt_1);  sub_4 = rsqrt_1 = None
        mul_7 = torch.ops.aten.mul.Tensor(mul_6, arg13_1);  mul_6 = arg13_1 = None
        add_10 = torch.ops.aten.add.Tensor(mul_7, arg12_1);  mul_7 = arg12_1 = None
        convert_element_type_20 = torch.ops.prims.convert_element_type.default(add_10, torch.float16);  add_10 = None
        return (convert_element_type_20,)
        
def load_args(reader):
    buf0 = reader.storage(None, 8388608, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf0, (8, 1024, 512), dtype=torch.float16, is_leaf=True)  # arg0_1
    buf1 = reader.storage(None, 1572864, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1, (512, 1536), dtype=torch.float16, is_leaf=True)  # arg1_1
    buf2 = reader.storage(None, 3072, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf2, (1536,), dtype=torch.float16, is_leaf=True)  # arg2_1
    buf3 = reader.storage(None, 33554432, device=device(type='cuda', index=0))
    reader.tensor(buf3, (8, 1024, 1024), is_leaf=True)  # arg3_1
    buf4 = reader.storage(None, 524288, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf4, (512, 512), dtype=torch.float16, is_leaf=True)  # arg4_1
    buf5 = reader.storage(None, 1024, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf5, (512,), dtype=torch.float16, is_leaf=True)  # arg5_1
    buf6 = reader.storage(None, 1024, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf6, (512,), dtype=torch.float16, is_leaf=True)  # arg6_1
    buf7 = reader.storage(None, 1024, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf7, (512,), dtype=torch.float16, is_leaf=True)  # arg7_1
    buf8 = reader.storage(None, 2097152, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf8, (512, 2048), dtype=torch.float16, is_leaf=True)  # arg8_1
    buf9 = reader.storage(None, 4096, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf9, (2048,), dtype=torch.float16, is_leaf=True)  # arg9_1
    buf10 = reader.storage(None, 2097152, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf10, (2048, 512), dtype=torch.float16, is_leaf=True)  # arg10_1
    buf11 = reader.storage(None, 1024, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf11, (512,), dtype=torch.float16, is_leaf=True)  # arg11_1
    buf12 = reader.storage(None, 1024, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf12, (512,), dtype=torch.float16, is_leaf=True)  # arg12_1
    buf13 = reader.storage(None, 1024, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf13, (512,), dtype=torch.float16, is_leaf=True)  # arg13_1
load_args._version = 0
mod = Repro()
if __name__ == '__main__':
    from torch._dynamo.repro.after_aot import run_repro
    with torch.no_grad():
        run_repro(mod, load_args, accuracy=False, command='run', save_dir=None, tracing_mode='real', check_str=None)
        # To run it separately, do 
        # mod, args = run_repro(mod, load_args, accuracy=False, command='get_args', save_dir=None, tracing_mode='real', check_str=None)
        # mod(*args)