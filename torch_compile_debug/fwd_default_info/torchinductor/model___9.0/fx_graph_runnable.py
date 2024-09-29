
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


torch._functorch.config.debug_partitioner = True



isolate_fails_code_str = None



# torch version: 2.3.0a0+6ddf5cf85e.nv24.04
# torch cuda version: 12.4
# torch git version: Unknown


# CUDA Info: 
# nvcc: NVIDIA (R) Cuda compiler driver 
# Copyright (c) 2005-2024 NVIDIA Corporation 
# Built on Thu_Mar_28_02:18:24_PDT_2024 
# Cuda compilation tools, release 12.4, V12.4.131 
# Build cuda_12.4.r12.4/compiler.34097967_0 

# GPU Hardware Info: 
# NVIDIA A100-PCIE-40GB : 1 
# NVIDIA TITAN RTX : 1 
# NVIDIA TITAN X (Pascal) : 1 


from torch.nn import *
class Repro(torch.nn.Module):
    def __init__(self):
        super().__init__()

    
    
    def forward(self, arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1):
        view = torch.ops.aten.view.default(arg0_1, [4096, 512])
        mm = torch.ops.aten.mm.default(view, arg1_1);  view = arg1_1 = None
        view_1 = torch.ops.aten.view.default(mm, [16, 256, 1536]);  mm = None
        add = torch.ops.aten.add.Tensor(view_1, arg2_1);  view_1 = arg2_1 = None
        split = torch.ops.aten.split.Tensor(add, 512, -1);  add = None
        getitem = split[0]
        getitem_1 = split[1]
        getitem_2 = split[2];  split = None
        view_2 = torch.ops.aten.view.default(getitem, [16, 256, 16, 32]);  getitem = None
        view_3 = torch.ops.aten.view.default(getitem_1, [16, 256, 16, 32]);  getitem_1 = None
        view_4 = torch.ops.aten.view.default(getitem_2, [16, 256, 16, 32]);  getitem_2 = None
        permute_default = torch.ops.aten.permute.default(view_2, [0, 2, 1, 3]);  view_2 = None
        permute_default_1 = torch.ops.aten.permute.default(view_3, [0, 2, 1, 3]);  view_3 = None
        permute_default_2 = torch.ops.aten.permute.default(view_4, [0, 2, 1, 3]);  view_4 = None
        _scaled_dot_product_flash_attention_default = torch.ops.aten._scaled_dot_product_flash_attention.default(permute_default, permute_default_1, permute_default_2, scale = 0.17677669529663687);  permute_default = permute_default_1 = permute_default_2 = None
        getitem_7 = _scaled_dot_product_flash_attention_default[0];  _scaled_dot_product_flash_attention_default = None
        permute_4 = torch.ops.aten.permute.default(getitem_7, [0, 2, 1, 3]);  getitem_7 = None
        clone_3 = torch.ops.aten.clone.default(permute_4, memory_format = torch.contiguous_format);  permute_4 = None
        view_11 = torch.ops.aten.view.default(clone_3, [16, 256, 512]);  clone_3 = None
        view_12 = torch.ops.aten.view.default(view_11, [4096, 512]);  view_11 = None
        mm_1 = torch.ops.aten.mm.default(view_12, arg3_1);  view_12 = arg3_1 = None
        view_13 = torch.ops.aten.view.default(mm_1, [16, 256, 512]);  mm_1 = None
        add_1 = torch.ops.aten.add.Tensor(view_13, arg4_1);  view_13 = arg4_1 = None
        add_2 = torch.ops.aten.add.Tensor(add_1, arg0_1);  add_1 = arg0_1 = None
        convert_element_type_10 = torch.ops.prims.convert_element_type.default(add_2, torch.float32)
        var_mean = torch.ops.aten.var_mean.correction(convert_element_type_10, [2], correction = 0, keepdim = True);  convert_element_type_10 = None
        getitem_3 = var_mean[0]
        getitem_4 = var_mean[1];  var_mean = None
        add_3 = torch.ops.aten.add.Tensor(getitem_3, 1e-05);  getitem_3 = None
        rsqrt = torch.ops.aten.rsqrt.default(add_3);  add_3 = None
        sub_1 = torch.ops.aten.sub.Tensor(add_2, getitem_4);  add_2 = getitem_4 = None
        mul = torch.ops.aten.mul.Tensor(sub_1, rsqrt);  sub_1 = rsqrt = None
        mul_1 = torch.ops.aten.mul.Tensor(mul, arg5_1);  mul = arg5_1 = None
        add_4 = torch.ops.aten.add.Tensor(mul_1, arg6_1);  mul_1 = arg6_1 = None
        convert_element_type_11 = torch.ops.prims.convert_element_type.default(add_4, torch.float16);  add_4 = None
        view_14 = torch.ops.aten.view.default(convert_element_type_11, [4096, 512])
        mm_2 = torch.ops.aten.mm.default(view_14, arg7_1);  view_14 = arg7_1 = None
        view_15 = torch.ops.aten.view.default(mm_2, [16, 256, 2048]);  mm_2 = None
        add_5 = torch.ops.aten.add.Tensor(view_15, arg8_1);  view_15 = arg8_1 = None
        convert_element_type_14 = torch.ops.prims.convert_element_type.default(add_5, torch.float32);  add_5 = None
        mul_2 = torch.ops.aten.mul.Tensor(convert_element_type_14, 0.5)
        mul_3 = torch.ops.aten.mul.Tensor(convert_element_type_14, 0.7071067811865476);  convert_element_type_14 = None
        erf = torch.ops.aten.erf.default(mul_3);  mul_3 = None
        add_6 = torch.ops.aten.add.Tensor(erf, 1);  erf = None
        mul_4 = torch.ops.aten.mul.Tensor(mul_2, add_6);  mul_2 = add_6 = None
        convert_element_type_15 = torch.ops.prims.convert_element_type.default(mul_4, torch.float16);  mul_4 = None
        view_16 = torch.ops.aten.view.default(convert_element_type_15, [4096, 2048]);  convert_element_type_15 = None
        mm_3 = torch.ops.aten.mm.default(view_16, arg9_1);  view_16 = arg9_1 = None
        view_17 = torch.ops.aten.view.default(mm_3, [16, 256, 512]);  mm_3 = None
        add_7 = torch.ops.aten.add.Tensor(view_17, arg10_1);  view_17 = arg10_1 = None
        add_8 = torch.ops.aten.add.Tensor(add_7, convert_element_type_11);  add_7 = convert_element_type_11 = None
        convert_element_type_18 = torch.ops.prims.convert_element_type.default(add_8, torch.float32)
        var_mean_1 = torch.ops.aten.var_mean.correction(convert_element_type_18, [2], correction = 0, keepdim = True);  convert_element_type_18 = None
        getitem_5 = var_mean_1[0]
        getitem_6 = var_mean_1[1];  var_mean_1 = None
        add_9 = torch.ops.aten.add.Tensor(getitem_5, 1e-05);  getitem_5 = None
        rsqrt_1 = torch.ops.aten.rsqrt.default(add_9);  add_9 = None
        sub_2 = torch.ops.aten.sub.Tensor(add_8, getitem_6);  add_8 = getitem_6 = None
        mul_5 = torch.ops.aten.mul.Tensor(sub_2, rsqrt_1);  sub_2 = rsqrt_1 = None
        mul_6 = torch.ops.aten.mul.Tensor(mul_5, arg11_1);  mul_5 = arg11_1 = None
        add_10 = torch.ops.aten.add.Tensor(mul_6, arg12_1);  mul_6 = arg12_1 = None
        convert_element_type_19 = torch.ops.prims.convert_element_type.default(add_10, torch.float16);  add_10 = None
        return (convert_element_type_19,)
        
def load_args(reader):
    buf0 = reader.storage(None, 4194304, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf0, (16, 256, 512), dtype=torch.float16, is_leaf=True)  # arg0_1
    buf1 = reader.storage(None, 1572864, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1, (512, 1536), dtype=torch.float16, is_leaf=True)  # arg1_1
    buf2 = reader.storage(None, 3072, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf2, (1536,), dtype=torch.float16, is_leaf=True)  # arg2_1
    buf3 = reader.storage(None, 524288, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf3, (512, 512), dtype=torch.float16, is_leaf=True)  # arg3_1
    buf4 = reader.storage(None, 1024, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf4, (512,), dtype=torch.float16, is_leaf=True)  # arg4_1
    buf5 = reader.storage(None, 1024, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf5, (512,), dtype=torch.float16, is_leaf=True)  # arg5_1
    buf6 = reader.storage(None, 1024, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf6, (512,), dtype=torch.float16, is_leaf=True)  # arg6_1
    buf7 = reader.storage(None, 2097152, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf7, (512, 2048), dtype=torch.float16, is_leaf=True)  # arg7_1
    buf8 = reader.storage(None, 4096, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf8, (2048,), dtype=torch.float16, is_leaf=True)  # arg8_1
    buf9 = reader.storage(None, 2097152, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf9, (2048, 512), dtype=torch.float16, is_leaf=True)  # arg9_1
    buf10 = reader.storage(None, 1024, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf10, (512,), dtype=torch.float16, is_leaf=True)  # arg10_1
    buf11 = reader.storage(None, 1024, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf11, (512,), dtype=torch.float16, is_leaf=True)  # arg11_1
    buf12 = reader.storage(None, 1024, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf12, (512,), dtype=torch.float16, is_leaf=True)  # arg12_1
load_args._version = 0
mod = Repro()
if __name__ == '__main__':
    from torch._dynamo.repro.after_aot import run_repro
    with torch.no_grad():
        run_repro(mod, load_args, accuracy=False, command='run', save_dir=None, tracing_mode='real', check_str=None)
