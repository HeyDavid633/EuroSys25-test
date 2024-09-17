
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

torch._inductor.config.triton.cudagraphs = True
torch._inductor.config.trace.enabled = True




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
# NVIDIA GeForce RTX 4080 Laptop GPU : 1 


from torch.nn import *
class Repro(torch.nn.Module):
    def __init__(self):
        super().__init__()

    
    
    def forward(self, arg0_1, arg1_1, arg2_1):
        permute = torch.ops.aten.permute.default(arg0_1, [0, 1, 3, 2]);  arg0_1 = None
        expand = torch.ops.aten.expand.default(arg1_1, [16, 16, 256, 32]);  arg1_1 = None
        clone = torch.ops.aten.clone.default(expand, memory_format = torch.contiguous_format);  expand = None
        view = torch.ops.aten.view.default(clone, [256, 256, 32]);  clone = None
        expand_1 = torch.ops.aten.expand.default(permute, [16, 16, 32, 256]);  permute = None
        clone_1 = torch.ops.aten.clone.default(expand_1, memory_format = torch.contiguous_format);  expand_1 = None
        view_1 = torch.ops.aten.view.default(clone_1, [256, 32, 256]);  clone_1 = None
        bmm = torch.ops.aten.bmm.default(view, view_1);  view = view_1 = None
        view_2 = torch.ops.aten.view.default(bmm, [16, 16, 256, 256]);  bmm = None
        div = torch.ops.aten.div.Tensor(view_2, 5.656854249492381);  view_2 = None
        convert_element_type_2 = torch.ops.prims.convert_element_type.default(div, torch.float32);  div = None
        amax = torch.ops.aten.amax.default(convert_element_type_2, [-1], True)
        sub = torch.ops.aten.sub.Tensor(convert_element_type_2, amax);  convert_element_type_2 = amax = None
        exp = torch.ops.aten.exp.default(sub);  sub = None
        sum_1 = torch.ops.aten.sum.dim_IntList(exp, [-1], True)
        div_1 = torch.ops.aten.div.Tensor(exp, sum_1);  exp = sum_1 = None
        convert_element_type_3 = torch.ops.prims.convert_element_type.default(div_1, torch.float16);  div_1 = None
        expand_2 = torch.ops.aten.expand.default(convert_element_type_3, [16, 16, 256, 256]);  convert_element_type_3 = None
        view_3 = torch.ops.aten.view.default(expand_2, [256, 256, 256]);  expand_2 = None
        expand_3 = torch.ops.aten.expand.default(arg2_1, [16, 16, 256, 32]);  arg2_1 = None
        clone_2 = torch.ops.aten.clone.default(expand_3, memory_format = torch.contiguous_format);  expand_3 = None
        view_4 = torch.ops.aten.view.default(clone_2, [256, 256, 32]);  clone_2 = None
        bmm_1 = torch.ops.aten.bmm.default(view_3, view_4);  view_3 = view_4 = None
        view_5 = torch.ops.aten.view.default(bmm_1, [16, 16, 256, 32]);  bmm_1 = None
        return (view_5,)
        
def load_args(reader):
    buf0 = reader.storage(None, 12582912, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf0, (16, 16, 256, 32), (393216, 32, 1536, 1), dtype=torch.float16, storage_offset=512, is_leaf=True)  # arg0_1
    reader.tensor(buf0, (16, 16, 256, 32), (393216, 32, 1536, 1), dtype=torch.float16, is_leaf=True)  # arg1_1
    reader.tensor(buf0, (16, 16, 256, 32), (393216, 32, 1536, 1), dtype=torch.float16, storage_offset=1024, is_leaf=True)  # arg2_1
load_args._version = 0
mod = Repro()
if __name__ == '__main__':
    from torch._dynamo.repro.after_aot import run_repro
    with torch.no_grad():
        run_repro(mod, load_args, accuracy=False, command='run', save_dir=None, tracing_mode='real', check_str=None)
