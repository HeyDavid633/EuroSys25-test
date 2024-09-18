
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

    
    
    def forward(self, arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1):
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
        permute_default_33 = torch.ops.aten.permute.default(view_2, [0, 2, 1, 3]);  view_2 = None
        permute_default_34 = torch.ops.aten.permute.default(view_3, [0, 2, 1, 3]);  view_3 = None
        permute_default_35 = torch.ops.aten.permute.default(view_4, [0, 2, 1, 3]);  view_4 = None
        _scaled_dot_product_flash_attention_default_11 = torch.ops.aten._scaled_dot_product_flash_attention.default(permute_default_33, permute_default_34, permute_default_35, scale = 0.17677669529663687);  permute_default_33 = permute_default_34 = permute_default_35 = None
        getitem_95 = _scaled_dot_product_flash_attention_default_11[0];  _scaled_dot_product_flash_attention_default_11 = None
        permute_4 = torch.ops.aten.permute.default(getitem_95, [0, 2, 1, 3]);  getitem_95 = None
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
        view_18 = torch.ops.aten.view.default(convert_element_type_19, [4096, 512])
        mm_4 = torch.ops.aten.mm.default(view_18, arg13_1);  view_18 = arg13_1 = None
        view_19 = torch.ops.aten.view.default(mm_4, [16, 256, 1536]);  mm_4 = None
        add_11 = torch.ops.aten.add.Tensor(view_19, arg14_1);  view_19 = arg14_1 = None
        split_1 = torch.ops.aten.split.Tensor(add_11, 512, -1);  add_11 = None
        getitem_7 = split_1[0]
        getitem_8 = split_1[1]
        getitem_9 = split_1[2];  split_1 = None
        view_20 = torch.ops.aten.view.default(getitem_7, [16, 256, 16, 32]);  getitem_7 = None
        view_21 = torch.ops.aten.view.default(getitem_8, [16, 256, 16, 32]);  getitem_8 = None
        view_22 = torch.ops.aten.view.default(getitem_9, [16, 256, 16, 32]);  getitem_9 = None
        permute_default_30 = torch.ops.aten.permute.default(view_20, [0, 2, 1, 3]);  view_20 = None
        permute_default_31 = torch.ops.aten.permute.default(view_21, [0, 2, 1, 3]);  view_21 = None
        permute_default_32 = torch.ops.aten.permute.default(view_22, [0, 2, 1, 3]);  view_22 = None
        _scaled_dot_product_flash_attention_default_10 = torch.ops.aten._scaled_dot_product_flash_attention.default(permute_default_30, permute_default_31, permute_default_32, scale = 0.17677669529663687);  permute_default_30 = permute_default_31 = permute_default_32 = None
        getitem_94 = _scaled_dot_product_flash_attention_default_10[0];  _scaled_dot_product_flash_attention_default_10 = None
        permute_9 = torch.ops.aten.permute.default(getitem_94, [0, 2, 1, 3]);  getitem_94 = None
        clone_7 = torch.ops.aten.clone.default(permute_9, memory_format = torch.contiguous_format);  permute_9 = None
        view_29 = torch.ops.aten.view.default(clone_7, [16, 256, 512]);  clone_7 = None
        view_30 = torch.ops.aten.view.default(view_29, [4096, 512]);  view_29 = None
        mm_5 = torch.ops.aten.mm.default(view_30, arg15_1);  view_30 = arg15_1 = None
        view_31 = torch.ops.aten.view.default(mm_5, [16, 256, 512]);  mm_5 = None
        add_12 = torch.ops.aten.add.Tensor(view_31, arg16_1);  view_31 = arg16_1 = None
        add_13 = torch.ops.aten.add.Tensor(add_12, convert_element_type_19);  add_12 = None
        convert_element_type_30 = torch.ops.prims.convert_element_type.default(add_13, torch.float32)
        var_mean_2 = torch.ops.aten.var_mean.correction(convert_element_type_30, [2], correction = 0, keepdim = True);  convert_element_type_30 = None
        getitem_10 = var_mean_2[0]
        getitem_11 = var_mean_2[1];  var_mean_2 = None
        add_14 = torch.ops.aten.add.Tensor(getitem_10, 1e-05);  getitem_10 = None
        rsqrt_2 = torch.ops.aten.rsqrt.default(add_14);  add_14 = None
        sub_4 = torch.ops.aten.sub.Tensor(add_13, getitem_11);  add_13 = getitem_11 = None
        mul_7 = torch.ops.aten.mul.Tensor(sub_4, rsqrt_2);  sub_4 = rsqrt_2 = None
        mul_8 = torch.ops.aten.mul.Tensor(mul_7, arg17_1);  mul_7 = arg17_1 = None
        add_15 = torch.ops.aten.add.Tensor(mul_8, arg18_1);  mul_8 = arg18_1 = None
        convert_element_type_31 = torch.ops.prims.convert_element_type.default(add_15, torch.float16);  add_15 = None
        view_32 = torch.ops.aten.view.default(convert_element_type_31, [4096, 512])
        mm_6 = torch.ops.aten.mm.default(view_32, arg19_1);  view_32 = arg19_1 = None
        view_33 = torch.ops.aten.view.default(mm_6, [16, 256, 2048]);  mm_6 = None
        add_16 = torch.ops.aten.add.Tensor(view_33, arg20_1);  view_33 = arg20_1 = None
        convert_element_type_34 = torch.ops.prims.convert_element_type.default(add_16, torch.float32);  add_16 = None
        mul_9 = torch.ops.aten.mul.Tensor(convert_element_type_34, 0.5)
        mul_10 = torch.ops.aten.mul.Tensor(convert_element_type_34, 0.7071067811865476);  convert_element_type_34 = None
        erf_1 = torch.ops.aten.erf.default(mul_10);  mul_10 = None
        add_17 = torch.ops.aten.add.Tensor(erf_1, 1);  erf_1 = None
        mul_11 = torch.ops.aten.mul.Tensor(mul_9, add_17);  mul_9 = add_17 = None
        convert_element_type_35 = torch.ops.prims.convert_element_type.default(mul_11, torch.float16);  mul_11 = None
        view_34 = torch.ops.aten.view.default(convert_element_type_35, [4096, 2048]);  convert_element_type_35 = None
        mm_7 = torch.ops.aten.mm.default(view_34, arg21_1);  view_34 = arg21_1 = None
        view_35 = torch.ops.aten.view.default(mm_7, [16, 256, 512]);  mm_7 = None
        add_18 = torch.ops.aten.add.Tensor(view_35, arg22_1);  view_35 = arg22_1 = None
        add_19 = torch.ops.aten.add.Tensor(add_18, convert_element_type_31);  add_18 = convert_element_type_31 = None
        convert_element_type_38 = torch.ops.prims.convert_element_type.default(add_19, torch.float32)
        var_mean_3 = torch.ops.aten.var_mean.correction(convert_element_type_38, [2], correction = 0, keepdim = True);  convert_element_type_38 = None
        getitem_12 = var_mean_3[0]
        getitem_13 = var_mean_3[1];  var_mean_3 = None
        add_20 = torch.ops.aten.add.Tensor(getitem_12, 1e-05);  getitem_12 = None
        rsqrt_3 = torch.ops.aten.rsqrt.default(add_20);  add_20 = None
        sub_5 = torch.ops.aten.sub.Tensor(add_19, getitem_13);  add_19 = getitem_13 = None
        mul_12 = torch.ops.aten.mul.Tensor(sub_5, rsqrt_3);  sub_5 = rsqrt_3 = None
        mul_13 = torch.ops.aten.mul.Tensor(mul_12, arg23_1);  mul_12 = arg23_1 = None
        add_21 = torch.ops.aten.add.Tensor(mul_13, arg24_1);  mul_13 = arg24_1 = None
        convert_element_type_39 = torch.ops.prims.convert_element_type.default(add_21, torch.float16);  add_21 = None
        view_36 = torch.ops.aten.view.default(convert_element_type_39, [4096, 512])
        mm_8 = torch.ops.aten.mm.default(view_36, arg25_1);  view_36 = arg25_1 = None
        view_37 = torch.ops.aten.view.default(mm_8, [16, 256, 1536]);  mm_8 = None
        add_22 = torch.ops.aten.add.Tensor(view_37, arg26_1);  view_37 = arg26_1 = None
        split_2 = torch.ops.aten.split.Tensor(add_22, 512, -1);  add_22 = None
        getitem_14 = split_2[0]
        getitem_15 = split_2[1]
        getitem_16 = split_2[2];  split_2 = None
        view_38 = torch.ops.aten.view.default(getitem_14, [16, 256, 16, 32]);  getitem_14 = None
        view_39 = torch.ops.aten.view.default(getitem_15, [16, 256, 16, 32]);  getitem_15 = None
        view_40 = torch.ops.aten.view.default(getitem_16, [16, 256, 16, 32]);  getitem_16 = None
        permute_default_27 = torch.ops.aten.permute.default(view_38, [0, 2, 1, 3]);  view_38 = None
        permute_default_28 = torch.ops.aten.permute.default(view_39, [0, 2, 1, 3]);  view_39 = None
        permute_default_29 = torch.ops.aten.permute.default(view_40, [0, 2, 1, 3]);  view_40 = None
        _scaled_dot_product_flash_attention_default_9 = torch.ops.aten._scaled_dot_product_flash_attention.default(permute_default_27, permute_default_28, permute_default_29, scale = 0.17677669529663687);  permute_default_27 = permute_default_28 = permute_default_29 = None
        getitem_93 = _scaled_dot_product_flash_attention_default_9[0];  _scaled_dot_product_flash_attention_default_9 = None
        permute_14 = torch.ops.aten.permute.default(getitem_93, [0, 2, 1, 3]);  getitem_93 = None
        clone_11 = torch.ops.aten.clone.default(permute_14, memory_format = torch.contiguous_format);  permute_14 = None
        view_47 = torch.ops.aten.view.default(clone_11, [16, 256, 512]);  clone_11 = None
        view_48 = torch.ops.aten.view.default(view_47, [4096, 512]);  view_47 = None
        mm_9 = torch.ops.aten.mm.default(view_48, arg27_1);  view_48 = arg27_1 = None
        view_49 = torch.ops.aten.view.default(mm_9, [16, 256, 512]);  mm_9 = None
        add_23 = torch.ops.aten.add.Tensor(view_49, arg28_1);  view_49 = arg28_1 = None
        add_24 = torch.ops.aten.add.Tensor(add_23, convert_element_type_39);  add_23 = None
        convert_element_type_50 = torch.ops.prims.convert_element_type.default(add_24, torch.float32)
        var_mean_4 = torch.ops.aten.var_mean.correction(convert_element_type_50, [2], correction = 0, keepdim = True);  convert_element_type_50 = None
        getitem_17 = var_mean_4[0]
        getitem_18 = var_mean_4[1];  var_mean_4 = None
        add_25 = torch.ops.aten.add.Tensor(getitem_17, 1e-05);  getitem_17 = None
        rsqrt_4 = torch.ops.aten.rsqrt.default(add_25);  add_25 = None
        sub_7 = torch.ops.aten.sub.Tensor(add_24, getitem_18);  add_24 = getitem_18 = None
        mul_14 = torch.ops.aten.mul.Tensor(sub_7, rsqrt_4);  sub_7 = rsqrt_4 = None
        mul_15 = torch.ops.aten.mul.Tensor(mul_14, arg29_1);  mul_14 = arg29_1 = None
        add_26 = torch.ops.aten.add.Tensor(mul_15, arg30_1);  mul_15 = arg30_1 = None
        convert_element_type_51 = torch.ops.prims.convert_element_type.default(add_26, torch.float16);  add_26 = None
        view_50 = torch.ops.aten.view.default(convert_element_type_51, [4096, 512])
        mm_10 = torch.ops.aten.mm.default(view_50, arg31_1);  view_50 = arg31_1 = None
        view_51 = torch.ops.aten.view.default(mm_10, [16, 256, 2048]);  mm_10 = None
        add_27 = torch.ops.aten.add.Tensor(view_51, arg32_1);  view_51 = arg32_1 = None
        convert_element_type_54 = torch.ops.prims.convert_element_type.default(add_27, torch.float32);  add_27 = None
        mul_16 = torch.ops.aten.mul.Tensor(convert_element_type_54, 0.5)
        mul_17 = torch.ops.aten.mul.Tensor(convert_element_type_54, 0.7071067811865476);  convert_element_type_54 = None
        erf_2 = torch.ops.aten.erf.default(mul_17);  mul_17 = None
        add_28 = torch.ops.aten.add.Tensor(erf_2, 1);  erf_2 = None
        mul_18 = torch.ops.aten.mul.Tensor(mul_16, add_28);  mul_16 = add_28 = None
        convert_element_type_55 = torch.ops.prims.convert_element_type.default(mul_18, torch.float16);  mul_18 = None
        view_52 = torch.ops.aten.view.default(convert_element_type_55, [4096, 2048]);  convert_element_type_55 = None
        mm_11 = torch.ops.aten.mm.default(view_52, arg33_1);  view_52 = arg33_1 = None
        view_53 = torch.ops.aten.view.default(mm_11, [16, 256, 512]);  mm_11 = None
        add_29 = torch.ops.aten.add.Tensor(view_53, arg34_1);  view_53 = arg34_1 = None
        add_30 = torch.ops.aten.add.Tensor(add_29, convert_element_type_51);  add_29 = convert_element_type_51 = None
        convert_element_type_58 = torch.ops.prims.convert_element_type.default(add_30, torch.float32)
        var_mean_5 = torch.ops.aten.var_mean.correction(convert_element_type_58, [2], correction = 0, keepdim = True);  convert_element_type_58 = None
        getitem_19 = var_mean_5[0]
        getitem_20 = var_mean_5[1];  var_mean_5 = None
        add_31 = torch.ops.aten.add.Tensor(getitem_19, 1e-05);  getitem_19 = None
        rsqrt_5 = torch.ops.aten.rsqrt.default(add_31);  add_31 = None
        sub_8 = torch.ops.aten.sub.Tensor(add_30, getitem_20);  add_30 = getitem_20 = None
        mul_19 = torch.ops.aten.mul.Tensor(sub_8, rsqrt_5);  sub_8 = rsqrt_5 = None
        mul_20 = torch.ops.aten.mul.Tensor(mul_19, arg35_1);  mul_19 = arg35_1 = None
        add_32 = torch.ops.aten.add.Tensor(mul_20, arg36_1);  mul_20 = arg36_1 = None
        convert_element_type_59 = torch.ops.prims.convert_element_type.default(add_32, torch.float16);  add_32 = None
        view_54 = torch.ops.aten.view.default(convert_element_type_59, [4096, 512])
        mm_12 = torch.ops.aten.mm.default(view_54, arg37_1);  view_54 = arg37_1 = None
        view_55 = torch.ops.aten.view.default(mm_12, [16, 256, 1536]);  mm_12 = None
        add_33 = torch.ops.aten.add.Tensor(view_55, arg38_1);  view_55 = arg38_1 = None
        split_3 = torch.ops.aten.split.Tensor(add_33, 512, -1);  add_33 = None
        getitem_21 = split_3[0]
        getitem_22 = split_3[1]
        getitem_23 = split_3[2];  split_3 = None
        view_56 = torch.ops.aten.view.default(getitem_21, [16, 256, 16, 32]);  getitem_21 = None
        view_57 = torch.ops.aten.view.default(getitem_22, [16, 256, 16, 32]);  getitem_22 = None
        view_58 = torch.ops.aten.view.default(getitem_23, [16, 256, 16, 32]);  getitem_23 = None
        permute_default_24 = torch.ops.aten.permute.default(view_56, [0, 2, 1, 3]);  view_56 = None
        permute_default_25 = torch.ops.aten.permute.default(view_57, [0, 2, 1, 3]);  view_57 = None
        permute_default_26 = torch.ops.aten.permute.default(view_58, [0, 2, 1, 3]);  view_58 = None
        _scaled_dot_product_flash_attention_default_8 = torch.ops.aten._scaled_dot_product_flash_attention.default(permute_default_24, permute_default_25, permute_default_26, scale = 0.17677669529663687);  permute_default_24 = permute_default_25 = permute_default_26 = None
        getitem_92 = _scaled_dot_product_flash_attention_default_8[0];  _scaled_dot_product_flash_attention_default_8 = None
        permute_19 = torch.ops.aten.permute.default(getitem_92, [0, 2, 1, 3]);  getitem_92 = None
        clone_15 = torch.ops.aten.clone.default(permute_19, memory_format = torch.contiguous_format);  permute_19 = None
        view_65 = torch.ops.aten.view.default(clone_15, [16, 256, 512]);  clone_15 = None
        view_66 = torch.ops.aten.view.default(view_65, [4096, 512]);  view_65 = None
        mm_13 = torch.ops.aten.mm.default(view_66, arg39_1);  view_66 = arg39_1 = None
        view_67 = torch.ops.aten.view.default(mm_13, [16, 256, 512]);  mm_13 = None
        add_34 = torch.ops.aten.add.Tensor(view_67, arg40_1);  view_67 = arg40_1 = None
        add_35 = torch.ops.aten.add.Tensor(add_34, convert_element_type_59);  add_34 = None
        convert_element_type_70 = torch.ops.prims.convert_element_type.default(add_35, torch.float32)
        var_mean_6 = torch.ops.aten.var_mean.correction(convert_element_type_70, [2], correction = 0, keepdim = True);  convert_element_type_70 = None
        getitem_24 = var_mean_6[0]
        getitem_25 = var_mean_6[1];  var_mean_6 = None
        add_36 = torch.ops.aten.add.Tensor(getitem_24, 1e-05);  getitem_24 = None
        rsqrt_6 = torch.ops.aten.rsqrt.default(add_36);  add_36 = None
        sub_10 = torch.ops.aten.sub.Tensor(add_35, getitem_25);  add_35 = getitem_25 = None
        mul_21 = torch.ops.aten.mul.Tensor(sub_10, rsqrt_6);  sub_10 = rsqrt_6 = None
        mul_22 = torch.ops.aten.mul.Tensor(mul_21, arg41_1);  mul_21 = arg41_1 = None
        add_37 = torch.ops.aten.add.Tensor(mul_22, arg42_1);  mul_22 = arg42_1 = None
        convert_element_type_71 = torch.ops.prims.convert_element_type.default(add_37, torch.float16);  add_37 = None
        view_68 = torch.ops.aten.view.default(convert_element_type_71, [4096, 512])
        mm_14 = torch.ops.aten.mm.default(view_68, arg43_1);  view_68 = arg43_1 = None
        view_69 = torch.ops.aten.view.default(mm_14, [16, 256, 2048]);  mm_14 = None
        add_38 = torch.ops.aten.add.Tensor(view_69, arg44_1);  view_69 = arg44_1 = None
        convert_element_type_74 = torch.ops.prims.convert_element_type.default(add_38, torch.float32);  add_38 = None
        mul_23 = torch.ops.aten.mul.Tensor(convert_element_type_74, 0.5)
        mul_24 = torch.ops.aten.mul.Tensor(convert_element_type_74, 0.7071067811865476);  convert_element_type_74 = None
        erf_3 = torch.ops.aten.erf.default(mul_24);  mul_24 = None
        add_39 = torch.ops.aten.add.Tensor(erf_3, 1);  erf_3 = None
        mul_25 = torch.ops.aten.mul.Tensor(mul_23, add_39);  mul_23 = add_39 = None
        convert_element_type_75 = torch.ops.prims.convert_element_type.default(mul_25, torch.float16);  mul_25 = None
        view_70 = torch.ops.aten.view.default(convert_element_type_75, [4096, 2048]);  convert_element_type_75 = None
        mm_15 = torch.ops.aten.mm.default(view_70, arg45_1);  view_70 = arg45_1 = None
        view_71 = torch.ops.aten.view.default(mm_15, [16, 256, 512]);  mm_15 = None
        add_40 = torch.ops.aten.add.Tensor(view_71, arg46_1);  view_71 = arg46_1 = None
        add_41 = torch.ops.aten.add.Tensor(add_40, convert_element_type_71);  add_40 = convert_element_type_71 = None
        convert_element_type_78 = torch.ops.prims.convert_element_type.default(add_41, torch.float32)
        var_mean_7 = torch.ops.aten.var_mean.correction(convert_element_type_78, [2], correction = 0, keepdim = True);  convert_element_type_78 = None
        getitem_26 = var_mean_7[0]
        getitem_27 = var_mean_7[1];  var_mean_7 = None
        add_42 = torch.ops.aten.add.Tensor(getitem_26, 1e-05);  getitem_26 = None
        rsqrt_7 = torch.ops.aten.rsqrt.default(add_42);  add_42 = None
        sub_11 = torch.ops.aten.sub.Tensor(add_41, getitem_27);  add_41 = getitem_27 = None
        mul_26 = torch.ops.aten.mul.Tensor(sub_11, rsqrt_7);  sub_11 = rsqrt_7 = None
        mul_27 = torch.ops.aten.mul.Tensor(mul_26, arg47_1);  mul_26 = arg47_1 = None
        add_43 = torch.ops.aten.add.Tensor(mul_27, arg48_1);  mul_27 = arg48_1 = None
        convert_element_type_79 = torch.ops.prims.convert_element_type.default(add_43, torch.float16);  add_43 = None
        view_72 = torch.ops.aten.view.default(convert_element_type_79, [4096, 512])
        mm_16 = torch.ops.aten.mm.default(view_72, arg49_1);  view_72 = arg49_1 = None
        view_73 = torch.ops.aten.view.default(mm_16, [16, 256, 1536]);  mm_16 = None
        add_44 = torch.ops.aten.add.Tensor(view_73, arg50_1);  view_73 = arg50_1 = None
        split_4 = torch.ops.aten.split.Tensor(add_44, 512, -1);  add_44 = None
        getitem_28 = split_4[0]
        getitem_29 = split_4[1]
        getitem_30 = split_4[2];  split_4 = None
        view_74 = torch.ops.aten.view.default(getitem_28, [16, 256, 16, 32]);  getitem_28 = None
        view_75 = torch.ops.aten.view.default(getitem_29, [16, 256, 16, 32]);  getitem_29 = None
        view_76 = torch.ops.aten.view.default(getitem_30, [16, 256, 16, 32]);  getitem_30 = None
        permute_default_21 = torch.ops.aten.permute.default(view_74, [0, 2, 1, 3]);  view_74 = None
        permute_default_22 = torch.ops.aten.permute.default(view_75, [0, 2, 1, 3]);  view_75 = None
        permute_default_23 = torch.ops.aten.permute.default(view_76, [0, 2, 1, 3]);  view_76 = None
        _scaled_dot_product_flash_attention_default_7 = torch.ops.aten._scaled_dot_product_flash_attention.default(permute_default_21, permute_default_22, permute_default_23, scale = 0.17677669529663687);  permute_default_21 = permute_default_22 = permute_default_23 = None
        getitem_91 = _scaled_dot_product_flash_attention_default_7[0];  _scaled_dot_product_flash_attention_default_7 = None
        permute_24 = torch.ops.aten.permute.default(getitem_91, [0, 2, 1, 3]);  getitem_91 = None
        clone_19 = torch.ops.aten.clone.default(permute_24, memory_format = torch.contiguous_format);  permute_24 = None
        view_83 = torch.ops.aten.view.default(clone_19, [16, 256, 512]);  clone_19 = None
        view_84 = torch.ops.aten.view.default(view_83, [4096, 512]);  view_83 = None
        mm_17 = torch.ops.aten.mm.default(view_84, arg51_1);  view_84 = arg51_1 = None
        view_85 = torch.ops.aten.view.default(mm_17, [16, 256, 512]);  mm_17 = None
        add_45 = torch.ops.aten.add.Tensor(view_85, arg52_1);  view_85 = arg52_1 = None
        add_46 = torch.ops.aten.add.Tensor(add_45, convert_element_type_79);  add_45 = None
        convert_element_type_90 = torch.ops.prims.convert_element_type.default(add_46, torch.float32)
        var_mean_8 = torch.ops.aten.var_mean.correction(convert_element_type_90, [2], correction = 0, keepdim = True);  convert_element_type_90 = None
        getitem_31 = var_mean_8[0]
        getitem_32 = var_mean_8[1];  var_mean_8 = None
        add_47 = torch.ops.aten.add.Tensor(getitem_31, 1e-05);  getitem_31 = None
        rsqrt_8 = torch.ops.aten.rsqrt.default(add_47);  add_47 = None
        sub_13 = torch.ops.aten.sub.Tensor(add_46, getitem_32);  add_46 = getitem_32 = None
        mul_28 = torch.ops.aten.mul.Tensor(sub_13, rsqrt_8);  sub_13 = rsqrt_8 = None
        mul_29 = torch.ops.aten.mul.Tensor(mul_28, arg53_1);  mul_28 = arg53_1 = None
        add_48 = torch.ops.aten.add.Tensor(mul_29, arg54_1);  mul_29 = arg54_1 = None
        convert_element_type_91 = torch.ops.prims.convert_element_type.default(add_48, torch.float16);  add_48 = None
        view_86 = torch.ops.aten.view.default(convert_element_type_91, [4096, 512])
        mm_18 = torch.ops.aten.mm.default(view_86, arg55_1);  view_86 = arg55_1 = None
        view_87 = torch.ops.aten.view.default(mm_18, [16, 256, 2048]);  mm_18 = None
        add_49 = torch.ops.aten.add.Tensor(view_87, arg56_1);  view_87 = arg56_1 = None
        convert_element_type_94 = torch.ops.prims.convert_element_type.default(add_49, torch.float32);  add_49 = None
        mul_30 = torch.ops.aten.mul.Tensor(convert_element_type_94, 0.5)
        mul_31 = torch.ops.aten.mul.Tensor(convert_element_type_94, 0.7071067811865476);  convert_element_type_94 = None
        erf_4 = torch.ops.aten.erf.default(mul_31);  mul_31 = None
        add_50 = torch.ops.aten.add.Tensor(erf_4, 1);  erf_4 = None
        mul_32 = torch.ops.aten.mul.Tensor(mul_30, add_50);  mul_30 = add_50 = None
        convert_element_type_95 = torch.ops.prims.convert_element_type.default(mul_32, torch.float16);  mul_32 = None
        view_88 = torch.ops.aten.view.default(convert_element_type_95, [4096, 2048]);  convert_element_type_95 = None
        mm_19 = torch.ops.aten.mm.default(view_88, arg57_1);  view_88 = arg57_1 = None
        view_89 = torch.ops.aten.view.default(mm_19, [16, 256, 512]);  mm_19 = None
        add_51 = torch.ops.aten.add.Tensor(view_89, arg58_1);  view_89 = arg58_1 = None
        add_52 = torch.ops.aten.add.Tensor(add_51, convert_element_type_91);  add_51 = convert_element_type_91 = None
        convert_element_type_98 = torch.ops.prims.convert_element_type.default(add_52, torch.float32)
        var_mean_9 = torch.ops.aten.var_mean.correction(convert_element_type_98, [2], correction = 0, keepdim = True);  convert_element_type_98 = None
        getitem_33 = var_mean_9[0]
        getitem_34 = var_mean_9[1];  var_mean_9 = None
        add_53 = torch.ops.aten.add.Tensor(getitem_33, 1e-05);  getitem_33 = None
        rsqrt_9 = torch.ops.aten.rsqrt.default(add_53);  add_53 = None
        sub_14 = torch.ops.aten.sub.Tensor(add_52, getitem_34);  add_52 = getitem_34 = None
        mul_33 = torch.ops.aten.mul.Tensor(sub_14, rsqrt_9);  sub_14 = rsqrt_9 = None
        mul_34 = torch.ops.aten.mul.Tensor(mul_33, arg59_1);  mul_33 = arg59_1 = None
        add_54 = torch.ops.aten.add.Tensor(mul_34, arg60_1);  mul_34 = arg60_1 = None
        convert_element_type_99 = torch.ops.prims.convert_element_type.default(add_54, torch.float16);  add_54 = None
        view_90 = torch.ops.aten.view.default(convert_element_type_99, [4096, 512])
        mm_20 = torch.ops.aten.mm.default(view_90, arg61_1);  view_90 = arg61_1 = None
        view_91 = torch.ops.aten.view.default(mm_20, [16, 256, 1536]);  mm_20 = None
        add_55 = torch.ops.aten.add.Tensor(view_91, arg62_1);  view_91 = arg62_1 = None
        split_5 = torch.ops.aten.split.Tensor(add_55, 512, -1);  add_55 = None
        getitem_35 = split_5[0]
        getitem_36 = split_5[1]
        getitem_37 = split_5[2];  split_5 = None
        view_92 = torch.ops.aten.view.default(getitem_35, [16, 256, 16, 32]);  getitem_35 = None
        view_93 = torch.ops.aten.view.default(getitem_36, [16, 256, 16, 32]);  getitem_36 = None
        view_94 = torch.ops.aten.view.default(getitem_37, [16, 256, 16, 32]);  getitem_37 = None
        permute_default_18 = torch.ops.aten.permute.default(view_92, [0, 2, 1, 3]);  view_92 = None
        permute_default_19 = torch.ops.aten.permute.default(view_93, [0, 2, 1, 3]);  view_93 = None
        permute_default_20 = torch.ops.aten.permute.default(view_94, [0, 2, 1, 3]);  view_94 = None
        _scaled_dot_product_flash_attention_default_6 = torch.ops.aten._scaled_dot_product_flash_attention.default(permute_default_18, permute_default_19, permute_default_20, scale = 0.17677669529663687);  permute_default_18 = permute_default_19 = permute_default_20 = None
        getitem_90 = _scaled_dot_product_flash_attention_default_6[0];  _scaled_dot_product_flash_attention_default_6 = None
        permute_29 = torch.ops.aten.permute.default(getitem_90, [0, 2, 1, 3]);  getitem_90 = None
        clone_23 = torch.ops.aten.clone.default(permute_29, memory_format = torch.contiguous_format);  permute_29 = None
        view_101 = torch.ops.aten.view.default(clone_23, [16, 256, 512]);  clone_23 = None
        view_102 = torch.ops.aten.view.default(view_101, [4096, 512]);  view_101 = None
        mm_21 = torch.ops.aten.mm.default(view_102, arg63_1);  view_102 = arg63_1 = None
        view_103 = torch.ops.aten.view.default(mm_21, [16, 256, 512]);  mm_21 = None
        add_56 = torch.ops.aten.add.Tensor(view_103, arg64_1);  view_103 = arg64_1 = None
        add_57 = torch.ops.aten.add.Tensor(add_56, convert_element_type_99);  add_56 = None
        convert_element_type_110 = torch.ops.prims.convert_element_type.default(add_57, torch.float32)
        var_mean_10 = torch.ops.aten.var_mean.correction(convert_element_type_110, [2], correction = 0, keepdim = True);  convert_element_type_110 = None
        getitem_38 = var_mean_10[0]
        getitem_39 = var_mean_10[1];  var_mean_10 = None
        add_58 = torch.ops.aten.add.Tensor(getitem_38, 1e-05);  getitem_38 = None
        rsqrt_10 = torch.ops.aten.rsqrt.default(add_58);  add_58 = None
        sub_16 = torch.ops.aten.sub.Tensor(add_57, getitem_39);  add_57 = getitem_39 = None
        mul_35 = torch.ops.aten.mul.Tensor(sub_16, rsqrt_10);  sub_16 = rsqrt_10 = None
        mul_36 = torch.ops.aten.mul.Tensor(mul_35, arg65_1);  mul_35 = arg65_1 = None
        add_59 = torch.ops.aten.add.Tensor(mul_36, arg66_1);  mul_36 = arg66_1 = None
        convert_element_type_111 = torch.ops.prims.convert_element_type.default(add_59, torch.float16);  add_59 = None
        view_104 = torch.ops.aten.view.default(convert_element_type_111, [4096, 512])
        mm_22 = torch.ops.aten.mm.default(view_104, arg67_1);  view_104 = arg67_1 = None
        view_105 = torch.ops.aten.view.default(mm_22, [16, 256, 2048]);  mm_22 = None
        add_60 = torch.ops.aten.add.Tensor(view_105, arg68_1);  view_105 = arg68_1 = None
        convert_element_type_114 = torch.ops.prims.convert_element_type.default(add_60, torch.float32);  add_60 = None
        mul_37 = torch.ops.aten.mul.Tensor(convert_element_type_114, 0.5)
        mul_38 = torch.ops.aten.mul.Tensor(convert_element_type_114, 0.7071067811865476);  convert_element_type_114 = None
        erf_5 = torch.ops.aten.erf.default(mul_38);  mul_38 = None
        add_61 = torch.ops.aten.add.Tensor(erf_5, 1);  erf_5 = None
        mul_39 = torch.ops.aten.mul.Tensor(mul_37, add_61);  mul_37 = add_61 = None
        convert_element_type_115 = torch.ops.prims.convert_element_type.default(mul_39, torch.float16);  mul_39 = None
        view_106 = torch.ops.aten.view.default(convert_element_type_115, [4096, 2048]);  convert_element_type_115 = None
        mm_23 = torch.ops.aten.mm.default(view_106, arg69_1);  view_106 = arg69_1 = None
        view_107 = torch.ops.aten.view.default(mm_23, [16, 256, 512]);  mm_23 = None
        add_62 = torch.ops.aten.add.Tensor(view_107, arg70_1);  view_107 = arg70_1 = None
        add_63 = torch.ops.aten.add.Tensor(add_62, convert_element_type_111);  add_62 = convert_element_type_111 = None
        convert_element_type_118 = torch.ops.prims.convert_element_type.default(add_63, torch.float32)
        var_mean_11 = torch.ops.aten.var_mean.correction(convert_element_type_118, [2], correction = 0, keepdim = True);  convert_element_type_118 = None
        getitem_40 = var_mean_11[0]
        getitem_41 = var_mean_11[1];  var_mean_11 = None
        add_64 = torch.ops.aten.add.Tensor(getitem_40, 1e-05);  getitem_40 = None
        rsqrt_11 = torch.ops.aten.rsqrt.default(add_64);  add_64 = None
        sub_17 = torch.ops.aten.sub.Tensor(add_63, getitem_41);  add_63 = getitem_41 = None
        mul_40 = torch.ops.aten.mul.Tensor(sub_17, rsqrt_11);  sub_17 = rsqrt_11 = None
        mul_41 = torch.ops.aten.mul.Tensor(mul_40, arg71_1);  mul_40 = arg71_1 = None
        add_65 = torch.ops.aten.add.Tensor(mul_41, arg72_1);  mul_41 = arg72_1 = None
        convert_element_type_119 = torch.ops.prims.convert_element_type.default(add_65, torch.float16);  add_65 = None
        view_108 = torch.ops.aten.view.default(convert_element_type_119, [4096, 512])
        mm_24 = torch.ops.aten.mm.default(view_108, arg73_1);  view_108 = arg73_1 = None
        view_109 = torch.ops.aten.view.default(mm_24, [16, 256, 1536]);  mm_24 = None
        add_66 = torch.ops.aten.add.Tensor(view_109, arg74_1);  view_109 = arg74_1 = None
        split_6 = torch.ops.aten.split.Tensor(add_66, 512, -1);  add_66 = None
        getitem_42 = split_6[0]
        getitem_43 = split_6[1]
        getitem_44 = split_6[2];  split_6 = None
        view_110 = torch.ops.aten.view.default(getitem_42, [16, 256, 16, 32]);  getitem_42 = None
        view_111 = torch.ops.aten.view.default(getitem_43, [16, 256, 16, 32]);  getitem_43 = None
        view_112 = torch.ops.aten.view.default(getitem_44, [16, 256, 16, 32]);  getitem_44 = None
        permute_default_15 = torch.ops.aten.permute.default(view_110, [0, 2, 1, 3]);  view_110 = None
        permute_default_16 = torch.ops.aten.permute.default(view_111, [0, 2, 1, 3]);  view_111 = None
        permute_default_17 = torch.ops.aten.permute.default(view_112, [0, 2, 1, 3]);  view_112 = None
        _scaled_dot_product_flash_attention_default_5 = torch.ops.aten._scaled_dot_product_flash_attention.default(permute_default_15, permute_default_16, permute_default_17, scale = 0.17677669529663687);  permute_default_15 = permute_default_16 = permute_default_17 = None
        getitem_89 = _scaled_dot_product_flash_attention_default_5[0];  _scaled_dot_product_flash_attention_default_5 = None
        permute_34 = torch.ops.aten.permute.default(getitem_89, [0, 2, 1, 3]);  getitem_89 = None
        clone_27 = torch.ops.aten.clone.default(permute_34, memory_format = torch.contiguous_format);  permute_34 = None
        view_119 = torch.ops.aten.view.default(clone_27, [16, 256, 512]);  clone_27 = None
        view_120 = torch.ops.aten.view.default(view_119, [4096, 512]);  view_119 = None
        mm_25 = torch.ops.aten.mm.default(view_120, arg75_1);  view_120 = arg75_1 = None
        view_121 = torch.ops.aten.view.default(mm_25, [16, 256, 512]);  mm_25 = None
        add_67 = torch.ops.aten.add.Tensor(view_121, arg76_1);  view_121 = arg76_1 = None
        add_68 = torch.ops.aten.add.Tensor(add_67, convert_element_type_119);  add_67 = None
        convert_element_type_130 = torch.ops.prims.convert_element_type.default(add_68, torch.float32)
        var_mean_12 = torch.ops.aten.var_mean.correction(convert_element_type_130, [2], correction = 0, keepdim = True);  convert_element_type_130 = None
        getitem_45 = var_mean_12[0]
        getitem_46 = var_mean_12[1];  var_mean_12 = None
        add_69 = torch.ops.aten.add.Tensor(getitem_45, 1e-05);  getitem_45 = None
        rsqrt_12 = torch.ops.aten.rsqrt.default(add_69);  add_69 = None
        sub_19 = torch.ops.aten.sub.Tensor(add_68, getitem_46);  add_68 = getitem_46 = None
        mul_42 = torch.ops.aten.mul.Tensor(sub_19, rsqrt_12);  sub_19 = rsqrt_12 = None
        mul_43 = torch.ops.aten.mul.Tensor(mul_42, arg77_1);  mul_42 = arg77_1 = None
        add_70 = torch.ops.aten.add.Tensor(mul_43, arg78_1);  mul_43 = arg78_1 = None
        convert_element_type_131 = torch.ops.prims.convert_element_type.default(add_70, torch.float16);  add_70 = None
        view_122 = torch.ops.aten.view.default(convert_element_type_131, [4096, 512])
        mm_26 = torch.ops.aten.mm.default(view_122, arg79_1);  view_122 = arg79_1 = None
        view_123 = torch.ops.aten.view.default(mm_26, [16, 256, 2048]);  mm_26 = None
        add_71 = torch.ops.aten.add.Tensor(view_123, arg80_1);  view_123 = arg80_1 = None
        convert_element_type_134 = torch.ops.prims.convert_element_type.default(add_71, torch.float32);  add_71 = None
        mul_44 = torch.ops.aten.mul.Tensor(convert_element_type_134, 0.5)
        mul_45 = torch.ops.aten.mul.Tensor(convert_element_type_134, 0.7071067811865476);  convert_element_type_134 = None
        erf_6 = torch.ops.aten.erf.default(mul_45);  mul_45 = None
        add_72 = torch.ops.aten.add.Tensor(erf_6, 1);  erf_6 = None
        mul_46 = torch.ops.aten.mul.Tensor(mul_44, add_72);  mul_44 = add_72 = None
        convert_element_type_135 = torch.ops.prims.convert_element_type.default(mul_46, torch.float16);  mul_46 = None
        view_124 = torch.ops.aten.view.default(convert_element_type_135, [4096, 2048]);  convert_element_type_135 = None
        mm_27 = torch.ops.aten.mm.default(view_124, arg81_1);  view_124 = arg81_1 = None
        view_125 = torch.ops.aten.view.default(mm_27, [16, 256, 512]);  mm_27 = None
        add_73 = torch.ops.aten.add.Tensor(view_125, arg82_1);  view_125 = arg82_1 = None
        add_74 = torch.ops.aten.add.Tensor(add_73, convert_element_type_131);  add_73 = convert_element_type_131 = None
        convert_element_type_138 = torch.ops.prims.convert_element_type.default(add_74, torch.float32)
        var_mean_13 = torch.ops.aten.var_mean.correction(convert_element_type_138, [2], correction = 0, keepdim = True);  convert_element_type_138 = None
        getitem_47 = var_mean_13[0]
        getitem_48 = var_mean_13[1];  var_mean_13 = None
        add_75 = torch.ops.aten.add.Tensor(getitem_47, 1e-05);  getitem_47 = None
        rsqrt_13 = torch.ops.aten.rsqrt.default(add_75);  add_75 = None
        sub_20 = torch.ops.aten.sub.Tensor(add_74, getitem_48);  add_74 = getitem_48 = None
        mul_47 = torch.ops.aten.mul.Tensor(sub_20, rsqrt_13);  sub_20 = rsqrt_13 = None
        mul_48 = torch.ops.aten.mul.Tensor(mul_47, arg83_1);  mul_47 = arg83_1 = None
        add_76 = torch.ops.aten.add.Tensor(mul_48, arg84_1);  mul_48 = arg84_1 = None
        convert_element_type_139 = torch.ops.prims.convert_element_type.default(add_76, torch.float16);  add_76 = None
        view_126 = torch.ops.aten.view.default(convert_element_type_139, [4096, 512])
        mm_28 = torch.ops.aten.mm.default(view_126, arg85_1);  view_126 = arg85_1 = None
        view_127 = torch.ops.aten.view.default(mm_28, [16, 256, 1536]);  mm_28 = None
        add_77 = torch.ops.aten.add.Tensor(view_127, arg86_1);  view_127 = arg86_1 = None
        split_7 = torch.ops.aten.split.Tensor(add_77, 512, -1);  add_77 = None
        getitem_49 = split_7[0]
        getitem_50 = split_7[1]
        getitem_51 = split_7[2];  split_7 = None
        view_128 = torch.ops.aten.view.default(getitem_49, [16, 256, 16, 32]);  getitem_49 = None
        view_129 = torch.ops.aten.view.default(getitem_50, [16, 256, 16, 32]);  getitem_50 = None
        view_130 = torch.ops.aten.view.default(getitem_51, [16, 256, 16, 32]);  getitem_51 = None
        permute_default_12 = torch.ops.aten.permute.default(view_128, [0, 2, 1, 3]);  view_128 = None
        permute_default_13 = torch.ops.aten.permute.default(view_129, [0, 2, 1, 3]);  view_129 = None
        permute_default_14 = torch.ops.aten.permute.default(view_130, [0, 2, 1, 3]);  view_130 = None
        _scaled_dot_product_flash_attention_default_4 = torch.ops.aten._scaled_dot_product_flash_attention.default(permute_default_12, permute_default_13, permute_default_14, scale = 0.17677669529663687);  permute_default_12 = permute_default_13 = permute_default_14 = None
        getitem_88 = _scaled_dot_product_flash_attention_default_4[0];  _scaled_dot_product_flash_attention_default_4 = None
        permute_39 = torch.ops.aten.permute.default(getitem_88, [0, 2, 1, 3]);  getitem_88 = None
        clone_31 = torch.ops.aten.clone.default(permute_39, memory_format = torch.contiguous_format);  permute_39 = None
        view_137 = torch.ops.aten.view.default(clone_31, [16, 256, 512]);  clone_31 = None
        view_138 = torch.ops.aten.view.default(view_137, [4096, 512]);  view_137 = None
        mm_29 = torch.ops.aten.mm.default(view_138, arg87_1);  view_138 = arg87_1 = None
        view_139 = torch.ops.aten.view.default(mm_29, [16, 256, 512]);  mm_29 = None
        add_78 = torch.ops.aten.add.Tensor(view_139, arg88_1);  view_139 = arg88_1 = None
        add_79 = torch.ops.aten.add.Tensor(add_78, convert_element_type_139);  add_78 = None
        convert_element_type_150 = torch.ops.prims.convert_element_type.default(add_79, torch.float32)
        var_mean_14 = torch.ops.aten.var_mean.correction(convert_element_type_150, [2], correction = 0, keepdim = True);  convert_element_type_150 = None
        getitem_52 = var_mean_14[0]
        getitem_53 = var_mean_14[1];  var_mean_14 = None
        add_80 = torch.ops.aten.add.Tensor(getitem_52, 1e-05);  getitem_52 = None
        rsqrt_14 = torch.ops.aten.rsqrt.default(add_80);  add_80 = None
        sub_22 = torch.ops.aten.sub.Tensor(add_79, getitem_53);  add_79 = getitem_53 = None
        mul_49 = torch.ops.aten.mul.Tensor(sub_22, rsqrt_14);  sub_22 = rsqrt_14 = None
        mul_50 = torch.ops.aten.mul.Tensor(mul_49, arg89_1);  mul_49 = arg89_1 = None
        add_81 = torch.ops.aten.add.Tensor(mul_50, arg90_1);  mul_50 = arg90_1 = None
        convert_element_type_151 = torch.ops.prims.convert_element_type.default(add_81, torch.float16);  add_81 = None
        view_140 = torch.ops.aten.view.default(convert_element_type_151, [4096, 512])
        mm_30 = torch.ops.aten.mm.default(view_140, arg91_1);  view_140 = arg91_1 = None
        view_141 = torch.ops.aten.view.default(mm_30, [16, 256, 2048]);  mm_30 = None
        add_82 = torch.ops.aten.add.Tensor(view_141, arg92_1);  view_141 = arg92_1 = None
        convert_element_type_154 = torch.ops.prims.convert_element_type.default(add_82, torch.float32);  add_82 = None
        mul_51 = torch.ops.aten.mul.Tensor(convert_element_type_154, 0.5)
        mul_52 = torch.ops.aten.mul.Tensor(convert_element_type_154, 0.7071067811865476);  convert_element_type_154 = None
        erf_7 = torch.ops.aten.erf.default(mul_52);  mul_52 = None
        add_83 = torch.ops.aten.add.Tensor(erf_7, 1);  erf_7 = None
        mul_53 = torch.ops.aten.mul.Tensor(mul_51, add_83);  mul_51 = add_83 = None
        convert_element_type_155 = torch.ops.prims.convert_element_type.default(mul_53, torch.float16);  mul_53 = None
        view_142 = torch.ops.aten.view.default(convert_element_type_155, [4096, 2048]);  convert_element_type_155 = None
        mm_31 = torch.ops.aten.mm.default(view_142, arg93_1);  view_142 = arg93_1 = None
        view_143 = torch.ops.aten.view.default(mm_31, [16, 256, 512]);  mm_31 = None
        add_84 = torch.ops.aten.add.Tensor(view_143, arg94_1);  view_143 = arg94_1 = None
        add_85 = torch.ops.aten.add.Tensor(add_84, convert_element_type_151);  add_84 = convert_element_type_151 = None
        convert_element_type_158 = torch.ops.prims.convert_element_type.default(add_85, torch.float32)
        var_mean_15 = torch.ops.aten.var_mean.correction(convert_element_type_158, [2], correction = 0, keepdim = True);  convert_element_type_158 = None
        getitem_54 = var_mean_15[0]
        getitem_55 = var_mean_15[1];  var_mean_15 = None
        add_86 = torch.ops.aten.add.Tensor(getitem_54, 1e-05);  getitem_54 = None
        rsqrt_15 = torch.ops.aten.rsqrt.default(add_86);  add_86 = None
        sub_23 = torch.ops.aten.sub.Tensor(add_85, getitem_55);  add_85 = getitem_55 = None
        mul_54 = torch.ops.aten.mul.Tensor(sub_23, rsqrt_15);  sub_23 = rsqrt_15 = None
        mul_55 = torch.ops.aten.mul.Tensor(mul_54, arg95_1);  mul_54 = arg95_1 = None
        add_87 = torch.ops.aten.add.Tensor(mul_55, arg96_1);  mul_55 = arg96_1 = None
        convert_element_type_159 = torch.ops.prims.convert_element_type.default(add_87, torch.float16);  add_87 = None
        view_144 = torch.ops.aten.view.default(convert_element_type_159, [4096, 512])
        mm_32 = torch.ops.aten.mm.default(view_144, arg97_1);  view_144 = arg97_1 = None
        view_145 = torch.ops.aten.view.default(mm_32, [16, 256, 1536]);  mm_32 = None
        add_88 = torch.ops.aten.add.Tensor(view_145, arg98_1);  view_145 = arg98_1 = None
        split_8 = torch.ops.aten.split.Tensor(add_88, 512, -1);  add_88 = None
        getitem_56 = split_8[0]
        getitem_57 = split_8[1]
        getitem_58 = split_8[2];  split_8 = None
        view_146 = torch.ops.aten.view.default(getitem_56, [16, 256, 16, 32]);  getitem_56 = None
        view_147 = torch.ops.aten.view.default(getitem_57, [16, 256, 16, 32]);  getitem_57 = None
        view_148 = torch.ops.aten.view.default(getitem_58, [16, 256, 16, 32]);  getitem_58 = None
        permute_default_9 = torch.ops.aten.permute.default(view_146, [0, 2, 1, 3]);  view_146 = None
        permute_default_10 = torch.ops.aten.permute.default(view_147, [0, 2, 1, 3]);  view_147 = None
        permute_default_11 = torch.ops.aten.permute.default(view_148, [0, 2, 1, 3]);  view_148 = None
        _scaled_dot_product_flash_attention_default_3 = torch.ops.aten._scaled_dot_product_flash_attention.default(permute_default_9, permute_default_10, permute_default_11, scale = 0.17677669529663687);  permute_default_9 = permute_default_10 = permute_default_11 = None
        getitem_87 = _scaled_dot_product_flash_attention_default_3[0];  _scaled_dot_product_flash_attention_default_3 = None
        permute_44 = torch.ops.aten.permute.default(getitem_87, [0, 2, 1, 3]);  getitem_87 = None
        clone_35 = torch.ops.aten.clone.default(permute_44, memory_format = torch.contiguous_format);  permute_44 = None
        view_155 = torch.ops.aten.view.default(clone_35, [16, 256, 512]);  clone_35 = None
        view_156 = torch.ops.aten.view.default(view_155, [4096, 512]);  view_155 = None
        mm_33 = torch.ops.aten.mm.default(view_156, arg99_1);  view_156 = arg99_1 = None
        view_157 = torch.ops.aten.view.default(mm_33, [16, 256, 512]);  mm_33 = None
        add_89 = torch.ops.aten.add.Tensor(view_157, arg100_1);  view_157 = arg100_1 = None
        add_90 = torch.ops.aten.add.Tensor(add_89, convert_element_type_159);  add_89 = None
        convert_element_type_170 = torch.ops.prims.convert_element_type.default(add_90, torch.float32)
        var_mean_16 = torch.ops.aten.var_mean.correction(convert_element_type_170, [2], correction = 0, keepdim = True);  convert_element_type_170 = None
        getitem_59 = var_mean_16[0]
        getitem_60 = var_mean_16[1];  var_mean_16 = None
        add_91 = torch.ops.aten.add.Tensor(getitem_59, 1e-05);  getitem_59 = None
        rsqrt_16 = torch.ops.aten.rsqrt.default(add_91);  add_91 = None
        sub_25 = torch.ops.aten.sub.Tensor(add_90, getitem_60);  add_90 = getitem_60 = None
        mul_56 = torch.ops.aten.mul.Tensor(sub_25, rsqrt_16);  sub_25 = rsqrt_16 = None
        mul_57 = torch.ops.aten.mul.Tensor(mul_56, arg101_1);  mul_56 = arg101_1 = None
        add_92 = torch.ops.aten.add.Tensor(mul_57, arg102_1);  mul_57 = arg102_1 = None
        convert_element_type_171 = torch.ops.prims.convert_element_type.default(add_92, torch.float16);  add_92 = None
        view_158 = torch.ops.aten.view.default(convert_element_type_171, [4096, 512])
        mm_34 = torch.ops.aten.mm.default(view_158, arg103_1);  view_158 = arg103_1 = None
        view_159 = torch.ops.aten.view.default(mm_34, [16, 256, 2048]);  mm_34 = None
        add_93 = torch.ops.aten.add.Tensor(view_159, arg104_1);  view_159 = arg104_1 = None
        convert_element_type_174 = torch.ops.prims.convert_element_type.default(add_93, torch.float32);  add_93 = None
        mul_58 = torch.ops.aten.mul.Tensor(convert_element_type_174, 0.5)
        mul_59 = torch.ops.aten.mul.Tensor(convert_element_type_174, 0.7071067811865476);  convert_element_type_174 = None
        erf_8 = torch.ops.aten.erf.default(mul_59);  mul_59 = None
        add_94 = torch.ops.aten.add.Tensor(erf_8, 1);  erf_8 = None
        mul_60 = torch.ops.aten.mul.Tensor(mul_58, add_94);  mul_58 = add_94 = None
        convert_element_type_175 = torch.ops.prims.convert_element_type.default(mul_60, torch.float16);  mul_60 = None
        view_160 = torch.ops.aten.view.default(convert_element_type_175, [4096, 2048]);  convert_element_type_175 = None
        mm_35 = torch.ops.aten.mm.default(view_160, arg105_1);  view_160 = arg105_1 = None
        view_161 = torch.ops.aten.view.default(mm_35, [16, 256, 512]);  mm_35 = None
        add_95 = torch.ops.aten.add.Tensor(view_161, arg106_1);  view_161 = arg106_1 = None
        add_96 = torch.ops.aten.add.Tensor(add_95, convert_element_type_171);  add_95 = convert_element_type_171 = None
        convert_element_type_178 = torch.ops.prims.convert_element_type.default(add_96, torch.float32)
        var_mean_17 = torch.ops.aten.var_mean.correction(convert_element_type_178, [2], correction = 0, keepdim = True);  convert_element_type_178 = None
        getitem_61 = var_mean_17[0]
        getitem_62 = var_mean_17[1];  var_mean_17 = None
        add_97 = torch.ops.aten.add.Tensor(getitem_61, 1e-05);  getitem_61 = None
        rsqrt_17 = torch.ops.aten.rsqrt.default(add_97);  add_97 = None
        sub_26 = torch.ops.aten.sub.Tensor(add_96, getitem_62);  add_96 = getitem_62 = None
        mul_61 = torch.ops.aten.mul.Tensor(sub_26, rsqrt_17);  sub_26 = rsqrt_17 = None
        mul_62 = torch.ops.aten.mul.Tensor(mul_61, arg107_1);  mul_61 = arg107_1 = None
        add_98 = torch.ops.aten.add.Tensor(mul_62, arg108_1);  mul_62 = arg108_1 = None
        convert_element_type_179 = torch.ops.prims.convert_element_type.default(add_98, torch.float16);  add_98 = None
        view_162 = torch.ops.aten.view.default(convert_element_type_179, [4096, 512])
        mm_36 = torch.ops.aten.mm.default(view_162, arg109_1);  view_162 = arg109_1 = None
        view_163 = torch.ops.aten.view.default(mm_36, [16, 256, 1536]);  mm_36 = None
        add_99 = torch.ops.aten.add.Tensor(view_163, arg110_1);  view_163 = arg110_1 = None
        split_9 = torch.ops.aten.split.Tensor(add_99, 512, -1);  add_99 = None
        getitem_63 = split_9[0]
        getitem_64 = split_9[1]
        getitem_65 = split_9[2];  split_9 = None
        view_164 = torch.ops.aten.view.default(getitem_63, [16, 256, 16, 32]);  getitem_63 = None
        view_165 = torch.ops.aten.view.default(getitem_64, [16, 256, 16, 32]);  getitem_64 = None
        view_166 = torch.ops.aten.view.default(getitem_65, [16, 256, 16, 32]);  getitem_65 = None
        permute_default_6 = torch.ops.aten.permute.default(view_164, [0, 2, 1, 3]);  view_164 = None
        permute_default_7 = torch.ops.aten.permute.default(view_165, [0, 2, 1, 3]);  view_165 = None
        permute_default_8 = torch.ops.aten.permute.default(view_166, [0, 2, 1, 3]);  view_166 = None
        _scaled_dot_product_flash_attention_default_2 = torch.ops.aten._scaled_dot_product_flash_attention.default(permute_default_6, permute_default_7, permute_default_8, scale = 0.17677669529663687);  permute_default_6 = permute_default_7 = permute_default_8 = None
        getitem_86 = _scaled_dot_product_flash_attention_default_2[0];  _scaled_dot_product_flash_attention_default_2 = None
        permute_49 = torch.ops.aten.permute.default(getitem_86, [0, 2, 1, 3]);  getitem_86 = None
        clone_39 = torch.ops.aten.clone.default(permute_49, memory_format = torch.contiguous_format);  permute_49 = None
        view_173 = torch.ops.aten.view.default(clone_39, [16, 256, 512]);  clone_39 = None
        view_174 = torch.ops.aten.view.default(view_173, [4096, 512]);  view_173 = None
        mm_37 = torch.ops.aten.mm.default(view_174, arg111_1);  view_174 = arg111_1 = None
        view_175 = torch.ops.aten.view.default(mm_37, [16, 256, 512]);  mm_37 = None
        add_100 = torch.ops.aten.add.Tensor(view_175, arg112_1);  view_175 = arg112_1 = None
        add_101 = torch.ops.aten.add.Tensor(add_100, convert_element_type_179);  add_100 = None
        convert_element_type_190 = torch.ops.prims.convert_element_type.default(add_101, torch.float32)
        var_mean_18 = torch.ops.aten.var_mean.correction(convert_element_type_190, [2], correction = 0, keepdim = True);  convert_element_type_190 = None
        getitem_66 = var_mean_18[0]
        getitem_67 = var_mean_18[1];  var_mean_18 = None
        add_102 = torch.ops.aten.add.Tensor(getitem_66, 1e-05);  getitem_66 = None
        rsqrt_18 = torch.ops.aten.rsqrt.default(add_102);  add_102 = None
        sub_28 = torch.ops.aten.sub.Tensor(add_101, getitem_67);  add_101 = getitem_67 = None
        mul_63 = torch.ops.aten.mul.Tensor(sub_28, rsqrt_18);  sub_28 = rsqrt_18 = None
        mul_64 = torch.ops.aten.mul.Tensor(mul_63, arg113_1);  mul_63 = arg113_1 = None
        add_103 = torch.ops.aten.add.Tensor(mul_64, arg114_1);  mul_64 = arg114_1 = None
        convert_element_type_191 = torch.ops.prims.convert_element_type.default(add_103, torch.float16);  add_103 = None
        view_176 = torch.ops.aten.view.default(convert_element_type_191, [4096, 512])
        mm_38 = torch.ops.aten.mm.default(view_176, arg115_1);  view_176 = arg115_1 = None
        view_177 = torch.ops.aten.view.default(mm_38, [16, 256, 2048]);  mm_38 = None
        add_104 = torch.ops.aten.add.Tensor(view_177, arg116_1);  view_177 = arg116_1 = None
        convert_element_type_194 = torch.ops.prims.convert_element_type.default(add_104, torch.float32);  add_104 = None
        mul_65 = torch.ops.aten.mul.Tensor(convert_element_type_194, 0.5)
        mul_66 = torch.ops.aten.mul.Tensor(convert_element_type_194, 0.7071067811865476);  convert_element_type_194 = None
        erf_9 = torch.ops.aten.erf.default(mul_66);  mul_66 = None
        add_105 = torch.ops.aten.add.Tensor(erf_9, 1);  erf_9 = None
        mul_67 = torch.ops.aten.mul.Tensor(mul_65, add_105);  mul_65 = add_105 = None
        convert_element_type_195 = torch.ops.prims.convert_element_type.default(mul_67, torch.float16);  mul_67 = None
        view_178 = torch.ops.aten.view.default(convert_element_type_195, [4096, 2048]);  convert_element_type_195 = None
        mm_39 = torch.ops.aten.mm.default(view_178, arg117_1);  view_178 = arg117_1 = None
        view_179 = torch.ops.aten.view.default(mm_39, [16, 256, 512]);  mm_39 = None
        add_106 = torch.ops.aten.add.Tensor(view_179, arg118_1);  view_179 = arg118_1 = None
        add_107 = torch.ops.aten.add.Tensor(add_106, convert_element_type_191);  add_106 = convert_element_type_191 = None
        convert_element_type_198 = torch.ops.prims.convert_element_type.default(add_107, torch.float32)
        var_mean_19 = torch.ops.aten.var_mean.correction(convert_element_type_198, [2], correction = 0, keepdim = True);  convert_element_type_198 = None
        getitem_68 = var_mean_19[0]
        getitem_69 = var_mean_19[1];  var_mean_19 = None
        add_108 = torch.ops.aten.add.Tensor(getitem_68, 1e-05);  getitem_68 = None
        rsqrt_19 = torch.ops.aten.rsqrt.default(add_108);  add_108 = None
        sub_29 = torch.ops.aten.sub.Tensor(add_107, getitem_69);  add_107 = getitem_69 = None
        mul_68 = torch.ops.aten.mul.Tensor(sub_29, rsqrt_19);  sub_29 = rsqrt_19 = None
        mul_69 = torch.ops.aten.mul.Tensor(mul_68, arg119_1);  mul_68 = arg119_1 = None
        add_109 = torch.ops.aten.add.Tensor(mul_69, arg120_1);  mul_69 = arg120_1 = None
        convert_element_type_199 = torch.ops.prims.convert_element_type.default(add_109, torch.float16);  add_109 = None
        view_180 = torch.ops.aten.view.default(convert_element_type_199, [4096, 512])
        mm_40 = torch.ops.aten.mm.default(view_180, arg121_1);  view_180 = arg121_1 = None
        view_181 = torch.ops.aten.view.default(mm_40, [16, 256, 1536]);  mm_40 = None
        add_110 = torch.ops.aten.add.Tensor(view_181, arg122_1);  view_181 = arg122_1 = None
        split_10 = torch.ops.aten.split.Tensor(add_110, 512, -1);  add_110 = None
        getitem_70 = split_10[0]
        getitem_71 = split_10[1]
        getitem_72 = split_10[2];  split_10 = None
        view_182 = torch.ops.aten.view.default(getitem_70, [16, 256, 16, 32]);  getitem_70 = None
        view_183 = torch.ops.aten.view.default(getitem_71, [16, 256, 16, 32]);  getitem_71 = None
        view_184 = torch.ops.aten.view.default(getitem_72, [16, 256, 16, 32]);  getitem_72 = None
        permute_default_3 = torch.ops.aten.permute.default(view_182, [0, 2, 1, 3]);  view_182 = None
        permute_default_4 = torch.ops.aten.permute.default(view_183, [0, 2, 1, 3]);  view_183 = None
        permute_default_5 = torch.ops.aten.permute.default(view_184, [0, 2, 1, 3]);  view_184 = None
        _scaled_dot_product_flash_attention_default_1 = torch.ops.aten._scaled_dot_product_flash_attention.default(permute_default_3, permute_default_4, permute_default_5, scale = 0.17677669529663687);  permute_default_3 = permute_default_4 = permute_default_5 = None
        getitem_85 = _scaled_dot_product_flash_attention_default_1[0];  _scaled_dot_product_flash_attention_default_1 = None
        permute_54 = torch.ops.aten.permute.default(getitem_85, [0, 2, 1, 3]);  getitem_85 = None
        clone_43 = torch.ops.aten.clone.default(permute_54, memory_format = torch.contiguous_format);  permute_54 = None
        view_191 = torch.ops.aten.view.default(clone_43, [16, 256, 512]);  clone_43 = None
        view_192 = torch.ops.aten.view.default(view_191, [4096, 512]);  view_191 = None
        mm_41 = torch.ops.aten.mm.default(view_192, arg123_1);  view_192 = arg123_1 = None
        view_193 = torch.ops.aten.view.default(mm_41, [16, 256, 512]);  mm_41 = None
        add_111 = torch.ops.aten.add.Tensor(view_193, arg124_1);  view_193 = arg124_1 = None
        add_112 = torch.ops.aten.add.Tensor(add_111, convert_element_type_199);  add_111 = None
        convert_element_type_210 = torch.ops.prims.convert_element_type.default(add_112, torch.float32)
        var_mean_20 = torch.ops.aten.var_mean.correction(convert_element_type_210, [2], correction = 0, keepdim = True);  convert_element_type_210 = None
        getitem_73 = var_mean_20[0]
        getitem_74 = var_mean_20[1];  var_mean_20 = None
        add_113 = torch.ops.aten.add.Tensor(getitem_73, 1e-05);  getitem_73 = None
        rsqrt_20 = torch.ops.aten.rsqrt.default(add_113);  add_113 = None
        sub_31 = torch.ops.aten.sub.Tensor(add_112, getitem_74);  add_112 = getitem_74 = None
        mul_70 = torch.ops.aten.mul.Tensor(sub_31, rsqrt_20);  sub_31 = rsqrt_20 = None
        mul_71 = torch.ops.aten.mul.Tensor(mul_70, arg125_1);  mul_70 = arg125_1 = None
        add_114 = torch.ops.aten.add.Tensor(mul_71, arg126_1);  mul_71 = arg126_1 = None
        convert_element_type_211 = torch.ops.prims.convert_element_type.default(add_114, torch.float16);  add_114 = None
        view_194 = torch.ops.aten.view.default(convert_element_type_211, [4096, 512])
        mm_42 = torch.ops.aten.mm.default(view_194, arg127_1);  view_194 = arg127_1 = None
        view_195 = torch.ops.aten.view.default(mm_42, [16, 256, 2048]);  mm_42 = None
        add_115 = torch.ops.aten.add.Tensor(view_195, arg128_1);  view_195 = arg128_1 = None
        convert_element_type_214 = torch.ops.prims.convert_element_type.default(add_115, torch.float32);  add_115 = None
        mul_72 = torch.ops.aten.mul.Tensor(convert_element_type_214, 0.5)
        mul_73 = torch.ops.aten.mul.Tensor(convert_element_type_214, 0.7071067811865476);  convert_element_type_214 = None
        erf_10 = torch.ops.aten.erf.default(mul_73);  mul_73 = None
        add_116 = torch.ops.aten.add.Tensor(erf_10, 1);  erf_10 = None
        mul_74 = torch.ops.aten.mul.Tensor(mul_72, add_116);  mul_72 = add_116 = None
        convert_element_type_215 = torch.ops.prims.convert_element_type.default(mul_74, torch.float16);  mul_74 = None
        view_196 = torch.ops.aten.view.default(convert_element_type_215, [4096, 2048]);  convert_element_type_215 = None
        mm_43 = torch.ops.aten.mm.default(view_196, arg129_1);  view_196 = arg129_1 = None
        view_197 = torch.ops.aten.view.default(mm_43, [16, 256, 512]);  mm_43 = None
        add_117 = torch.ops.aten.add.Tensor(view_197, arg130_1);  view_197 = arg130_1 = None
        add_118 = torch.ops.aten.add.Tensor(add_117, convert_element_type_211);  add_117 = convert_element_type_211 = None
        convert_element_type_218 = torch.ops.prims.convert_element_type.default(add_118, torch.float32)
        var_mean_21 = torch.ops.aten.var_mean.correction(convert_element_type_218, [2], correction = 0, keepdim = True);  convert_element_type_218 = None
        getitem_75 = var_mean_21[0]
        getitem_76 = var_mean_21[1];  var_mean_21 = None
        add_119 = torch.ops.aten.add.Tensor(getitem_75, 1e-05);  getitem_75 = None
        rsqrt_21 = torch.ops.aten.rsqrt.default(add_119);  add_119 = None
        sub_32 = torch.ops.aten.sub.Tensor(add_118, getitem_76);  add_118 = getitem_76 = None
        mul_75 = torch.ops.aten.mul.Tensor(sub_32, rsqrt_21);  sub_32 = rsqrt_21 = None
        mul_76 = torch.ops.aten.mul.Tensor(mul_75, arg131_1);  mul_75 = arg131_1 = None
        add_120 = torch.ops.aten.add.Tensor(mul_76, arg132_1);  mul_76 = arg132_1 = None
        convert_element_type_219 = torch.ops.prims.convert_element_type.default(add_120, torch.float16);  add_120 = None
        view_198 = torch.ops.aten.view.default(convert_element_type_219, [4096, 512])
        mm_44 = torch.ops.aten.mm.default(view_198, arg133_1);  view_198 = arg133_1 = None
        view_199 = torch.ops.aten.view.default(mm_44, [16, 256, 1536]);  mm_44 = None
        add_121 = torch.ops.aten.add.Tensor(view_199, arg134_1);  view_199 = arg134_1 = None
        split_11 = torch.ops.aten.split.Tensor(add_121, 512, -1);  add_121 = None
        getitem_77 = split_11[0]
        getitem_78 = split_11[1]
        getitem_79 = split_11[2];  split_11 = None
        view_200 = torch.ops.aten.view.default(getitem_77, [16, 256, 16, 32]);  getitem_77 = None
        view_201 = torch.ops.aten.view.default(getitem_78, [16, 256, 16, 32]);  getitem_78 = None
        view_202 = torch.ops.aten.view.default(getitem_79, [16, 256, 16, 32]);  getitem_79 = None
        permute_default = torch.ops.aten.permute.default(view_200, [0, 2, 1, 3]);  view_200 = None
        permute_default_1 = torch.ops.aten.permute.default(view_201, [0, 2, 1, 3]);  view_201 = None
        permute_default_2 = torch.ops.aten.permute.default(view_202, [0, 2, 1, 3]);  view_202 = None
        _scaled_dot_product_flash_attention_default = torch.ops.aten._scaled_dot_product_flash_attention.default(permute_default, permute_default_1, permute_default_2, scale = 0.17677669529663687);  permute_default = permute_default_1 = permute_default_2 = None
        getitem_84 = _scaled_dot_product_flash_attention_default[0];  _scaled_dot_product_flash_attention_default = None
        permute_59 = torch.ops.aten.permute.default(getitem_84, [0, 2, 1, 3]);  getitem_84 = None
        clone_47 = torch.ops.aten.clone.default(permute_59, memory_format = torch.contiguous_format);  permute_59 = None
        view_209 = torch.ops.aten.view.default(clone_47, [16, 256, 512]);  clone_47 = None
        view_210 = torch.ops.aten.view.default(view_209, [4096, 512]);  view_209 = None
        mm_45 = torch.ops.aten.mm.default(view_210, arg135_1);  view_210 = arg135_1 = None
        view_211 = torch.ops.aten.view.default(mm_45, [16, 256, 512]);  mm_45 = None
        add_122 = torch.ops.aten.add.Tensor(view_211, arg136_1);  view_211 = arg136_1 = None
        add_123 = torch.ops.aten.add.Tensor(add_122, convert_element_type_219);  add_122 = None
        convert_element_type_230 = torch.ops.prims.convert_element_type.default(add_123, torch.float32)
        var_mean_22 = torch.ops.aten.var_mean.correction(convert_element_type_230, [2], correction = 0, keepdim = True);  convert_element_type_230 = None
        getitem_80 = var_mean_22[0]
        getitem_81 = var_mean_22[1];  var_mean_22 = None
        add_124 = torch.ops.aten.add.Tensor(getitem_80, 1e-05);  getitem_80 = None
        rsqrt_22 = torch.ops.aten.rsqrt.default(add_124);  add_124 = None
        sub_34 = torch.ops.aten.sub.Tensor(add_123, getitem_81);  add_123 = getitem_81 = None
        mul_77 = torch.ops.aten.mul.Tensor(sub_34, rsqrt_22);  sub_34 = rsqrt_22 = None
        mul_78 = torch.ops.aten.mul.Tensor(mul_77, arg137_1);  mul_77 = arg137_1 = None
        add_125 = torch.ops.aten.add.Tensor(mul_78, arg138_1);  mul_78 = arg138_1 = None
        convert_element_type_231 = torch.ops.prims.convert_element_type.default(add_125, torch.float16);  add_125 = None
        view_212 = torch.ops.aten.view.default(convert_element_type_231, [4096, 512])
        mm_46 = torch.ops.aten.mm.default(view_212, arg139_1);  view_212 = arg139_1 = None
        view_213 = torch.ops.aten.view.default(mm_46, [16, 256, 2048]);  mm_46 = None
        add_126 = torch.ops.aten.add.Tensor(view_213, arg140_1);  view_213 = arg140_1 = None
        convert_element_type_234 = torch.ops.prims.convert_element_type.default(add_126, torch.float32);  add_126 = None
        mul_79 = torch.ops.aten.mul.Tensor(convert_element_type_234, 0.5)
        mul_80 = torch.ops.aten.mul.Tensor(convert_element_type_234, 0.7071067811865476);  convert_element_type_234 = None
        erf_11 = torch.ops.aten.erf.default(mul_80);  mul_80 = None
        add_127 = torch.ops.aten.add.Tensor(erf_11, 1);  erf_11 = None
        mul_81 = torch.ops.aten.mul.Tensor(mul_79, add_127);  mul_79 = add_127 = None
        convert_element_type_235 = torch.ops.prims.convert_element_type.default(mul_81, torch.float16);  mul_81 = None
        view_214 = torch.ops.aten.view.default(convert_element_type_235, [4096, 2048]);  convert_element_type_235 = None
        mm_47 = torch.ops.aten.mm.default(view_214, arg141_1);  view_214 = arg141_1 = None
        view_215 = torch.ops.aten.view.default(mm_47, [16, 256, 512]);  mm_47 = None
        add_128 = torch.ops.aten.add.Tensor(view_215, arg142_1);  view_215 = arg142_1 = None
        add_129 = torch.ops.aten.add.Tensor(add_128, convert_element_type_231);  add_128 = convert_element_type_231 = None
        convert_element_type_238 = torch.ops.prims.convert_element_type.default(add_129, torch.float32)
        var_mean_23 = torch.ops.aten.var_mean.correction(convert_element_type_238, [2], correction = 0, keepdim = True);  convert_element_type_238 = None
        getitem_82 = var_mean_23[0]
        getitem_83 = var_mean_23[1];  var_mean_23 = None
        add_130 = torch.ops.aten.add.Tensor(getitem_82, 1e-05);  getitem_82 = None
        rsqrt_23 = torch.ops.aten.rsqrt.default(add_130);  add_130 = None
        sub_35 = torch.ops.aten.sub.Tensor(add_129, getitem_83);  add_129 = getitem_83 = None
        mul_82 = torch.ops.aten.mul.Tensor(sub_35, rsqrt_23);  sub_35 = rsqrt_23 = None
        mul_83 = torch.ops.aten.mul.Tensor(mul_82, arg143_1);  mul_82 = arg143_1 = None
        add_131 = torch.ops.aten.add.Tensor(mul_83, arg144_1);  mul_83 = arg144_1 = None
        convert_element_type_239 = torch.ops.prims.convert_element_type.default(add_131, torch.float16);  add_131 = None
        return (convert_element_type_19, convert_element_type_39, convert_element_type_59, convert_element_type_79, convert_element_type_99, convert_element_type_119, convert_element_type_139, convert_element_type_159, convert_element_type_179, convert_element_type_199, convert_element_type_219, convert_element_type_239)
        
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
    buf13 = reader.storage(None, 1572864, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf13, (512, 1536), dtype=torch.float16, is_leaf=True)  # arg13_1
    buf14 = reader.storage(None, 3072, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf14, (1536,), dtype=torch.float16, is_leaf=True)  # arg14_1
    buf15 = reader.storage(None, 524288, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf15, (512, 512), dtype=torch.float16, is_leaf=True)  # arg15_1
    buf16 = reader.storage(None, 1024, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf16, (512,), dtype=torch.float16, is_leaf=True)  # arg16_1
    buf17 = reader.storage(None, 1024, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf17, (512,), dtype=torch.float16, is_leaf=True)  # arg17_1
    buf18 = reader.storage(None, 1024, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf18, (512,), dtype=torch.float16, is_leaf=True)  # arg18_1
    buf19 = reader.storage(None, 2097152, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf19, (512, 2048), dtype=torch.float16, is_leaf=True)  # arg19_1
    buf20 = reader.storage(None, 4096, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf20, (2048,), dtype=torch.float16, is_leaf=True)  # arg20_1
    buf21 = reader.storage(None, 2097152, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf21, (2048, 512), dtype=torch.float16, is_leaf=True)  # arg21_1
    buf22 = reader.storage(None, 1024, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf22, (512,), dtype=torch.float16, is_leaf=True)  # arg22_1
    buf23 = reader.storage(None, 1024, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf23, (512,), dtype=torch.float16, is_leaf=True)  # arg23_1
    buf24 = reader.storage(None, 1024, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf24, (512,), dtype=torch.float16, is_leaf=True)  # arg24_1
    buf25 = reader.storage(None, 1572864, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf25, (512, 1536), dtype=torch.float16, is_leaf=True)  # arg25_1
    buf26 = reader.storage(None, 3072, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf26, (1536,), dtype=torch.float16, is_leaf=True)  # arg26_1
    buf27 = reader.storage(None, 524288, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf27, (512, 512), dtype=torch.float16, is_leaf=True)  # arg27_1
    buf28 = reader.storage(None, 1024, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf28, (512,), dtype=torch.float16, is_leaf=True)  # arg28_1
    buf29 = reader.storage(None, 1024, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf29, (512,), dtype=torch.float16, is_leaf=True)  # arg29_1
    buf30 = reader.storage(None, 1024, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf30, (512,), dtype=torch.float16, is_leaf=True)  # arg30_1
    buf31 = reader.storage(None, 2097152, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf31, (512, 2048), dtype=torch.float16, is_leaf=True)  # arg31_1
    buf32 = reader.storage(None, 4096, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf32, (2048,), dtype=torch.float16, is_leaf=True)  # arg32_1
    buf33 = reader.storage(None, 2097152, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf33, (2048, 512), dtype=torch.float16, is_leaf=True)  # arg33_1
    buf34 = reader.storage(None, 1024, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf34, (512,), dtype=torch.float16, is_leaf=True)  # arg34_1
    buf35 = reader.storage(None, 1024, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf35, (512,), dtype=torch.float16, is_leaf=True)  # arg35_1
    buf36 = reader.storage(None, 1024, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf36, (512,), dtype=torch.float16, is_leaf=True)  # arg36_1
    buf37 = reader.storage(None, 1572864, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf37, (512, 1536), dtype=torch.float16, is_leaf=True)  # arg37_1
    buf38 = reader.storage(None, 3072, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf38, (1536,), dtype=torch.float16, is_leaf=True)  # arg38_1
    buf39 = reader.storage(None, 524288, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf39, (512, 512), dtype=torch.float16, is_leaf=True)  # arg39_1
    buf40 = reader.storage(None, 1024, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf40, (512,), dtype=torch.float16, is_leaf=True)  # arg40_1
    buf41 = reader.storage(None, 1024, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf41, (512,), dtype=torch.float16, is_leaf=True)  # arg41_1
    buf42 = reader.storage(None, 1024, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf42, (512,), dtype=torch.float16, is_leaf=True)  # arg42_1
    buf43 = reader.storage(None, 2097152, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf43, (512, 2048), dtype=torch.float16, is_leaf=True)  # arg43_1
    buf44 = reader.storage(None, 4096, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf44, (2048,), dtype=torch.float16, is_leaf=True)  # arg44_1
    buf45 = reader.storage(None, 2097152, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf45, (2048, 512), dtype=torch.float16, is_leaf=True)  # arg45_1
    buf46 = reader.storage(None, 1024, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf46, (512,), dtype=torch.float16, is_leaf=True)  # arg46_1
    buf47 = reader.storage(None, 1024, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf47, (512,), dtype=torch.float16, is_leaf=True)  # arg47_1
    buf48 = reader.storage(None, 1024, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf48, (512,), dtype=torch.float16, is_leaf=True)  # arg48_1
    buf49 = reader.storage(None, 1572864, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf49, (512, 1536), dtype=torch.float16, is_leaf=True)  # arg49_1
    buf50 = reader.storage(None, 3072, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf50, (1536,), dtype=torch.float16, is_leaf=True)  # arg50_1
    buf51 = reader.storage(None, 524288, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf51, (512, 512), dtype=torch.float16, is_leaf=True)  # arg51_1
    buf52 = reader.storage(None, 1024, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf52, (512,), dtype=torch.float16, is_leaf=True)  # arg52_1
    buf53 = reader.storage(None, 1024, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf53, (512,), dtype=torch.float16, is_leaf=True)  # arg53_1
    buf54 = reader.storage(None, 1024, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf54, (512,), dtype=torch.float16, is_leaf=True)  # arg54_1
    buf55 = reader.storage(None, 2097152, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf55, (512, 2048), dtype=torch.float16, is_leaf=True)  # arg55_1
    buf56 = reader.storage(None, 4096, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf56, (2048,), dtype=torch.float16, is_leaf=True)  # arg56_1
    buf57 = reader.storage(None, 2097152, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf57, (2048, 512), dtype=torch.float16, is_leaf=True)  # arg57_1
    buf58 = reader.storage(None, 1024, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf58, (512,), dtype=torch.float16, is_leaf=True)  # arg58_1
    buf59 = reader.storage(None, 1024, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf59, (512,), dtype=torch.float16, is_leaf=True)  # arg59_1
    buf60 = reader.storage(None, 1024, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf60, (512,), dtype=torch.float16, is_leaf=True)  # arg60_1
    buf61 = reader.storage(None, 1572864, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf61, (512, 1536), dtype=torch.float16, is_leaf=True)  # arg61_1
    buf62 = reader.storage(None, 3072, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf62, (1536,), dtype=torch.float16, is_leaf=True)  # arg62_1
    buf63 = reader.storage(None, 524288, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf63, (512, 512), dtype=torch.float16, is_leaf=True)  # arg63_1
    buf64 = reader.storage(None, 1024, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf64, (512,), dtype=torch.float16, is_leaf=True)  # arg64_1
    buf65 = reader.storage(None, 1024, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf65, (512,), dtype=torch.float16, is_leaf=True)  # arg65_1
    buf66 = reader.storage(None, 1024, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf66, (512,), dtype=torch.float16, is_leaf=True)  # arg66_1
    buf67 = reader.storage(None, 2097152, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf67, (512, 2048), dtype=torch.float16, is_leaf=True)  # arg67_1
    buf68 = reader.storage(None, 4096, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf68, (2048,), dtype=torch.float16, is_leaf=True)  # arg68_1
    buf69 = reader.storage(None, 2097152, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf69, (2048, 512), dtype=torch.float16, is_leaf=True)  # arg69_1
    buf70 = reader.storage(None, 1024, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf70, (512,), dtype=torch.float16, is_leaf=True)  # arg70_1
    buf71 = reader.storage(None, 1024, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf71, (512,), dtype=torch.float16, is_leaf=True)  # arg71_1
    buf72 = reader.storage(None, 1024, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf72, (512,), dtype=torch.float16, is_leaf=True)  # arg72_1
    buf73 = reader.storage(None, 1572864, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf73, (512, 1536), dtype=torch.float16, is_leaf=True)  # arg73_1
    buf74 = reader.storage(None, 3072, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf74, (1536,), dtype=torch.float16, is_leaf=True)  # arg74_1
    buf75 = reader.storage(None, 524288, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf75, (512, 512), dtype=torch.float16, is_leaf=True)  # arg75_1
    buf76 = reader.storage(None, 1024, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf76, (512,), dtype=torch.float16, is_leaf=True)  # arg76_1
    buf77 = reader.storage(None, 1024, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf77, (512,), dtype=torch.float16, is_leaf=True)  # arg77_1
    buf78 = reader.storage(None, 1024, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf78, (512,), dtype=torch.float16, is_leaf=True)  # arg78_1
    buf79 = reader.storage(None, 2097152, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf79, (512, 2048), dtype=torch.float16, is_leaf=True)  # arg79_1
    buf80 = reader.storage(None, 4096, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf80, (2048,), dtype=torch.float16, is_leaf=True)  # arg80_1
    buf81 = reader.storage(None, 2097152, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf81, (2048, 512), dtype=torch.float16, is_leaf=True)  # arg81_1
    buf82 = reader.storage(None, 1024, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf82, (512,), dtype=torch.float16, is_leaf=True)  # arg82_1
    buf83 = reader.storage(None, 1024, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf83, (512,), dtype=torch.float16, is_leaf=True)  # arg83_1
    buf84 = reader.storage(None, 1024, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf84, (512,), dtype=torch.float16, is_leaf=True)  # arg84_1
    buf85 = reader.storage(None, 1572864, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf85, (512, 1536), dtype=torch.float16, is_leaf=True)  # arg85_1
    buf86 = reader.storage(None, 3072, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf86, (1536,), dtype=torch.float16, is_leaf=True)  # arg86_1
    buf87 = reader.storage(None, 524288, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf87, (512, 512), dtype=torch.float16, is_leaf=True)  # arg87_1
    buf88 = reader.storage(None, 1024, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf88, (512,), dtype=torch.float16, is_leaf=True)  # arg88_1
    buf89 = reader.storage(None, 1024, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf89, (512,), dtype=torch.float16, is_leaf=True)  # arg89_1
    buf90 = reader.storage(None, 1024, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf90, (512,), dtype=torch.float16, is_leaf=True)  # arg90_1
    buf91 = reader.storage(None, 2097152, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf91, (512, 2048), dtype=torch.float16, is_leaf=True)  # arg91_1
    buf92 = reader.storage(None, 4096, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf92, (2048,), dtype=torch.float16, is_leaf=True)  # arg92_1
    buf93 = reader.storage(None, 2097152, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf93, (2048, 512), dtype=torch.float16, is_leaf=True)  # arg93_1
    buf94 = reader.storage(None, 1024, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf94, (512,), dtype=torch.float16, is_leaf=True)  # arg94_1
    buf95 = reader.storage(None, 1024, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf95, (512,), dtype=torch.float16, is_leaf=True)  # arg95_1
    buf96 = reader.storage(None, 1024, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf96, (512,), dtype=torch.float16, is_leaf=True)  # arg96_1
    buf97 = reader.storage(None, 1572864, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf97, (512, 1536), dtype=torch.float16, is_leaf=True)  # arg97_1
    buf98 = reader.storage(None, 3072, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf98, (1536,), dtype=torch.float16, is_leaf=True)  # arg98_1
    buf99 = reader.storage(None, 524288, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf99, (512, 512), dtype=torch.float16, is_leaf=True)  # arg99_1
    buf100 = reader.storage(None, 1024, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf100, (512,), dtype=torch.float16, is_leaf=True)  # arg100_1
    buf101 = reader.storage(None, 1024, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf101, (512,), dtype=torch.float16, is_leaf=True)  # arg101_1
    buf102 = reader.storage(None, 1024, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf102, (512,), dtype=torch.float16, is_leaf=True)  # arg102_1
    buf103 = reader.storage(None, 2097152, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf103, (512, 2048), dtype=torch.float16, is_leaf=True)  # arg103_1
    buf104 = reader.storage(None, 4096, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf104, (2048,), dtype=torch.float16, is_leaf=True)  # arg104_1
    buf105 = reader.storage(None, 2097152, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf105, (2048, 512), dtype=torch.float16, is_leaf=True)  # arg105_1
    buf106 = reader.storage(None, 1024, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf106, (512,), dtype=torch.float16, is_leaf=True)  # arg106_1
    buf107 = reader.storage(None, 1024, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf107, (512,), dtype=torch.float16, is_leaf=True)  # arg107_1
    buf108 = reader.storage(None, 1024, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf108, (512,), dtype=torch.float16, is_leaf=True)  # arg108_1
    buf109 = reader.storage(None, 1572864, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf109, (512, 1536), dtype=torch.float16, is_leaf=True)  # arg109_1
    buf110 = reader.storage(None, 3072, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf110, (1536,), dtype=torch.float16, is_leaf=True)  # arg110_1
    buf111 = reader.storage(None, 524288, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf111, (512, 512), dtype=torch.float16, is_leaf=True)  # arg111_1
    buf112 = reader.storage(None, 1024, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf112, (512,), dtype=torch.float16, is_leaf=True)  # arg112_1
    buf113 = reader.storage(None, 1024, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf113, (512,), dtype=torch.float16, is_leaf=True)  # arg113_1
    buf114 = reader.storage(None, 1024, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf114, (512,), dtype=torch.float16, is_leaf=True)  # arg114_1
    buf115 = reader.storage(None, 2097152, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf115, (512, 2048), dtype=torch.float16, is_leaf=True)  # arg115_1
    buf116 = reader.storage(None, 4096, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf116, (2048,), dtype=torch.float16, is_leaf=True)  # arg116_1
    buf117 = reader.storage(None, 2097152, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf117, (2048, 512), dtype=torch.float16, is_leaf=True)  # arg117_1
    buf118 = reader.storage(None, 1024, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf118, (512,), dtype=torch.float16, is_leaf=True)  # arg118_1
    buf119 = reader.storage(None, 1024, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf119, (512,), dtype=torch.float16, is_leaf=True)  # arg119_1
    buf120 = reader.storage(None, 1024, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf120, (512,), dtype=torch.float16, is_leaf=True)  # arg120_1
    buf121 = reader.storage(None, 1572864, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf121, (512, 1536), dtype=torch.float16, is_leaf=True)  # arg121_1
    buf122 = reader.storage(None, 3072, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf122, (1536,), dtype=torch.float16, is_leaf=True)  # arg122_1
    buf123 = reader.storage(None, 524288, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf123, (512, 512), dtype=torch.float16, is_leaf=True)  # arg123_1
    buf124 = reader.storage(None, 1024, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf124, (512,), dtype=torch.float16, is_leaf=True)  # arg124_1
    buf125 = reader.storage(None, 1024, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf125, (512,), dtype=torch.float16, is_leaf=True)  # arg125_1
    buf126 = reader.storage(None, 1024, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf126, (512,), dtype=torch.float16, is_leaf=True)  # arg126_1
    buf127 = reader.storage(None, 2097152, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf127, (512, 2048), dtype=torch.float16, is_leaf=True)  # arg127_1
    buf128 = reader.storage(None, 4096, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf128, (2048,), dtype=torch.float16, is_leaf=True)  # arg128_1
    buf129 = reader.storage(None, 2097152, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf129, (2048, 512), dtype=torch.float16, is_leaf=True)  # arg129_1
    buf130 = reader.storage(None, 1024, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf130, (512,), dtype=torch.float16, is_leaf=True)  # arg130_1
    buf131 = reader.storage(None, 1024, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf131, (512,), dtype=torch.float16, is_leaf=True)  # arg131_1
    buf132 = reader.storage(None, 1024, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf132, (512,), dtype=torch.float16, is_leaf=True)  # arg132_1
    buf133 = reader.storage(None, 1572864, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf133, (512, 1536), dtype=torch.float16, is_leaf=True)  # arg133_1
    buf134 = reader.storage(None, 3072, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf134, (1536,), dtype=torch.float16, is_leaf=True)  # arg134_1
    buf135 = reader.storage(None, 524288, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf135, (512, 512), dtype=torch.float16, is_leaf=True)  # arg135_1
    buf136 = reader.storage(None, 1024, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf136, (512,), dtype=torch.float16, is_leaf=True)  # arg136_1
    buf137 = reader.storage(None, 1024, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf137, (512,), dtype=torch.float16, is_leaf=True)  # arg137_1
    buf138 = reader.storage(None, 1024, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf138, (512,), dtype=torch.float16, is_leaf=True)  # arg138_1
    buf139 = reader.storage(None, 2097152, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf139, (512, 2048), dtype=torch.float16, is_leaf=True)  # arg139_1
    buf140 = reader.storage(None, 4096, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf140, (2048,), dtype=torch.float16, is_leaf=True)  # arg140_1
    buf141 = reader.storage(None, 2097152, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf141, (2048, 512), dtype=torch.float16, is_leaf=True)  # arg141_1
    buf142 = reader.storage(None, 1024, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf142, (512,), dtype=torch.float16, is_leaf=True)  # arg142_1
    buf143 = reader.storage(None, 1024, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf143, (512,), dtype=torch.float16, is_leaf=True)  # arg143_1
    buf144 = reader.storage(None, 1024, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf144, (512,), dtype=torch.float16, is_leaf=True)  # arg144_1
load_args._version = 0
mod = Repro()
if __name__ == '__main__':
    from torch._dynamo.repro.after_aot import run_repro
    with torch.no_grad():
        run_repro(mod, load_args, accuracy=False, command='run', save_dir=None, tracing_mode='real', check_str=None)
