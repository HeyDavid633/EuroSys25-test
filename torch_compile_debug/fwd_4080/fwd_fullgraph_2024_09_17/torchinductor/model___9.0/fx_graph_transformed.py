class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "f16[16, 256, 512]", arg1_1: "f16[512, 1536]", arg2_1: "f16[1536]", arg3_1: "f16[512, 512]", arg4_1: "f16[512]", arg5_1: "f16[512]", arg6_1: "f16[512]", arg7_1: "f16[512, 2048]", arg8_1: "f16[2048]", arg9_1: "f16[2048, 512]", arg10_1: "f16[512]", arg11_1: "f16[512]", arg12_1: "f16[512]", arg13_1: "f16[512, 1536]", arg14_1: "f16[1536]", arg15_1: "f16[512, 512]", arg16_1: "f16[512]", arg17_1: "f16[512]", arg18_1: "f16[512]", arg19_1: "f16[512, 2048]", arg20_1: "f16[2048]", arg21_1: "f16[2048, 512]", arg22_1: "f16[512]", arg23_1: "f16[512]", arg24_1: "f16[512]", arg25_1: "f16[512, 1536]", arg26_1: "f16[1536]", arg27_1: "f16[512, 512]", arg28_1: "f16[512]", arg29_1: "f16[512]", arg30_1: "f16[512]", arg31_1: "f16[512, 2048]", arg32_1: "f16[2048]", arg33_1: "f16[2048, 512]", arg34_1: "f16[512]", arg35_1: "f16[512]", arg36_1: "f16[512]", arg37_1: "f16[512, 1536]", arg38_1: "f16[1536]", arg39_1: "f16[512, 512]", arg40_1: "f16[512]", arg41_1: "f16[512]", arg42_1: "f16[512]", arg43_1: "f16[512, 2048]", arg44_1: "f16[2048]", arg45_1: "f16[2048, 512]", arg46_1: "f16[512]", arg47_1: "f16[512]", arg48_1: "f16[512]", arg49_1: "f16[512, 1536]", arg50_1: "f16[1536]", arg51_1: "f16[512, 512]", arg52_1: "f16[512]", arg53_1: "f16[512]", arg54_1: "f16[512]", arg55_1: "f16[512, 2048]", arg56_1: "f16[2048]", arg57_1: "f16[2048, 512]", arg58_1: "f16[512]", arg59_1: "f16[512]", arg60_1: "f16[512]", arg61_1: "f16[512, 1536]", arg62_1: "f16[1536]", arg63_1: "f16[512, 512]", arg64_1: "f16[512]", arg65_1: "f16[512]", arg66_1: "f16[512]", arg67_1: "f16[512, 2048]", arg68_1: "f16[2048]", arg69_1: "f16[2048, 512]", arg70_1: "f16[512]", arg71_1: "f16[512]", arg72_1: "f16[512]", arg73_1: "f16[512, 1536]", arg74_1: "f16[1536]", arg75_1: "f16[512, 512]", arg76_1: "f16[512]", arg77_1: "f16[512]", arg78_1: "f16[512]", arg79_1: "f16[512, 2048]", arg80_1: "f16[2048]", arg81_1: "f16[2048, 512]", arg82_1: "f16[512]", arg83_1: "f16[512]", arg84_1: "f16[512]", arg85_1: "f16[512, 1536]", arg86_1: "f16[1536]", arg87_1: "f16[512, 512]", arg88_1: "f16[512]", arg89_1: "f16[512]", arg90_1: "f16[512]", arg91_1: "f16[512, 2048]", arg92_1: "f16[2048]", arg93_1: "f16[2048, 512]", arg94_1: "f16[512]", arg95_1: "f16[512]", arg96_1: "f16[512]", arg97_1: "f16[512, 1536]", arg98_1: "f16[1536]", arg99_1: "f16[512, 512]", arg100_1: "f16[512]", arg101_1: "f16[512]", arg102_1: "f16[512]", arg103_1: "f16[512, 2048]", arg104_1: "f16[2048]", arg105_1: "f16[2048, 512]", arg106_1: "f16[512]", arg107_1: "f16[512]", arg108_1: "f16[512]", arg109_1: "f16[512, 1536]", arg110_1: "f16[1536]", arg111_1: "f16[512, 512]", arg112_1: "f16[512]", arg113_1: "f16[512]", arg114_1: "f16[512]", arg115_1: "f16[512, 2048]", arg116_1: "f16[2048]", arg117_1: "f16[2048, 512]", arg118_1: "f16[512]", arg119_1: "f16[512]", arg120_1: "f16[512]", arg121_1: "f16[512, 1536]", arg122_1: "f16[1536]", arg123_1: "f16[512, 512]", arg124_1: "f16[512]", arg125_1: "f16[512]", arg126_1: "f16[512]", arg127_1: "f16[512, 2048]", arg128_1: "f16[2048]", arg129_1: "f16[2048, 512]", arg130_1: "f16[512]", arg131_1: "f16[512]", arg132_1: "f16[512]", arg133_1: "f16[512, 1536]", arg134_1: "f16[1536]", arg135_1: "f16[512, 512]", arg136_1: "f16[512]", arg137_1: "f16[512]", arg138_1: "f16[512]", arg139_1: "f16[512, 2048]", arg140_1: "f16[2048]", arg141_1: "f16[2048, 512]", arg142_1: "f16[512]", arg143_1: "f16[512]", arg144_1: "f16[512]"):
        # File: /EuroSys25/fwd-compile-print.py:22 in fwd_bert_std, code: qkv = torch.matmul(hidden_states, qkv_kernel[layer]) + qkv_bias[layer]
        view: "f16[4096, 512]" = torch.ops.aten.reshape.default(arg0_1, [4096, 512])
        mm: "f16[4096, 1536]" = torch.ops.aten.mm.default(view, arg1_1);  view = arg1_1 = None
        view_1: "f16[16, 256, 1536]" = torch.ops.aten.reshape.default(mm, [16, 256, 1536]);  mm = None
        add: "f16[16, 256, 1536]" = torch.ops.aten.add.Tensor(view_1, arg2_1);  view_1 = arg2_1 = None
        
        # File: /EuroSys25/fwd-compile-print.py:23 in fwd_bert_std, code: q, k, v = qkv.chunk(3, dim=-1)
        split = torch.ops.aten.split.Tensor(add, 512, -1);  add = None
        getitem: "f16[16, 256, 512]" = split[0]
        getitem_1: "f16[16, 256, 512]" = split[1]
        getitem_2: "f16[16, 256, 512]" = split[2];  split = None
        
        # File: /EuroSys25/utils/utils.py:47 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_2: "f16[16, 256, 16, 32]" = torch.ops.aten.reshape.default(getitem, [16, 256, 16, 32]);  getitem = None
        
        # No stacktrace found for following nodes
        permute_default_33: "f16[16, 16, 256, 32]" = torch.ops.aten.permute.default(view_2, [0, 2, 1, 3]);  view_2 = None
        
        # File: /EuroSys25/utils/utils.py:47 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_3: "f16[16, 256, 16, 32]" = torch.ops.aten.reshape.default(getitem_1, [16, 256, 16, 32]);  getitem_1 = None
        
        # No stacktrace found for following nodes
        permute_default_34: "f16[16, 16, 256, 32]" = torch.ops.aten.permute.default(view_3, [0, 2, 1, 3]);  view_3 = None
        
        # File: /EuroSys25/utils/utils.py:47 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_4: "f16[16, 256, 16, 32]" = torch.ops.aten.reshape.default(getitem_2, [16, 256, 16, 32]);  getitem_2 = None
        
        # No stacktrace found for following nodes
        permute_default_35: "f16[16, 16, 256, 32]" = torch.ops.aten.permute.default(view_4, [0, 2, 1, 3]);  view_4 = None
        _scaled_dot_product_flash_attention_default_11 = torch.ops.aten._scaled_dot_product_flash_attention.default(permute_default_33, permute_default_34, permute_default_35, scale = 0.17677669529663687);  permute_default_33 = permute_default_34 = permute_default_35 = None
        getitem_95: "f16[16, 16, 256, 32]" = _scaled_dot_product_flash_attention_default_11[0];  _scaled_dot_product_flash_attention_default_11 = None
        
        # File: /EuroSys25/fwd-compile-print.py:34 in fwd_bert_std, code: h = torch.matmul(probs, v).permute(0, 2, 1, 3).contiguous()
        permute_4: "f16[16, 256, 16, 32]" = torch.ops.aten.permute.default(getitem_95, [0, 2, 1, 3]);  getitem_95 = None
        
        # File: /EuroSys25/fwd-compile-print.py:38 in fwd_bert_std, code: hidden_states = h.view(new_context_layer_shape)
        view_11: "f16[16, 256, 512]" = torch.ops.aten.reshape.default(permute_4, [16, 256, 512]);  permute_4 = None
        
        # File: /EuroSys25/fwd-compile-print.py:42 in fwd_bert_std, code: hidden_states = torch.matmul(hidden_states, attr_output_kernel[layer]) + attr_output_bias[layer]
        view_12: "f16[4096, 512]" = torch.ops.aten.reshape.default(view_11, [4096, 512]);  view_11 = None
        mm_1: "f16[4096, 512]" = torch.ops.aten.mm.default(view_12, arg3_1);  view_12 = arg3_1 = None
        view_13: "f16[16, 256, 512]" = torch.ops.aten.reshape.default(mm_1, [16, 256, 512]);  mm_1 = None
        add_1: "f16[16, 256, 512]" = torch.ops.aten.add.Tensor(view_13, arg4_1);  view_13 = arg4_1 = None
        
        # File: /EuroSys25/fwd-compile-print.py:43 in fwd_bert_std, code: hidden_states = hidden_states + input_tensor  # 残差连接
        add_2: "f16[16, 256, 512]" = torch.ops.aten.add.Tensor(add_1, arg0_1);  add_1 = arg0_1 = None
        
        # File: /EuroSys25/fwd-compile-print.py:46 in fwd_bert_std, code: hidden_states = F.layer_norm(hidden_states, (hidden_dim, ),
        convert_element_type_10: "f32[16, 256, 512]" = torch.ops.prims.convert_element_type.default(add_2, torch.float32)
        var_mean = torch.ops.aten.var_mean.correction(convert_element_type_10, [2], correction = 0, keepdim = True);  convert_element_type_10 = None
        getitem_3: "f32[16, 256, 1]" = var_mean[0]
        getitem_4: "f32[16, 256, 1]" = var_mean[1];  var_mean = None
        sub_1: "f32[16, 256, 512]" = torch.ops.aten.sub.Tensor(add_2, getitem_4);  add_2 = getitem_4 = None
        add_3: "f32[16, 256, 1]" = torch.ops.aten.add.Tensor(getitem_3, 1e-05);  getitem_3 = None
        rsqrt: "f32[16, 256, 1]" = torch.ops.aten.rsqrt.default(add_3);  add_3 = None
        mul: "f32[16, 256, 512]" = torch.ops.aten.mul.Tensor(sub_1, rsqrt);  sub_1 = rsqrt = None
        mul_1: "f32[16, 256, 512]" = torch.ops.aten.mul.Tensor(mul, arg5_1);  mul = arg5_1 = None
        add_4: "f32[16, 256, 512]" = torch.ops.aten.add.Tensor(mul_1, arg6_1);  mul_1 = arg6_1 = None
        convert_element_type_11: "f16[16, 256, 512]" = torch.ops.prims.convert_element_type.default(add_4, torch.float16);  add_4 = None
        
        # File: /EuroSys25/fwd-compile-print.py:51 in fwd_bert_std, code: hidden_states = torch.matmul(hidden_states, inter_kernel[layer]) + inter_bias[layer]
        view_14: "f16[4096, 512]" = torch.ops.aten.reshape.default(convert_element_type_11, [4096, 512])
        mm_2: "f16[4096, 2048]" = torch.ops.aten.mm.default(view_14, arg7_1);  view_14 = arg7_1 = None
        view_15: "f16[16, 256, 2048]" = torch.ops.aten.reshape.default(mm_2, [16, 256, 2048]);  mm_2 = None
        add_5: "f16[16, 256, 2048]" = torch.ops.aten.add.Tensor(view_15, arg8_1);  view_15 = arg8_1 = None
        
        # File: /EuroSys25/fwd-compile-print.py:52 in fwd_bert_std, code: hidden_states = F.gelu(hidden_states)  #激活函数
        convert_element_type_14: "f32[16, 256, 2048]" = torch.ops.prims.convert_element_type.default(add_5, torch.float32);  add_5 = None
        mul_2: "f32[16, 256, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_14, 0.5)
        mul_3: "f32[16, 256, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_14, 0.7071067811865476);  convert_element_type_14 = None
        erf: "f32[16, 256, 2048]" = torch.ops.aten.erf.default(mul_3);  mul_3 = None
        add_6: "f32[16, 256, 2048]" = torch.ops.aten.add.Tensor(erf, 1);  erf = None
        mul_4: "f32[16, 256, 2048]" = torch.ops.aten.mul.Tensor(mul_2, add_6);  mul_2 = add_6 = None
        convert_element_type_15: "f16[16, 256, 2048]" = torch.ops.prims.convert_element_type.default(mul_4, torch.float16);  mul_4 = None
        
        # File: /EuroSys25/fwd-compile-print.py:54 in fwd_bert_std, code: hidden_states = torch.matmul(hidden_states, output_kernel[layer]) + output_bias[layer]
        view_16: "f16[4096, 2048]" = torch.ops.aten.reshape.default(convert_element_type_15, [4096, 2048]);  convert_element_type_15 = None
        mm_3: "f16[4096, 512]" = torch.ops.aten.mm.default(view_16, arg9_1);  view_16 = arg9_1 = None
        view_17: "f16[16, 256, 512]" = torch.ops.aten.reshape.default(mm_3, [16, 256, 512]);  mm_3 = None
        add_7: "f16[16, 256, 512]" = torch.ops.aten.add.Tensor(view_17, arg10_1);  view_17 = arg10_1 = None
        
        # File: /EuroSys25/fwd-compile-print.py:55 in fwd_bert_std, code: hidden_states = hidden_states + residual  #残差连接
        add_8: "f16[16, 256, 512]" = torch.ops.aten.add.Tensor(add_7, convert_element_type_11);  add_7 = convert_element_type_11 = None
        
        # File: /EuroSys25/fwd-compile-print.py:58 in fwd_bert_std, code: hidden_states = F.layer_norm(hidden_states, (hidden_dim, ),
        convert_element_type_18: "f32[16, 256, 512]" = torch.ops.prims.convert_element_type.default(add_8, torch.float32)
        var_mean_1 = torch.ops.aten.var_mean.correction(convert_element_type_18, [2], correction = 0, keepdim = True);  convert_element_type_18 = None
        getitem_5: "f32[16, 256, 1]" = var_mean_1[0]
        getitem_6: "f32[16, 256, 1]" = var_mean_1[1];  var_mean_1 = None
        sub_2: "f32[16, 256, 512]" = torch.ops.aten.sub.Tensor(add_8, getitem_6);  add_8 = getitem_6 = None
        add_9: "f32[16, 256, 1]" = torch.ops.aten.add.Tensor(getitem_5, 1e-05);  getitem_5 = None
        rsqrt_1: "f32[16, 256, 1]" = torch.ops.aten.rsqrt.default(add_9);  add_9 = None
        mul_5: "f32[16, 256, 512]" = torch.ops.aten.mul.Tensor(sub_2, rsqrt_1);  sub_2 = rsqrt_1 = None
        mul_6: "f32[16, 256, 512]" = torch.ops.aten.mul.Tensor(mul_5, arg11_1);  mul_5 = arg11_1 = None
        add_10: "f32[16, 256, 512]" = torch.ops.aten.add.Tensor(mul_6, arg12_1);  mul_6 = arg12_1 = None
        convert_element_type_19: "f16[16, 256, 512]" = torch.ops.prims.convert_element_type.default(add_10, torch.float16);  add_10 = None
        
        # File: /EuroSys25/fwd-compile-print.py:22 in fwd_bert_std, code: qkv = torch.matmul(hidden_states, qkv_kernel[layer]) + qkv_bias[layer]
        view_18: "f16[4096, 512]" = torch.ops.aten.reshape.default(convert_element_type_19, [4096, 512])
        mm_4: "f16[4096, 1536]" = torch.ops.aten.mm.default(view_18, arg13_1);  view_18 = arg13_1 = None
        view_19: "f16[16, 256, 1536]" = torch.ops.aten.reshape.default(mm_4, [16, 256, 1536]);  mm_4 = None
        add_11: "f16[16, 256, 1536]" = torch.ops.aten.add.Tensor(view_19, arg14_1);  view_19 = arg14_1 = None
        
        # File: /EuroSys25/fwd-compile-print.py:23 in fwd_bert_std, code: q, k, v = qkv.chunk(3, dim=-1)
        split_1 = torch.ops.aten.split.Tensor(add_11, 512, -1);  add_11 = None
        getitem_7: "f16[16, 256, 512]" = split_1[0]
        getitem_8: "f16[16, 256, 512]" = split_1[1]
        getitem_9: "f16[16, 256, 512]" = split_1[2];  split_1 = None
        
        # File: /EuroSys25/utils/utils.py:47 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_20: "f16[16, 256, 16, 32]" = torch.ops.aten.reshape.default(getitem_7, [16, 256, 16, 32]);  getitem_7 = None
        
        # No stacktrace found for following nodes
        permute_default_30: "f16[16, 16, 256, 32]" = torch.ops.aten.permute.default(view_20, [0, 2, 1, 3]);  view_20 = None
        
        # File: /EuroSys25/utils/utils.py:47 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_21: "f16[16, 256, 16, 32]" = torch.ops.aten.reshape.default(getitem_8, [16, 256, 16, 32]);  getitem_8 = None
        
        # No stacktrace found for following nodes
        permute_default_31: "f16[16, 16, 256, 32]" = torch.ops.aten.permute.default(view_21, [0, 2, 1, 3]);  view_21 = None
        
        # File: /EuroSys25/utils/utils.py:47 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_22: "f16[16, 256, 16, 32]" = torch.ops.aten.reshape.default(getitem_9, [16, 256, 16, 32]);  getitem_9 = None
        
        # No stacktrace found for following nodes
        permute_default_32: "f16[16, 16, 256, 32]" = torch.ops.aten.permute.default(view_22, [0, 2, 1, 3]);  view_22 = None
        _scaled_dot_product_flash_attention_default_10 = torch.ops.aten._scaled_dot_product_flash_attention.default(permute_default_30, permute_default_31, permute_default_32, scale = 0.17677669529663687);  permute_default_30 = permute_default_31 = permute_default_32 = None
        getitem_94: "f16[16, 16, 256, 32]" = _scaled_dot_product_flash_attention_default_10[0];  _scaled_dot_product_flash_attention_default_10 = None
        
        # File: /EuroSys25/fwd-compile-print.py:34 in fwd_bert_std, code: h = torch.matmul(probs, v).permute(0, 2, 1, 3).contiguous()
        permute_9: "f16[16, 256, 16, 32]" = torch.ops.aten.permute.default(getitem_94, [0, 2, 1, 3]);  getitem_94 = None
        
        # File: /EuroSys25/fwd-compile-print.py:38 in fwd_bert_std, code: hidden_states = h.view(new_context_layer_shape)
        view_29: "f16[16, 256, 512]" = torch.ops.aten.reshape.default(permute_9, [16, 256, 512]);  permute_9 = None
        
        # File: /EuroSys25/fwd-compile-print.py:42 in fwd_bert_std, code: hidden_states = torch.matmul(hidden_states, attr_output_kernel[layer]) + attr_output_bias[layer]
        view_30: "f16[4096, 512]" = torch.ops.aten.reshape.default(view_29, [4096, 512]);  view_29 = None
        mm_5: "f16[4096, 512]" = torch.ops.aten.mm.default(view_30, arg15_1);  view_30 = arg15_1 = None
        view_31: "f16[16, 256, 512]" = torch.ops.aten.reshape.default(mm_5, [16, 256, 512]);  mm_5 = None
        add_12: "f16[16, 256, 512]" = torch.ops.aten.add.Tensor(view_31, arg16_1);  view_31 = arg16_1 = None
        
        # File: /EuroSys25/fwd-compile-print.py:43 in fwd_bert_std, code: hidden_states = hidden_states + input_tensor  # 残差连接
        add_13: "f16[16, 256, 512]" = torch.ops.aten.add.Tensor(add_12, convert_element_type_19);  add_12 = None
        
        # File: /EuroSys25/fwd-compile-print.py:46 in fwd_bert_std, code: hidden_states = F.layer_norm(hidden_states, (hidden_dim, ),
        convert_element_type_30: "f32[16, 256, 512]" = torch.ops.prims.convert_element_type.default(add_13, torch.float32)
        var_mean_2 = torch.ops.aten.var_mean.correction(convert_element_type_30, [2], correction = 0, keepdim = True);  convert_element_type_30 = None
        getitem_10: "f32[16, 256, 1]" = var_mean_2[0]
        getitem_11: "f32[16, 256, 1]" = var_mean_2[1];  var_mean_2 = None
        sub_4: "f32[16, 256, 512]" = torch.ops.aten.sub.Tensor(add_13, getitem_11);  add_13 = getitem_11 = None
        add_14: "f32[16, 256, 1]" = torch.ops.aten.add.Tensor(getitem_10, 1e-05);  getitem_10 = None
        rsqrt_2: "f32[16, 256, 1]" = torch.ops.aten.rsqrt.default(add_14);  add_14 = None
        mul_7: "f32[16, 256, 512]" = torch.ops.aten.mul.Tensor(sub_4, rsqrt_2);  sub_4 = rsqrt_2 = None
        mul_8: "f32[16, 256, 512]" = torch.ops.aten.mul.Tensor(mul_7, arg17_1);  mul_7 = arg17_1 = None
        add_15: "f32[16, 256, 512]" = torch.ops.aten.add.Tensor(mul_8, arg18_1);  mul_8 = arg18_1 = None
        convert_element_type_31: "f16[16, 256, 512]" = torch.ops.prims.convert_element_type.default(add_15, torch.float16);  add_15 = None
        
        # File: /EuroSys25/fwd-compile-print.py:51 in fwd_bert_std, code: hidden_states = torch.matmul(hidden_states, inter_kernel[layer]) + inter_bias[layer]
        view_32: "f16[4096, 512]" = torch.ops.aten.reshape.default(convert_element_type_31, [4096, 512])
        mm_6: "f16[4096, 2048]" = torch.ops.aten.mm.default(view_32, arg19_1);  view_32 = arg19_1 = None
        view_33: "f16[16, 256, 2048]" = torch.ops.aten.reshape.default(mm_6, [16, 256, 2048]);  mm_6 = None
        add_16: "f16[16, 256, 2048]" = torch.ops.aten.add.Tensor(view_33, arg20_1);  view_33 = arg20_1 = None
        
        # File: /EuroSys25/fwd-compile-print.py:52 in fwd_bert_std, code: hidden_states = F.gelu(hidden_states)  #激活函数
        convert_element_type_34: "f32[16, 256, 2048]" = torch.ops.prims.convert_element_type.default(add_16, torch.float32);  add_16 = None
        mul_9: "f32[16, 256, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_34, 0.5)
        mul_10: "f32[16, 256, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_34, 0.7071067811865476);  convert_element_type_34 = None
        erf_1: "f32[16, 256, 2048]" = torch.ops.aten.erf.default(mul_10);  mul_10 = None
        add_17: "f32[16, 256, 2048]" = torch.ops.aten.add.Tensor(erf_1, 1);  erf_1 = None
        mul_11: "f32[16, 256, 2048]" = torch.ops.aten.mul.Tensor(mul_9, add_17);  mul_9 = add_17 = None
        convert_element_type_35: "f16[16, 256, 2048]" = torch.ops.prims.convert_element_type.default(mul_11, torch.float16);  mul_11 = None
        
        # File: /EuroSys25/fwd-compile-print.py:54 in fwd_bert_std, code: hidden_states = torch.matmul(hidden_states, output_kernel[layer]) + output_bias[layer]
        view_34: "f16[4096, 2048]" = torch.ops.aten.reshape.default(convert_element_type_35, [4096, 2048]);  convert_element_type_35 = None
        mm_7: "f16[4096, 512]" = torch.ops.aten.mm.default(view_34, arg21_1);  view_34 = arg21_1 = None
        view_35: "f16[16, 256, 512]" = torch.ops.aten.reshape.default(mm_7, [16, 256, 512]);  mm_7 = None
        add_18: "f16[16, 256, 512]" = torch.ops.aten.add.Tensor(view_35, arg22_1);  view_35 = arg22_1 = None
        
        # File: /EuroSys25/fwd-compile-print.py:55 in fwd_bert_std, code: hidden_states = hidden_states + residual  #残差连接
        add_19: "f16[16, 256, 512]" = torch.ops.aten.add.Tensor(add_18, convert_element_type_31);  add_18 = convert_element_type_31 = None
        
        # File: /EuroSys25/fwd-compile-print.py:58 in fwd_bert_std, code: hidden_states = F.layer_norm(hidden_states, (hidden_dim, ),
        convert_element_type_38: "f32[16, 256, 512]" = torch.ops.prims.convert_element_type.default(add_19, torch.float32)
        var_mean_3 = torch.ops.aten.var_mean.correction(convert_element_type_38, [2], correction = 0, keepdim = True);  convert_element_type_38 = None
        getitem_12: "f32[16, 256, 1]" = var_mean_3[0]
        getitem_13: "f32[16, 256, 1]" = var_mean_3[1];  var_mean_3 = None
        sub_5: "f32[16, 256, 512]" = torch.ops.aten.sub.Tensor(add_19, getitem_13);  add_19 = getitem_13 = None
        add_20: "f32[16, 256, 1]" = torch.ops.aten.add.Tensor(getitem_12, 1e-05);  getitem_12 = None
        rsqrt_3: "f32[16, 256, 1]" = torch.ops.aten.rsqrt.default(add_20);  add_20 = None
        mul_12: "f32[16, 256, 512]" = torch.ops.aten.mul.Tensor(sub_5, rsqrt_3);  sub_5 = rsqrt_3 = None
        mul_13: "f32[16, 256, 512]" = torch.ops.aten.mul.Tensor(mul_12, arg23_1);  mul_12 = arg23_1 = None
        add_21: "f32[16, 256, 512]" = torch.ops.aten.add.Tensor(mul_13, arg24_1);  mul_13 = arg24_1 = None
        convert_element_type_39: "f16[16, 256, 512]" = torch.ops.prims.convert_element_type.default(add_21, torch.float16);  add_21 = None
        
        # File: /EuroSys25/fwd-compile-print.py:22 in fwd_bert_std, code: qkv = torch.matmul(hidden_states, qkv_kernel[layer]) + qkv_bias[layer]
        view_36: "f16[4096, 512]" = torch.ops.aten.reshape.default(convert_element_type_39, [4096, 512])
        mm_8: "f16[4096, 1536]" = torch.ops.aten.mm.default(view_36, arg25_1);  view_36 = arg25_1 = None
        view_37: "f16[16, 256, 1536]" = torch.ops.aten.reshape.default(mm_8, [16, 256, 1536]);  mm_8 = None
        add_22: "f16[16, 256, 1536]" = torch.ops.aten.add.Tensor(view_37, arg26_1);  view_37 = arg26_1 = None
        
        # File: /EuroSys25/fwd-compile-print.py:23 in fwd_bert_std, code: q, k, v = qkv.chunk(3, dim=-1)
        split_2 = torch.ops.aten.split.Tensor(add_22, 512, -1);  add_22 = None
        getitem_14: "f16[16, 256, 512]" = split_2[0]
        getitem_15: "f16[16, 256, 512]" = split_2[1]
        getitem_16: "f16[16, 256, 512]" = split_2[2];  split_2 = None
        
        # File: /EuroSys25/utils/utils.py:47 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_38: "f16[16, 256, 16, 32]" = torch.ops.aten.reshape.default(getitem_14, [16, 256, 16, 32]);  getitem_14 = None
        
        # No stacktrace found for following nodes
        permute_default_27: "f16[16, 16, 256, 32]" = torch.ops.aten.permute.default(view_38, [0, 2, 1, 3]);  view_38 = None
        
        # File: /EuroSys25/utils/utils.py:47 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_39: "f16[16, 256, 16, 32]" = torch.ops.aten.reshape.default(getitem_15, [16, 256, 16, 32]);  getitem_15 = None
        
        # No stacktrace found for following nodes
        permute_default_28: "f16[16, 16, 256, 32]" = torch.ops.aten.permute.default(view_39, [0, 2, 1, 3]);  view_39 = None
        
        # File: /EuroSys25/utils/utils.py:47 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_40: "f16[16, 256, 16, 32]" = torch.ops.aten.reshape.default(getitem_16, [16, 256, 16, 32]);  getitem_16 = None
        
        # No stacktrace found for following nodes
        permute_default_29: "f16[16, 16, 256, 32]" = torch.ops.aten.permute.default(view_40, [0, 2, 1, 3]);  view_40 = None
        _scaled_dot_product_flash_attention_default_9 = torch.ops.aten._scaled_dot_product_flash_attention.default(permute_default_27, permute_default_28, permute_default_29, scale = 0.17677669529663687);  permute_default_27 = permute_default_28 = permute_default_29 = None
        getitem_93: "f16[16, 16, 256, 32]" = _scaled_dot_product_flash_attention_default_9[0];  _scaled_dot_product_flash_attention_default_9 = None
        
        # File: /EuroSys25/fwd-compile-print.py:34 in fwd_bert_std, code: h = torch.matmul(probs, v).permute(0, 2, 1, 3).contiguous()
        permute_14: "f16[16, 256, 16, 32]" = torch.ops.aten.permute.default(getitem_93, [0, 2, 1, 3]);  getitem_93 = None
        
        # File: /EuroSys25/fwd-compile-print.py:38 in fwd_bert_std, code: hidden_states = h.view(new_context_layer_shape)
        view_47: "f16[16, 256, 512]" = torch.ops.aten.reshape.default(permute_14, [16, 256, 512]);  permute_14 = None
        
        # File: /EuroSys25/fwd-compile-print.py:42 in fwd_bert_std, code: hidden_states = torch.matmul(hidden_states, attr_output_kernel[layer]) + attr_output_bias[layer]
        view_48: "f16[4096, 512]" = torch.ops.aten.reshape.default(view_47, [4096, 512]);  view_47 = None
        mm_9: "f16[4096, 512]" = torch.ops.aten.mm.default(view_48, arg27_1);  view_48 = arg27_1 = None
        view_49: "f16[16, 256, 512]" = torch.ops.aten.reshape.default(mm_9, [16, 256, 512]);  mm_9 = None
        add_23: "f16[16, 256, 512]" = torch.ops.aten.add.Tensor(view_49, arg28_1);  view_49 = arg28_1 = None
        
        # File: /EuroSys25/fwd-compile-print.py:43 in fwd_bert_std, code: hidden_states = hidden_states + input_tensor  # 残差连接
        add_24: "f16[16, 256, 512]" = torch.ops.aten.add.Tensor(add_23, convert_element_type_39);  add_23 = None
        
        # File: /EuroSys25/fwd-compile-print.py:46 in fwd_bert_std, code: hidden_states = F.layer_norm(hidden_states, (hidden_dim, ),
        convert_element_type_50: "f32[16, 256, 512]" = torch.ops.prims.convert_element_type.default(add_24, torch.float32)
        var_mean_4 = torch.ops.aten.var_mean.correction(convert_element_type_50, [2], correction = 0, keepdim = True);  convert_element_type_50 = None
        getitem_17: "f32[16, 256, 1]" = var_mean_4[0]
        getitem_18: "f32[16, 256, 1]" = var_mean_4[1];  var_mean_4 = None
        sub_7: "f32[16, 256, 512]" = torch.ops.aten.sub.Tensor(add_24, getitem_18);  add_24 = getitem_18 = None
        add_25: "f32[16, 256, 1]" = torch.ops.aten.add.Tensor(getitem_17, 1e-05);  getitem_17 = None
        rsqrt_4: "f32[16, 256, 1]" = torch.ops.aten.rsqrt.default(add_25);  add_25 = None
        mul_14: "f32[16, 256, 512]" = torch.ops.aten.mul.Tensor(sub_7, rsqrt_4);  sub_7 = rsqrt_4 = None
        mul_15: "f32[16, 256, 512]" = torch.ops.aten.mul.Tensor(mul_14, arg29_1);  mul_14 = arg29_1 = None
        add_26: "f32[16, 256, 512]" = torch.ops.aten.add.Tensor(mul_15, arg30_1);  mul_15 = arg30_1 = None
        convert_element_type_51: "f16[16, 256, 512]" = torch.ops.prims.convert_element_type.default(add_26, torch.float16);  add_26 = None
        
        # File: /EuroSys25/fwd-compile-print.py:51 in fwd_bert_std, code: hidden_states = torch.matmul(hidden_states, inter_kernel[layer]) + inter_bias[layer]
        view_50: "f16[4096, 512]" = torch.ops.aten.reshape.default(convert_element_type_51, [4096, 512])
        mm_10: "f16[4096, 2048]" = torch.ops.aten.mm.default(view_50, arg31_1);  view_50 = arg31_1 = None
        view_51: "f16[16, 256, 2048]" = torch.ops.aten.reshape.default(mm_10, [16, 256, 2048]);  mm_10 = None
        add_27: "f16[16, 256, 2048]" = torch.ops.aten.add.Tensor(view_51, arg32_1);  view_51 = arg32_1 = None
        
        # File: /EuroSys25/fwd-compile-print.py:52 in fwd_bert_std, code: hidden_states = F.gelu(hidden_states)  #激活函数
        convert_element_type_54: "f32[16, 256, 2048]" = torch.ops.prims.convert_element_type.default(add_27, torch.float32);  add_27 = None
        mul_16: "f32[16, 256, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_54, 0.5)
        mul_17: "f32[16, 256, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_54, 0.7071067811865476);  convert_element_type_54 = None
        erf_2: "f32[16, 256, 2048]" = torch.ops.aten.erf.default(mul_17);  mul_17 = None
        add_28: "f32[16, 256, 2048]" = torch.ops.aten.add.Tensor(erf_2, 1);  erf_2 = None
        mul_18: "f32[16, 256, 2048]" = torch.ops.aten.mul.Tensor(mul_16, add_28);  mul_16 = add_28 = None
        convert_element_type_55: "f16[16, 256, 2048]" = torch.ops.prims.convert_element_type.default(mul_18, torch.float16);  mul_18 = None
        
        # File: /EuroSys25/fwd-compile-print.py:54 in fwd_bert_std, code: hidden_states = torch.matmul(hidden_states, output_kernel[layer]) + output_bias[layer]
        view_52: "f16[4096, 2048]" = torch.ops.aten.reshape.default(convert_element_type_55, [4096, 2048]);  convert_element_type_55 = None
        mm_11: "f16[4096, 512]" = torch.ops.aten.mm.default(view_52, arg33_1);  view_52 = arg33_1 = None
        view_53: "f16[16, 256, 512]" = torch.ops.aten.reshape.default(mm_11, [16, 256, 512]);  mm_11 = None
        add_29: "f16[16, 256, 512]" = torch.ops.aten.add.Tensor(view_53, arg34_1);  view_53 = arg34_1 = None
        
        # File: /EuroSys25/fwd-compile-print.py:55 in fwd_bert_std, code: hidden_states = hidden_states + residual  #残差连接
        add_30: "f16[16, 256, 512]" = torch.ops.aten.add.Tensor(add_29, convert_element_type_51);  add_29 = convert_element_type_51 = None
        
        # File: /EuroSys25/fwd-compile-print.py:58 in fwd_bert_std, code: hidden_states = F.layer_norm(hidden_states, (hidden_dim, ),
        convert_element_type_58: "f32[16, 256, 512]" = torch.ops.prims.convert_element_type.default(add_30, torch.float32)
        var_mean_5 = torch.ops.aten.var_mean.correction(convert_element_type_58, [2], correction = 0, keepdim = True);  convert_element_type_58 = None
        getitem_19: "f32[16, 256, 1]" = var_mean_5[0]
        getitem_20: "f32[16, 256, 1]" = var_mean_5[1];  var_mean_5 = None
        sub_8: "f32[16, 256, 512]" = torch.ops.aten.sub.Tensor(add_30, getitem_20);  add_30 = getitem_20 = None
        add_31: "f32[16, 256, 1]" = torch.ops.aten.add.Tensor(getitem_19, 1e-05);  getitem_19 = None
        rsqrt_5: "f32[16, 256, 1]" = torch.ops.aten.rsqrt.default(add_31);  add_31 = None
        mul_19: "f32[16, 256, 512]" = torch.ops.aten.mul.Tensor(sub_8, rsqrt_5);  sub_8 = rsqrt_5 = None
        mul_20: "f32[16, 256, 512]" = torch.ops.aten.mul.Tensor(mul_19, arg35_1);  mul_19 = arg35_1 = None
        add_32: "f32[16, 256, 512]" = torch.ops.aten.add.Tensor(mul_20, arg36_1);  mul_20 = arg36_1 = None
        convert_element_type_59: "f16[16, 256, 512]" = torch.ops.prims.convert_element_type.default(add_32, torch.float16);  add_32 = None
        
        # File: /EuroSys25/fwd-compile-print.py:22 in fwd_bert_std, code: qkv = torch.matmul(hidden_states, qkv_kernel[layer]) + qkv_bias[layer]
        view_54: "f16[4096, 512]" = torch.ops.aten.reshape.default(convert_element_type_59, [4096, 512])
        mm_12: "f16[4096, 1536]" = torch.ops.aten.mm.default(view_54, arg37_1);  view_54 = arg37_1 = None
        view_55: "f16[16, 256, 1536]" = torch.ops.aten.reshape.default(mm_12, [16, 256, 1536]);  mm_12 = None
        add_33: "f16[16, 256, 1536]" = torch.ops.aten.add.Tensor(view_55, arg38_1);  view_55 = arg38_1 = None
        
        # File: /EuroSys25/fwd-compile-print.py:23 in fwd_bert_std, code: q, k, v = qkv.chunk(3, dim=-1)
        split_3 = torch.ops.aten.split.Tensor(add_33, 512, -1);  add_33 = None
        getitem_21: "f16[16, 256, 512]" = split_3[0]
        getitem_22: "f16[16, 256, 512]" = split_3[1]
        getitem_23: "f16[16, 256, 512]" = split_3[2];  split_3 = None
        
        # File: /EuroSys25/utils/utils.py:47 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_56: "f16[16, 256, 16, 32]" = torch.ops.aten.reshape.default(getitem_21, [16, 256, 16, 32]);  getitem_21 = None
        
        # No stacktrace found for following nodes
        permute_default_24: "f16[16, 16, 256, 32]" = torch.ops.aten.permute.default(view_56, [0, 2, 1, 3]);  view_56 = None
        
        # File: /EuroSys25/utils/utils.py:47 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_57: "f16[16, 256, 16, 32]" = torch.ops.aten.reshape.default(getitem_22, [16, 256, 16, 32]);  getitem_22 = None
        
        # No stacktrace found for following nodes
        permute_default_25: "f16[16, 16, 256, 32]" = torch.ops.aten.permute.default(view_57, [0, 2, 1, 3]);  view_57 = None
        
        # File: /EuroSys25/utils/utils.py:47 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_58: "f16[16, 256, 16, 32]" = torch.ops.aten.reshape.default(getitem_23, [16, 256, 16, 32]);  getitem_23 = None
        
        # No stacktrace found for following nodes
        permute_default_26: "f16[16, 16, 256, 32]" = torch.ops.aten.permute.default(view_58, [0, 2, 1, 3]);  view_58 = None
        _scaled_dot_product_flash_attention_default_8 = torch.ops.aten._scaled_dot_product_flash_attention.default(permute_default_24, permute_default_25, permute_default_26, scale = 0.17677669529663687);  permute_default_24 = permute_default_25 = permute_default_26 = None
        getitem_92: "f16[16, 16, 256, 32]" = _scaled_dot_product_flash_attention_default_8[0];  _scaled_dot_product_flash_attention_default_8 = None
        
        # File: /EuroSys25/fwd-compile-print.py:34 in fwd_bert_std, code: h = torch.matmul(probs, v).permute(0, 2, 1, 3).contiguous()
        permute_19: "f16[16, 256, 16, 32]" = torch.ops.aten.permute.default(getitem_92, [0, 2, 1, 3]);  getitem_92 = None
        
        # File: /EuroSys25/fwd-compile-print.py:38 in fwd_bert_std, code: hidden_states = h.view(new_context_layer_shape)
        view_65: "f16[16, 256, 512]" = torch.ops.aten.reshape.default(permute_19, [16, 256, 512]);  permute_19 = None
        
        # File: /EuroSys25/fwd-compile-print.py:42 in fwd_bert_std, code: hidden_states = torch.matmul(hidden_states, attr_output_kernel[layer]) + attr_output_bias[layer]
        view_66: "f16[4096, 512]" = torch.ops.aten.reshape.default(view_65, [4096, 512]);  view_65 = None
        mm_13: "f16[4096, 512]" = torch.ops.aten.mm.default(view_66, arg39_1);  view_66 = arg39_1 = None
        view_67: "f16[16, 256, 512]" = torch.ops.aten.reshape.default(mm_13, [16, 256, 512]);  mm_13 = None
        add_34: "f16[16, 256, 512]" = torch.ops.aten.add.Tensor(view_67, arg40_1);  view_67 = arg40_1 = None
        
        # File: /EuroSys25/fwd-compile-print.py:43 in fwd_bert_std, code: hidden_states = hidden_states + input_tensor  # 残差连接
        add_35: "f16[16, 256, 512]" = torch.ops.aten.add.Tensor(add_34, convert_element_type_59);  add_34 = None
        
        # File: /EuroSys25/fwd-compile-print.py:46 in fwd_bert_std, code: hidden_states = F.layer_norm(hidden_states, (hidden_dim, ),
        convert_element_type_70: "f32[16, 256, 512]" = torch.ops.prims.convert_element_type.default(add_35, torch.float32)
        var_mean_6 = torch.ops.aten.var_mean.correction(convert_element_type_70, [2], correction = 0, keepdim = True);  convert_element_type_70 = None
        getitem_24: "f32[16, 256, 1]" = var_mean_6[0]
        getitem_25: "f32[16, 256, 1]" = var_mean_6[1];  var_mean_6 = None
        sub_10: "f32[16, 256, 512]" = torch.ops.aten.sub.Tensor(add_35, getitem_25);  add_35 = getitem_25 = None
        add_36: "f32[16, 256, 1]" = torch.ops.aten.add.Tensor(getitem_24, 1e-05);  getitem_24 = None
        rsqrt_6: "f32[16, 256, 1]" = torch.ops.aten.rsqrt.default(add_36);  add_36 = None
        mul_21: "f32[16, 256, 512]" = torch.ops.aten.mul.Tensor(sub_10, rsqrt_6);  sub_10 = rsqrt_6 = None
        mul_22: "f32[16, 256, 512]" = torch.ops.aten.mul.Tensor(mul_21, arg41_1);  mul_21 = arg41_1 = None
        add_37: "f32[16, 256, 512]" = torch.ops.aten.add.Tensor(mul_22, arg42_1);  mul_22 = arg42_1 = None
        convert_element_type_71: "f16[16, 256, 512]" = torch.ops.prims.convert_element_type.default(add_37, torch.float16);  add_37 = None
        
        # File: /EuroSys25/fwd-compile-print.py:51 in fwd_bert_std, code: hidden_states = torch.matmul(hidden_states, inter_kernel[layer]) + inter_bias[layer]
        view_68: "f16[4096, 512]" = torch.ops.aten.reshape.default(convert_element_type_71, [4096, 512])
        mm_14: "f16[4096, 2048]" = torch.ops.aten.mm.default(view_68, arg43_1);  view_68 = arg43_1 = None
        view_69: "f16[16, 256, 2048]" = torch.ops.aten.reshape.default(mm_14, [16, 256, 2048]);  mm_14 = None
        add_38: "f16[16, 256, 2048]" = torch.ops.aten.add.Tensor(view_69, arg44_1);  view_69 = arg44_1 = None
        
        # File: /EuroSys25/fwd-compile-print.py:52 in fwd_bert_std, code: hidden_states = F.gelu(hidden_states)  #激活函数
        convert_element_type_74: "f32[16, 256, 2048]" = torch.ops.prims.convert_element_type.default(add_38, torch.float32);  add_38 = None
        mul_23: "f32[16, 256, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_74, 0.5)
        mul_24: "f32[16, 256, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_74, 0.7071067811865476);  convert_element_type_74 = None
        erf_3: "f32[16, 256, 2048]" = torch.ops.aten.erf.default(mul_24);  mul_24 = None
        add_39: "f32[16, 256, 2048]" = torch.ops.aten.add.Tensor(erf_3, 1);  erf_3 = None
        mul_25: "f32[16, 256, 2048]" = torch.ops.aten.mul.Tensor(mul_23, add_39);  mul_23 = add_39 = None
        convert_element_type_75: "f16[16, 256, 2048]" = torch.ops.prims.convert_element_type.default(mul_25, torch.float16);  mul_25 = None
        
        # File: /EuroSys25/fwd-compile-print.py:54 in fwd_bert_std, code: hidden_states = torch.matmul(hidden_states, output_kernel[layer]) + output_bias[layer]
        view_70: "f16[4096, 2048]" = torch.ops.aten.reshape.default(convert_element_type_75, [4096, 2048]);  convert_element_type_75 = None
        mm_15: "f16[4096, 512]" = torch.ops.aten.mm.default(view_70, arg45_1);  view_70 = arg45_1 = None
        view_71: "f16[16, 256, 512]" = torch.ops.aten.reshape.default(mm_15, [16, 256, 512]);  mm_15 = None
        add_40: "f16[16, 256, 512]" = torch.ops.aten.add.Tensor(view_71, arg46_1);  view_71 = arg46_1 = None
        
        # File: /EuroSys25/fwd-compile-print.py:55 in fwd_bert_std, code: hidden_states = hidden_states + residual  #残差连接
        add_41: "f16[16, 256, 512]" = torch.ops.aten.add.Tensor(add_40, convert_element_type_71);  add_40 = convert_element_type_71 = None
        
        # File: /EuroSys25/fwd-compile-print.py:58 in fwd_bert_std, code: hidden_states = F.layer_norm(hidden_states, (hidden_dim, ),
        convert_element_type_78: "f32[16, 256, 512]" = torch.ops.prims.convert_element_type.default(add_41, torch.float32)
        var_mean_7 = torch.ops.aten.var_mean.correction(convert_element_type_78, [2], correction = 0, keepdim = True);  convert_element_type_78 = None
        getitem_26: "f32[16, 256, 1]" = var_mean_7[0]
        getitem_27: "f32[16, 256, 1]" = var_mean_7[1];  var_mean_7 = None
        sub_11: "f32[16, 256, 512]" = torch.ops.aten.sub.Tensor(add_41, getitem_27);  add_41 = getitem_27 = None
        add_42: "f32[16, 256, 1]" = torch.ops.aten.add.Tensor(getitem_26, 1e-05);  getitem_26 = None
        rsqrt_7: "f32[16, 256, 1]" = torch.ops.aten.rsqrt.default(add_42);  add_42 = None
        mul_26: "f32[16, 256, 512]" = torch.ops.aten.mul.Tensor(sub_11, rsqrt_7);  sub_11 = rsqrt_7 = None
        mul_27: "f32[16, 256, 512]" = torch.ops.aten.mul.Tensor(mul_26, arg47_1);  mul_26 = arg47_1 = None
        add_43: "f32[16, 256, 512]" = torch.ops.aten.add.Tensor(mul_27, arg48_1);  mul_27 = arg48_1 = None
        convert_element_type_79: "f16[16, 256, 512]" = torch.ops.prims.convert_element_type.default(add_43, torch.float16);  add_43 = None
        
        # File: /EuroSys25/fwd-compile-print.py:22 in fwd_bert_std, code: qkv = torch.matmul(hidden_states, qkv_kernel[layer]) + qkv_bias[layer]
        view_72: "f16[4096, 512]" = torch.ops.aten.reshape.default(convert_element_type_79, [4096, 512])
        mm_16: "f16[4096, 1536]" = torch.ops.aten.mm.default(view_72, arg49_1);  view_72 = arg49_1 = None
        view_73: "f16[16, 256, 1536]" = torch.ops.aten.reshape.default(mm_16, [16, 256, 1536]);  mm_16 = None
        add_44: "f16[16, 256, 1536]" = torch.ops.aten.add.Tensor(view_73, arg50_1);  view_73 = arg50_1 = None
        
        # File: /EuroSys25/fwd-compile-print.py:23 in fwd_bert_std, code: q, k, v = qkv.chunk(3, dim=-1)
        split_4 = torch.ops.aten.split.Tensor(add_44, 512, -1);  add_44 = None
        getitem_28: "f16[16, 256, 512]" = split_4[0]
        getitem_29: "f16[16, 256, 512]" = split_4[1]
        getitem_30: "f16[16, 256, 512]" = split_4[2];  split_4 = None
        
        # File: /EuroSys25/utils/utils.py:47 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_74: "f16[16, 256, 16, 32]" = torch.ops.aten.reshape.default(getitem_28, [16, 256, 16, 32]);  getitem_28 = None
        
        # No stacktrace found for following nodes
        permute_default_21: "f16[16, 16, 256, 32]" = torch.ops.aten.permute.default(view_74, [0, 2, 1, 3]);  view_74 = None
        
        # File: /EuroSys25/utils/utils.py:47 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_75: "f16[16, 256, 16, 32]" = torch.ops.aten.reshape.default(getitem_29, [16, 256, 16, 32]);  getitem_29 = None
        
        # No stacktrace found for following nodes
        permute_default_22: "f16[16, 16, 256, 32]" = torch.ops.aten.permute.default(view_75, [0, 2, 1, 3]);  view_75 = None
        
        # File: /EuroSys25/utils/utils.py:47 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_76: "f16[16, 256, 16, 32]" = torch.ops.aten.reshape.default(getitem_30, [16, 256, 16, 32]);  getitem_30 = None
        
        # No stacktrace found for following nodes
        permute_default_23: "f16[16, 16, 256, 32]" = torch.ops.aten.permute.default(view_76, [0, 2, 1, 3]);  view_76 = None
        _scaled_dot_product_flash_attention_default_7 = torch.ops.aten._scaled_dot_product_flash_attention.default(permute_default_21, permute_default_22, permute_default_23, scale = 0.17677669529663687);  permute_default_21 = permute_default_22 = permute_default_23 = None
        getitem_91: "f16[16, 16, 256, 32]" = _scaled_dot_product_flash_attention_default_7[0];  _scaled_dot_product_flash_attention_default_7 = None
        
        # File: /EuroSys25/fwd-compile-print.py:34 in fwd_bert_std, code: h = torch.matmul(probs, v).permute(0, 2, 1, 3).contiguous()
        permute_24: "f16[16, 256, 16, 32]" = torch.ops.aten.permute.default(getitem_91, [0, 2, 1, 3]);  getitem_91 = None
        
        # File: /EuroSys25/fwd-compile-print.py:38 in fwd_bert_std, code: hidden_states = h.view(new_context_layer_shape)
        view_83: "f16[16, 256, 512]" = torch.ops.aten.reshape.default(permute_24, [16, 256, 512]);  permute_24 = None
        
        # File: /EuroSys25/fwd-compile-print.py:42 in fwd_bert_std, code: hidden_states = torch.matmul(hidden_states, attr_output_kernel[layer]) + attr_output_bias[layer]
        view_84: "f16[4096, 512]" = torch.ops.aten.reshape.default(view_83, [4096, 512]);  view_83 = None
        mm_17: "f16[4096, 512]" = torch.ops.aten.mm.default(view_84, arg51_1);  view_84 = arg51_1 = None
        view_85: "f16[16, 256, 512]" = torch.ops.aten.reshape.default(mm_17, [16, 256, 512]);  mm_17 = None
        add_45: "f16[16, 256, 512]" = torch.ops.aten.add.Tensor(view_85, arg52_1);  view_85 = arg52_1 = None
        
        # File: /EuroSys25/fwd-compile-print.py:43 in fwd_bert_std, code: hidden_states = hidden_states + input_tensor  # 残差连接
        add_46: "f16[16, 256, 512]" = torch.ops.aten.add.Tensor(add_45, convert_element_type_79);  add_45 = None
        
        # File: /EuroSys25/fwd-compile-print.py:46 in fwd_bert_std, code: hidden_states = F.layer_norm(hidden_states, (hidden_dim, ),
        convert_element_type_90: "f32[16, 256, 512]" = torch.ops.prims.convert_element_type.default(add_46, torch.float32)
        var_mean_8 = torch.ops.aten.var_mean.correction(convert_element_type_90, [2], correction = 0, keepdim = True);  convert_element_type_90 = None
        getitem_31: "f32[16, 256, 1]" = var_mean_8[0]
        getitem_32: "f32[16, 256, 1]" = var_mean_8[1];  var_mean_8 = None
        sub_13: "f32[16, 256, 512]" = torch.ops.aten.sub.Tensor(add_46, getitem_32);  add_46 = getitem_32 = None
        add_47: "f32[16, 256, 1]" = torch.ops.aten.add.Tensor(getitem_31, 1e-05);  getitem_31 = None
        rsqrt_8: "f32[16, 256, 1]" = torch.ops.aten.rsqrt.default(add_47);  add_47 = None
        mul_28: "f32[16, 256, 512]" = torch.ops.aten.mul.Tensor(sub_13, rsqrt_8);  sub_13 = rsqrt_8 = None
        mul_29: "f32[16, 256, 512]" = torch.ops.aten.mul.Tensor(mul_28, arg53_1);  mul_28 = arg53_1 = None
        add_48: "f32[16, 256, 512]" = torch.ops.aten.add.Tensor(mul_29, arg54_1);  mul_29 = arg54_1 = None
        convert_element_type_91: "f16[16, 256, 512]" = torch.ops.prims.convert_element_type.default(add_48, torch.float16);  add_48 = None
        
        # File: /EuroSys25/fwd-compile-print.py:51 in fwd_bert_std, code: hidden_states = torch.matmul(hidden_states, inter_kernel[layer]) + inter_bias[layer]
        view_86: "f16[4096, 512]" = torch.ops.aten.reshape.default(convert_element_type_91, [4096, 512])
        mm_18: "f16[4096, 2048]" = torch.ops.aten.mm.default(view_86, arg55_1);  view_86 = arg55_1 = None
        view_87: "f16[16, 256, 2048]" = torch.ops.aten.reshape.default(mm_18, [16, 256, 2048]);  mm_18 = None
        add_49: "f16[16, 256, 2048]" = torch.ops.aten.add.Tensor(view_87, arg56_1);  view_87 = arg56_1 = None
        
        # File: /EuroSys25/fwd-compile-print.py:52 in fwd_bert_std, code: hidden_states = F.gelu(hidden_states)  #激活函数
        convert_element_type_94: "f32[16, 256, 2048]" = torch.ops.prims.convert_element_type.default(add_49, torch.float32);  add_49 = None
        mul_30: "f32[16, 256, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_94, 0.5)
        mul_31: "f32[16, 256, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_94, 0.7071067811865476);  convert_element_type_94 = None
        erf_4: "f32[16, 256, 2048]" = torch.ops.aten.erf.default(mul_31);  mul_31 = None
        add_50: "f32[16, 256, 2048]" = torch.ops.aten.add.Tensor(erf_4, 1);  erf_4 = None
        mul_32: "f32[16, 256, 2048]" = torch.ops.aten.mul.Tensor(mul_30, add_50);  mul_30 = add_50 = None
        convert_element_type_95: "f16[16, 256, 2048]" = torch.ops.prims.convert_element_type.default(mul_32, torch.float16);  mul_32 = None
        
        # File: /EuroSys25/fwd-compile-print.py:54 in fwd_bert_std, code: hidden_states = torch.matmul(hidden_states, output_kernel[layer]) + output_bias[layer]
        view_88: "f16[4096, 2048]" = torch.ops.aten.reshape.default(convert_element_type_95, [4096, 2048]);  convert_element_type_95 = None
        mm_19: "f16[4096, 512]" = torch.ops.aten.mm.default(view_88, arg57_1);  view_88 = arg57_1 = None
        view_89: "f16[16, 256, 512]" = torch.ops.aten.reshape.default(mm_19, [16, 256, 512]);  mm_19 = None
        add_51: "f16[16, 256, 512]" = torch.ops.aten.add.Tensor(view_89, arg58_1);  view_89 = arg58_1 = None
        
        # File: /EuroSys25/fwd-compile-print.py:55 in fwd_bert_std, code: hidden_states = hidden_states + residual  #残差连接
        add_52: "f16[16, 256, 512]" = torch.ops.aten.add.Tensor(add_51, convert_element_type_91);  add_51 = convert_element_type_91 = None
        
        # File: /EuroSys25/fwd-compile-print.py:58 in fwd_bert_std, code: hidden_states = F.layer_norm(hidden_states, (hidden_dim, ),
        convert_element_type_98: "f32[16, 256, 512]" = torch.ops.prims.convert_element_type.default(add_52, torch.float32)
        var_mean_9 = torch.ops.aten.var_mean.correction(convert_element_type_98, [2], correction = 0, keepdim = True);  convert_element_type_98 = None
        getitem_33: "f32[16, 256, 1]" = var_mean_9[0]
        getitem_34: "f32[16, 256, 1]" = var_mean_9[1];  var_mean_9 = None
        sub_14: "f32[16, 256, 512]" = torch.ops.aten.sub.Tensor(add_52, getitem_34);  add_52 = getitem_34 = None
        add_53: "f32[16, 256, 1]" = torch.ops.aten.add.Tensor(getitem_33, 1e-05);  getitem_33 = None
        rsqrt_9: "f32[16, 256, 1]" = torch.ops.aten.rsqrt.default(add_53);  add_53 = None
        mul_33: "f32[16, 256, 512]" = torch.ops.aten.mul.Tensor(sub_14, rsqrt_9);  sub_14 = rsqrt_9 = None
        mul_34: "f32[16, 256, 512]" = torch.ops.aten.mul.Tensor(mul_33, arg59_1);  mul_33 = arg59_1 = None
        add_54: "f32[16, 256, 512]" = torch.ops.aten.add.Tensor(mul_34, arg60_1);  mul_34 = arg60_1 = None
        convert_element_type_99: "f16[16, 256, 512]" = torch.ops.prims.convert_element_type.default(add_54, torch.float16);  add_54 = None
        
        # File: /EuroSys25/fwd-compile-print.py:22 in fwd_bert_std, code: qkv = torch.matmul(hidden_states, qkv_kernel[layer]) + qkv_bias[layer]
        view_90: "f16[4096, 512]" = torch.ops.aten.reshape.default(convert_element_type_99, [4096, 512])
        mm_20: "f16[4096, 1536]" = torch.ops.aten.mm.default(view_90, arg61_1);  view_90 = arg61_1 = None
        view_91: "f16[16, 256, 1536]" = torch.ops.aten.reshape.default(mm_20, [16, 256, 1536]);  mm_20 = None
        add_55: "f16[16, 256, 1536]" = torch.ops.aten.add.Tensor(view_91, arg62_1);  view_91 = arg62_1 = None
        
        # File: /EuroSys25/fwd-compile-print.py:23 in fwd_bert_std, code: q, k, v = qkv.chunk(3, dim=-1)
        split_5 = torch.ops.aten.split.Tensor(add_55, 512, -1);  add_55 = None
        getitem_35: "f16[16, 256, 512]" = split_5[0]
        getitem_36: "f16[16, 256, 512]" = split_5[1]
        getitem_37: "f16[16, 256, 512]" = split_5[2];  split_5 = None
        
        # File: /EuroSys25/utils/utils.py:47 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_92: "f16[16, 256, 16, 32]" = torch.ops.aten.reshape.default(getitem_35, [16, 256, 16, 32]);  getitem_35 = None
        
        # No stacktrace found for following nodes
        permute_default_18: "f16[16, 16, 256, 32]" = torch.ops.aten.permute.default(view_92, [0, 2, 1, 3]);  view_92 = None
        
        # File: /EuroSys25/utils/utils.py:47 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_93: "f16[16, 256, 16, 32]" = torch.ops.aten.reshape.default(getitem_36, [16, 256, 16, 32]);  getitem_36 = None
        
        # No stacktrace found for following nodes
        permute_default_19: "f16[16, 16, 256, 32]" = torch.ops.aten.permute.default(view_93, [0, 2, 1, 3]);  view_93 = None
        
        # File: /EuroSys25/utils/utils.py:47 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_94: "f16[16, 256, 16, 32]" = torch.ops.aten.reshape.default(getitem_37, [16, 256, 16, 32]);  getitem_37 = None
        
        # No stacktrace found for following nodes
        permute_default_20: "f16[16, 16, 256, 32]" = torch.ops.aten.permute.default(view_94, [0, 2, 1, 3]);  view_94 = None
        _scaled_dot_product_flash_attention_default_6 = torch.ops.aten._scaled_dot_product_flash_attention.default(permute_default_18, permute_default_19, permute_default_20, scale = 0.17677669529663687);  permute_default_18 = permute_default_19 = permute_default_20 = None
        getitem_90: "f16[16, 16, 256, 32]" = _scaled_dot_product_flash_attention_default_6[0];  _scaled_dot_product_flash_attention_default_6 = None
        
        # File: /EuroSys25/fwd-compile-print.py:34 in fwd_bert_std, code: h = torch.matmul(probs, v).permute(0, 2, 1, 3).contiguous()
        permute_29: "f16[16, 256, 16, 32]" = torch.ops.aten.permute.default(getitem_90, [0, 2, 1, 3]);  getitem_90 = None
        
        # File: /EuroSys25/fwd-compile-print.py:38 in fwd_bert_std, code: hidden_states = h.view(new_context_layer_shape)
        view_101: "f16[16, 256, 512]" = torch.ops.aten.reshape.default(permute_29, [16, 256, 512]);  permute_29 = None
        
        # File: /EuroSys25/fwd-compile-print.py:42 in fwd_bert_std, code: hidden_states = torch.matmul(hidden_states, attr_output_kernel[layer]) + attr_output_bias[layer]
        view_102: "f16[4096, 512]" = torch.ops.aten.reshape.default(view_101, [4096, 512]);  view_101 = None
        mm_21: "f16[4096, 512]" = torch.ops.aten.mm.default(view_102, arg63_1);  view_102 = arg63_1 = None
        view_103: "f16[16, 256, 512]" = torch.ops.aten.reshape.default(mm_21, [16, 256, 512]);  mm_21 = None
        add_56: "f16[16, 256, 512]" = torch.ops.aten.add.Tensor(view_103, arg64_1);  view_103 = arg64_1 = None
        
        # File: /EuroSys25/fwd-compile-print.py:43 in fwd_bert_std, code: hidden_states = hidden_states + input_tensor  # 残差连接
        add_57: "f16[16, 256, 512]" = torch.ops.aten.add.Tensor(add_56, convert_element_type_99);  add_56 = None
        
        # File: /EuroSys25/fwd-compile-print.py:46 in fwd_bert_std, code: hidden_states = F.layer_norm(hidden_states, (hidden_dim, ),
        convert_element_type_110: "f32[16, 256, 512]" = torch.ops.prims.convert_element_type.default(add_57, torch.float32)
        var_mean_10 = torch.ops.aten.var_mean.correction(convert_element_type_110, [2], correction = 0, keepdim = True);  convert_element_type_110 = None
        getitem_38: "f32[16, 256, 1]" = var_mean_10[0]
        getitem_39: "f32[16, 256, 1]" = var_mean_10[1];  var_mean_10 = None
        sub_16: "f32[16, 256, 512]" = torch.ops.aten.sub.Tensor(add_57, getitem_39);  add_57 = getitem_39 = None
        add_58: "f32[16, 256, 1]" = torch.ops.aten.add.Tensor(getitem_38, 1e-05);  getitem_38 = None
        rsqrt_10: "f32[16, 256, 1]" = torch.ops.aten.rsqrt.default(add_58);  add_58 = None
        mul_35: "f32[16, 256, 512]" = torch.ops.aten.mul.Tensor(sub_16, rsqrt_10);  sub_16 = rsqrt_10 = None
        mul_36: "f32[16, 256, 512]" = torch.ops.aten.mul.Tensor(mul_35, arg65_1);  mul_35 = arg65_1 = None
        add_59: "f32[16, 256, 512]" = torch.ops.aten.add.Tensor(mul_36, arg66_1);  mul_36 = arg66_1 = None
        convert_element_type_111: "f16[16, 256, 512]" = torch.ops.prims.convert_element_type.default(add_59, torch.float16);  add_59 = None
        
        # File: /EuroSys25/fwd-compile-print.py:51 in fwd_bert_std, code: hidden_states = torch.matmul(hidden_states, inter_kernel[layer]) + inter_bias[layer]
        view_104: "f16[4096, 512]" = torch.ops.aten.reshape.default(convert_element_type_111, [4096, 512])
        mm_22: "f16[4096, 2048]" = torch.ops.aten.mm.default(view_104, arg67_1);  view_104 = arg67_1 = None
        view_105: "f16[16, 256, 2048]" = torch.ops.aten.reshape.default(mm_22, [16, 256, 2048]);  mm_22 = None
        add_60: "f16[16, 256, 2048]" = torch.ops.aten.add.Tensor(view_105, arg68_1);  view_105 = arg68_1 = None
        
        # File: /EuroSys25/fwd-compile-print.py:52 in fwd_bert_std, code: hidden_states = F.gelu(hidden_states)  #激活函数
        convert_element_type_114: "f32[16, 256, 2048]" = torch.ops.prims.convert_element_type.default(add_60, torch.float32);  add_60 = None
        mul_37: "f32[16, 256, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_114, 0.5)
        mul_38: "f32[16, 256, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_114, 0.7071067811865476);  convert_element_type_114 = None
        erf_5: "f32[16, 256, 2048]" = torch.ops.aten.erf.default(mul_38);  mul_38 = None
        add_61: "f32[16, 256, 2048]" = torch.ops.aten.add.Tensor(erf_5, 1);  erf_5 = None
        mul_39: "f32[16, 256, 2048]" = torch.ops.aten.mul.Tensor(mul_37, add_61);  mul_37 = add_61 = None
        convert_element_type_115: "f16[16, 256, 2048]" = torch.ops.prims.convert_element_type.default(mul_39, torch.float16);  mul_39 = None
        
        # File: /EuroSys25/fwd-compile-print.py:54 in fwd_bert_std, code: hidden_states = torch.matmul(hidden_states, output_kernel[layer]) + output_bias[layer]
        view_106: "f16[4096, 2048]" = torch.ops.aten.reshape.default(convert_element_type_115, [4096, 2048]);  convert_element_type_115 = None
        mm_23: "f16[4096, 512]" = torch.ops.aten.mm.default(view_106, arg69_1);  view_106 = arg69_1 = None
        view_107: "f16[16, 256, 512]" = torch.ops.aten.reshape.default(mm_23, [16, 256, 512]);  mm_23 = None
        add_62: "f16[16, 256, 512]" = torch.ops.aten.add.Tensor(view_107, arg70_1);  view_107 = arg70_1 = None
        
        # File: /EuroSys25/fwd-compile-print.py:55 in fwd_bert_std, code: hidden_states = hidden_states + residual  #残差连接
        add_63: "f16[16, 256, 512]" = torch.ops.aten.add.Tensor(add_62, convert_element_type_111);  add_62 = convert_element_type_111 = None
        
        # File: /EuroSys25/fwd-compile-print.py:58 in fwd_bert_std, code: hidden_states = F.layer_norm(hidden_states, (hidden_dim, ),
        convert_element_type_118: "f32[16, 256, 512]" = torch.ops.prims.convert_element_type.default(add_63, torch.float32)
        var_mean_11 = torch.ops.aten.var_mean.correction(convert_element_type_118, [2], correction = 0, keepdim = True);  convert_element_type_118 = None
        getitem_40: "f32[16, 256, 1]" = var_mean_11[0]
        getitem_41: "f32[16, 256, 1]" = var_mean_11[1];  var_mean_11 = None
        sub_17: "f32[16, 256, 512]" = torch.ops.aten.sub.Tensor(add_63, getitem_41);  add_63 = getitem_41 = None
        add_64: "f32[16, 256, 1]" = torch.ops.aten.add.Tensor(getitem_40, 1e-05);  getitem_40 = None
        rsqrt_11: "f32[16, 256, 1]" = torch.ops.aten.rsqrt.default(add_64);  add_64 = None
        mul_40: "f32[16, 256, 512]" = torch.ops.aten.mul.Tensor(sub_17, rsqrt_11);  sub_17 = rsqrt_11 = None
        mul_41: "f32[16, 256, 512]" = torch.ops.aten.mul.Tensor(mul_40, arg71_1);  mul_40 = arg71_1 = None
        add_65: "f32[16, 256, 512]" = torch.ops.aten.add.Tensor(mul_41, arg72_1);  mul_41 = arg72_1 = None
        convert_element_type_119: "f16[16, 256, 512]" = torch.ops.prims.convert_element_type.default(add_65, torch.float16);  add_65 = None
        
        # File: /EuroSys25/fwd-compile-print.py:22 in fwd_bert_std, code: qkv = torch.matmul(hidden_states, qkv_kernel[layer]) + qkv_bias[layer]
        view_108: "f16[4096, 512]" = torch.ops.aten.reshape.default(convert_element_type_119, [4096, 512])
        mm_24: "f16[4096, 1536]" = torch.ops.aten.mm.default(view_108, arg73_1);  view_108 = arg73_1 = None
        view_109: "f16[16, 256, 1536]" = torch.ops.aten.reshape.default(mm_24, [16, 256, 1536]);  mm_24 = None
        add_66: "f16[16, 256, 1536]" = torch.ops.aten.add.Tensor(view_109, arg74_1);  view_109 = arg74_1 = None
        
        # File: /EuroSys25/fwd-compile-print.py:23 in fwd_bert_std, code: q, k, v = qkv.chunk(3, dim=-1)
        split_6 = torch.ops.aten.split.Tensor(add_66, 512, -1);  add_66 = None
        getitem_42: "f16[16, 256, 512]" = split_6[0]
        getitem_43: "f16[16, 256, 512]" = split_6[1]
        getitem_44: "f16[16, 256, 512]" = split_6[2];  split_6 = None
        
        # File: /EuroSys25/utils/utils.py:47 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_110: "f16[16, 256, 16, 32]" = torch.ops.aten.reshape.default(getitem_42, [16, 256, 16, 32]);  getitem_42 = None
        
        # No stacktrace found for following nodes
        permute_default_15: "f16[16, 16, 256, 32]" = torch.ops.aten.permute.default(view_110, [0, 2, 1, 3]);  view_110 = None
        
        # File: /EuroSys25/utils/utils.py:47 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_111: "f16[16, 256, 16, 32]" = torch.ops.aten.reshape.default(getitem_43, [16, 256, 16, 32]);  getitem_43 = None
        
        # No stacktrace found for following nodes
        permute_default_16: "f16[16, 16, 256, 32]" = torch.ops.aten.permute.default(view_111, [0, 2, 1, 3]);  view_111 = None
        
        # File: /EuroSys25/utils/utils.py:47 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_112: "f16[16, 256, 16, 32]" = torch.ops.aten.reshape.default(getitem_44, [16, 256, 16, 32]);  getitem_44 = None
        
        # No stacktrace found for following nodes
        permute_default_17: "f16[16, 16, 256, 32]" = torch.ops.aten.permute.default(view_112, [0, 2, 1, 3]);  view_112 = None
        _scaled_dot_product_flash_attention_default_5 = torch.ops.aten._scaled_dot_product_flash_attention.default(permute_default_15, permute_default_16, permute_default_17, scale = 0.17677669529663687);  permute_default_15 = permute_default_16 = permute_default_17 = None
        getitem_89: "f16[16, 16, 256, 32]" = _scaled_dot_product_flash_attention_default_5[0];  _scaled_dot_product_flash_attention_default_5 = None
        
        # File: /EuroSys25/fwd-compile-print.py:34 in fwd_bert_std, code: h = torch.matmul(probs, v).permute(0, 2, 1, 3).contiguous()
        permute_34: "f16[16, 256, 16, 32]" = torch.ops.aten.permute.default(getitem_89, [0, 2, 1, 3]);  getitem_89 = None
        
        # File: /EuroSys25/fwd-compile-print.py:38 in fwd_bert_std, code: hidden_states = h.view(new_context_layer_shape)
        view_119: "f16[16, 256, 512]" = torch.ops.aten.reshape.default(permute_34, [16, 256, 512]);  permute_34 = None
        
        # File: /EuroSys25/fwd-compile-print.py:42 in fwd_bert_std, code: hidden_states = torch.matmul(hidden_states, attr_output_kernel[layer]) + attr_output_bias[layer]
        view_120: "f16[4096, 512]" = torch.ops.aten.reshape.default(view_119, [4096, 512]);  view_119 = None
        mm_25: "f16[4096, 512]" = torch.ops.aten.mm.default(view_120, arg75_1);  view_120 = arg75_1 = None
        view_121: "f16[16, 256, 512]" = torch.ops.aten.reshape.default(mm_25, [16, 256, 512]);  mm_25 = None
        add_67: "f16[16, 256, 512]" = torch.ops.aten.add.Tensor(view_121, arg76_1);  view_121 = arg76_1 = None
        
        # File: /EuroSys25/fwd-compile-print.py:43 in fwd_bert_std, code: hidden_states = hidden_states + input_tensor  # 残差连接
        add_68: "f16[16, 256, 512]" = torch.ops.aten.add.Tensor(add_67, convert_element_type_119);  add_67 = None
        
        # File: /EuroSys25/fwd-compile-print.py:46 in fwd_bert_std, code: hidden_states = F.layer_norm(hidden_states, (hidden_dim, ),
        convert_element_type_130: "f32[16, 256, 512]" = torch.ops.prims.convert_element_type.default(add_68, torch.float32)
        var_mean_12 = torch.ops.aten.var_mean.correction(convert_element_type_130, [2], correction = 0, keepdim = True);  convert_element_type_130 = None
        getitem_45: "f32[16, 256, 1]" = var_mean_12[0]
        getitem_46: "f32[16, 256, 1]" = var_mean_12[1];  var_mean_12 = None
        sub_19: "f32[16, 256, 512]" = torch.ops.aten.sub.Tensor(add_68, getitem_46);  add_68 = getitem_46 = None
        add_69: "f32[16, 256, 1]" = torch.ops.aten.add.Tensor(getitem_45, 1e-05);  getitem_45 = None
        rsqrt_12: "f32[16, 256, 1]" = torch.ops.aten.rsqrt.default(add_69);  add_69 = None
        mul_42: "f32[16, 256, 512]" = torch.ops.aten.mul.Tensor(sub_19, rsqrt_12);  sub_19 = rsqrt_12 = None
        mul_43: "f32[16, 256, 512]" = torch.ops.aten.mul.Tensor(mul_42, arg77_1);  mul_42 = arg77_1 = None
        add_70: "f32[16, 256, 512]" = torch.ops.aten.add.Tensor(mul_43, arg78_1);  mul_43 = arg78_1 = None
        convert_element_type_131: "f16[16, 256, 512]" = torch.ops.prims.convert_element_type.default(add_70, torch.float16);  add_70 = None
        
        # File: /EuroSys25/fwd-compile-print.py:51 in fwd_bert_std, code: hidden_states = torch.matmul(hidden_states, inter_kernel[layer]) + inter_bias[layer]
        view_122: "f16[4096, 512]" = torch.ops.aten.reshape.default(convert_element_type_131, [4096, 512])
        mm_26: "f16[4096, 2048]" = torch.ops.aten.mm.default(view_122, arg79_1);  view_122 = arg79_1 = None
        view_123: "f16[16, 256, 2048]" = torch.ops.aten.reshape.default(mm_26, [16, 256, 2048]);  mm_26 = None
        add_71: "f16[16, 256, 2048]" = torch.ops.aten.add.Tensor(view_123, arg80_1);  view_123 = arg80_1 = None
        
        # File: /EuroSys25/fwd-compile-print.py:52 in fwd_bert_std, code: hidden_states = F.gelu(hidden_states)  #激活函数
        convert_element_type_134: "f32[16, 256, 2048]" = torch.ops.prims.convert_element_type.default(add_71, torch.float32);  add_71 = None
        mul_44: "f32[16, 256, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_134, 0.5)
        mul_45: "f32[16, 256, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_134, 0.7071067811865476);  convert_element_type_134 = None
        erf_6: "f32[16, 256, 2048]" = torch.ops.aten.erf.default(mul_45);  mul_45 = None
        add_72: "f32[16, 256, 2048]" = torch.ops.aten.add.Tensor(erf_6, 1);  erf_6 = None
        mul_46: "f32[16, 256, 2048]" = torch.ops.aten.mul.Tensor(mul_44, add_72);  mul_44 = add_72 = None
        convert_element_type_135: "f16[16, 256, 2048]" = torch.ops.prims.convert_element_type.default(mul_46, torch.float16);  mul_46 = None
        
        # File: /EuroSys25/fwd-compile-print.py:54 in fwd_bert_std, code: hidden_states = torch.matmul(hidden_states, output_kernel[layer]) + output_bias[layer]
        view_124: "f16[4096, 2048]" = torch.ops.aten.reshape.default(convert_element_type_135, [4096, 2048]);  convert_element_type_135 = None
        mm_27: "f16[4096, 512]" = torch.ops.aten.mm.default(view_124, arg81_1);  view_124 = arg81_1 = None
        view_125: "f16[16, 256, 512]" = torch.ops.aten.reshape.default(mm_27, [16, 256, 512]);  mm_27 = None
        add_73: "f16[16, 256, 512]" = torch.ops.aten.add.Tensor(view_125, arg82_1);  view_125 = arg82_1 = None
        
        # File: /EuroSys25/fwd-compile-print.py:55 in fwd_bert_std, code: hidden_states = hidden_states + residual  #残差连接
        add_74: "f16[16, 256, 512]" = torch.ops.aten.add.Tensor(add_73, convert_element_type_131);  add_73 = convert_element_type_131 = None
        
        # File: /EuroSys25/fwd-compile-print.py:58 in fwd_bert_std, code: hidden_states = F.layer_norm(hidden_states, (hidden_dim, ),
        convert_element_type_138: "f32[16, 256, 512]" = torch.ops.prims.convert_element_type.default(add_74, torch.float32)
        var_mean_13 = torch.ops.aten.var_mean.correction(convert_element_type_138, [2], correction = 0, keepdim = True);  convert_element_type_138 = None
        getitem_47: "f32[16, 256, 1]" = var_mean_13[0]
        getitem_48: "f32[16, 256, 1]" = var_mean_13[1];  var_mean_13 = None
        sub_20: "f32[16, 256, 512]" = torch.ops.aten.sub.Tensor(add_74, getitem_48);  add_74 = getitem_48 = None
        add_75: "f32[16, 256, 1]" = torch.ops.aten.add.Tensor(getitem_47, 1e-05);  getitem_47 = None
        rsqrt_13: "f32[16, 256, 1]" = torch.ops.aten.rsqrt.default(add_75);  add_75 = None
        mul_47: "f32[16, 256, 512]" = torch.ops.aten.mul.Tensor(sub_20, rsqrt_13);  sub_20 = rsqrt_13 = None
        mul_48: "f32[16, 256, 512]" = torch.ops.aten.mul.Tensor(mul_47, arg83_1);  mul_47 = arg83_1 = None
        add_76: "f32[16, 256, 512]" = torch.ops.aten.add.Tensor(mul_48, arg84_1);  mul_48 = arg84_1 = None
        convert_element_type_139: "f16[16, 256, 512]" = torch.ops.prims.convert_element_type.default(add_76, torch.float16);  add_76 = None
        
        # File: /EuroSys25/fwd-compile-print.py:22 in fwd_bert_std, code: qkv = torch.matmul(hidden_states, qkv_kernel[layer]) + qkv_bias[layer]
        view_126: "f16[4096, 512]" = torch.ops.aten.reshape.default(convert_element_type_139, [4096, 512])
        mm_28: "f16[4096, 1536]" = torch.ops.aten.mm.default(view_126, arg85_1);  view_126 = arg85_1 = None
        view_127: "f16[16, 256, 1536]" = torch.ops.aten.reshape.default(mm_28, [16, 256, 1536]);  mm_28 = None
        add_77: "f16[16, 256, 1536]" = torch.ops.aten.add.Tensor(view_127, arg86_1);  view_127 = arg86_1 = None
        
        # File: /EuroSys25/fwd-compile-print.py:23 in fwd_bert_std, code: q, k, v = qkv.chunk(3, dim=-1)
        split_7 = torch.ops.aten.split.Tensor(add_77, 512, -1);  add_77 = None
        getitem_49: "f16[16, 256, 512]" = split_7[0]
        getitem_50: "f16[16, 256, 512]" = split_7[1]
        getitem_51: "f16[16, 256, 512]" = split_7[2];  split_7 = None
        
        # File: /EuroSys25/utils/utils.py:47 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_128: "f16[16, 256, 16, 32]" = torch.ops.aten.reshape.default(getitem_49, [16, 256, 16, 32]);  getitem_49 = None
        
        # No stacktrace found for following nodes
        permute_default_12: "f16[16, 16, 256, 32]" = torch.ops.aten.permute.default(view_128, [0, 2, 1, 3]);  view_128 = None
        
        # File: /EuroSys25/utils/utils.py:47 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_129: "f16[16, 256, 16, 32]" = torch.ops.aten.reshape.default(getitem_50, [16, 256, 16, 32]);  getitem_50 = None
        
        # No stacktrace found for following nodes
        permute_default_13: "f16[16, 16, 256, 32]" = torch.ops.aten.permute.default(view_129, [0, 2, 1, 3]);  view_129 = None
        
        # File: /EuroSys25/utils/utils.py:47 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_130: "f16[16, 256, 16, 32]" = torch.ops.aten.reshape.default(getitem_51, [16, 256, 16, 32]);  getitem_51 = None
        
        # No stacktrace found for following nodes
        permute_default_14: "f16[16, 16, 256, 32]" = torch.ops.aten.permute.default(view_130, [0, 2, 1, 3]);  view_130 = None
        _scaled_dot_product_flash_attention_default_4 = torch.ops.aten._scaled_dot_product_flash_attention.default(permute_default_12, permute_default_13, permute_default_14, scale = 0.17677669529663687);  permute_default_12 = permute_default_13 = permute_default_14 = None
        getitem_88: "f16[16, 16, 256, 32]" = _scaled_dot_product_flash_attention_default_4[0];  _scaled_dot_product_flash_attention_default_4 = None
        
        # File: /EuroSys25/fwd-compile-print.py:34 in fwd_bert_std, code: h = torch.matmul(probs, v).permute(0, 2, 1, 3).contiguous()
        permute_39: "f16[16, 256, 16, 32]" = torch.ops.aten.permute.default(getitem_88, [0, 2, 1, 3]);  getitem_88 = None
        
        # File: /EuroSys25/fwd-compile-print.py:38 in fwd_bert_std, code: hidden_states = h.view(new_context_layer_shape)
        view_137: "f16[16, 256, 512]" = torch.ops.aten.reshape.default(permute_39, [16, 256, 512]);  permute_39 = None
        
        # File: /EuroSys25/fwd-compile-print.py:42 in fwd_bert_std, code: hidden_states = torch.matmul(hidden_states, attr_output_kernel[layer]) + attr_output_bias[layer]
        view_138: "f16[4096, 512]" = torch.ops.aten.reshape.default(view_137, [4096, 512]);  view_137 = None
        mm_29: "f16[4096, 512]" = torch.ops.aten.mm.default(view_138, arg87_1);  view_138 = arg87_1 = None
        view_139: "f16[16, 256, 512]" = torch.ops.aten.reshape.default(mm_29, [16, 256, 512]);  mm_29 = None
        add_78: "f16[16, 256, 512]" = torch.ops.aten.add.Tensor(view_139, arg88_1);  view_139 = arg88_1 = None
        
        # File: /EuroSys25/fwd-compile-print.py:43 in fwd_bert_std, code: hidden_states = hidden_states + input_tensor  # 残差连接
        add_79: "f16[16, 256, 512]" = torch.ops.aten.add.Tensor(add_78, convert_element_type_139);  add_78 = None
        
        # File: /EuroSys25/fwd-compile-print.py:46 in fwd_bert_std, code: hidden_states = F.layer_norm(hidden_states, (hidden_dim, ),
        convert_element_type_150: "f32[16, 256, 512]" = torch.ops.prims.convert_element_type.default(add_79, torch.float32)
        var_mean_14 = torch.ops.aten.var_mean.correction(convert_element_type_150, [2], correction = 0, keepdim = True);  convert_element_type_150 = None
        getitem_52: "f32[16, 256, 1]" = var_mean_14[0]
        getitem_53: "f32[16, 256, 1]" = var_mean_14[1];  var_mean_14 = None
        sub_22: "f32[16, 256, 512]" = torch.ops.aten.sub.Tensor(add_79, getitem_53);  add_79 = getitem_53 = None
        add_80: "f32[16, 256, 1]" = torch.ops.aten.add.Tensor(getitem_52, 1e-05);  getitem_52 = None
        rsqrt_14: "f32[16, 256, 1]" = torch.ops.aten.rsqrt.default(add_80);  add_80 = None
        mul_49: "f32[16, 256, 512]" = torch.ops.aten.mul.Tensor(sub_22, rsqrt_14);  sub_22 = rsqrt_14 = None
        mul_50: "f32[16, 256, 512]" = torch.ops.aten.mul.Tensor(mul_49, arg89_1);  mul_49 = arg89_1 = None
        add_81: "f32[16, 256, 512]" = torch.ops.aten.add.Tensor(mul_50, arg90_1);  mul_50 = arg90_1 = None
        convert_element_type_151: "f16[16, 256, 512]" = torch.ops.prims.convert_element_type.default(add_81, torch.float16);  add_81 = None
        
        # File: /EuroSys25/fwd-compile-print.py:51 in fwd_bert_std, code: hidden_states = torch.matmul(hidden_states, inter_kernel[layer]) + inter_bias[layer]
        view_140: "f16[4096, 512]" = torch.ops.aten.reshape.default(convert_element_type_151, [4096, 512])
        mm_30: "f16[4096, 2048]" = torch.ops.aten.mm.default(view_140, arg91_1);  view_140 = arg91_1 = None
        view_141: "f16[16, 256, 2048]" = torch.ops.aten.reshape.default(mm_30, [16, 256, 2048]);  mm_30 = None
        add_82: "f16[16, 256, 2048]" = torch.ops.aten.add.Tensor(view_141, arg92_1);  view_141 = arg92_1 = None
        
        # File: /EuroSys25/fwd-compile-print.py:52 in fwd_bert_std, code: hidden_states = F.gelu(hidden_states)  #激活函数
        convert_element_type_154: "f32[16, 256, 2048]" = torch.ops.prims.convert_element_type.default(add_82, torch.float32);  add_82 = None
        mul_51: "f32[16, 256, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_154, 0.5)
        mul_52: "f32[16, 256, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_154, 0.7071067811865476);  convert_element_type_154 = None
        erf_7: "f32[16, 256, 2048]" = torch.ops.aten.erf.default(mul_52);  mul_52 = None
        add_83: "f32[16, 256, 2048]" = torch.ops.aten.add.Tensor(erf_7, 1);  erf_7 = None
        mul_53: "f32[16, 256, 2048]" = torch.ops.aten.mul.Tensor(mul_51, add_83);  mul_51 = add_83 = None
        convert_element_type_155: "f16[16, 256, 2048]" = torch.ops.prims.convert_element_type.default(mul_53, torch.float16);  mul_53 = None
        
        # File: /EuroSys25/fwd-compile-print.py:54 in fwd_bert_std, code: hidden_states = torch.matmul(hidden_states, output_kernel[layer]) + output_bias[layer]
        view_142: "f16[4096, 2048]" = torch.ops.aten.reshape.default(convert_element_type_155, [4096, 2048]);  convert_element_type_155 = None
        mm_31: "f16[4096, 512]" = torch.ops.aten.mm.default(view_142, arg93_1);  view_142 = arg93_1 = None
        view_143: "f16[16, 256, 512]" = torch.ops.aten.reshape.default(mm_31, [16, 256, 512]);  mm_31 = None
        add_84: "f16[16, 256, 512]" = torch.ops.aten.add.Tensor(view_143, arg94_1);  view_143 = arg94_1 = None
        
        # File: /EuroSys25/fwd-compile-print.py:55 in fwd_bert_std, code: hidden_states = hidden_states + residual  #残差连接
        add_85: "f16[16, 256, 512]" = torch.ops.aten.add.Tensor(add_84, convert_element_type_151);  add_84 = convert_element_type_151 = None
        
        # File: /EuroSys25/fwd-compile-print.py:58 in fwd_bert_std, code: hidden_states = F.layer_norm(hidden_states, (hidden_dim, ),
        convert_element_type_158: "f32[16, 256, 512]" = torch.ops.prims.convert_element_type.default(add_85, torch.float32)
        var_mean_15 = torch.ops.aten.var_mean.correction(convert_element_type_158, [2], correction = 0, keepdim = True);  convert_element_type_158 = None
        getitem_54: "f32[16, 256, 1]" = var_mean_15[0]
        getitem_55: "f32[16, 256, 1]" = var_mean_15[1];  var_mean_15 = None
        sub_23: "f32[16, 256, 512]" = torch.ops.aten.sub.Tensor(add_85, getitem_55);  add_85 = getitem_55 = None
        add_86: "f32[16, 256, 1]" = torch.ops.aten.add.Tensor(getitem_54, 1e-05);  getitem_54 = None
        rsqrt_15: "f32[16, 256, 1]" = torch.ops.aten.rsqrt.default(add_86);  add_86 = None
        mul_54: "f32[16, 256, 512]" = torch.ops.aten.mul.Tensor(sub_23, rsqrt_15);  sub_23 = rsqrt_15 = None
        mul_55: "f32[16, 256, 512]" = torch.ops.aten.mul.Tensor(mul_54, arg95_1);  mul_54 = arg95_1 = None
        add_87: "f32[16, 256, 512]" = torch.ops.aten.add.Tensor(mul_55, arg96_1);  mul_55 = arg96_1 = None
        convert_element_type_159: "f16[16, 256, 512]" = torch.ops.prims.convert_element_type.default(add_87, torch.float16);  add_87 = None
        
        # File: /EuroSys25/fwd-compile-print.py:22 in fwd_bert_std, code: qkv = torch.matmul(hidden_states, qkv_kernel[layer]) + qkv_bias[layer]
        view_144: "f16[4096, 512]" = torch.ops.aten.reshape.default(convert_element_type_159, [4096, 512])
        mm_32: "f16[4096, 1536]" = torch.ops.aten.mm.default(view_144, arg97_1);  view_144 = arg97_1 = None
        view_145: "f16[16, 256, 1536]" = torch.ops.aten.reshape.default(mm_32, [16, 256, 1536]);  mm_32 = None
        add_88: "f16[16, 256, 1536]" = torch.ops.aten.add.Tensor(view_145, arg98_1);  view_145 = arg98_1 = None
        
        # File: /EuroSys25/fwd-compile-print.py:23 in fwd_bert_std, code: q, k, v = qkv.chunk(3, dim=-1)
        split_8 = torch.ops.aten.split.Tensor(add_88, 512, -1);  add_88 = None
        getitem_56: "f16[16, 256, 512]" = split_8[0]
        getitem_57: "f16[16, 256, 512]" = split_8[1]
        getitem_58: "f16[16, 256, 512]" = split_8[2];  split_8 = None
        
        # File: /EuroSys25/utils/utils.py:47 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_146: "f16[16, 256, 16, 32]" = torch.ops.aten.reshape.default(getitem_56, [16, 256, 16, 32]);  getitem_56 = None
        
        # No stacktrace found for following nodes
        permute_default_9: "f16[16, 16, 256, 32]" = torch.ops.aten.permute.default(view_146, [0, 2, 1, 3]);  view_146 = None
        
        # File: /EuroSys25/utils/utils.py:47 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_147: "f16[16, 256, 16, 32]" = torch.ops.aten.reshape.default(getitem_57, [16, 256, 16, 32]);  getitem_57 = None
        
        # No stacktrace found for following nodes
        permute_default_10: "f16[16, 16, 256, 32]" = torch.ops.aten.permute.default(view_147, [0, 2, 1, 3]);  view_147 = None
        
        # File: /EuroSys25/utils/utils.py:47 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_148: "f16[16, 256, 16, 32]" = torch.ops.aten.reshape.default(getitem_58, [16, 256, 16, 32]);  getitem_58 = None
        
        # No stacktrace found for following nodes
        permute_default_11: "f16[16, 16, 256, 32]" = torch.ops.aten.permute.default(view_148, [0, 2, 1, 3]);  view_148 = None
        _scaled_dot_product_flash_attention_default_3 = torch.ops.aten._scaled_dot_product_flash_attention.default(permute_default_9, permute_default_10, permute_default_11, scale = 0.17677669529663687);  permute_default_9 = permute_default_10 = permute_default_11 = None
        getitem_87: "f16[16, 16, 256, 32]" = _scaled_dot_product_flash_attention_default_3[0];  _scaled_dot_product_flash_attention_default_3 = None
        
        # File: /EuroSys25/fwd-compile-print.py:34 in fwd_bert_std, code: h = torch.matmul(probs, v).permute(0, 2, 1, 3).contiguous()
        permute_44: "f16[16, 256, 16, 32]" = torch.ops.aten.permute.default(getitem_87, [0, 2, 1, 3]);  getitem_87 = None
        
        # File: /EuroSys25/fwd-compile-print.py:38 in fwd_bert_std, code: hidden_states = h.view(new_context_layer_shape)
        view_155: "f16[16, 256, 512]" = torch.ops.aten.reshape.default(permute_44, [16, 256, 512]);  permute_44 = None
        
        # File: /EuroSys25/fwd-compile-print.py:42 in fwd_bert_std, code: hidden_states = torch.matmul(hidden_states, attr_output_kernel[layer]) + attr_output_bias[layer]
        view_156: "f16[4096, 512]" = torch.ops.aten.reshape.default(view_155, [4096, 512]);  view_155 = None
        mm_33: "f16[4096, 512]" = torch.ops.aten.mm.default(view_156, arg99_1);  view_156 = arg99_1 = None
        view_157: "f16[16, 256, 512]" = torch.ops.aten.reshape.default(mm_33, [16, 256, 512]);  mm_33 = None
        add_89: "f16[16, 256, 512]" = torch.ops.aten.add.Tensor(view_157, arg100_1);  view_157 = arg100_1 = None
        
        # File: /EuroSys25/fwd-compile-print.py:43 in fwd_bert_std, code: hidden_states = hidden_states + input_tensor  # 残差连接
        add_90: "f16[16, 256, 512]" = torch.ops.aten.add.Tensor(add_89, convert_element_type_159);  add_89 = None
        
        # File: /EuroSys25/fwd-compile-print.py:46 in fwd_bert_std, code: hidden_states = F.layer_norm(hidden_states, (hidden_dim, ),
        convert_element_type_170: "f32[16, 256, 512]" = torch.ops.prims.convert_element_type.default(add_90, torch.float32)
        var_mean_16 = torch.ops.aten.var_mean.correction(convert_element_type_170, [2], correction = 0, keepdim = True);  convert_element_type_170 = None
        getitem_59: "f32[16, 256, 1]" = var_mean_16[0]
        getitem_60: "f32[16, 256, 1]" = var_mean_16[1];  var_mean_16 = None
        sub_25: "f32[16, 256, 512]" = torch.ops.aten.sub.Tensor(add_90, getitem_60);  add_90 = getitem_60 = None
        add_91: "f32[16, 256, 1]" = torch.ops.aten.add.Tensor(getitem_59, 1e-05);  getitem_59 = None
        rsqrt_16: "f32[16, 256, 1]" = torch.ops.aten.rsqrt.default(add_91);  add_91 = None
        mul_56: "f32[16, 256, 512]" = torch.ops.aten.mul.Tensor(sub_25, rsqrt_16);  sub_25 = rsqrt_16 = None
        mul_57: "f32[16, 256, 512]" = torch.ops.aten.mul.Tensor(mul_56, arg101_1);  mul_56 = arg101_1 = None
        add_92: "f32[16, 256, 512]" = torch.ops.aten.add.Tensor(mul_57, arg102_1);  mul_57 = arg102_1 = None
        convert_element_type_171: "f16[16, 256, 512]" = torch.ops.prims.convert_element_type.default(add_92, torch.float16);  add_92 = None
        
        # File: /EuroSys25/fwd-compile-print.py:51 in fwd_bert_std, code: hidden_states = torch.matmul(hidden_states, inter_kernel[layer]) + inter_bias[layer]
        view_158: "f16[4096, 512]" = torch.ops.aten.reshape.default(convert_element_type_171, [4096, 512])
        mm_34: "f16[4096, 2048]" = torch.ops.aten.mm.default(view_158, arg103_1);  view_158 = arg103_1 = None
        view_159: "f16[16, 256, 2048]" = torch.ops.aten.reshape.default(mm_34, [16, 256, 2048]);  mm_34 = None
        add_93: "f16[16, 256, 2048]" = torch.ops.aten.add.Tensor(view_159, arg104_1);  view_159 = arg104_1 = None
        
        # File: /EuroSys25/fwd-compile-print.py:52 in fwd_bert_std, code: hidden_states = F.gelu(hidden_states)  #激活函数
        convert_element_type_174: "f32[16, 256, 2048]" = torch.ops.prims.convert_element_type.default(add_93, torch.float32);  add_93 = None
        mul_58: "f32[16, 256, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_174, 0.5)
        mul_59: "f32[16, 256, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_174, 0.7071067811865476);  convert_element_type_174 = None
        erf_8: "f32[16, 256, 2048]" = torch.ops.aten.erf.default(mul_59);  mul_59 = None
        add_94: "f32[16, 256, 2048]" = torch.ops.aten.add.Tensor(erf_8, 1);  erf_8 = None
        mul_60: "f32[16, 256, 2048]" = torch.ops.aten.mul.Tensor(mul_58, add_94);  mul_58 = add_94 = None
        convert_element_type_175: "f16[16, 256, 2048]" = torch.ops.prims.convert_element_type.default(mul_60, torch.float16);  mul_60 = None
        
        # File: /EuroSys25/fwd-compile-print.py:54 in fwd_bert_std, code: hidden_states = torch.matmul(hidden_states, output_kernel[layer]) + output_bias[layer]
        view_160: "f16[4096, 2048]" = torch.ops.aten.reshape.default(convert_element_type_175, [4096, 2048]);  convert_element_type_175 = None
        mm_35: "f16[4096, 512]" = torch.ops.aten.mm.default(view_160, arg105_1);  view_160 = arg105_1 = None
        view_161: "f16[16, 256, 512]" = torch.ops.aten.reshape.default(mm_35, [16, 256, 512]);  mm_35 = None
        add_95: "f16[16, 256, 512]" = torch.ops.aten.add.Tensor(view_161, arg106_1);  view_161 = arg106_1 = None
        
        # File: /EuroSys25/fwd-compile-print.py:55 in fwd_bert_std, code: hidden_states = hidden_states + residual  #残差连接
        add_96: "f16[16, 256, 512]" = torch.ops.aten.add.Tensor(add_95, convert_element_type_171);  add_95 = convert_element_type_171 = None
        
        # File: /EuroSys25/fwd-compile-print.py:58 in fwd_bert_std, code: hidden_states = F.layer_norm(hidden_states, (hidden_dim, ),
        convert_element_type_178: "f32[16, 256, 512]" = torch.ops.prims.convert_element_type.default(add_96, torch.float32)
        var_mean_17 = torch.ops.aten.var_mean.correction(convert_element_type_178, [2], correction = 0, keepdim = True);  convert_element_type_178 = None
        getitem_61: "f32[16, 256, 1]" = var_mean_17[0]
        getitem_62: "f32[16, 256, 1]" = var_mean_17[1];  var_mean_17 = None
        sub_26: "f32[16, 256, 512]" = torch.ops.aten.sub.Tensor(add_96, getitem_62);  add_96 = getitem_62 = None
        add_97: "f32[16, 256, 1]" = torch.ops.aten.add.Tensor(getitem_61, 1e-05);  getitem_61 = None
        rsqrt_17: "f32[16, 256, 1]" = torch.ops.aten.rsqrt.default(add_97);  add_97 = None
        mul_61: "f32[16, 256, 512]" = torch.ops.aten.mul.Tensor(sub_26, rsqrt_17);  sub_26 = rsqrt_17 = None
        mul_62: "f32[16, 256, 512]" = torch.ops.aten.mul.Tensor(mul_61, arg107_1);  mul_61 = arg107_1 = None
        add_98: "f32[16, 256, 512]" = torch.ops.aten.add.Tensor(mul_62, arg108_1);  mul_62 = arg108_1 = None
        convert_element_type_179: "f16[16, 256, 512]" = torch.ops.prims.convert_element_type.default(add_98, torch.float16);  add_98 = None
        
        # File: /EuroSys25/fwd-compile-print.py:22 in fwd_bert_std, code: qkv = torch.matmul(hidden_states, qkv_kernel[layer]) + qkv_bias[layer]
        view_162: "f16[4096, 512]" = torch.ops.aten.reshape.default(convert_element_type_179, [4096, 512])
        mm_36: "f16[4096, 1536]" = torch.ops.aten.mm.default(view_162, arg109_1);  view_162 = arg109_1 = None
        view_163: "f16[16, 256, 1536]" = torch.ops.aten.reshape.default(mm_36, [16, 256, 1536]);  mm_36 = None
        add_99: "f16[16, 256, 1536]" = torch.ops.aten.add.Tensor(view_163, arg110_1);  view_163 = arg110_1 = None
        
        # File: /EuroSys25/fwd-compile-print.py:23 in fwd_bert_std, code: q, k, v = qkv.chunk(3, dim=-1)
        split_9 = torch.ops.aten.split.Tensor(add_99, 512, -1);  add_99 = None
        getitem_63: "f16[16, 256, 512]" = split_9[0]
        getitem_64: "f16[16, 256, 512]" = split_9[1]
        getitem_65: "f16[16, 256, 512]" = split_9[2];  split_9 = None
        
        # File: /EuroSys25/utils/utils.py:47 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_164: "f16[16, 256, 16, 32]" = torch.ops.aten.reshape.default(getitem_63, [16, 256, 16, 32]);  getitem_63 = None
        
        # No stacktrace found for following nodes
        permute_default_6: "f16[16, 16, 256, 32]" = torch.ops.aten.permute.default(view_164, [0, 2, 1, 3]);  view_164 = None
        
        # File: /EuroSys25/utils/utils.py:47 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_165: "f16[16, 256, 16, 32]" = torch.ops.aten.reshape.default(getitem_64, [16, 256, 16, 32]);  getitem_64 = None
        
        # No stacktrace found for following nodes
        permute_default_7: "f16[16, 16, 256, 32]" = torch.ops.aten.permute.default(view_165, [0, 2, 1, 3]);  view_165 = None
        
        # File: /EuroSys25/utils/utils.py:47 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_166: "f16[16, 256, 16, 32]" = torch.ops.aten.reshape.default(getitem_65, [16, 256, 16, 32]);  getitem_65 = None
        
        # No stacktrace found for following nodes
        permute_default_8: "f16[16, 16, 256, 32]" = torch.ops.aten.permute.default(view_166, [0, 2, 1, 3]);  view_166 = None
        _scaled_dot_product_flash_attention_default_2 = torch.ops.aten._scaled_dot_product_flash_attention.default(permute_default_6, permute_default_7, permute_default_8, scale = 0.17677669529663687);  permute_default_6 = permute_default_7 = permute_default_8 = None
        getitem_86: "f16[16, 16, 256, 32]" = _scaled_dot_product_flash_attention_default_2[0];  _scaled_dot_product_flash_attention_default_2 = None
        
        # File: /EuroSys25/fwd-compile-print.py:34 in fwd_bert_std, code: h = torch.matmul(probs, v).permute(0, 2, 1, 3).contiguous()
        permute_49: "f16[16, 256, 16, 32]" = torch.ops.aten.permute.default(getitem_86, [0, 2, 1, 3]);  getitem_86 = None
        
        # File: /EuroSys25/fwd-compile-print.py:38 in fwd_bert_std, code: hidden_states = h.view(new_context_layer_shape)
        view_173: "f16[16, 256, 512]" = torch.ops.aten.reshape.default(permute_49, [16, 256, 512]);  permute_49 = None
        
        # File: /EuroSys25/fwd-compile-print.py:42 in fwd_bert_std, code: hidden_states = torch.matmul(hidden_states, attr_output_kernel[layer]) + attr_output_bias[layer]
        view_174: "f16[4096, 512]" = torch.ops.aten.reshape.default(view_173, [4096, 512]);  view_173 = None
        mm_37: "f16[4096, 512]" = torch.ops.aten.mm.default(view_174, arg111_1);  view_174 = arg111_1 = None
        view_175: "f16[16, 256, 512]" = torch.ops.aten.reshape.default(mm_37, [16, 256, 512]);  mm_37 = None
        add_100: "f16[16, 256, 512]" = torch.ops.aten.add.Tensor(view_175, arg112_1);  view_175 = arg112_1 = None
        
        # File: /EuroSys25/fwd-compile-print.py:43 in fwd_bert_std, code: hidden_states = hidden_states + input_tensor  # 残差连接
        add_101: "f16[16, 256, 512]" = torch.ops.aten.add.Tensor(add_100, convert_element_type_179);  add_100 = None
        
        # File: /EuroSys25/fwd-compile-print.py:46 in fwd_bert_std, code: hidden_states = F.layer_norm(hidden_states, (hidden_dim, ),
        convert_element_type_190: "f32[16, 256, 512]" = torch.ops.prims.convert_element_type.default(add_101, torch.float32)
        var_mean_18 = torch.ops.aten.var_mean.correction(convert_element_type_190, [2], correction = 0, keepdim = True);  convert_element_type_190 = None
        getitem_66: "f32[16, 256, 1]" = var_mean_18[0]
        getitem_67: "f32[16, 256, 1]" = var_mean_18[1];  var_mean_18 = None
        sub_28: "f32[16, 256, 512]" = torch.ops.aten.sub.Tensor(add_101, getitem_67);  add_101 = getitem_67 = None
        add_102: "f32[16, 256, 1]" = torch.ops.aten.add.Tensor(getitem_66, 1e-05);  getitem_66 = None
        rsqrt_18: "f32[16, 256, 1]" = torch.ops.aten.rsqrt.default(add_102);  add_102 = None
        mul_63: "f32[16, 256, 512]" = torch.ops.aten.mul.Tensor(sub_28, rsqrt_18);  sub_28 = rsqrt_18 = None
        mul_64: "f32[16, 256, 512]" = torch.ops.aten.mul.Tensor(mul_63, arg113_1);  mul_63 = arg113_1 = None
        add_103: "f32[16, 256, 512]" = torch.ops.aten.add.Tensor(mul_64, arg114_1);  mul_64 = arg114_1 = None
        convert_element_type_191: "f16[16, 256, 512]" = torch.ops.prims.convert_element_type.default(add_103, torch.float16);  add_103 = None
        
        # File: /EuroSys25/fwd-compile-print.py:51 in fwd_bert_std, code: hidden_states = torch.matmul(hidden_states, inter_kernel[layer]) + inter_bias[layer]
        view_176: "f16[4096, 512]" = torch.ops.aten.reshape.default(convert_element_type_191, [4096, 512])
        mm_38: "f16[4096, 2048]" = torch.ops.aten.mm.default(view_176, arg115_1);  view_176 = arg115_1 = None
        view_177: "f16[16, 256, 2048]" = torch.ops.aten.reshape.default(mm_38, [16, 256, 2048]);  mm_38 = None
        add_104: "f16[16, 256, 2048]" = torch.ops.aten.add.Tensor(view_177, arg116_1);  view_177 = arg116_1 = None
        
        # File: /EuroSys25/fwd-compile-print.py:52 in fwd_bert_std, code: hidden_states = F.gelu(hidden_states)  #激活函数
        convert_element_type_194: "f32[16, 256, 2048]" = torch.ops.prims.convert_element_type.default(add_104, torch.float32);  add_104 = None
        mul_65: "f32[16, 256, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_194, 0.5)
        mul_66: "f32[16, 256, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_194, 0.7071067811865476);  convert_element_type_194 = None
        erf_9: "f32[16, 256, 2048]" = torch.ops.aten.erf.default(mul_66);  mul_66 = None
        add_105: "f32[16, 256, 2048]" = torch.ops.aten.add.Tensor(erf_9, 1);  erf_9 = None
        mul_67: "f32[16, 256, 2048]" = torch.ops.aten.mul.Tensor(mul_65, add_105);  mul_65 = add_105 = None
        convert_element_type_195: "f16[16, 256, 2048]" = torch.ops.prims.convert_element_type.default(mul_67, torch.float16);  mul_67 = None
        
        # File: /EuroSys25/fwd-compile-print.py:54 in fwd_bert_std, code: hidden_states = torch.matmul(hidden_states, output_kernel[layer]) + output_bias[layer]
        view_178: "f16[4096, 2048]" = torch.ops.aten.reshape.default(convert_element_type_195, [4096, 2048]);  convert_element_type_195 = None
        mm_39: "f16[4096, 512]" = torch.ops.aten.mm.default(view_178, arg117_1);  view_178 = arg117_1 = None
        view_179: "f16[16, 256, 512]" = torch.ops.aten.reshape.default(mm_39, [16, 256, 512]);  mm_39 = None
        add_106: "f16[16, 256, 512]" = torch.ops.aten.add.Tensor(view_179, arg118_1);  view_179 = arg118_1 = None
        
        # File: /EuroSys25/fwd-compile-print.py:55 in fwd_bert_std, code: hidden_states = hidden_states + residual  #残差连接
        add_107: "f16[16, 256, 512]" = torch.ops.aten.add.Tensor(add_106, convert_element_type_191);  add_106 = convert_element_type_191 = None
        
        # File: /EuroSys25/fwd-compile-print.py:58 in fwd_bert_std, code: hidden_states = F.layer_norm(hidden_states, (hidden_dim, ),
        convert_element_type_198: "f32[16, 256, 512]" = torch.ops.prims.convert_element_type.default(add_107, torch.float32)
        var_mean_19 = torch.ops.aten.var_mean.correction(convert_element_type_198, [2], correction = 0, keepdim = True);  convert_element_type_198 = None
        getitem_68: "f32[16, 256, 1]" = var_mean_19[0]
        getitem_69: "f32[16, 256, 1]" = var_mean_19[1];  var_mean_19 = None
        sub_29: "f32[16, 256, 512]" = torch.ops.aten.sub.Tensor(add_107, getitem_69);  add_107 = getitem_69 = None
        add_108: "f32[16, 256, 1]" = torch.ops.aten.add.Tensor(getitem_68, 1e-05);  getitem_68 = None
        rsqrt_19: "f32[16, 256, 1]" = torch.ops.aten.rsqrt.default(add_108);  add_108 = None
        mul_68: "f32[16, 256, 512]" = torch.ops.aten.mul.Tensor(sub_29, rsqrt_19);  sub_29 = rsqrt_19 = None
        mul_69: "f32[16, 256, 512]" = torch.ops.aten.mul.Tensor(mul_68, arg119_1);  mul_68 = arg119_1 = None
        add_109: "f32[16, 256, 512]" = torch.ops.aten.add.Tensor(mul_69, arg120_1);  mul_69 = arg120_1 = None
        convert_element_type_199: "f16[16, 256, 512]" = torch.ops.prims.convert_element_type.default(add_109, torch.float16);  add_109 = None
        
        # File: /EuroSys25/fwd-compile-print.py:22 in fwd_bert_std, code: qkv = torch.matmul(hidden_states, qkv_kernel[layer]) + qkv_bias[layer]
        view_180: "f16[4096, 512]" = torch.ops.aten.reshape.default(convert_element_type_199, [4096, 512])
        mm_40: "f16[4096, 1536]" = torch.ops.aten.mm.default(view_180, arg121_1);  view_180 = arg121_1 = None
        view_181: "f16[16, 256, 1536]" = torch.ops.aten.reshape.default(mm_40, [16, 256, 1536]);  mm_40 = None
        add_110: "f16[16, 256, 1536]" = torch.ops.aten.add.Tensor(view_181, arg122_1);  view_181 = arg122_1 = None
        
        # File: /EuroSys25/fwd-compile-print.py:23 in fwd_bert_std, code: q, k, v = qkv.chunk(3, dim=-1)
        split_10 = torch.ops.aten.split.Tensor(add_110, 512, -1);  add_110 = None
        getitem_70: "f16[16, 256, 512]" = split_10[0]
        getitem_71: "f16[16, 256, 512]" = split_10[1]
        getitem_72: "f16[16, 256, 512]" = split_10[2];  split_10 = None
        
        # File: /EuroSys25/utils/utils.py:47 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_182: "f16[16, 256, 16, 32]" = torch.ops.aten.reshape.default(getitem_70, [16, 256, 16, 32]);  getitem_70 = None
        
        # No stacktrace found for following nodes
        permute_default_3: "f16[16, 16, 256, 32]" = torch.ops.aten.permute.default(view_182, [0, 2, 1, 3]);  view_182 = None
        
        # File: /EuroSys25/utils/utils.py:47 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_183: "f16[16, 256, 16, 32]" = torch.ops.aten.reshape.default(getitem_71, [16, 256, 16, 32]);  getitem_71 = None
        
        # No stacktrace found for following nodes
        permute_default_4: "f16[16, 16, 256, 32]" = torch.ops.aten.permute.default(view_183, [0, 2, 1, 3]);  view_183 = None
        
        # File: /EuroSys25/utils/utils.py:47 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_184: "f16[16, 256, 16, 32]" = torch.ops.aten.reshape.default(getitem_72, [16, 256, 16, 32]);  getitem_72 = None
        
        # No stacktrace found for following nodes
        permute_default_5: "f16[16, 16, 256, 32]" = torch.ops.aten.permute.default(view_184, [0, 2, 1, 3]);  view_184 = None
        _scaled_dot_product_flash_attention_default_1 = torch.ops.aten._scaled_dot_product_flash_attention.default(permute_default_3, permute_default_4, permute_default_5, scale = 0.17677669529663687);  permute_default_3 = permute_default_4 = permute_default_5 = None
        getitem_85: "f16[16, 16, 256, 32]" = _scaled_dot_product_flash_attention_default_1[0];  _scaled_dot_product_flash_attention_default_1 = None
        
        # File: /EuroSys25/fwd-compile-print.py:34 in fwd_bert_std, code: h = torch.matmul(probs, v).permute(0, 2, 1, 3).contiguous()
        permute_54: "f16[16, 256, 16, 32]" = torch.ops.aten.permute.default(getitem_85, [0, 2, 1, 3]);  getitem_85 = None
        
        # File: /EuroSys25/fwd-compile-print.py:38 in fwd_bert_std, code: hidden_states = h.view(new_context_layer_shape)
        view_191: "f16[16, 256, 512]" = torch.ops.aten.reshape.default(permute_54, [16, 256, 512]);  permute_54 = None
        
        # File: /EuroSys25/fwd-compile-print.py:42 in fwd_bert_std, code: hidden_states = torch.matmul(hidden_states, attr_output_kernel[layer]) + attr_output_bias[layer]
        view_192: "f16[4096, 512]" = torch.ops.aten.reshape.default(view_191, [4096, 512]);  view_191 = None
        mm_41: "f16[4096, 512]" = torch.ops.aten.mm.default(view_192, arg123_1);  view_192 = arg123_1 = None
        view_193: "f16[16, 256, 512]" = torch.ops.aten.reshape.default(mm_41, [16, 256, 512]);  mm_41 = None
        add_111: "f16[16, 256, 512]" = torch.ops.aten.add.Tensor(view_193, arg124_1);  view_193 = arg124_1 = None
        
        # File: /EuroSys25/fwd-compile-print.py:43 in fwd_bert_std, code: hidden_states = hidden_states + input_tensor  # 残差连接
        add_112: "f16[16, 256, 512]" = torch.ops.aten.add.Tensor(add_111, convert_element_type_199);  add_111 = None
        
        # File: /EuroSys25/fwd-compile-print.py:46 in fwd_bert_std, code: hidden_states = F.layer_norm(hidden_states, (hidden_dim, ),
        convert_element_type_210: "f32[16, 256, 512]" = torch.ops.prims.convert_element_type.default(add_112, torch.float32)
        var_mean_20 = torch.ops.aten.var_mean.correction(convert_element_type_210, [2], correction = 0, keepdim = True);  convert_element_type_210 = None
        getitem_73: "f32[16, 256, 1]" = var_mean_20[0]
        getitem_74: "f32[16, 256, 1]" = var_mean_20[1];  var_mean_20 = None
        sub_31: "f32[16, 256, 512]" = torch.ops.aten.sub.Tensor(add_112, getitem_74);  add_112 = getitem_74 = None
        add_113: "f32[16, 256, 1]" = torch.ops.aten.add.Tensor(getitem_73, 1e-05);  getitem_73 = None
        rsqrt_20: "f32[16, 256, 1]" = torch.ops.aten.rsqrt.default(add_113);  add_113 = None
        mul_70: "f32[16, 256, 512]" = torch.ops.aten.mul.Tensor(sub_31, rsqrt_20);  sub_31 = rsqrt_20 = None
        mul_71: "f32[16, 256, 512]" = torch.ops.aten.mul.Tensor(mul_70, arg125_1);  mul_70 = arg125_1 = None
        add_114: "f32[16, 256, 512]" = torch.ops.aten.add.Tensor(mul_71, arg126_1);  mul_71 = arg126_1 = None
        convert_element_type_211: "f16[16, 256, 512]" = torch.ops.prims.convert_element_type.default(add_114, torch.float16);  add_114 = None
        
        # File: /EuroSys25/fwd-compile-print.py:51 in fwd_bert_std, code: hidden_states = torch.matmul(hidden_states, inter_kernel[layer]) + inter_bias[layer]
        view_194: "f16[4096, 512]" = torch.ops.aten.reshape.default(convert_element_type_211, [4096, 512])
        mm_42: "f16[4096, 2048]" = torch.ops.aten.mm.default(view_194, arg127_1);  view_194 = arg127_1 = None
        view_195: "f16[16, 256, 2048]" = torch.ops.aten.reshape.default(mm_42, [16, 256, 2048]);  mm_42 = None
        add_115: "f16[16, 256, 2048]" = torch.ops.aten.add.Tensor(view_195, arg128_1);  view_195 = arg128_1 = None
        
        # File: /EuroSys25/fwd-compile-print.py:52 in fwd_bert_std, code: hidden_states = F.gelu(hidden_states)  #激活函数
        convert_element_type_214: "f32[16, 256, 2048]" = torch.ops.prims.convert_element_type.default(add_115, torch.float32);  add_115 = None
        mul_72: "f32[16, 256, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_214, 0.5)
        mul_73: "f32[16, 256, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_214, 0.7071067811865476);  convert_element_type_214 = None
        erf_10: "f32[16, 256, 2048]" = torch.ops.aten.erf.default(mul_73);  mul_73 = None
        add_116: "f32[16, 256, 2048]" = torch.ops.aten.add.Tensor(erf_10, 1);  erf_10 = None
        mul_74: "f32[16, 256, 2048]" = torch.ops.aten.mul.Tensor(mul_72, add_116);  mul_72 = add_116 = None
        convert_element_type_215: "f16[16, 256, 2048]" = torch.ops.prims.convert_element_type.default(mul_74, torch.float16);  mul_74 = None
        
        # File: /EuroSys25/fwd-compile-print.py:54 in fwd_bert_std, code: hidden_states = torch.matmul(hidden_states, output_kernel[layer]) + output_bias[layer]
        view_196: "f16[4096, 2048]" = torch.ops.aten.reshape.default(convert_element_type_215, [4096, 2048]);  convert_element_type_215 = None
        mm_43: "f16[4096, 512]" = torch.ops.aten.mm.default(view_196, arg129_1);  view_196 = arg129_1 = None
        view_197: "f16[16, 256, 512]" = torch.ops.aten.reshape.default(mm_43, [16, 256, 512]);  mm_43 = None
        add_117: "f16[16, 256, 512]" = torch.ops.aten.add.Tensor(view_197, arg130_1);  view_197 = arg130_1 = None
        
        # File: /EuroSys25/fwd-compile-print.py:55 in fwd_bert_std, code: hidden_states = hidden_states + residual  #残差连接
        add_118: "f16[16, 256, 512]" = torch.ops.aten.add.Tensor(add_117, convert_element_type_211);  add_117 = convert_element_type_211 = None
        
        # File: /EuroSys25/fwd-compile-print.py:58 in fwd_bert_std, code: hidden_states = F.layer_norm(hidden_states, (hidden_dim, ),
        convert_element_type_218: "f32[16, 256, 512]" = torch.ops.prims.convert_element_type.default(add_118, torch.float32)
        var_mean_21 = torch.ops.aten.var_mean.correction(convert_element_type_218, [2], correction = 0, keepdim = True);  convert_element_type_218 = None
        getitem_75: "f32[16, 256, 1]" = var_mean_21[0]
        getitem_76: "f32[16, 256, 1]" = var_mean_21[1];  var_mean_21 = None
        sub_32: "f32[16, 256, 512]" = torch.ops.aten.sub.Tensor(add_118, getitem_76);  add_118 = getitem_76 = None
        add_119: "f32[16, 256, 1]" = torch.ops.aten.add.Tensor(getitem_75, 1e-05);  getitem_75 = None
        rsqrt_21: "f32[16, 256, 1]" = torch.ops.aten.rsqrt.default(add_119);  add_119 = None
        mul_75: "f32[16, 256, 512]" = torch.ops.aten.mul.Tensor(sub_32, rsqrt_21);  sub_32 = rsqrt_21 = None
        mul_76: "f32[16, 256, 512]" = torch.ops.aten.mul.Tensor(mul_75, arg131_1);  mul_75 = arg131_1 = None
        add_120: "f32[16, 256, 512]" = torch.ops.aten.add.Tensor(mul_76, arg132_1);  mul_76 = arg132_1 = None
        convert_element_type_219: "f16[16, 256, 512]" = torch.ops.prims.convert_element_type.default(add_120, torch.float16);  add_120 = None
        
        # File: /EuroSys25/fwd-compile-print.py:22 in fwd_bert_std, code: qkv = torch.matmul(hidden_states, qkv_kernel[layer]) + qkv_bias[layer]
        view_198: "f16[4096, 512]" = torch.ops.aten.reshape.default(convert_element_type_219, [4096, 512])
        mm_44: "f16[4096, 1536]" = torch.ops.aten.mm.default(view_198, arg133_1);  view_198 = arg133_1 = None
        view_199: "f16[16, 256, 1536]" = torch.ops.aten.reshape.default(mm_44, [16, 256, 1536]);  mm_44 = None
        add_121: "f16[16, 256, 1536]" = torch.ops.aten.add.Tensor(view_199, arg134_1);  view_199 = arg134_1 = None
        
        # File: /EuroSys25/fwd-compile-print.py:23 in fwd_bert_std, code: q, k, v = qkv.chunk(3, dim=-1)
        split_11 = torch.ops.aten.split.Tensor(add_121, 512, -1);  add_121 = None
        getitem_77: "f16[16, 256, 512]" = split_11[0]
        getitem_78: "f16[16, 256, 512]" = split_11[1]
        getitem_79: "f16[16, 256, 512]" = split_11[2];  split_11 = None
        
        # File: /EuroSys25/utils/utils.py:47 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_200: "f16[16, 256, 16, 32]" = torch.ops.aten.reshape.default(getitem_77, [16, 256, 16, 32]);  getitem_77 = None
        
        # No stacktrace found for following nodes
        permute_default: "f16[16, 16, 256, 32]" = torch.ops.aten.permute.default(view_200, [0, 2, 1, 3]);  view_200 = None
        
        # File: /EuroSys25/utils/utils.py:47 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_201: "f16[16, 256, 16, 32]" = torch.ops.aten.reshape.default(getitem_78, [16, 256, 16, 32]);  getitem_78 = None
        
        # No stacktrace found for following nodes
        permute_default_1: "f16[16, 16, 256, 32]" = torch.ops.aten.permute.default(view_201, [0, 2, 1, 3]);  view_201 = None
        
        # File: /EuroSys25/utils/utils.py:47 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_202: "f16[16, 256, 16, 32]" = torch.ops.aten.reshape.default(getitem_79, [16, 256, 16, 32]);  getitem_79 = None
        
        # No stacktrace found for following nodes
        permute_default_2: "f16[16, 16, 256, 32]" = torch.ops.aten.permute.default(view_202, [0, 2, 1, 3]);  view_202 = None
        _scaled_dot_product_flash_attention_default = torch.ops.aten._scaled_dot_product_flash_attention.default(permute_default, permute_default_1, permute_default_2, scale = 0.17677669529663687);  permute_default = permute_default_1 = permute_default_2 = None
        getitem_84: "f16[16, 16, 256, 32]" = _scaled_dot_product_flash_attention_default[0];  _scaled_dot_product_flash_attention_default = None
        
        # File: /EuroSys25/fwd-compile-print.py:34 in fwd_bert_std, code: h = torch.matmul(probs, v).permute(0, 2, 1, 3).contiguous()
        permute_59: "f16[16, 256, 16, 32]" = torch.ops.aten.permute.default(getitem_84, [0, 2, 1, 3]);  getitem_84 = None
        
        # File: /EuroSys25/fwd-compile-print.py:38 in fwd_bert_std, code: hidden_states = h.view(new_context_layer_shape)
        view_209: "f16[16, 256, 512]" = torch.ops.aten.reshape.default(permute_59, [16, 256, 512]);  permute_59 = None
        
        # File: /EuroSys25/fwd-compile-print.py:42 in fwd_bert_std, code: hidden_states = torch.matmul(hidden_states, attr_output_kernel[layer]) + attr_output_bias[layer]
        view_210: "f16[4096, 512]" = torch.ops.aten.reshape.default(view_209, [4096, 512]);  view_209 = None
        mm_45: "f16[4096, 512]" = torch.ops.aten.mm.default(view_210, arg135_1);  view_210 = arg135_1 = None
        view_211: "f16[16, 256, 512]" = torch.ops.aten.reshape.default(mm_45, [16, 256, 512]);  mm_45 = None
        add_122: "f16[16, 256, 512]" = torch.ops.aten.add.Tensor(view_211, arg136_1);  view_211 = arg136_1 = None
        
        # File: /EuroSys25/fwd-compile-print.py:43 in fwd_bert_std, code: hidden_states = hidden_states + input_tensor  # 残差连接
        add_123: "f16[16, 256, 512]" = torch.ops.aten.add.Tensor(add_122, convert_element_type_219);  add_122 = None
        
        # File: /EuroSys25/fwd-compile-print.py:46 in fwd_bert_std, code: hidden_states = F.layer_norm(hidden_states, (hidden_dim, ),
        convert_element_type_230: "f32[16, 256, 512]" = torch.ops.prims.convert_element_type.default(add_123, torch.float32)
        var_mean_22 = torch.ops.aten.var_mean.correction(convert_element_type_230, [2], correction = 0, keepdim = True);  convert_element_type_230 = None
        getitem_80: "f32[16, 256, 1]" = var_mean_22[0]
        getitem_81: "f32[16, 256, 1]" = var_mean_22[1];  var_mean_22 = None
        sub_34: "f32[16, 256, 512]" = torch.ops.aten.sub.Tensor(add_123, getitem_81);  add_123 = getitem_81 = None
        add_124: "f32[16, 256, 1]" = torch.ops.aten.add.Tensor(getitem_80, 1e-05);  getitem_80 = None
        rsqrt_22: "f32[16, 256, 1]" = torch.ops.aten.rsqrt.default(add_124);  add_124 = None
        mul_77: "f32[16, 256, 512]" = torch.ops.aten.mul.Tensor(sub_34, rsqrt_22);  sub_34 = rsqrt_22 = None
        mul_78: "f32[16, 256, 512]" = torch.ops.aten.mul.Tensor(mul_77, arg137_1);  mul_77 = arg137_1 = None
        add_125: "f32[16, 256, 512]" = torch.ops.aten.add.Tensor(mul_78, arg138_1);  mul_78 = arg138_1 = None
        convert_element_type_231: "f16[16, 256, 512]" = torch.ops.prims.convert_element_type.default(add_125, torch.float16);  add_125 = None
        
        # File: /EuroSys25/fwd-compile-print.py:51 in fwd_bert_std, code: hidden_states = torch.matmul(hidden_states, inter_kernel[layer]) + inter_bias[layer]
        view_212: "f16[4096, 512]" = torch.ops.aten.reshape.default(convert_element_type_231, [4096, 512])
        mm_46: "f16[4096, 2048]" = torch.ops.aten.mm.default(view_212, arg139_1);  view_212 = arg139_1 = None
        view_213: "f16[16, 256, 2048]" = torch.ops.aten.reshape.default(mm_46, [16, 256, 2048]);  mm_46 = None
        add_126: "f16[16, 256, 2048]" = torch.ops.aten.add.Tensor(view_213, arg140_1);  view_213 = arg140_1 = None
        
        # File: /EuroSys25/fwd-compile-print.py:52 in fwd_bert_std, code: hidden_states = F.gelu(hidden_states)  #激活函数
        convert_element_type_234: "f32[16, 256, 2048]" = torch.ops.prims.convert_element_type.default(add_126, torch.float32);  add_126 = None
        mul_79: "f32[16, 256, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_234, 0.5)
        mul_80: "f32[16, 256, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_234, 0.7071067811865476);  convert_element_type_234 = None
        erf_11: "f32[16, 256, 2048]" = torch.ops.aten.erf.default(mul_80);  mul_80 = None
        add_127: "f32[16, 256, 2048]" = torch.ops.aten.add.Tensor(erf_11, 1);  erf_11 = None
        mul_81: "f32[16, 256, 2048]" = torch.ops.aten.mul.Tensor(mul_79, add_127);  mul_79 = add_127 = None
        convert_element_type_235: "f16[16, 256, 2048]" = torch.ops.prims.convert_element_type.default(mul_81, torch.float16);  mul_81 = None
        
        # File: /EuroSys25/fwd-compile-print.py:54 in fwd_bert_std, code: hidden_states = torch.matmul(hidden_states, output_kernel[layer]) + output_bias[layer]
        view_214: "f16[4096, 2048]" = torch.ops.aten.reshape.default(convert_element_type_235, [4096, 2048]);  convert_element_type_235 = None
        mm_47: "f16[4096, 512]" = torch.ops.aten.mm.default(view_214, arg141_1);  view_214 = arg141_1 = None
        view_215: "f16[16, 256, 512]" = torch.ops.aten.reshape.default(mm_47, [16, 256, 512]);  mm_47 = None
        add_128: "f16[16, 256, 512]" = torch.ops.aten.add.Tensor(view_215, arg142_1);  view_215 = arg142_1 = None
        
        # File: /EuroSys25/fwd-compile-print.py:55 in fwd_bert_std, code: hidden_states = hidden_states + residual  #残差连接
        add_129: "f16[16, 256, 512]" = torch.ops.aten.add.Tensor(add_128, convert_element_type_231);  add_128 = convert_element_type_231 = None
        
        # File: /EuroSys25/fwd-compile-print.py:58 in fwd_bert_std, code: hidden_states = F.layer_norm(hidden_states, (hidden_dim, ),
        convert_element_type_238: "f32[16, 256, 512]" = torch.ops.prims.convert_element_type.default(add_129, torch.float32)
        var_mean_23 = torch.ops.aten.var_mean.correction(convert_element_type_238, [2], correction = 0, keepdim = True);  convert_element_type_238 = None
        getitem_82: "f32[16, 256, 1]" = var_mean_23[0]
        getitem_83: "f32[16, 256, 1]" = var_mean_23[1];  var_mean_23 = None
        sub_35: "f32[16, 256, 512]" = torch.ops.aten.sub.Tensor(add_129, getitem_83);  add_129 = getitem_83 = None
        add_130: "f32[16, 256, 1]" = torch.ops.aten.add.Tensor(getitem_82, 1e-05);  getitem_82 = None
        rsqrt_23: "f32[16, 256, 1]" = torch.ops.aten.rsqrt.default(add_130);  add_130 = None
        mul_82: "f32[16, 256, 512]" = torch.ops.aten.mul.Tensor(sub_35, rsqrt_23);  sub_35 = rsqrt_23 = None
        mul_83: "f32[16, 256, 512]" = torch.ops.aten.mul.Tensor(mul_82, arg143_1);  mul_82 = arg143_1 = None
        add_131: "f32[16, 256, 512]" = torch.ops.aten.add.Tensor(mul_83, arg144_1);  mul_83 = arg144_1 = None
        convert_element_type_239: "f16[16, 256, 512]" = torch.ops.prims.convert_element_type.default(add_131, torch.float16);  add_131 = None
        return (convert_element_type_19, convert_element_type_39, convert_element_type_59, convert_element_type_79, convert_element_type_99, convert_element_type_119, convert_element_type_139, convert_element_type_159, convert_element_type_179, convert_element_type_199, convert_element_type_219, convert_element_type_239)
        