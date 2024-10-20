class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "f16[8, 1024, 512]", arg1_1: "f16[512, 1536]", arg2_1: "f16[1536]", arg3_1: "f32[8, 1024, 1024]", arg4_1: "f16[512, 512]", arg5_1: "f16[512]", arg6_1: "f16[512]", arg7_1: "f16[512]", arg8_1: "f16[512, 2048]", arg9_1: "f16[2048]", arg10_1: "f16[2048, 512]", arg11_1: "f16[512]", arg12_1: "f16[512]", arg13_1: "f16[512]"):
         # File: /EuroSys25/fwd-compile-print-withmask.py:22 in fwd_bert_std, code: qkv = torch.matmul(hidden_states, qkv_kernel[layer]) + qkv_bias[layer]
        view: "f16[8192, 512]" = torch.ops.aten.view.default(arg0_1, [8192, 512])
        mm: "f16[8192, 1536]" = torch.ops.aten.mm.default(view, arg1_1);  view = arg1_1 = None
        view_1: "f16[8, 1024, 1536]" = torch.ops.aten.view.default(mm, [8, 1024, 1536]);  mm = None
        add: "f16[8, 1024, 1536]" = torch.ops.aten.add.Tensor(view_1, arg2_1);  view_1 = arg2_1 = None
        
         # File: /EuroSys25/fwd-compile-print-withmask.py:23 in fwd_bert_std, code: q, k, v = qkv.chunk(3, dim=-1)
        split = torch.ops.aten.split.Tensor(add, 512, -1);  add = None
        getitem: "f16[8, 1024, 512]" = split[0]
        getitem_1: "f16[8, 1024, 512]" = split[1]
        getitem_2: "f16[8, 1024, 512]" = split[2];  split = None
        
         # File: /EuroSys25/utils/utils.py:48 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_2: "f16[8, 1024, 16, 32]" = torch.ops.aten.view.default(getitem, [8, 1024, 16, 32]);  getitem = None
        
         # File: /EuroSys25/utils/utils.py:51 in transpose_for_scores, code: return x.permute(0, 2, 1, 3)
        permute: "f16[8, 16, 1024, 32]" = torch.ops.aten.permute.default(view_2, [0, 2, 1, 3]);  view_2 = None
        
         # File: /EuroSys25/utils/utils.py:48 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_3: "f16[8, 1024, 16, 32]" = torch.ops.aten.view.default(getitem_1, [8, 1024, 16, 32]);  getitem_1 = None
        
         # File: /EuroSys25/utils/utils.py:51 in transpose_for_scores, code: return x.permute(0, 2, 1, 3)
        permute_1: "f16[8, 16, 1024, 32]" = torch.ops.aten.permute.default(view_3, [0, 2, 1, 3]);  view_3 = None
        
         # File: /EuroSys25/utils/utils.py:48 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_4: "f16[8, 1024, 16, 32]" = torch.ops.aten.view.default(getitem_2, [8, 1024, 16, 32]);  getitem_2 = None
        
         # File: /EuroSys25/utils/utils.py:51 in transpose_for_scores, code: return x.permute(0, 2, 1, 3)
        permute_2: "f16[8, 16, 1024, 32]" = torch.ops.aten.permute.default(view_4, [0, 2, 1, 3]);  view_4 = None
        
         # File: /EuroSys25/fwd-compile-print-withmask.py:30 in fwd_bert_std, code: scores = torch.matmul(q, k.transpose(-2, -1)) / (head_size ** .5)
        permute_3: "f16[8, 16, 32, 1024]" = torch.ops.aten.permute.default(permute_1, [0, 1, 3, 2]);  permute_1 = None
        expand: "f16[8, 16, 1024, 32]" = torch.ops.aten.expand.default(permute, [8, 16, 1024, 32]);  permute = None
        clone: "f16[8, 16, 1024, 32]" = torch.ops.aten.clone.default(expand, memory_format = torch.contiguous_format);  expand = None
        view_5: "f16[128, 1024, 32]" = torch.ops.aten.view.default(clone, [128, 1024, 32]);  clone = None
        expand_1: "f16[8, 16, 32, 1024]" = torch.ops.aten.expand.default(permute_3, [8, 16, 32, 1024]);  permute_3 = None
        clone_1: "f16[8, 16, 32, 1024]" = torch.ops.aten.clone.default(expand_1, memory_format = torch.contiguous_format);  expand_1 = None
        view_6: "f16[128, 32, 1024]" = torch.ops.aten.view.default(clone_1, [128, 32, 1024]);  clone_1 = None
        bmm: "f16[128, 1024, 1024]" = torch.ops.aten.bmm.default(view_5, view_6);  view_5 = view_6 = None
        view_7: "f16[8, 16, 1024, 1024]" = torch.ops.aten.view.default(bmm, [8, 16, 1024, 1024]);  bmm = None
        div: "f16[8, 16, 1024, 1024]" = torch.ops.aten.div.Tensor(view_7, 5.656854249492381);  view_7 = None
        
         # File: /EuroSys25/fwd-compile-print-withmask.py:31 in fwd_bert_std, code: scores -= 10000.0 * (1.0 - mask.unsqueeze(1))
        unsqueeze: "f32[8, 1, 1024, 1024]" = torch.ops.aten.unsqueeze.default(arg3_1, 1);  arg3_1 = None
        sub: "f32[8, 1, 1024, 1024]" = torch.ops.aten.sub.Tensor(1.0, unsqueeze);  unsqueeze = None
        mul: "f32[8, 1, 1024, 1024]" = torch.ops.aten.mul.Tensor(sub, 10000.0);  sub = None
        sub_1: "f32[8, 16, 1024, 1024]" = torch.ops.aten.sub.Tensor(div, mul);  div = mul = None
        
         # File: /EuroSys25/fwd-compile-print-withmask.py:32 in fwd_bert_std, code: probs = F.softmax(scores, dim=-1)
        convert_element_type_default: "f32[8, 16, 1024, 1024]" = torch.ops.prims.convert_element_type.default(sub_1, torch.float32);  sub_1 = None
        amax: "f32[8, 16, 1024, 1]" = torch.ops.aten.amax.default(convert_element_type_default, [-1], True)
        sub_2: "f32[8, 16, 1024, 1024]" = torch.ops.aten.sub.Tensor(convert_element_type_default, amax);  convert_element_type_default = amax = None
        exp: "f32[8, 16, 1024, 1024]" = torch.ops.aten.exp.default(sub_2);  sub_2 = None
        sum_1: "f32[8, 16, 1024, 1]" = torch.ops.aten.sum.dim_IntList(exp, [-1], True)
        div_1: "f32[8, 16, 1024, 1024]" = torch.ops.aten.div.Tensor(exp, sum_1);  exp = sum_1 = None
        convert_element_type_6: "f16[8, 16, 1024, 1024]" = torch.ops.prims.convert_element_type.default(div_1, torch.float16);  div_1 = None
        
         # File: /EuroSys25/fwd-compile-print-withmask.py:34 in fwd_bert_std, code: h = torch.matmul(probs, v)
        expand_2: "f16[8, 16, 1024, 1024]" = torch.ops.aten.expand.default(convert_element_type_6, [8, 16, 1024, 1024]);  convert_element_type_6 = None
        view_8: "f16[128, 1024, 1024]" = torch.ops.aten.view.default(expand_2, [128, 1024, 1024]);  expand_2 = None
        expand_3: "f16[8, 16, 1024, 32]" = torch.ops.aten.expand.default(permute_2, [8, 16, 1024, 32]);  permute_2 = None
        clone_2: "f16[8, 16, 1024, 32]" = torch.ops.aten.clone.default(expand_3, memory_format = torch.contiguous_format);  expand_3 = None
        view_9: "f16[128, 1024, 32]" = torch.ops.aten.view.default(clone_2, [128, 1024, 32]);  clone_2 = None
        bmm_1: "f16[128, 1024, 32]" = torch.ops.aten.bmm.default(view_8, view_9);  view_8 = view_9 = None
        view_10: "f16[8, 16, 1024, 32]" = torch.ops.aten.view.default(bmm_1, [8, 16, 1024, 32]);  bmm_1 = None
        
         # File: /EuroSys25/fwd-compile-print-withmask.py:37 in fwd_bert_std, code: h = h.permute(0, 2, 1, 3).contiguous()
        permute_4: "f16[8, 1024, 16, 32]" = torch.ops.aten.permute.default(view_10, [0, 2, 1, 3]);  view_10 = None
        clone_3: "f16[8, 1024, 16, 32]" = torch.ops.aten.clone.default(permute_4, memory_format = torch.contiguous_format);  permute_4 = None
        
         # File: /EuroSys25/fwd-compile-print-withmask.py:40 in fwd_bert_std, code: hidden_states = h.view(new_context_layer_shape)
        view_11: "f16[8, 1024, 512]" = torch.ops.aten.view.default(clone_3, [8, 1024, 512]);  clone_3 = None
        
         # File: /EuroSys25/fwd-compile-print-withmask.py:43 in fwd_bert_std, code: hidden_states = torch.matmul(hidden_states, attr_output_kernel[layer]) + attr_output_bias[layer]
        view_12: "f16[8192, 512]" = torch.ops.aten.view.default(view_11, [8192, 512]);  view_11 = None
        mm_1: "f16[8192, 512]" = torch.ops.aten.mm.default(view_12, arg4_1);  view_12 = arg4_1 = None
        view_13: "f16[8, 1024, 512]" = torch.ops.aten.view.default(mm_1, [8, 1024, 512]);  mm_1 = None
        add_1: "f16[8, 1024, 512]" = torch.ops.aten.add.Tensor(view_13, arg5_1);  view_13 = arg5_1 = None
        
         # File: /EuroSys25/fwd-compile-print-withmask.py:44 in fwd_bert_std, code: hidden_states = hidden_states + input_tensor  # 残差连接
        add_2: "f16[8, 1024, 512]" = torch.ops.aten.add.Tensor(add_1, arg0_1);  add_1 = arg0_1 = None
        
         # File: /EuroSys25/fwd-compile-print-withmask.py:47 in fwd_bert_std, code: hidden_states = F.layer_norm(hidden_states, (hidden_dim, ), weight=attr_output_layernorm_gamma[layer], bias=attr_output_layernorm_beta[layer])
        convert_element_type_11: "f32[8, 1024, 512]" = torch.ops.prims.convert_element_type.default(add_2, torch.float32)
        var_mean = torch.ops.aten.var_mean.correction(convert_element_type_11, [2], correction = 0, keepdim = True);  convert_element_type_11 = None
        getitem_3: "f32[8, 1024, 1]" = var_mean[0]
        getitem_4: "f32[8, 1024, 1]" = var_mean[1];  var_mean = None
        add_3: "f32[8, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_3, 1e-05);  getitem_3 = None
        rsqrt: "f32[8, 1024, 1]" = torch.ops.aten.rsqrt.default(add_3);  add_3 = None
        sub_3: "f32[8, 1024, 512]" = torch.ops.aten.sub.Tensor(add_2, getitem_4);  add_2 = getitem_4 = None
        mul_1: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(sub_3, rsqrt);  sub_3 = rsqrt = None
        mul_2: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_1, arg7_1);  mul_1 = arg7_1 = None
        add_4: "f32[8, 1024, 512]" = torch.ops.aten.add.Tensor(mul_2, arg6_1);  mul_2 = arg6_1 = None
        convert_element_type_12: "f16[8, 1024, 512]" = torch.ops.prims.convert_element_type.default(add_4, torch.float16);  add_4 = None
        
         # File: /EuroSys25/fwd-compile-print-withmask.py:51 in fwd_bert_std, code: hidden_states = torch.matmul(hidden_states, inter_kernel[layer]) + inter_bias[layer]
        view_14: "f16[8192, 512]" = torch.ops.aten.view.default(convert_element_type_12, [8192, 512])
        mm_2: "f16[8192, 2048]" = torch.ops.aten.mm.default(view_14, arg8_1);  view_14 = arg8_1 = None
        view_15: "f16[8, 1024, 2048]" = torch.ops.aten.view.default(mm_2, [8, 1024, 2048]);  mm_2 = None
        add_5: "f16[8, 1024, 2048]" = torch.ops.aten.add.Tensor(view_15, arg9_1);  view_15 = arg9_1 = None
        
         # File: /EuroSys25/fwd-compile-print-withmask.py:52 in fwd_bert_std, code: hidden_states = F.gelu(hidden_states)  #激活函数
        convert_element_type_15: "f32[8, 1024, 2048]" = torch.ops.prims.convert_element_type.default(add_5, torch.float32);  add_5 = None
        mul_3: "f32[8, 1024, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_15, 0.5)
        mul_4: "f32[8, 1024, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_15, 0.7071067811865476);  convert_element_type_15 = None
        erf: "f32[8, 1024, 2048]" = torch.ops.aten.erf.default(mul_4);  mul_4 = None
        add_6: "f32[8, 1024, 2048]" = torch.ops.aten.add.Tensor(erf, 1);  erf = None
        mul_5: "f32[8, 1024, 2048]" = torch.ops.aten.mul.Tensor(mul_3, add_6);  mul_3 = add_6 = None
        convert_element_type_16: "f16[8, 1024, 2048]" = torch.ops.prims.convert_element_type.default(mul_5, torch.float16);  mul_5 = None
        
         # File: /EuroSys25/fwd-compile-print-withmask.py:54 in fwd_bert_std, code: hidden_states = torch.matmul(hidden_states, output_kernel[layer]) + output_bias[layer]
        view_16: "f16[8192, 2048]" = torch.ops.aten.view.default(convert_element_type_16, [8192, 2048]);  convert_element_type_16 = None
        mm_3: "f16[8192, 512]" = torch.ops.aten.mm.default(view_16, arg10_1);  view_16 = arg10_1 = None
        view_17: "f16[8, 1024, 512]" = torch.ops.aten.view.default(mm_3, [8, 1024, 512]);  mm_3 = None
        add_7: "f16[8, 1024, 512]" = torch.ops.aten.add.Tensor(view_17, arg11_1);  view_17 = arg11_1 = None
        
         # File: /EuroSys25/fwd-compile-print-withmask.py:55 in fwd_bert_std, code: hidden_states = hidden_states + residual  #残差连接
        add_8: "f16[8, 1024, 512]" = torch.ops.aten.add.Tensor(add_7, convert_element_type_12);  add_7 = convert_element_type_12 = None
        
         # File: /EuroSys25/fwd-compile-print-withmask.py:58 in fwd_bert_std, code: hidden_states = F.layer_norm(hidden_states, (hidden_dim, ),  weight=output_layernorm_gamma[layer], bias=output_layernorm_beta[layer])
        convert_element_type_19: "f32[8, 1024, 512]" = torch.ops.prims.convert_element_type.default(add_8, torch.float32)
        var_mean_1 = torch.ops.aten.var_mean.correction(convert_element_type_19, [2], correction = 0, keepdim = True);  convert_element_type_19 = None
        getitem_5: "f32[8, 1024, 1]" = var_mean_1[0]
        getitem_6: "f32[8, 1024, 1]" = var_mean_1[1];  var_mean_1 = None
        add_9: "f32[8, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_5, 1e-05);  getitem_5 = None
        rsqrt_1: "f32[8, 1024, 1]" = torch.ops.aten.rsqrt.default(add_9);  add_9 = None
        sub_4: "f32[8, 1024, 512]" = torch.ops.aten.sub.Tensor(add_8, getitem_6);  add_8 = getitem_6 = None
        mul_6: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(sub_4, rsqrt_1);  sub_4 = rsqrt_1 = None
        mul_7: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_6, arg13_1);  mul_6 = arg13_1 = None
        add_10: "f32[8, 1024, 512]" = torch.ops.aten.add.Tensor(mul_7, arg12_1);  mul_7 = arg12_1 = None
        convert_element_type_20: "f16[8, 1024, 512]" = torch.ops.prims.convert_element_type.default(add_10, torch.float16);  add_10 = None
        return (convert_element_type_20,)
        