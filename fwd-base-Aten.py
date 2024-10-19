# 10.18 fwd-base-Aten.py
# 
# 以fwd的前向过程为基准不动（里面不包含masking）区别于fwd-base-origin.py是pytorch算子代码
# 本代码拆分到了meta算子（ATen算子）这一层
# 程序来源于 fwd-compile-print.py的debug输出，
# 存于`./torch_compile_debug/fwd-default-as-base/torchinductor/model__0_inference_0.0/fx_graph_runnable.py `
# python fwd-base-Aten.py

import torch
from utils.utils import torch_cuda_identify, time_stamp_cudasync
import config


def fwd_bert_std_Aten(arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1):
    
    #  qkv = torch.matmul(hidden_states, qkv_kernel[layer]) + qkv_bias[layer]
    view = torch.ops.aten.view.default(arg0_1, [8192, 512])
    mm = torch.ops.aten.mm.default(view, arg1_1);  view = arg1_1 = None
    view_1 = torch.ops.aten.view.default(mm, [8, 1024, 1536]);  mm = None
    add = torch.ops.aten.add.Tensor(view_1, arg2_1);  view_1 = arg2_1 = None
    
    #q, k, v = qkv.chunk(3, dim=-1)
    split = torch.ops.aten.split.Tensor(add, 512, -1);  add = None
    getitem = split[0]
    getitem_1 = split[1]
    getitem_2 = split[2];  split = None
    
    #q = transpose_for_scores(q, head_num, head_size)
    view_2 = torch.ops.aten.view.default(getitem, [8, 1024, 16, 32]);  getitem = None
    view_3 = torch.ops.aten.view.default(getitem_1, [8, 1024, 16, 32]);  getitem_1 = None
    view_4 = torch.ops.aten.view.default(getitem_2, [8, 1024, 16, 32]);  getitem_2 = None
    permute_default = torch.ops.aten.permute.default(view_2, [0, 2, 1, 3]);  view_2 = None
    permute_default_1 = torch.ops.aten.permute.default(view_3, [0, 2, 1, 3]);  view_3 = None
    permute_default_2 = torch.ops.aten.permute.default(view_4, [0, 2, 1, 3]);  view_4 = None
    
    # scores = torch.matmul(q, k.transpose(-2, -1)) / (head_size ** .5)
    # probs = F.softmax(scores, dim=-1)
    # h = torch.matmul(probs, v)
    _scaled_dot_product_flash_attention_default = torch.ops.aten._scaled_dot_product_flash_attention.default(permute_default, permute_default_1, permute_default_2, scale = 0.17677669529663687);  permute_default = permute_default_1 = permute_default_2 = None
    getitem_7 = _scaled_dot_product_flash_attention_default[0];  _scaled_dot_product_flash_attention_default = None
    
    # h = h.permute(0, 2, 1, 3).contiguous()
    permute_4 = torch.ops.aten.permute.default(getitem_7, [0, 2, 1, 3]);  getitem_7 = None
    clone_3 = torch.ops.aten.clone.default(permute_4, memory_format = torch.contiguous_format);  permute_4 = None
    
    # hidden_states = h.view(new_context_layer_shape)
    view_11 = torch.ops.aten.view.default(clone_3, [8, 1024, 512]);  clone_3 = None
    
    # hidden_states = torch.matmul(hidden_states, attr_output_kernel[layer]) + attr_output_bias[layer]
    view_12 = torch.ops.aten.view.default(view_11, [8192, 512]);  view_11 = None
    mm_1 = torch.ops.aten.mm.default(view_12, arg3_1);  view_12 = arg3_1 = None
    view_13 = torch.ops.aten.view.default(mm_1, [8, 1024, 512]);  mm_1 = None
    add_1 = torch.ops.aten.add.Tensor(view_13, arg4_1);  view_13 = arg4_1 = None
    
    # hidden_states = hidden_states + input_tensor
    add_2 = torch.ops.aten.add.Tensor(add_1, arg0_1);  add_1 = arg0_1 = None
    
    # hidden_states = F.layer_norm(hidden_states, (hidden_dim, ),  weight=attr_output_layernorm_gamma[layer], bias=attr_output_layernorm_beta[layer])
    convert_element_type_10 = torch.ops.prims.convert_element_type.default(add_2, torch.float32)
    var_mean = torch.ops.aten.var_mean.correction(convert_element_type_10, [2], correction = 0, keepdim = True);  convert_element_type_10 = None
    getitem_3 = var_mean[0]
    getitem_4 = var_mean[1];  var_mean = None
    add_3 = torch.ops.aten.add.Tensor(getitem_3, 1e-05);  getitem_3 = None
    rsqrt = torch.ops.aten.rsqrt.default(add_3);  add_3 = None
    sub_1 = torch.ops.aten.sub.Tensor(add_2, getitem_4);  add_2 = getitem_4 = None
    mul = torch.ops.aten.mul.Tensor(sub_1, rsqrt);  sub_1 = rsqrt = None
    mul_1 = torch.ops.aten.mul.Tensor(mul, arg6_1);  mul = arg6_1 = None
    add_4 = torch.ops.aten.add.Tensor(mul_1, arg5_1);  mul_1 = arg5_1 = None
    convert_element_type_11 = torch.ops.prims.convert_element_type.default(add_4, torch.float16);  add_4 = None
    
    # hidden_states = torch.matmul(hidden_states, inter_kernel[layer]) + inter_bias[layer]
    view_14 = torch.ops.aten.view.default(convert_element_type_11, [8192, 512])
    mm_2 = torch.ops.aten.mm.default(view_14, arg7_1);  view_14 = arg7_1 = None
    view_15 = torch.ops.aten.view.default(mm_2, [8, 1024, 2048]);  mm_2 = None
    add_5 = torch.ops.aten.add.Tensor(view_15, arg8_1);  view_15 = arg8_1 = None
    
    # hidden_states = F.gelu(hidden_states)  #激活函数
    convert_element_type_14 = torch.ops.prims.convert_element_type.default(add_5, torch.float32);  add_5 = None
    mul_2 = torch.ops.aten.mul.Tensor(convert_element_type_14, 0.5)
    mul_3 = torch.ops.aten.mul.Tensor(convert_element_type_14, 0.7071067811865476);  convert_element_type_14 = None
    erf = torch.ops.aten.erf.default(mul_3);  mul_3 = None
    add_6 = torch.ops.aten.add.Tensor(erf, 1);  erf = None
    mul_4 = torch.ops.aten.mul.Tensor(mul_2, add_6);  mul_2 = add_6 = None
    convert_element_type_15 = torch.ops.prims.convert_element_type.default(mul_4, torch.float16);  mul_4 = None
    
    # hidden_states = torch.matmul(hidden_states, output_kernel[layer]) + output_bias[layer]
    view_16 = torch.ops.aten.view.default(convert_element_type_15, [8192, 2048]);  convert_element_type_15 = None
    mm_3 = torch.ops.aten.mm.default(view_16, arg9_1);  view_16 = arg9_1 = None
    view_17 = torch.ops.aten.view.default(mm_3, [8, 1024, 512]);  mm_3 = None
    add_7 = torch.ops.aten.add.Tensor(view_17, arg10_1);  view_17 = arg10_1 = None
    
    # hidden_states = hidden_states + residual  #残差连接
    add_8 = torch.ops.aten.add.Tensor(add_7, convert_element_type_11);  add_7 = convert_element_type_11 = None
    
    # hidden_states = F.layer_norm(hidden_states, (hidden_dim, ),w eight=output_layernorm_gamma[layer], bias=output_layernorm_beta[layer])  
    convert_element_type_18 = torch.ops.prims.convert_element_type.default(add_8, torch.float32)
    var_mean_1 = torch.ops.aten.var_mean.correction(convert_element_type_18, [2], correction = 0, keepdim = True);  convert_element_type_18 = None
    getitem_5 = var_mean_1[0]
    getitem_6 = var_mean_1[1];  var_mean_1 = None
    add_9 = torch.ops.aten.add.Tensor(getitem_5, 1e-05);  getitem_5 = None
    rsqrt_1 = torch.ops.aten.rsqrt.default(add_9);  add_9 = None
    sub_2 = torch.ops.aten.sub.Tensor(add_8, getitem_6);  add_8 = getitem_6 = None
    mul_5 = torch.ops.aten.mul.Tensor(sub_2, rsqrt_1);  sub_2 = rsqrt_1 = None
    mul_6 = torch.ops.aten.mul.Tensor(mul_5, arg12_1);  mul_5 = arg12_1 = None
    add_10 = torch.ops.aten.add.Tensor(mul_6, arg11_1);  mul_6 = arg11_1 = None
    convert_element_type_19 = torch.ops.prims.convert_element_type.default(add_10, torch.float16);  add_10 = None
    
    return (convert_element_type_19,)
        

if __name__ == '__main__':
    device = torch_cuda_identify(print_info=False)
    
    torch.manual_seed(0)
    
    head_size = config.HEAD_DIM   # head_dim aka. head_size
    seq_len = config.SEQ_LEN
    head_num = config.HEADS_NUM
    batch_size = config.BATCH_SIZE # (8k) 8192 / seq_len
    data_type = torch.float16
    
    warmup_iters = config.WARMUP_TIME
    running_iters = config.RUNNING_TIME
    
    hidden_dim = head_num * head_size
    
    arg0_1 = torch.rand((batch_size, seq_len, hidden_dim), device=device, dtype=torch.float16)
    arg1_1 = torch.rand((hidden_dim, hidden_dim * 3), device=device, dtype=torch.float16)
    arg2_1 = torch.rand((hidden_dim * 3, ), device=device, dtype=torch.float16)
    arg3_1 = torch.rand((hidden_dim, hidden_dim), device=device, dtype=torch.float16)
    arg4_1 = torch.rand((hidden_dim, ), device=device, dtype=torch.float16)
    arg5_1 = torch.rand((hidden_dim, ), device=device, dtype=torch.float16)
    arg6_1 = torch.rand((hidden_dim, ), device=device, dtype=torch.float16)
    arg7_1 = torch.rand((hidden_dim, hidden_dim * 4), device=device, dtype=torch.float16)
    arg8_1 = torch.rand((hidden_dim * 4, ), device=device, dtype=torch.float16)
    arg9_1 = torch.rand((hidden_dim * 4, hidden_dim), device=device, dtype=torch.float16)
    arg10_1 = torch.rand((hidden_dim, ), device=device, dtype=torch.float16)
    arg11_1 = torch.rand((hidden_dim, ), device=device, dtype=torch.float16)
    arg12_1 = torch.rand((hidden_dim, ), device=device, dtype=torch.float16)
    
    
    # output = fwd_bert_std_Aten(arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1)
    
    for iter in range(warmup_iters + running_iters):
        if iter == warmup_iters:
            t0_start = time_stamp_cudasync()
        output = fwd_bert_std_Aten(arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1)
    t0_end = time_stamp_cudasync()    
    
    print("Aten Base time:  \t{:.2f} ms / iter".format((t0_end - t0_start) * 1000 / running_iters)) 
    
    
    