# 10.18 fwd-base-Aten.py 无注释纯净版
# 但没有 mask
import torch
from utils.utils import torch_cuda_identify, time_stamp_cudasync
import config


def fwd_bert_std_Aten():
    view = torch.ops.aten.view.default(arg0_1, [batch_size * seq_len, hidden_dim])
    mm = torch.ops.aten.mm.default(view, arg1_1)
    view_1 = torch.ops.aten.view.default(mm, [batch_size, seq_len, hidden_dim * 3])
    add = torch.ops.aten.add.Tensor(view_1, arg2_1)
    split = torch.ops.aten.split.Tensor(add, hidden_dim, -1)
    getitem = split[0]
    getitem_1 = split[1]
    getitem_2 = split[2]
    view_2 = torch.ops.aten.view.default(getitem, [batch_size, seq_len, head_num, head_size])
    view_3 = torch.ops.aten.view.default(getitem_1, [batch_size, seq_len, head_num, head_size])
    view_4 = torch.ops.aten.view.default(getitem_2, [batch_size, seq_len, head_num, head_size])
    permute_default = torch.ops.aten.permute.default(view_2, [0, 2, 1, 3])
    permute_default_1 = torch.ops.aten.permute.default(view_3, [0, 2, 1, 3])
    permute_default_2 = torch.ops.aten.permute.default(view_4, [0, 2, 1, 3])
    
    _scaled_dot_product_flash_attention_default = torch.ops.aten._scaled_dot_product_flash_attention.default(permute_default, permute_default_1, permute_default_2, scale=(head_size ** .5))
    getitem_7 = _scaled_dot_product_flash_attention_default[0]
    
    permute_4 = torch.ops.aten.permute.default(getitem_7, [0, 2, 1, 3])
    clone_3 = torch.ops.aten.clone.default(permute_4, memory_format=torch.contiguous_format)
    view_11 = torch.ops.aten.view.default(clone_3, [batch_size, seq_len, hidden_dim])
    view_12 = torch.ops.aten.view.default(view_11, [batch_size * seq_len, hidden_dim])
    mm_1 = torch.ops.aten.mm.default(view_12, arg3_1)
    view_13 = torch.ops.aten.view.default(mm_1, [batch_size, seq_len, hidden_dim])
    add_1 = torch.ops.aten.add.Tensor(view_13, arg4_1)
    add_2 = torch.ops.aten.add.Tensor(add_1, arg0_1)
    convert_element_type_10 = torch.ops.prims.convert_element_type.default(add_2, torch.float32)
    var_mean = torch.ops.aten.var_mean.correction(convert_element_type_10, [2], correction=0, keepdim=True)
    getitem_3 = var_mean[0]
    getitem_4 = var_mean[1]
    add_3 = torch.ops.aten.add.Tensor(getitem_3, 1e-05)
    rsqrt = torch.ops.aten.rsqrt.default(add_3)
    sub_1 = torch.ops.aten.sub.Tensor(add_2, getitem_4)
    mul = torch.ops.aten.mul.Tensor(sub_1, rsqrt)
    mul_1 = torch.ops.aten.mul.Tensor(mul, arg6_1)
    add_4 = torch.ops.aten.add.Tensor(mul_1, arg5_1)
    convert_element_type_11 = torch.ops.prims.convert_element_type.default(add_4, torch.float16)
    view_14 = torch.ops.aten.view.default(convert_element_type_11, [batch_size * seq_len, hidden_dim])
    mm_2 = torch.ops.aten.mm.default(view_14, arg7_1)
    view_15 = torch.ops.aten.view.default(mm_2, [batch_size, seq_len, seq_len * 2])
    add_5 = torch.ops.aten.add.Tensor(view_15, arg8_1)
    convert_element_type_14 = torch.ops.prims.convert_element_type.default(add_5, torch.float32)
    mul_2 = torch.ops.aten.mul.Tensor(convert_element_type_14, 0.5)
    mul_3 = torch.ops.aten.mul.Tensor(convert_element_type_14, 0.7071067811865476)
    erf = torch.ops.aten.erf.default(mul_3)
    add_6 = torch.ops.aten.add.Tensor(erf, 1)
    mul_4 = torch.ops.aten.mul.Tensor(mul_2, add_6)
    convert_element_type_15 = torch.ops.prims.convert_element_type.default(mul_4, torch.float16)
    view_16 = torch.ops.aten.view.default(convert_element_type_15, [batch_size * seq_len, seq_len * 2])
    mm_3 = torch.ops.aten.mm.default(view_16, arg9_1)
    view_17 = torch.ops.aten.view.default(mm_3, [batch_size, seq_len, hidden_dim])
    add_7 = torch.ops.aten.add.Tensor(view_17, arg10_1)
    add_8 = torch.ops.aten.add.Tensor(add_7, convert_element_type_11)
    convert_element_type_18 = torch.ops.prims.convert_element_type.default(add_8, torch.float32)
    var_mean_1 = torch.ops.aten.var_mean.correction(convert_element_type_18, [2], correction=0, keepdim=True)
    getitem_5 = var_mean_1[0]
    getitem_6 = var_mean_1[1]
    add_9 = torch.ops.aten.add.Tensor(getitem_5, 1e-05)
    rsqrt_1 = torch.ops.aten.rsqrt.default(add_9)
    sub_2 = torch.ops.aten.sub.Tensor(add_8, getitem_6)
    mul_5 = torch.ops.aten.mul.Tensor(sub_2, rsqrt_1)
    mul_6 = torch.ops.aten.mul.Tensor(mul_5, arg12_1)
    add_10 = torch.ops.aten.add.Tensor(mul_6, arg11_1)
    convert_element_type_19 = torch.ops.prims.convert_element_type.default(add_10, torch.float16)
    
    return (convert_element_type_19,)
        

if __name__ == '__main__':
    device = torch_cuda_identify(print_info=False)
    
    torch.manual_seed(0)
    
    head_size = config.HEAD_DIM   # head_dim aka. head_size
    seq_len = config.SEQ_LEN
    head_num = config.HEADS_NUM
    batch_size = config.BATCH_SIZE # (8k) batch_size * seq_len / seq_len
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

    for iter in range(warmup_iters + running_iters):
        if iter == warmup_iters:
            t0_start = time_stamp_cudasync()
        output = fwd_bert_std_Aten()
    t0_end = time_stamp_cudasync()    
    
    print("Aten Base time:  \t{:.2f} ms / iter".format((t0_end - t0_start) * 1000 / running_iters)) 
    
    
    