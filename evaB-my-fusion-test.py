# 10.21 evaB 测试脚本
# 端到端实验，使用我的融合策略和算子
# 

import torch
from utils.utils import torch_cuda_identify, time_stamp_cudasync, seqlen_to_mask
from utils.masks import seqlen_to_mask, generate_triangle_mask
import config
from ops.package_op import syncfree_strided_attn_op, syncfree_fixed_attn_op, syncfree_band_attn_op, syncfree_dilated_attn_op

def fwd_bert_seg1():
    view = torch.ops.aten.view.default(arg0, [batch_size * seq_len, hidden_dim])
    mm = torch.ops.aten.mm.default(view, arg1)              # mm 1
    view_1 = torch.ops.aten.view.default(mm, [batch_size, seq_len, hidden_dim * 3])
    add = torch.ops.aten.add.Tensor(view_1, arg2)  
    split = torch.ops.aten.split.Tensor(add, hidden_dim, -1)
    getitem = split[0]
    getitem_1 = split[1]
    getitem_2 = split[2]
    view_2 = torch.ops.aten.view.default(getitem, [batch_size, seq_len, head_num, head_size])
    view_3 = torch.ops.aten.view.default(getitem_1, [batch_size, seq_len, head_num, head_size])
    view_4 = torch.ops.aten.view.default(getitem_2, [batch_size, seq_len, head_num, head_size])
    permute = torch.ops.aten.permute.default(view_2, [0, 2, 1, 3])
    permute_1 = torch.ops.aten.permute.default(view_3, [0, 2, 1, 3])
    permute_2 = torch.ops.aten.permute.default(view_4, [0, 2, 1, 3])
    
    return permute, permute_1, permute_2
    
    # Attention ------ Start
    # insert my kernel
    # Attention ------ End
    
def fwd_bert_seg2():
    view_12 = torch.ops.aten.view.default(view_11, [batch_size * seq_len, hidden_dim])
    mm_1 = torch.ops.aten.mm.default(view_12, arg4)           # mm 2
    view_13 = torch.ops.aten.view.default(mm_1, [batch_size, seq_len, hidden_dim])
    add_1 = torch.ops.aten.add.Tensor(view_13, arg5)
    add_2 = torch.ops.aten.add.Tensor(add_1, arg0)
    convert_element_type_11 = torch.ops.prims.convert_element_type.default(add_2, torch.float32)
    var_mean = torch.ops.aten.var_mean.correction(convert_element_type_11, [2], correction=0, keepdim=True)
    getitem_3 = var_mean[0]
    getitem_4 = var_mean[1]
    add_3 = torch.ops.aten.add.Tensor(getitem_3, 1e-05)
    rsqrt = torch.ops.aten.rsqrt.default(add_3)
    sub_3 = torch.ops.aten.sub.Tensor(add_2, getitem_4)
    mul_1 = torch.ops.aten.mul.Tensor(sub_3, rsqrt)
    mul_2 = torch.ops.aten.mul.Tensor(mul_1, arg7)
    add_4 = torch.ops.aten.add.Tensor(mul_2, arg6)
    convert_element_type_12 = torch.ops.prims.convert_element_type.default(add_4, torch.float16)
    view_14 = torch.ops.aten.view.default(convert_element_type_12, [batch_size * seq_len, hidden_dim])
    
    return view_14, convert_element_type_12
    
def fwd_bert_seg3(view_14):
    mm_2 = torch.ops.aten.mm.default(view_14, arg8)            # mm 3
    view_15 = torch.ops.aten.view.default(mm_2, [batch_size, seq_len, hidden_dim * 4])
    add_5 = torch.ops.aten.add.Tensor(view_15, arg9)
    convert_element_type_15 = torch.ops.prims.convert_element_type.default(add_5, torch.float32)
    mul_3 = torch.ops.aten.mul.Tensor(convert_element_type_15, 0.5)
    mul_4 = torch.ops.aten.mul.Tensor(convert_element_type_15, 0.7071067811865476)
    erf = torch.ops.aten.erf.default(mul_4)
    add_6 = torch.ops.aten.add.Tensor(erf, 1)
    mul_5 = torch.ops.aten.mul.Tensor(mul_3, add_6)
    convert_element_type_16 = torch.ops.prims.convert_element_type.default(mul_5, torch.float16)
    view_16 = torch.ops.aten.view.default(convert_element_type_16, [batch_size * seq_len, hidden_dim * 4])
    
    return view_16
    
def fwd_bert_seg4(view_16, convert_element_type_12):
    mm_3 = torch.ops.aten.mm.default(view_16, arg10)           # mm 4
    view_17 = torch.ops.aten.view.default(mm_3, [batch_size, seq_len, hidden_dim])
    add_7 = torch.ops.aten.add.Tensor(view_17, arg11)
    add_8 = torch.ops.aten.add.Tensor(add_7, convert_element_type_12)
    convert_element_type_19 = torch.ops.prims.convert_element_type.default(add_8, torch.float32)
    var_mean_1 = torch.ops.aten.var_mean.correction(convert_element_type_19, [2], correction=0, keepdim=True)
    getitem_5 = var_mean_1[0]
    getitem_6 = var_mean_1[1]
    add_9 = torch.ops.aten.add.Tensor(getitem_5, 1e-05)
    rsqrt_1 = torch.ops.aten.rsqrt.default(add_9)
    sub_4 = torch.ops.aten.sub.Tensor(add_8, getitem_6)
    mul_6 = torch.ops.aten.mul.Tensor(sub_4, rsqrt_1)
    mul_7 = torch.ops.aten.mul.Tensor(mul_6, arg13)
    add_10 = torch.ops.aten.add.Tensor(mul_7, arg12)
    convert_element_type_20 = torch.ops.prims.convert_element_type.default(add_10, torch.float16)
    
    return (convert_element_type_20,)


if __name__ == '__main__':
    device = torch_cuda_identify(print_info=True)
    
    torch.manual_seed(0)
    
    batch_size = config.BATCH_SIZE # (8k) batch_size * seq_len / seq_len
    head_size = config.HEAD_DIM   # head_dim aka. head_size
    seq_len = config.SEQ_LEN
    head_num = config.HEAD_NUM
    hidden_dim = head_num * head_size
    mask_id = config.MASK_ID
    
    data_type = torch.float16
    warmup_iters = config.WARMUP_TIME
    running_iters = config.RUNNING_TIME
    
    avg_seq_len = seq_len
    low, high = (2 * avg_seq_len - seq_len, seq_len + 1)
    input_lens = torch.randint(low=low, high=high, size=(batch_size,))
    seqlen_mask = seqlen_to_mask(input_lens, seq_len)
    attr_mask   = torch.tile(seqlen_mask, dims=(seq_len,)).reshape(batch_size, seq_len, seq_len)
    
    lower_triangle_mask = generate_triangle_mask(attr_mask)
    mask = lower_triangle_mask.to(torch.float16).cuda()
    
    # para
    arg0 = torch.rand((batch_size, seq_len, hidden_dim), device=device, dtype=torch.float16)
    arg1 = torch.rand((hidden_dim, hidden_dim * 3), device=device, dtype=torch.float16)
    arg2 = torch.rand((hidden_dim * 3, ), device=device, dtype=torch.float16)
    arg3 = mask
    arg4 = torch.rand((hidden_dim, hidden_dim), device=device, dtype=torch.float16)
    arg5 = torch.rand((hidden_dim, ), device=device, dtype=torch.float16)
    arg6 = torch.rand((hidden_dim, ), device=device, dtype=torch.float16)
    arg7 = torch.rand((hidden_dim, ), device=device, dtype=torch.float16)
    arg8 = torch.rand((hidden_dim, hidden_dim * 4), device=device, dtype=torch.float16)
    arg9 = torch.rand((hidden_dim * 4, ), device=device, dtype=torch.float16)
    arg10 = torch.rand((hidden_dim * 4, hidden_dim), device=device, dtype=torch.float16)
    arg11 = torch.rand((hidden_dim, ), device=device, dtype=torch.float16)
    arg12 = torch.rand((hidden_dim, ), device=device, dtype=torch.float16)
    arg13 = torch.rand((hidden_dim, ), device=device, dtype=torch.float16)

    


    # torch._dynamo.reset()
    # fwd_compile_autotune = torch.compile(, mode="default")
    
    permute, permute_1, permute_2 = fwd_bert_seg1()
    view_11 = syncfree_strided_attn_op(permute, permute_1, permute_2)
    view14, convert_element_type_12 = fwd_bert_seg2()
    view_16 = fwd_bert_seg3(view14)
    result = fwd_bert_seg4(view_16, convert_element_type_12)
    
        
    print("pass !")

    # for iter in range(warmup_iters + running_iters):
    #     if iter == warmup_iters:
    #         t1_start = time_stamp_cudasync()
            
    #     output = fwd_compile_autotune()
    # t1_end = time_stamp_cudasync()    
    
    # print("Torch Autotune | bs:{}\t| seqlen:{}\t| {:.2f}\tms / iter".format(batch_size, seq_len, (t1_end - t1_start) * 1000 / running_iters)) 

    
    