# 10.20 evaA 测试脚本 
#
# pytorch原版 和 flash-attm版，均以Aten算子组合为基础
# 当前这个版本没有包含 ours的
# batch_size 8 16
# seqlen: 128 256 512 1024 2048 4096 (对于PyTorch 8192爆显存)
# 

import torch
import torch.nn.functional as F
from utils.utils import torch_cuda_identify, time_stamp_cudasync, seqlen_to_mask, set_dtype, transpose_for_scores
from utils.masks import seqlen_to_mask, generate_triangle_mask
import config

def Attention_std():
    scores = torch.matmul(q, k.transpose(-2, -1)) / (q.shape[3] ** .5)
    scores -= 10000.0 * (1.0 - mask.unsqueeze(1))
    probs = F.softmax(scores, dim=-1)
    h = torch.matmul(probs, v)

def Flash_Attention(score_for_FA):
    score_for_FA -= 10000.0 * (1.0 - mask.unsqueeze(1))
    _scaled_dot_product_flash_attention_default = torch.ops.aten._scaled_dot_product_flash_attention.default(q, k, v, is_causal=True, scale=(head_size ** .5))    # is_causal设置mask
    h = _scaled_dot_product_flash_attention_default[0]
    

if __name__ == '__main__':
    device = torch_cuda_identify(print_info=True)
    
    torch.manual_seed(0)
    
    batch_size = config.BATCH_SIZE # (8k) batch_size * seq_len / seq_len
    head_size = config.HEAD_DIM   # head_dim aka. head_size
    seq_len = config.SEQ_LEN
    head_num = config.HEAD_NUM
    hidden_dim = head_num * head_size
    layer_num = config.LAYER_NUM
    
    data_type = torch.float16
    dtype = "fp16"
    warmup_iters = config.WARMUP_TIME
    running_iters = config.RUNNING_TIME
    

    # Navie PyTorch Attn Time -----------------------------------------
    print("Attn base time cost:")
    
    for batch_test in [8, 16]:
        batch_size = batch_test
        for seqlen_test in [128, 256, 512, 1024, 2048, 4096]: # 8192 的时候爆显存了
            seq_len = seqlen_test
            
            # 此处由于 原始的总是稠密处理，所以mask形状对性能没有影响
            mask_id = 0
            avg_seq_len = seq_len
            low, high = (2 * avg_seq_len - seq_len, seq_len + 1)
            input_lens = torch.randint(low=low, high=high, size=(batch_size,))
            seqlen_mask = seqlen_to_mask(input_lens, seq_len)
            attr_mask   = torch.tile(seqlen_mask, dims=(seq_len,)).reshape(batch_size, seq_len, seq_len)
            
            lower_triangle_mask = generate_triangle_mask(attr_mask)
            mask_name = 'Lower_triangle_mask'
            mask = lower_triangle_mask.to(torch.float16).cuda()
            
            input_from_tensor = set_dtype(torch.empty(batch_size, seq_len, hidden_dim).uniform_(-0.4, 0.4).cuda(), dtype)
            qkv_kernel = [set_dtype(torch.zeros(hidden_dim, hidden_dim * 3).uniform_(-0.4, 0.4).cuda(), dtype) for _ in range(layer_num)]
            qkv_bias = [set_dtype(torch.zeros(hidden_dim * 3).uniform_(-0.4, 0.4).cuda(), dtype) for _ in range(layer_num)]
            
            hidden_states = input_from_tensor
            layer = 0 
            qkv = torch.matmul(hidden_states, qkv_kernel[layer]) + qkv_bias[layer]
            q, k, v = qkv.chunk(3, dim=-1)
            q = transpose_for_scores(q, head_num, head_size)
            k = transpose_for_scores(k, head_num, head_size)
            v = transpose_for_scores(v, head_num, head_size)
            
            for iter in range(warmup_iters + running_iters):
                if iter == warmup_iters:
                    t0_start = time_stamp_cudasync()
                output = Attention_std()
            t0_end = time_stamp_cudasync() 
            
            print("bs:{}\t| seqlen:{}\t| {:.2f} ms / iter".format(batch_size, seq_len, (t0_end - t0_start) * 1000 / running_iters)) 
    
    
    # Flash Attention Time ----------------------------------------------
    print("Flash-Attn time cost:")
    
    for batch_test in [8,16]:
        batch_size = batch_test
        for seqlen_test in [128, 256, 512, 1024, 2048, 4096, 8192]:
            seq_len = seqlen_test
            
            mask_id = 0
            avg_seq_len = seq_len
            low, high = (2 * avg_seq_len - seq_len, seq_len + 1)
            input_lens = torch.randint(low=low, high=high, size=(batch_size,))
            seqlen_mask = seqlen_to_mask(input_lens, seq_len)
            attr_mask   = torch.tile(seqlen_mask, dims=(seq_len,)).reshape(batch_size, seq_len, seq_len)
            
            lower_triangle_mask = generate_triangle_mask(attr_mask)
            mask_name = 'Lower_triangle_mask'
            mask = lower_triangle_mask.to(torch.float16).cuda()
            
            input_from_tensor = set_dtype(torch.empty(batch_size, seq_len, hidden_dim).uniform_(-0.4, 0.4).cuda(), dtype)
            qkv_kernel = [set_dtype(torch.zeros(hidden_dim, hidden_dim * 3).uniform_(-0.4, 0.4).cuda(), dtype) for _ in range(layer_num)]
            qkv_bias = [set_dtype(torch.zeros(hidden_dim * 3).uniform_(-0.4, 0.4).cuda(), dtype) for _ in range(layer_num)]
            score_for_FA = set_dtype(torch.zeros(batch_size, 1, seq_len, seq_len).uniform_(-0.4, 0.4).cuda(), dtype)
            
            hidden_states = input_from_tensor
            layer = 0 
            qkv = torch.matmul(hidden_states, qkv_kernel[layer]) + qkv_bias[layer]
            q, k, v = qkv.chunk(3, dim=-1)
            q = transpose_for_scores(q, head_num, head_size)
            k = transpose_for_scores(k, head_num, head_size)
            v = transpose_for_scores(v, head_num, head_size)
            
            for iter in range(warmup_iters + running_iters):
                if iter == warmup_iters:
                    t1_start = time_stamp_cudasync()
                output = Flash_Attention(score_for_FA)
            t1_end = time_stamp_cudasync() 
            
            print("bs:{}\t| seqlen:{}\t| {:.3f} ms / iter".format(batch_size, seq_len, (t1_end - t1_start) * 1000 / running_iters)) 
            