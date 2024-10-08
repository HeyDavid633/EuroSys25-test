# 2024.9.15 Sun.
# EuroSys的初步实验，多种类型的attn组合方法 
# 对比测试多种 torch的优化方案，暂时不要mask
# base  |  Jit  |  Torch.compile(Triton)
# 
# python attn-base-jit-compile.py 

import timeit
import torch
import numpy as np
import random
import torch.nn.functional as F
from utils.utils import *
import config

@torch.jit.script
def Attention_Jit(q, k, v):
    scores = torch.matmul(q, k.transpose(-2, -1)) / (q.shape[3] ** .5)
    probs = F.softmax(scores, dim=-1)
    h = torch.matmul(probs, v).permute(0, 2, 1, 3).contiguous() 
    new_context_layer_shape = h.size()[:-2] + (q.shape[1]*q.shape[3], )
    hidden_states = h.view(new_context_layer_shape) 
    return hidden_states

@torch.compile
def Attention_Compile(q, k, v):
    scores = torch.matmul(q, k.transpose(-2, -1)) / (q.shape[3] ** .5)
    probs = F.softmax(scores, dim=-1)
    h = torch.matmul(probs, v).permute(0, 2, 1, 3).contiguous() 
    new_context_layer_shape = h.size()[:-2] + (q.shape[1]*q.shape[3], )
    hidden_states = h.view(new_context_layer_shape) 
    return hidden_states


def attn_example():
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)
    
    torch_cuda_identify()
    
    avg_seq_len = config.AVG_SEQ_LEN
    batch_size = config.BATCH_SIZE
    head_size = config.HEAD_DIM # head_dim aka. head_size
    seq_len = config.SEQ_LEN
    head_num = config.HEADS_NUM
    layer_num = config.LAYER_NUM
    hidden_dim = head_num * head_size
    
    warmup_iters = config.WARMUP_TIME
    running_iters = config.RUNNING_TIME
    dtype = "fp16"


    if avg_seq_len <= 0:
        avg_seq_len = seq_len
    low, high = (2 * avg_seq_len - seq_len, seq_len + 1) if 2 * avg_seq_len > seq_len else (0, 2 * avg_seq_len + 1)
    input_lens = torch.randint(low=low, high=high, size=(batch_size,))
    seqlen_mask = seqlen_to_mask(input_lens, seq_len)
    
    input_from_tensor           = set_dtype(torch.empty(batch_size, seq_len, hidden_dim).uniform_(-0.4, 0.4).cuda(), dtype)
    qkv_kernel                  = [set_dtype(torch.zeros(hidden_dim, hidden_dim * 3).uniform_(-0.4, 0.4).cuda(), dtype) for _ in range(layer_num)]
    qkv_bias                    = [set_dtype(torch.zeros(hidden_dim * 3).uniform_(-0.4, 0.4).cuda(), dtype) for _ in range(layer_num)]
    
        
    hidden_states = input_from_tensor
    layer = 0 
    qkv = torch.matmul(hidden_states, qkv_kernel[layer]) + qkv_bias[layer]
    q, k, v = qkv.chunk(3, dim=-1)
    q = transpose_for_scores(q, head_num, head_size)
    k = transpose_for_scores(k, head_num, head_size)
    v = transpose_for_scores(v, head_num, head_size)


    for i in range(warmup_iters + running_iters):
        if i == warmup_iters:    
            torch.cuda.synchronize()
            t0_start = timeit.default_timer()                   
        scores = torch.matmul(q, k.transpose(-2, -1)) / (head_size ** .5)
        probs = F.softmax(scores, dim=-1)
        h = torch.matmul(probs, v).permute(0, 2, 1, 3).contiguous() 
        new_context_layer_shape = h.size()[:-2] + (hidden_dim, )
        hidden_states0 = h.view(new_context_layer_shape)    
    torch.cuda.synchronize()
    t0_end = timeit.default_timer()
    
    base_time = (t0_end - t0_start) * 1000 / running_iters
    print("bs:{} | seq:{} | Base   : {:.3f} ms / iter".format(batch_size, seq_len, base_time))  
    
    for i in range(warmup_iters + running_iters):
        if i == warmup_iters:    
            torch.cuda.synchronize()
            t1_start = timeit.default_timer()   
        hidden_states1 = Attention_Jit(q, k, v) 
    torch.cuda.synchronize()
    t1_end = timeit.default_timer()
    
    jit_time = (t1_end - t1_start) * 1000 / running_iters
    print("bs:{} | seq:{} | Jit    : {:.3f} ms / iter".format(batch_size, seq_len, jit_time))  
    
    for i in range(warmup_iters + running_iters):
        if i == warmup_iters:    
            torch.cuda.synchronize()
            t2_start = timeit.default_timer()
        hidden_states2 = Attention_Compile(q, k, v) 
    torch.cuda.synchronize()
    t2_end = timeit.default_timer()
    
    torch_compile_time = (t2_end - t2_start) * 1000 / running_iters
    print("bs:{} | seq:{} | compile: {:.3f} ms / iter\n".format(batch_size, seq_len, torch_compile_time))  

                
                            
if __name__ == '__main__':
        
    attn_example()
