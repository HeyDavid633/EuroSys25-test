# 2024.9.16 Mon. 
# 使用 torch.compile在不同mode下的性能差异
# Torch.compile(Triton) 
# 
# python attn-diff-compile-mode.py 

import timeit
import torch
import numpy as np
import random
import torch.nn.functional as F
import torch._dynamo
from utils.utils import *
import config

def Attention_std(q, k, v):
    scores = torch.matmul(q, k.transpose(-2, -1)) / (q.shape[3] ** .5)
    probs = F.softmax(scores, dim=-1)
    h = torch.matmul(probs, v)
    # h = h.permute(0, 2, 1, 3).contiguous() 
    # new_context_layer_shape = h.size()[:-2] + (q.shape[1]*q.shape[3], )
    # hidden_states = h.view(new_context_layer_shape) 
    return h


@torch.jit.script
def Attention_Jit(q, k, v):
    scores = torch.matmul(q, k.transpose(-2, -1)) / (q.shape[3] ** .5)
    probs = F.softmax(scores, dim=-1)
    h = torch.matmul(probs, v)
    # h = h.permute(0, 2, 1, 3).contiguous() 
    # new_context_layer_shape = h.size()[:-2] + (q.shape[1]*q.shape[3], )
    # hidden_states = h.view(new_context_layer_shape) 
    return h

@torch.compile
def Attention_Compile(q, k, v):
    scores = torch.matmul(q, k.transpose(-2, -1)) / (q.shape[3] ** .5)
    probs = F.softmax(scores, dim=-1)
    h = torch.matmul(probs, v)
    h = h.permute(0, 2, 1, 3).contiguous() 
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
            t0_start = time_stamp_cudasync() 
        scores = torch.matmul(q, k.transpose(-2, -1)) / (head_size ** .5)
        probs = F.softmax(scores, dim=-1)
        h = torch.matmul(probs, v)
        # h = h.permute(0, 2, 1, 3).contiguous() 
        # new_context_layer_shape = h.size()[:-2] + (hidden_dim, )
        # hidden_states0 = h.view(new_context_layer_shape)    
    t0_end = time_stamp_cudasync()
    
    base_time = (t0_end - t0_start) * 1000 / running_iters
    print("bs:{} | seq:{} | Base   : {:.3f} ms / iter".format(batch_size, seq_len, base_time))  
    # print('Max GPU memory: {:.2f} MB'.format(torch.cuda.max_memory_allocated() / (1024 ** 2)))
    
    
    # for i in range(warmup_iters + running_iters):
    #     if i == warmup_iters:    
    #         t1_start = time_stamp_cudasync()
    #     hidden_states1 = Attention_Jit(q, k, v) 
    # t1_end = time_stamp_cudasync()
    
    # jit_time = (t1_end - t1_start) * 1000 / running_iters
    # print("bs:{} | seq:{} | Jit    : {:.3f} ms / iter".format(batch_size, seq_len, jit_time)) 
    # print('Max GPU memory: {:.2f} MB'.format(torch.cuda.max_memory_allocated() / (1024 ** 2)))

    
    # # 测试不同的方式打开 torch.compile会不会有差异 ---> 结论是@和=效果完全一致
    # for i in range(warmup_iters + running_iters):
    #     if i == warmup_iters:    
    #         torch.cuda.synchronize()
    #         t2_start = timeit.default_timer()
    #     hidden_states2 = Attention_Compile(q, k, v) 
    # torch.cuda.synchronize()
    # t2_end = timeit.default_timer()
    
    # torch_compile_time = (t2_end - t2_start) * 1000 / running_iters
    # print("bs:{} | seq:{} | (default)@compile: {:.3f} ms / iter\n".format(batch_size, seq_len, torch_compile_time))  
    
    
    # 切换mode以前需要reset
    torch._dynamo.reset()
    attn_compile_default = torch.compile(Attention_std, mode="default")
    
    for i in range(warmup_iters + running_iters):
        if i == warmup_iters:    
            t3_start = time_stamp_cudasync()
        hidden_states3 = attn_compile_default(q, k, v) 
    t3_end = time_stamp_cudasync()
    
    torch_compile_time = (t3_end - t3_start) * 1000 / running_iters
    print("bs:{} | seq:{} | (default) compile: {:.3f} ms / iter".format(batch_size, seq_len, torch_compile_time))  
            
                
    # 切换mode以前需要reset
    torch._dynamo.reset()
    attn_compile_reduce = torch.compile(Attention_std, mode="reduce-overhead")
    
    for i in range(warmup_iters + running_iters):
        if i == warmup_iters:    
            t4_start = time_stamp_cudasync()
        hidden_states4 = attn_compile_reduce(q, k, v) 
    t4_end = time_stamp_cudasync()
    
    torch_compile_time = (t4_end - t4_start) * 1000 / running_iters
    print("bs:{} | seq:{} | (reduce)  compile: {:.3f} ms / iter".format(batch_size, seq_len, torch_compile_time))  
    
    
    # 切换mode以前需要reset
    torch._dynamo.reset()
    # 在4080-laptop上无法执行 -- not enough SMs to use max_autotune_gemm mode
    attn_compile_max_autotune = torch.compile(Attention_std, mode="max-autotune") 
    # attn_compile_max_autotune = torch.compile(Attention_std, options ="epilogue_fusion") 
    for i in range(warmup_iters + running_iters):
        if i == warmup_iters:    
            t5_start = time_stamp_cudasync()
        hidden_states5 = attn_compile_max_autotune(q, k, v) 
    t5_end = time_stamp_cudasync()
    
    torch_compile_time = (t5_end - t5_start) * 1000 / running_iters
    print("bs:{} | seq:{} | (autotune)compile: {:.3f} ms / iter\n".format(batch_size, seq_len, torch_compile_time))  
    
    
    # 切换mode以前需要reset --- 重新配置compile
    # torch._dynamo.reset()
    # attn_compile_fullgraph = torch.compile(Attention_std, fullgraph=True) 
    # for i in range(warmup_iters + running_iters):
        
    #     if i == warmup_iters:    
    #         torch.cuda.synchronize()
    #         t6_start = timeit.default_timer()
    #     hidden_states6 = attn_compile_fullgraph(q, k, v) 
    # torch.cuda.synchronize()
    # t6_end = timeit.default_timer()
    
    # torch_compile_time = (t6_end - t6_start) * 1000 / running_iters
    # print("bs:{} | seq:{} | (fullgraph)compile: {:.3f} ms / iter".format(batch_size, seq_len, torch_compile_time))  
    
    # print('Max GPU memory: {:.2f} MB'.format(torch.cuda.max_memory_allocated() / (1024 ** 2)))
                            
                            
if __name__ == '__main__':
    attn_example()
