# 2024.9.16 Mon. 
# 期望打印出 Torch.compile 在不同模式下的融合信息
# Torch.compile(Triton) 
# 信息打印专用，故不再多次运行；而且不需要base和计时
# 
# python attn-compile-print.py 

import torch
import numpy as np
import random
import torch.nn.functional as F
from utils.utils import *
import torch._dynamo
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
    
    batch_size = config.BATCH_SIZE
    head_size = config.HEAD_DIM # head_dim aka. head_size
    seq_len = config.SEQ_LEN
    head_num = config.HEADS_NUM
    layer_num = config.LAYER_NUM
    hidden_dim = head_num * head_size
    
    # warmup_iters = config.WARMUP_TIME
    # running_iters = config.RUNNING_TIME
    dtype = "fp16"

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

    
    
    hidden_states1 = Attention_Jit(q, k, v) 

    
    hidden_states2 = Attention_Compile(q, k, v) 
    
    
    
                            
if __name__ == '__main__':
        
    attn_example()
