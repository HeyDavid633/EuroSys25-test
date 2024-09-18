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
from typing import List

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

def Attention_std(q, k, v):
    scores = torch.matmul(q, k.transpose(-2, -1)) / (q.shape[3] ** .5)
    probs = F.softmax(scores, dim=-1)
    h = torch.matmul(probs, v)
    
    # h = torch.matmul(probs, v).permute(0, 2, 1, 3).contiguous() 
    # new_context_layer_shape = h.size()[:-2] + (q.shape[1]*q.shape[3], )
    # hidden_states = h.view(new_context_layer_shape) 
    # return hidden_states
    return h

def custom_backend(gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor]): #使用用户自定义的后端 输出FX图信息
    print("\ncustom backend called with FX graph:")
    gm.graph.print_tabular()
    return gm.forward

def inspect_backend(gm, sample_inputs):
    code = gm.print_readable()
    with open("forward.svg", "wb") as file:
        file.write(torch.fx.passes.graph_drawer.FxGraphDrawer(gm,'f').get_dot_graph().create_svg())
    return gm.forward


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
    
    # hidden_states1 = Attention_Jit(q, k, v) 
    # hidden_states2 = Attention_Compile(q, k, v) 
    
    
    
    # TorchInductor 调试日志记录: 打印一般的 TorchInductor 调试信息以及生成的 Triton/C++ 代码
    # torch._inductor.config.debug = True 实测没什么输出？
    
    # TorchInductor 跟踪: 显示每个 TorchInductor 阶段所花费的时间 + 输出代码和图可视化
    torch._inductor.config.trace.enabled = True
     
    torch._dynamo.reset()
    # attn_compile_default = torch.compile(Attention_std, mode="default", backend=custom_backend)
    # attn_compile_default = torch.compile(Attention_std, mode="default", backend=inspect_backend) # 错误
    # attn_compile_default = torch.compile(Attention_std, mode="default")
    # hidden_states = attn_compile_default(q, k, v) 
    
    # attn_compile_fullgraph = torch.compile(Attention_std, fullgraph=True) 
    # hidden_states = attn_compile_fullgraph(q, k, v) 
    
    attn_compile_reduce = torch.compile(Attention_std, mode="reduce-overhead") 
    hidden_states = attn_compile_reduce(q, k, v)   # 如果存在循环，并不影响这里的打印效果 --- 毕竟是计算图
    
                            
if __name__ == '__main__':
        
    attn_example()