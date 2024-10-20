# 2024.9.17 Tue.
# 
# attn-compile-print延伸 打印fwd过程中哪些部分被选择融合在了一起；
# 代码基准基于 fwd-diff-mode-diff 的不同模式设定
# 没有 mask，只是为了看在这种前向的组合下 融合策略情况
#
# python fwd-compile-print.py
# 
# 打印 更多融合debug信息和依赖图svg
# TORCH_LOGS=fusion TORCH_COMPILE_DEBUG=1 INDUCTOR_ORIG_FX_SVG=1 INDUCTOR_POST_FUSION_SVG=1 python fwd-compile-print.py

import torch
import numpy as np
import random
import torch.nn.functional as F
import torch._dynamo
from utils.utils import *
import config

def fwd_bert_std():
    hidden_states = input_from_tensor
    # for layer in range(layer_num):
    layer = 0
    input_tensor = hidden_states

    qkv = torch.matmul(hidden_states, qkv_kernel[layer]) + qkv_bias[layer]
    q, k, v = qkv.chunk(3, dim=-1)
    q = transpose_for_scores(q, head_num, head_size)
    k = transpose_for_scores(k, head_num, head_size)
    v = transpose_for_scores(v, head_num, head_size)

    # ------------------------------------------------------------- Attention start
    # (B, H, S, W) @ (B, H, W, S) -> (B, H, S, S) -softmax-> (B, H, S, S)
    scores = torch.matmul(q, k.transpose(-2, -1)) / (head_size ** .5)
    probs = F.softmax(scores, dim=-1)
    # (B, H, S, S) @ (B, H, S, W) -> (B, H, S, W) -trans-> (B, S, H, W)
    h = torch.matmul(probs, v)
    # ------------------------------------------------------------ Attention End 
    
    h = h.permute(0, 2, 1, 3).contiguous()
    # -merge-> (B, S, D)
    new_context_layer_shape = h.size()[:-2] + (hidden_dim, )
    hidden_states = h.view(new_context_layer_shape)
    
    #attention output projection GEMM
    hidden_states = torch.matmul(hidden_states, attr_output_kernel[layer]) + attr_output_bias[layer]
    hidden_states = hidden_states + input_tensor  # 残差连接
    
    # layer_Norm
    hidden_states = F.layer_norm(hidden_states, (hidden_dim, ),
                                weight=attr_output_layernorm_gamma[layer], bias=attr_output_layernorm_beta[layer])
    
    residual = hidden_states       # 为残差连接做好准备
    #FFN GEMM 1 + add bias 
    hidden_states = torch.matmul(hidden_states, inter_kernel[layer]) + inter_bias[layer] 
    hidden_states = F.gelu(hidden_states)  #激活函数
    #FFN GEMM 2 + add bias
    hidden_states = torch.matmul(hidden_states, output_kernel[layer]) + output_bias[layer]  
    hidden_states = hidden_states + residual  #残差连接

    # layer_Norm
    hidden_states = F.layer_norm(hidden_states, (hidden_dim, ),  
                                weight=output_layernorm_gamma[layer], bias=output_layernorm_beta[layer])  
    transformer_output[layer] = hidden_states
        
        

    
if __name__ == '__main__':
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

    
    warmup_iters = config.WARMUP_TIME
    running_iters = config.RUNNING_TIME
    dtype = config.DATA_TYPE
    
    input_from_tensor           = set_dtype(torch.empty(batch_size, seq_len, hidden_dim).uniform_(-0.4, 0.4).cuda(), dtype)
    qkv_kernel                  = [set_dtype(torch.zeros(hidden_dim, hidden_dim * 3).uniform_(-0.4, 0.4).cuda(), dtype) for _ in range(layer_num)]
    qkv_bias                    = [set_dtype(torch.zeros(hidden_dim * 3).uniform_(-0.4, 0.4).cuda(), dtype) for _ in range(layer_num)]
    
    attr_output_kernel          = [set_dtype(torch.zeros(hidden_dim, hidden_dim).uniform_(-0.4, 0.4).cuda(), dtype) for _ in range(layer_num)]
    attr_output_bias            = [set_dtype(torch.zeros(hidden_dim).uniform_(-0.4, 0.4).cuda(), dtype) for _ in range(layer_num)]
    attr_output_layernorm_gamma = [set_dtype(torch.zeros(hidden_dim).uniform_(-0.4, 0.4).cuda(), dtype) for _ in range(layer_num)]
    attr_output_layernorm_beta  = [set_dtype(torch.zeros(hidden_dim).uniform_(-0.4, 0.4).cuda(), dtype) for _ in range(layer_num)]
    inter_kernel                = [set_dtype(torch.zeros(hidden_dim, hidden_dim * 4).uniform_(-0.4, 0.4).cuda(), dtype) for _ in range(layer_num)]
    inter_bias                  = [set_dtype(torch.zeros(hidden_dim * 4).uniform_(-0.4, 0.4).cuda(), dtype) for _ in range(layer_num)]
    output_kernel               = [set_dtype(torch.zeros(hidden_dim * 4, hidden_dim).uniform_(-0.4, 0.4).cuda(), dtype) for _ in range(layer_num)]
    output_bias                 = [set_dtype(torch.zeros(hidden_dim).uniform_(-0.4, 0.4).cuda(), dtype) for _ in range(layer_num)]
    output_layernorm_gamma      = [set_dtype(torch.zeros(hidden_dim).uniform_(-0.4, 0.4).cuda(), dtype) for _ in range(layer_num)]
    output_layernorm_beta       = [set_dtype(torch.zeros(hidden_dim).uniform_(-0.4, 0.4).cuda(), dtype) for _ in range(layer_num)]

    transformer_output = [None for _ in range(layer_num)]
    
    
    # TorchInductor 跟踪: 显示每个 TorchInductor 阶段所花费的时间 + 输出代码和图可视化
    torch._inductor.config.trace.enabled = True
    
    # torch.compile --- mode:default
    torch._dynamo.reset()
    fwd_compile_default = torch.compile(fwd_bert_std, mode="default")
    fwd_compile_default()
    
    # torch.compile --- mode:max-autotune
    # torch._dynamo.reset()
    # fwd_compile_autotune = torch.compile(fwd_bert_std, mode="max-autotune")
    # fwd_compile_autotune()
        
    # torch.compile --- mode:reduce-overhead
    # torch._dynamo.reset()
    # fwd_compile_reduce = torch.compile(fwd_bert_std, mode="reduce-overhead")
    # fwd_compile_reduce()
    
    # torch.compile --- mode:default + fullgraph
    # torch._dynamo.reset()
    # fwd_compile_fullgraph = torch.compile(fwd_bert_std, fullgraph=True)
    # fwd_compile_fullgraph()

    
    
    
