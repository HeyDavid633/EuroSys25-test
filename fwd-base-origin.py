# 9.15 base-fwd.py
# 文件内容上完全同 SC24的准备工作base1.py
# 完整的前向bert计算流程，以此为基准不作变动
#
# python base-fwd.py

import argparse
import timeit
import torch
import numpy as np
import random
import torch.nn.functional as F

def sequence_mask(lengths, max_len=None, is_2d=True):
    batch_size = lengths.numel()
    max_len = max_len or lengths.max()
    mask = (torch.arange(0, max_len, device=lengths.device)
            .type_as(lengths)
            .repeat(batch_size, 1)
            .lt(lengths.unsqueeze(1)))
    if is_2d:
        return mask
    else:
        mask = mask.view(-1, 1, 1, max_len)
        m2 = mask.transpose(2, 3)
        return mask * m2

def transpose_for_scores(x, n_heads, head_size):
    # (B, S, D) -split-> (B, S, H, W) -trans-> (B, H, S, W)
    
    # 取x的除最后一个维度外的所有维度 加完了以后 = (batch_size, seq_len, head_num, head_size)
    new_x_shape = x.size()[:-1] + (n_heads, head_size)
    x = x.view(new_x_shape)
    # x的维度变化 (batch_size, seq_len, hidden_dim) --- (batch_size, head_num, seq_len, head_size)
    # 自动的拆开了 最后一个维度 hidden_dim
    return x.permute(0, 2, 1, 3)

def set_dtype(ts: torch.Tensor, dtype: str):
    if dtype == "fp32":
        return ts.float()
    elif dtype == "fp16":
        return ts.half()
    raise RuntimeError(f"Unsupported dtype {dtype}")


def bert_example(args):
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)

    batch_size = args['batch_size']  # 1
    layer_num = args['layer_num']  # 12
    seq_len = args['seq_len']  # 64
    head_num = args['head_num']   # 12
    head_size = args['head_size']   # 64
    avg_seq_len = args['avg_seq_len']  #  -1
    hidden_dim = head_num * head_size
    
    dtype = "fp32"

    for key in args:
        print("{:13} : {:5} ".format(key, args[key]))
    print("-"*21, "Argument", "-"*21)
    

    if avg_seq_len > 0:     
        mem_seq_lens = torch.ones((batch_size,)) * avg_seq_len
        mem_seq_lens = mem_seq_lens.to(torch.int32).cuda()
    elif avg_seq_len == -1:
        mem_seq_lens = torch.randint(1, seq_len + 1, (batch_size,), dtype=torch.int32).cuda()
    else:
        raise ValueError("wrong avg_seq_len")
    
    mask = set_dtype(sequence_mask(mem_seq_lens, seq_len, False), dtype)   
    output_mask = sequence_mask(mem_seq_lens, seq_len).to(mask.dtype).unsqueeze(-1)
    
    print("mask.shape", mask.shape)


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
    
    
    # with torch.no_grad():
    warmup_iters = 10
    iters = 100
    for i in range(warmup_iters + iters):
        if i == warmup_iters:    
            t0_start = timeit.default_timer()
    
        hidden_states = input_from_tensor
        for layer in range(layer_num):
            input_tensor = hidden_states
            
            #在这个matmul中没有广播， + 中发生了广播 
            qkv = torch.matmul(hidden_states, qkv_kernel[layer]) + qkv_bias[layer]
            # chunk函数将qkv张量在最后一个维度（dim=-1）上分割成3个相等的部分。每个部分对应于Q、K、V
            q, k, v = qkv.chunk(3, dim=-1)
            q = transpose_for_scores(q, head_num, head_size)
            k = transpose_for_scores(k, head_num, head_size)
            v = transpose_for_scores(v, head_num, head_size)
            # 此时q.k.v的维度为：(batch_size, head_num, seq_len, head_size)
            

            # ------------------------------------------------------------- Attention start
            # (B, H, S, W) @ (B, H, W, S) -> (B, H, S, S) -softmax-> (B, H, S, S)
            scores = torch.matmul(q, k.transpose(-2, -1)) / (head_size ** .5)
            scores -= 10000.0 * (1.0 - mask)
            probs = F.softmax(scores, dim=-1)
            
            # (B, H, S, S) @ (B, H, S, W) -> (B, H, S, W) -trans-> (B, S, H, W)
            h = torch.matmul(probs, v).permute(0, 2, 1, 3).contiguous()
            
            # -merge-> (B, S, D)
            new_context_layer_shape = h.size()[:-2] + (hidden_dim, )
            hidden_states = h.view(new_context_layer_shape)
            # ------------------------------------------------------------ Attention End                
            
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
        
        t0_end = timeit.default_timer()
        
    print("Base time costs:      \t{:.2f} ms / iter".format((t0_end - t0_start) * 1000 / iters)) 
    
    
       
    # Golden_output = transformer_output[-1]
    # masked_output = Golden_output * output_mask
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('batch_size', type=int, default=1, help='batch size')
    parser.add_argument('layer_num', type=int, default=12, help='number of layers')
    parser.add_argument('seq_len', type=int, default=64, help='sequence length')
    parser.add_argument('head_num', type=int, default=12, help='head number')
    parser.add_argument('head_size', type=int, default=64, help='size per head')
    parser.add_argument('--avg_seq_len', type=int, default=-1, metavar='NUMBER', help='average sequence length (default: -1)')
    args = parser.parse_args()
    bert_example(vars(args))