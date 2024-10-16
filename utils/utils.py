# some basic function 

import torch 
import math
import timeit

def torch_cuda_identify(print_info = True):        
    if torch.cuda.is_available():
        if print_info:
            print(' PyTorch version:', torch.__version__)
            print(' CUDA version \t:', torch.version.cuda)
            print(' GPU cuda:({}) \t: {}'.format(torch.cuda.current_device(), torch.cuda.get_device_name()),'\n', "-" * 50)
        return torch.device('cuda:{}'.format(torch.cuda.current_device()))
    else:
        print('cuda is not avaliable !')
        return torch.device('cpu')
    
def time_stamp_cudasync():
    torch.cuda.synchronize()
    return timeit.default_timer()   
    
def set_dtype(ts: torch.Tensor, dtype: str):
    if dtype == "fp32":
        return ts.float()
    elif dtype == "fp16":
        return ts.half()
    raise RuntimeError(f"Unsupported dtype {dtype}")

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
    
    # 取x的除最后一个维度外的所有维度 加完了以后 = (batch_size, seq_len, n_heads, head_size)
    new_x_shape = x.size()[:-1] + (n_heads, head_size)
    x = x.view(new_x_shape)
    # x的维度变化 (batch_size, seq_len, hidden_dim) --- (batch_size, head_num, seq_len, head_size)
    # 自动的拆开了 最后一个维度 hidden_dim
    return x.permute(0, 2, 1, 3)

def transpose_for_scores1(x):
    new_x_shape = x.size()[:-1] + (12, 64)
    x = x.view(new_x_shape)
    return x.permute(0, 2, 1, 3)

def seqlen_to_mask(lengths, max_len):
    batch_size = lengths.numel()
    mask = (torch.arange(0, max_len, device=lengths.device)
            .type_as(lengths)
            .repeat(batch_size, 1)
            .lt(lengths.unsqueeze(1)))
    return mask

def generate_triangle_mask(attr_mask):
    # gernerate lower triangle mask
    seq_len = attr_mask.shape[1]
    triangle_mask = torch.tril(torch.ones(seq_len, seq_len))
    triangle_mask = triangle_mask.unsqueeze(0).repeat(attr_mask.shape[0], 1, 1)
    return triangle_mask


def generate_strided_mask(attr_mask):
    # gernerate stride mask
    stride_step = int(math.sqrt(attr_mask.shape[1]))
    seq_len = attr_mask.shape[1]
    strided_mask = torch.zeros_like(attr_mask)
    
    for batch in range(strided_mask.shape[0]):
        for i in range(seq_len):
            for j in range(i+1):
                if((i - j) % stride_step == 0):
                    strided_mask[batch, i, j] = 1.0  
                if(j > i - stride_step):
                    strided_mask[batch, i, j] = 1.0  
    return strided_mask
    
   
def generate_fixed_mask(attr_mask):
    # gernerate stride mask
    fixed_step = int(math.sqrt(attr_mask.shape[1]))
    seq_len = attr_mask.shape[1]
    fixed_mask = torch.zeros_like(attr_mask)
    
    for batch in range(fixed_mask.shape[0]):
        for i in range(seq_len):
            for j in range(i+1):
                if(j % fixed_step == fixed_step-1):
                    fixed_mask[batch, i, j] = 1.0  
                if(j > i + (j % fixed_step) - fixed_step):
                    fixed_mask[batch, i, j] = 1.0  
    return fixed_mask