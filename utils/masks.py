import math
import torch
# 实际上应该有4种 Atom 类型的 mask：a)global  b)band  c)dilated  e)block local
# 然后组合为5种mask,    (3)b:sliding_windows  (4)c:dilated_sliding  (5)a+b:global_sliding
# 所有都是下三角的基础上  (1)b+c:strided  (2)a+d:fixed   id(0)留给下三角的稠密情况
# 

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

# 对于atomic系列，实际上应确保不越界，即参数应该小于seq_len，但此处没有注意
# atomic mask(a) global attention
def atomic_a_global(attr_mask, globalwidth = 1):
    batch_size = attr_mask.shape[0]
    seq_len = attr_mask.shape[1]
    global_mask = torch.zeros_like(attr_mask)
    
    for batch in range(batch_size):
        global_mask[batch, :globalwidth, :] = 1  
        global_mask[batch, :, :globalwidth] = 1  
    
    # 刷成下三角的 
    for i in range(seq_len-1):
        global_mask[batch, i, i+1:] = 0
            
    return global_mask

# atomic mask(b) band attention
def atomic_b_band(attr_mask, bandwidth = 1):
    batch_size = attr_mask.shape[0]
    seq_len = attr_mask.shape[1]
    band_mask = torch.zeros_like(attr_mask)

    for batch in range(batch_size):
        for i in range(seq_len):
            start = max(0, i - bandwidth)  # 确保起点不越界
            end = min(seq_len, i + bandwidth + 1)  # 确保终点不越界
            band_mask[batch, i, start:end] = 1
    
    for i in range(seq_len-1):
        band_mask[batch, i, i+1:] = 0        

    return band_mask


# atomic mask(c) dilated attention
def atomic_c_dilated(attr_mask, bandwidth = 1, dilation_rate = 1):
    batch_size = attr_mask.shape[0]
    seq_len = attr_mask.shape[1]
    dilated_mask = torch.zeros_like(attr_mask)

    for batch in range(batch_size):
        for i in range(seq_len):
            start = i - bandwidth - dilation_rate  # 需确保起点不越界
            end = min(seq_len, i + bandwidth + dilation_rate + 1)  # 确保终点不越界
            for row_idx in range(start, end, dilation_rate + 1):
                if(row_idx > -1):
                    dilated_mask[batch, i, row_idx] = 1
                    
    for i in range(seq_len-1):
        dilated_mask[:, i, i+1:] = 0
            
    return dilated_mask


# atomic mask(d) block local attention
def atomic_d_block(attr_mask, block_size = 2):
    
    batch_size = attr_mask.shape[0]
    seq_len = attr_mask.shape[1]
    block_mask = torch.zeros_like(attr_mask)

    for batch in range(batch_size):
        num_blocks = seq_len // block_size 
        for i in range(num_blocks):
            start = i * block_size
            end = start + block_size
            block_mask[batch, start:end, start:end] = 1
            
    for i in range(seq_len-1):
        block_mask[:, i, i+1:] = 0

    return block_mask


