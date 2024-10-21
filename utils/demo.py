import torch
import numpy as np
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
    #for i in range(seq_len-1):
        #global_mask[batch, i, i+1:] = 0
            
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
    
    # for i in range(seq_len-1):
    #     band_mask[batch, i, i+1:] = 0        

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
                    
    # for i in range(seq_len-1):
    #     dilated_mask[:, i, i+1:] = 0
            
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
            
    #for i in range(seq_len-1):
    #    block_mask[:, i, i+1:] = 0

    return block_mask

def calculate_sparsity(matrix):
    # 计算矩阵的总元素个数
    total_elements = matrix.numel()
    print(total_elements)
    # 计算矩阵中零元素的个数
    zero_elements = np.count_nonzero(matrix == 0)
    print(zero_elements)
    # 计算稀疏度
    sparsity = zero_elements / total_elements
    return sparsity

# 测试代码
attr_mask = torch.zeros((1, 1024, 1024))  # 2个批次，序列长度为5
#result = atomic_b_band(attr_mask, 32)
result = atomic_a_global(attr_mask, 32)
print(result)

# 计算result矩阵的稀疏度
sparsity = calculate_sparsity(result)
print(sparsity)
