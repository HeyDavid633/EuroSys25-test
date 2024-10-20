import torch
from utils.utils import set_dtype
from utils.masks import seqlen_to_mask, generate_triangle_mask, generate_strided_mask, generate_fixed_mask
from utils.masks import atomic_a_global, atomic_b_band, atomic_c_dilated, atomic_d_block
 
if __name__ == '__main__':
    mask_id = 5
    seq_len = 8
    batch_size = 1
    
    # 4类Atom mask叠加组合成5种Mask
    avg_seq_len = seq_len
    low, high = (2 * avg_seq_len - seq_len, seq_len + 1)
    input_lens = torch.randint(low=low, high=high, size=(batch_size,))
    seqlen_mask = seqlen_to_mask(input_lens, seq_len)
    attr_mask   = torch.tile(seqlen_mask, dims=(seq_len,)).reshape(batch_size, seq_len, seq_len)
    
    lower_triangle_mask = generate_triangle_mask(attr_mask)
    strided_mask = generate_strided_mask(attr_mask)  # a+d
    fixed_mask = generate_fixed_mask(attr_mask)      # b+c
    sliding_windows = atomic_b_band(attr_mask)       #b
    dilated_sliding = atomic_c_dilated(attr_mask)    #c
    global_sliding = (atomic_a_global(attr_mask) | atomic_b_band(attr_mask)).float()  # a + b
    
    # 0-lower triangle, 1-strided, 2-fixed, 3-sliding_windows, 4-dilated_sliding, 5-global_sliding
    mask_name_list = ['Lower_triangle_mask', 'Strided_mask', 'Fixed_mask', 'Sliding_windows', 'Dilated_sliding', 'Global_sliding']
    mask_matrix_list = [lower_triangle_mask, strided_mask, fixed_mask, sliding_windows, dilated_sliding, global_sliding]
    mask_name = mask_name_list[mask_id]
    mask = mask_matrix_list[mask_id].to(torch.float16).cuda()
    
    print("{} Mask with shape {}:\n".format(mask_name, mask.shape))
    print(mask)
    
    
    
    