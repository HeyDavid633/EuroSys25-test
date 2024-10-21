# 10.21 evaA 我自己的kernel 测试脚本
#
import torch
import torch.nn.functional as F
from utils.utils import torch_cuda_identify, time_stamp_cudasync, seqlen_to_mask, set_dtype, transpose_for_scores
import config
from ops.package_op import syncfree_strided_attn_op, syncfree_fixed_attn_op, syncfree_band_attn_op, syncfree_dilated_attn_op
    

if __name__ == '__main__':
    device = torch_cuda_identify(print_info=False)
    
    torch.manual_seed(0)
    
    batch_size = config.BATCH_SIZE # (8k) batch_size * seq_len / seq_len
    head_size = config.HEAD_DIM   # head_dim aka. head_size
    seq_len = config.SEQ_LEN
    head_num = config.HEAD_NUM
    hidden_dim = head_num * head_size
    layer_num = config.LAYER_NUM
    
    data_type = torch.float16
    dtype = "fp16"
    warmup_iters = config.WARMUP_TIME
    running_iters = config.RUNNING_TIME
    
    print("My Attn time cost:")
    
    for batch_test in [8,16]:
        batch_size = batch_test
        for seqlen_test in [128, 256, 512, 1024, 2048, 4096, 8192]: # 8192 的时候爆显存了
        # for seqlen_test in [128, 256, 512, 1024]: #快速获得性能反馈
            seq_len = seqlen_test
        
            input_from_tensor = set_dtype(torch.empty(batch_size, seq_len, hidden_dim).uniform_(-0.4, 0.4).cuda(), dtype)
            qkv_kernel = [set_dtype(torch.zeros(hidden_dim, hidden_dim * 3).uniform_(-0.4, 0.4).cuda(), dtype) for _ in range(layer_num)]
            qkv_bias = [set_dtype(torch.zeros(hidden_dim * 3).uniform_(-0.4, 0.4).cuda(), dtype) for _ in range(layer_num)]
            
            hidden_states = input_from_tensor
            layer = 0 
            qkv = torch.matmul(hidden_states, qkv_kernel[layer]) + qkv_bias[layer]
            q, k, v = qkv.chunk(3, dim=-1)
            q = transpose_for_scores(q, head_num, head_size)
            k = transpose_for_scores(k, head_num, head_size)
            v = transpose_for_scores(v, head_num, head_size)
            
            
            for iter in range(warmup_iters + running_iters):
                if iter == warmup_iters:
                    t1_start = time_stamp_cudasync()
                # output = syncfree_strided_attn_op(q, k, v)
                # output = syncfree_fixed_attn_op(q, k, v)
                # output = syncfree_band_attn_op(q, k, v)
                output = syncfree_dilated_attn_op(q, k, v)
            t1_end = time_stamp_cudasync() 
    
            print("bs:{}\t| seqlen:{}\t| {:.3f} ms / iter".format(batch_size, seq_len, (t1_end - t1_start) * 1000 / running_iters)) 
            