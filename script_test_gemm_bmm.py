# 2024.10.17  脚本测试cutlass和原生算子的性能
#
#
import numpy as np
import random 
import torch 
from ops.package_op import cutlass_00_basic_gemm_op, cutlass_05_batched_gemm_op
from utils.utils import torch_cuda_identify, time_stamp_cudasync
import config

if __name__ == '__main__':
    device = torch_cuda_identify(print_info = False)
    
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)
    

    head_size = config.HEAD_DIM   # head_dim aka. head_size
    seq_len = config.SEQ_LEN
    head_num = config.HEADS_NUM
    batch_size = config.BATCH_SIZE # (8k) 8192 / seq_len
    m = seq_len
    n = seq_len
    k = head_size
    alpha = 1.0
    beta = 0.0 
    data_type = torch.float16   # 务必保持精度为 fp16
    
    warmup_iters = config.WARMUP_TIME
    running_iters = config.RUNNING_TIME
    
    
    mat_A = torch.rand((m, k), dtype=data_type, device=device, requires_grad=False)
    mat_B = torch.rand((k, n), dtype=data_type, device=device, requires_grad=False)
    
    for i in range(warmup_iters + running_iters):
        if i == warmup_iters:    
            t1_start = time_stamp_cudasync()
        golden_resultAB = torch.matmul(mat_A, mat_B)
    t1_end = time_stamp_cudasync()
    
    for i in range(warmup_iters + running_iters):
        if i == warmup_iters:    
            t2_start = time_stamp_cudasync()
        my_cutlass_basic_gemm_float = cutlass_00_basic_gemm_op(mat_A, mat_B, alpha, beta) 
    t2_end = time_stamp_cudasync()
    
    compute_flop = (2*k - 1) * m * n
    GFlops_torch =  compute_flop / ((t1_end - t1_start)/running_iters) * 1e-9
    GFlops_cutlass = 0
    GFlops_cutlass =  compute_flop / ((t2_end - t2_start)/running_iters) * 1e-9
    print("GEMM {} \tm:{} k:{} n:{} | Torch Gflops: {:.2f} | Cutlass Gflops: {:.2f}".format(data_type, m, n, k, GFlops_torch, GFlops_cutlass))
    
    
    
    # batch_mat_A = torch.rand((batch_size * head_num, m, k), dtype=data_type, device=device, requires_grad=False)
    # batch_mat_B = torch.rand((batch_size * head_num, k, n), dtype=data_type, device=device, requires_grad=False)    
    
    # for i in range(warmup_iters + running_iters):
    #     if i == warmup_iters:    
    #         t1_start = time_stamp_cudasync()
    #     golden_resultAB = torch.matmul(batch_mat_A, batch_mat_B)
    # t1_end = time_stamp_cudasync()
    
    # for i in range(warmup_iters + running_iters):
    #     if i == warmup_iters:    
    #         t2_start = time_stamp_cudasync()
    #     my_cutlass_batched_gemm_result = cutlass_05_batched_gemm_op(batch_mat_A, batch_mat_B, alpha, beta)  
    # t2_end = time_stamp_cudasync()
     
    # compute_flop = (2*k - 1) * m * n * batch_size * head_num
    # GFlops_torch =  compute_flop / ((t1_end - t1_start)/running_iters) * 1e-9
    # GFlops_cutlass =  compute_flop / ((t2_end - t2_start)/running_iters) * 1e-9
    # print("BMM  {} \tm:{} k:{} n:{} | Torch Gflops: {:.2f} | Cutlass Gflops: {:.2f}".format(data_type, m, n, k, GFlops_torch, GFlops_cutlass))