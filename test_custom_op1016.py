# 2024.10.16  多个cutlass算子验证
#
# 包含的kernel：
# 1. cutlass_00_basic_gemm,     OK
# 2. cutlass_05_batched_gemm,   OK
# 3. cutlass_12_gemm_bias_relu, 失败 （非批量的）
# 4. cutlass_35_gemm_softmax,   失败 （矩阵传入方式奇怪） --- float16精度传入
# 5. cutlass_37_gemm_layernrom_gemm_fusion,
# 6. cutlass_41_fusion_multi_head_attention,
#
# 上面的实现精度还是 float32的居多
# 从/cutlass/example/xxx.cu抽取 ---> 对齐接口 ---> 封装调用验证正确性

import torch 
from ops.package_op import cutlass_00_basic_gemm_op, cutlass_05_batched_gemm_op
from utils.utils import torch_cuda_identify, time_stamp_cudasync

if __name__ == '__main__':
    device = torch_cuda_identify()
    
    batch_size = 16
    m = 1024
    n = 1024
    k = 256
    alpha = 1.0
    beta = 0.0
    data_type = torch.float32
   
    # 【验证成功】cutlass_00_basic_gemm  
    # mat_A = torch.rand((m, k), dtype=torch.float32, device=device, requires_grad=False)
    # mat_B = torch.rand((k, n), dtype=torch.float32, device=device, requires_grad=False)
    # golden_resultAB = torch.matmul(mat_A, mat_B)
    # my_cutlass_basic_gemm_float = cutlass_00_basic_gemm_op(mat_A, mat_B, alpha, beta) 
    
    # print("\n golden_resultAB.size(): ",golden_resultAB.size())
    # print(" my_cutlass_basic_gemm_float.size(): ", my_cutlass_basic_gemm_float.size())    

    # print(" Mat_A:\n", mat_A)
    # print(" Mat_B:\n", mat_B)    
    # print("\n golden_resultAB:\n ",golden_resultAB)
    # print("\n my_cutlass_basic_gemm_float:\n", my_cutlass_basic_gemm_float)
    
    # diff = torch.abs(golden_resultAB - my_cutlass_basic_gemm_float)
    # print('\n00 Mean diff: {:.8f}'.format(torch.mean(diff).item()))
    
    
    # 【验证成功】 cutlass_05_batched_gemm
    # mat_A = torch.rand((batch_size ,m, k), dtype=data_type, device=device, requires_grad=False)
    # mat_B = torch.rand((batch_size, k, n), dtype=data_type, device=device, requires_grad=False)
    mat_A = torch.rand((batch_size ,m, k), dtype=data_type, requires_grad=False).cuda()
    mat_B = torch.rand((batch_size, k, n), dtype=data_type, requires_grad=False).cuda()
    golden_resultAB = torch.matmul(mat_A, mat_B)
    my_cutlass_batched_gemm_result = cutlass_05_batched_gemm_op(mat_A, mat_B, alpha, beta) 
    
    diff = torch.abs(golden_resultAB - my_cutlass_batched_gemm_result)
    print('\n05 Mean diff: {:.8f}'.format(torch.mean(diff).item()))
    
    
    # 【失败】cutlass_35_gemm_softmax
    # 样例程序中的 矩阵传入方式奇怪，不知道怎么操作；调用没有问题
    # mat_A = torch.rand((batch_size ,m, k), dtype=data_type, device=device, requires_grad=False)
    # mat_B = torch.rand((batch_size, k, n), dtype=data_type, device=device, requires_grad=False)
    # golden_resultAB = torch.matmul(mat_A, mat_B)
    # my_cutlass_batched_gemm_result = cutlass_05_batched_gemm_op(mat_A, mat_B, alpha, beta) 
    
    # diff = torch.abs(golden_resultAB - my_cutlass_batched_gemm_result)
    # print('\n05 Mean diff: {:.8f}'.format(torch.mean(diff).item()))
    # cutlass_35_gemm_softmax_op(mat_A, mat_B, alpha, beta)


    
    
    
    
    # time_start = time_stamp_cudasync()
    # time_end = time_stamp_cudasync()
    # print("Real Cutlass_00_gemm time: {:.3f} ms".format((time_end - time_start)*1000))