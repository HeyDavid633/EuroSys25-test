# 2024.10.14  ctlass gemm 测试文件
# 
# 对齐接口的 cutlass basic gemm
# 带有传入与传出参数的过程

import torch 
import timeit
from ops.package_op import float_basic_gemm_op

def torch_cuda_identify():        
    if torch.cuda.is_available():
        print('PyTorch version\t:', torch.__version__)
        print('CUDA version\t:', torch.version.cuda)
        print('GPU\t\t:',torch.cuda.get_device_name(), '\n', "-" * 50)
        return torch.device("cuda:0")
    else:
        print('cuda is not avaliable !')
        return torch.device('cpu')
    
def time_stamp_cudasync():
    torch.cuda.synchronize()
    return timeit.default_timer()   
    

if __name__ == '__main__':
    device = torch_cuda_identify()
    
    m = 1024
    n = 1024
    k = 256
    alpha = 1.0
    beta = 0.0
    
    # A(4, 4) * B(4, 4) = (4, 4)  总是对 因为A和B 被摊开成一维的时候，内容一样 
    # mat_A = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8], [1, 2, 3, 4], [5, 6, 7, 8]], dtype=torch.float32, device=device, requires_grad=False)
    # mat_B = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8], [1, 2, 3, 4], [5, 6, 7, 8]], dtype=torch.float32, device=device, requires_grad=False)
    
    # mat_A = torch.tensor([[2, 1, 4, 3], [6, 5, 8, 7], [2, 1, 4, 3], [6, 5, 8, 7]], dtype=torch.float32, device=device, requires_grad=False)
    # mat_B = torch.tensor([[2, 1, 4, 3], [6, 5, 8, 7], [2, 1, 4, 3], [6, 5, 8, 7]], dtype=torch.float32, device=device, requires_grad=False)
    
    
    # 这个的结果就对不上了
    # mat_A = torch.tensor([[2, 1, 4, 3], [6, 5, 8, 7], [2, 1, 4, 3], [6, 5, 8, 7]], dtype=torch.float32, device=device, requires_grad=False)
    # mat_B = torch.tensor([[5, 6, 7, 8], [1, 2, 3, 4], [5, 6, 7, 8], [1, 2, 3, 4]], dtype=torch.float32, device=device, requires_grad=False)

    
    
    # A(4, 2) * B(2, 4) = (4, 4)  总是对 因为A和B 被摊开成一维的时候，内容一样 
    # mat_A = torch.tensor([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=torch.float32, device=device, requires_grad=False)
    # mat_B = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=torch.float32, device=device, requires_grad=False)
    
    # mat_A = torch.tensor([[2, 1], [4, 3], [6, 5], [8, 7]], dtype=torch.float32, device=device, requires_grad=False)
    # mat_B = torch.tensor([[2, 1, 4, 3], [6, 5, 8, 7]], dtype=torch.float32, device=device, requires_grad=False)
    

    
    # # A(4, 2) * B(2, 4) = (4, 4)
    # mat_A = torch.tensor([[2, 1], [4, 3], [6, 5], [8, 7]], dtype=torch.float32, device=device, requires_grad=False)
    # mat_B = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=torch.float32, device=device, requires_grad=False)
    
    # # A(4, 2) * B(2, 4) = (4, 4)
    # mat_A = torch.tensor([[2, 1], [4, 3], [6, 5], [8, 7]], dtype=torch.float32, device=device, requires_grad=False)
    # mat_B = torch.tensor([[5, 6, 7, 8], [1, 2, 3, 4]], dtype=torch.float32, device=device, requires_grad=False)
    
    # # A(4, 2) * B(2, 4) = (4, 4)
    # mat_A = torch.tensor([[2, 1], [4, 3], [6, 5], [8, 7]], dtype=torch.float32, device=device, requires_grad=False)
    # mat_B = torch.tensor([[6, 5, 8, 7], [2, 1, 4, 3]], dtype=torch.float32, device=device, requires_grad=False)
    
    
    
    mat_A = torch.rand((m, k), dtype=torch.float32, device=device, requires_grad=False)
    mat_B = torch.rand((k, n), dtype=torch.float32, device=device, requires_grad=False)
    
    
    golden_resultAB = torch.matmul(mat_A, mat_B)
    golden_resultBA = torch.matmul(mat_B, mat_A)
    
    
    # 在cutlass函数中写 Mat_A * Mat_B还不对，需要写 Mat_B * Mat_A --- 但这样的话MNK维度变化后就会错
    my_cutlass_basic_gemm_float = float_basic_gemm_op(mat_A, mat_B, alpha, beta) 
    
    # print("\n golden_resultAB.size(): ",golden_resultAB.size())
    # print(" my_cutlass_basic_gemm_float.size(): ", my_cutlass_basic_gemm_float.size())    

    print(" Mat_A:\n", mat_A)
    print(" Mat_B:\n", mat_B)
    

    print("\n golden_resultAB:\n ",golden_resultAB)
    # print("\n golden_resultBA:\n ",golden_resultBA)
    print("\n my_cutlass_basic_gemm_float:\n", my_cutlass_basic_gemm_float)
    
    
    diff = torch.abs(golden_resultAB - my_cutlass_basic_gemm_float)
    print('\nMean diff: {:.8f}'.format(torch.mean(diff).item()))

    
    
    # time_start = time_stamp_cudasync()
    # time_end = time_stamp_cudasync()
    # print("Real Cutlass_00_gemm time: {:.3f} ms".format((time_end - time_start)*1000))