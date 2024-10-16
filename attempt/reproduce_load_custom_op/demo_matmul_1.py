import torch 
import os
import time
from utils.utils import *
from torch.utils.cpp_extension import load

path = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
# path = os.path.join(path,'EuroSys25')

matrix_mul = load(
    name = "matrix_mul",
    sources = [os.path.join(path, "src/matrix_mul.cu"), os.path.join(path, "src/matrix_mul.cpp")],
    verbose = True,
)

if __name__ == '__main__':
    torch_cuda_identify()

    mat_A = torch.rand((400, 1024, 512), dtype=torch.float32)
    mat_B = torch.rand((400, 512, 1024), dtype=torch.float32)
    # mat_A = torch.rand((2, 2, 2), dtype=torch.float32)
    # mat_B = torch.rand((2, 2, 2), dtype=torch.float32)
    
    # GPU 上运行的 golden_result2
    
    mat_A_gpu = mat_A.cuda()
    mat_B_gpu = mat_B.cuda()
    
    start2 = time_stamp_cudasync()
    golden_result2 = torch.matmul(mat_A_gpu, mat_B_gpu)
    end2 = time_stamp_cudasync()
    Golden_time2 = (end2 - start2) * 1000  # 此处时间单位为毫秒
    
    start3 = time_stamp_cudasync()
    my_result = matrix_mul.run_batchMatrixMul(mat_A, mat_B)
    end3=time_stamp_cudasync()
    Mine_time = (end3 - start3) * 1000  
    # print("My Result Matrix_mul of (A * B):\n", my_result)
    
    print("Matrix A shape:", mat_A.shape)
    print("Matrix B shape:", mat_B.shape)
    
    print("Golden_time2 GPU : {:.2f} ms".format(Golden_time2))
    print("Mine_time  GPU\t : {:.2f} ms".format(Mine_time))

    epsilon = 1e-6
    result_equal = torch.allclose(golden_result2.cpu(), my_result.cpu(), atol=epsilon)

    if result_equal:
        print("Pass !")
    else:
        print("Failed !")
    
    
    
