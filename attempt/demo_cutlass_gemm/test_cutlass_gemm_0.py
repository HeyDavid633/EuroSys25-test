# 2024.10.13  ctlass gemm 测试文件
# 
# 通过demo_add_reduction简化了流程，此处无脑链进来一个cutlass的gemm，没有对齐接口
# 没有传入传出参数，内存管理在cuda代码内手动

import torch 
import timeit
from ops.package_op import cutlass_00_gemm_op

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
    
    time_start = time_stamp_cudasync()
    cutlass_00_gemm_op()
    time_end = time_stamp_cudasync()
    print("Cutlass_00_gemm time: {:.3f} ms".format((time_end - time_start)*1000))