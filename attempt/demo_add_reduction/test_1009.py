# 2024.10.08  测试文件
# 自定义CUDA算子嵌入PyTorch
#  
# 内容上完全照搬 ../demo_op_add 的内容，熟悉过程然后爆改
# 尤其注意其中的命名，如何封装了命令以及调用关系 --- 自下而上来写
import torch 
import timeit
from ops import tensoradd_op  # import都是 XXX_op 了，即pytorch已经包装过的接口
# 但从 ops 中有__init__ 从ops_py中 import 了一定 *，所以 不用 

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
    n = 10000000
    device = torch_cuda_identify()
    array1 = torch.ones(n, dtype=torch.float32, device=device, requires_grad=False)
    array2 = torch.ones(n, dtype=torch.float32, device=device, requires_grad=False)
    
    time_start = time_stamp_cudasync()
    my_result = tensoradd_op(array1, array2)
    time_end = time_stamp_cudasync()
    print("My CUDA exten  time: {:.3f} ms".format((time_end - time_start)*1000))
    
    time_start = time_stamp_cudasync()
    golden_result = array1 + array2
    time_end = time_stamp_cudasync()
    print("Origin PyTorch time: {:.3f} ms".format((time_end - time_start)*1000))

