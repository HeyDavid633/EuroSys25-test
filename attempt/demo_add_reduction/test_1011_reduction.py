# 2024.10.11  reduction 测试文件
# 熟悉了 tensor add的基础上 更复杂的实例 ---> 两个算子都在时应如何编译外链的算子
#  
# 理解自下而上的关系， /src 和 /ops 中变量名完全解耦，这里需要盯紧 /ops的注册过程
import torch 
import timeit
from ops.package_op import reduction_op 

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
    n = 100000000
    device = torch_cuda_identify()
    array1 = torch.ones(n, dtype=torch.float32, device=device, requires_grad=False) # 1的正态分布
    
    time_start = time_stamp_cudasync()
    my_result = reduction_op(array1)
    time_end = time_stamp_cudasync()
    print("My CUDA exten  time: {:.3f} ms".format((time_end - time_start)*1000))
    
    time_start = time_stamp_cudasync()
    golden_result = torch.sum(array1)
    time_end = time_stamp_cudasync()
    print("Origin PyTorch time: {:.3f} ms".format((time_end - time_start)*1000))
    
    # print("Golden result = {:.3f}, type(golden_result = {})".format(golden_result, type(golden_result)))
    
    print(golden_result, ', type(golden_result):', type(golden_result))
    print(my_result, ', type(my_result):', type(my_result))
    
    print("My result = {:.3f} | Golden result = {:.3f}".format(my_result.item(), golden_result.item()))

