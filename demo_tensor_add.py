# 2024.10.03 复现SC24的demo tensor-add
#  
# 所谓JIT的形式来编译，但结果不对？

import torch 
import os
from utils.utils import *
from torch.utils.cpp_extension import load

path = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
# path = os.path.join(path,'EuroSys25')

tensor_add = load(
    name = "tensor_add",
    sources = [os.path.join(path, "src/tensor_add.cu"), os.path.join(path, "src/tensor_add.cpp")],
    verbose = True,
)


if __name__ == '__main__':
    torch_cuda_identify()

    tensor1 = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)
    tensor2 = torch.tensor([[7, 8], [5, 6]], dtype=torch.float32)
    golden_result = tensor1 + tensor2 #torch张量加

    print("Result Tensor (Tensor 1 + Tensor 2):\n", golden_result)

    my_result = tensor_add.tensor_add(tensor1, tensor2)
    print("My Result Tensor (Tensor 1 + Tensor 2):\n", my_result)
