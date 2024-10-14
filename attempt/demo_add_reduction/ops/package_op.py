import torch
from torch.autograd import Function
import tensoradd  # 这个名称来自 setup.py中的命名
import reduction  # 到python这一层，只和setup编译安装然后import进来的这个名称有关

__all__ = ['tensoradd_op', 'reduction_op']

class TensorAdd(Function):

    @staticmethod
    def forward(ctx, array1, array2):
        """sum_double function forward.
        Args:
            array1 (torch.Tensor): [n,]
            array2 (torch.Tensor): [n,]
        
        Returns:
            ans (torch.Tensor): [n,]
        """
        array1 = array1.float()
        array2 = array2.float()
        ans = array1.new_zeros(array1.shape)
        
        tensoradd.forward(array1.contiguous(), array2.contiguous(), ans)

        ctx.mark_non_differentiable(ans) # if the function is no need for backpropogation

        return ans

    # @staticmethod
    # def backward(ctx, g_out):
    #     # return None, None   # if the function is no need for backpropogation

    #     g_in1 = g_out.clone()
    #     g_in2 = g_out.clone()
    #     return g_in1, g_in2

class Recduction(Function):

    @staticmethod
    def forward(ctx, array):
        """sum_single function forward.
        Args:
            array (torch.Tensor): [n,]
        
        Returns:
            ans (torch.Tensor): [1,]
        """
        array = array.float()
        ans = array.new_zeros(1)
        reduction.forward(array.contiguous(), ans)

        ctx.mark_non_differentiable(ans) # if the function is no need for backpropogation
        ctx.shape = array.shape[0]
        
        return ans


tensoradd_op = TensorAdd.apply
reduction_op = Recduction.apply