import torch
from torch.autograd import Function
import cutlass_00_basic_gemm, cutlass_05_batched_gemm
# import cutlass_12_gemm_bias_relu, cutlass_35_gemm_softmax

__all__ = ['cutlass_00_basic_gemm_op', 'cutlass_05_batched_gemm_op']

class Cutlass_00_Basic_Gemm(Function):
    
    @staticmethod
    def forward(ctx, mat_A, mat_B, alpha, beta):  
        """Cutlass GEMM function forward: D = aplha (A @ B) + beta C
        Args:
            mat_A (torch.Tensor): [m, k]
            mat_B (torch.Tensor): [k, n]
            alpha & beta: const float
        Returns:
            mat_C (torch.Tensor): [m, n]
        """
        m = mat_A.size(-2)
        n = mat_B.size(-1)
        mat_C = torch.zeros((m, n), device=mat_A.device, dtype=mat_A.dtype)
        
        cutlass_00_basic_gemm.forward(mat_A.contiguous(), mat_B.contiguous(), mat_C, alpha, beta)
        # 这个函数直接与 cutlass_00_gemm_float_gpu()对接匹配
        # 此处还 与数据类型并无相关
        
        ctx.mark_non_differentiable(mat_C)
        
        return mat_C


class Cutlass_05_batched_Gemm(Function):
    
    @staticmethod
    def forward(ctx, mat_A, mat_B, alpha, beta):  
        """Cutlass batched GEMM function forward: D = aplha (A @ B) + beta C
        Args:
            mat_A (torch.Tensor): [batch, m, k]
            mat_B (torch.Tensor): [batch, k, n]
            alpha & beta: const float
        Returns:
            mat_C (torch.Tensor): [batch, m, n]
        """
        m = mat_A.size(-2)
        n = mat_B.size(-1)
        batch_count = mat_A.size(0)
        mat_C = torch.zeros((batch_count, m, n), device=mat_A.device, dtype=mat_A.dtype)
        
        cutlass_05_batched_gemm.forward(mat_A.contiguous(), mat_B.contiguous(), mat_C, alpha, beta)
        
        ctx.mark_non_differentiable(mat_C)
        
        return mat_C


cutlass_00_basic_gemm_op = Cutlass_00_Basic_Gemm.apply
cutlass_05_batched_gemm_op = Cutlass_05_batched_Gemm.apply

