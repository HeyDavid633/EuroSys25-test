import torch
from torch.autograd import Function
import cutlass_00_gemm, float_basic_gemm

__all__ = ['cutlass_00_gemm_op', 'float_basic_gemm_op']

class Cutlass_00_Gemm(Function):
    
    @staticmethod
    def forward(ctx):   # 此处的ctx不可省，尽管 CUDA函数 cutlass_00_gemm 并没有入口参数
        cutlass_00_gemm.forward()



class Float_Basic_Gemm(Function):
    
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
        
        mat_A = mat_A.float()
        mat_B = mat_B.float()
        mat_C = torch.zeros((m, n), device=mat_A.device, dtype = torch.float32)
        
        float_basic_gemm.forward(mat_A.contiguous(), mat_B.contiguous(), mat_C, alpha, beta)
        # 这个函数直接与 cutlass_00_gemm_float_gpu()对接匹配
        
        ctx.mark_non_differentiable(mat_C)
        
        return mat_C


cutlass_00_gemm_op = Cutlass_00_Gemm.apply
float_basic_gemm_op = Float_Basic_Gemm.apply