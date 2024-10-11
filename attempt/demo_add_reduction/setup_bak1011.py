# 2024.10.08 实现自己的 cpp extension
#
# 小跨一步熟悉流程，基本上照搬 /demo_op_add的内容
# 实现了 张量加 以及 reducetion操作

from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

extra_compile_args = {
    "nvcc" : ["-O3"]
}  # 单独控制编译标志

setup(
    name='demo-copy-from-others',     
    packages=find_packages(),
    version='0.0.1',
    author='EuroSys25 by David',
    
    ext_modules=[
        CUDAExtension(
            'reduction', # operator name 需要import到 package_op.py 文件中
            ['./ops/src/reduction.cpp',
             './ops/src/reduction_cuda.cu',],
            extra_compile_args = extra_compile_args       
        ),
        
        CUDAExtension(
            'tensoradd',
            ['./ops/src/tensoradd.cpp',
             './ops/src/tensoradd_cuda.cu',],
            extra_compile_args = extra_compile_args
        ),
    ],
    
    cmdclass={'build_ext': BuildExtension}
)