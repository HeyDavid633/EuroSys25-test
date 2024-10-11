# 2024.10.08 实现自己的 cpp extension
#
# 小跨一步熟悉流程，基本上照搬 /demo_op_add的内容
# 实现了 张量加 以及 reducetion操作

import os
import glob
from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

extra_compile_args = {
    "nvcc" : ["-O3"]
}  # 单独控制编译标志

def get_extensions():
    
    this_dir = os.path.dirname(os.path.curdir)
    extensions_dir = os.path.join(this_dir, "ops", "src")
    
    sources = list(glob.glob(os.path.join(extensions_dir, "*.cpp")))
    cuda_sources = list(glob.glob(os.path.join(extensions_dir, "*_cuda.cu")))
    
    # 从文件名中提取算子名称
    extension_names = set()
    for source in sources + cuda_sources:
        base_name = os.path.basename(source)
        name, _ = os.path.splitext(base_name)
        if name.endswith("_cuda"):
            name = name[:-5]  # 去掉 "_cuda" 后缀
        extension_names.add(name)
    
    ext_modules = []
    for name in extension_names:
        # 动态生成每个扩展的源文件列表
        extension_sources = [s for s in sources + cuda_sources if name in os.path.basename(s)]
        ext_modules.append(
            CUDAExtension(
                name=name,
                sources=extension_sources,
                extra_compile_args=extra_compile_args
            )
        )
    
    return ext_modules

setup(
    name='optimized demo',     
    packages=find_packages(),
    version='0.0.1',
    author='EuroSys25 by David',
    
    ext_modules=get_extensions(),
    
    install_requires=["torch"],
    description="My Demo of PyTorch cpp and CUDA extensions, Auto install, learn from extension-cpp",
    
    cmdclass={'build_ext': BuildExtension}
)