#include <torch/extension.h>
#include <torch/serialize/tensor.h>

#define CHECK_CUDA(x) \
  TORCH_CHECK(x.type().is_cuda(), #x, " must be a CUDAtensor ")
#define CHECK_CONTIGUOUS(x) \
  TORCH_CHECK(x.is_contiguous(), #x, " must be contiguous ")
#define CHECK_INPUT(x) \
  CHECK_CUDA(x);       \
  CHECK_CONTIGUOUS(x)


void tensoradd_launcher(const float* a, const float* b, float* c, int n);


void tensoradd_cuda(at::Tensor a_tensor, at::Tensor b_tensor, at::Tensor c_tensor){
    CHECK_INPUT(a_tensor);
    CHECK_INPUT(b_tensor);
    CHECK_INPUT(c_tensor);

    const float* a = a_tensor.data_ptr<float>();
    const float* b = b_tensor.data_ptr<float>();
    float* c = c_tensor.data_ptr<float>();
    int n = a_tensor.size(0);
    tensoradd_launcher(a, b, c, n);
}


// 绑定函数
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &tensoradd_cuda, "add two tensor of my implement (CUDA)");
}
