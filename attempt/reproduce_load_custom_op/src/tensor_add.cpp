#include <torch/extension.h>
#include <iostream>
#include <vector>
#include <pybind11/pybind11.h>

//std::vector<torch::Tensor>
torch::Tensor tensor_add(
    torch::Tensor tensor1,
    torch::Tensor tensor2);


PYBIND11_MODULE(tensor_add, m)
{
    m.doc() = "Demo: tensor add work with torch ";
    m.def("tensor_add", &tensor_add, "Exact Tensor Add OP");
}
