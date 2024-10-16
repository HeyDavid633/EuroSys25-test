#include <torch/extension.h>
#include <iostream>
#include <vector>
#include <pybind11/pybind11.h>


torch::Tensor run_matrix_mul(
    torch::Tensor tensor1,
    torch::Tensor tensor2);

torch::Tensor run_matrix_mul2(
    torch::Tensor tensor1,
    torch::Tensor tensor2);

torch::Tensor run_batchMatrixMul(
    torch::Tensor tensor1,
    torch::Tensor tensor2);


PYBIND11_MODULE(matrix_mul, m)
{
    m.doc() = "Demo4: Matrix Mul for torch working with my-own-cuda. ";
    m.def("run_matrix_mul", &run_matrix_mul, "Exact  Matrix Mul OP");
    m.def("run_matrix_mul2", &run_matrix_mul2, "Exact  Matrix Mul OP");
    m.def("run_batchMatrixMul", &run_batchMatrixMul, "to Replace torch.bmm() Matrix Mul OP");
}