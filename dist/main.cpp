#include <torch/extension.h>
torch::Tensor forward(torch::Tensor out, torch::Tensor inp, int B, int T, int C, int NH, int block_size);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
m.def("forward", torch::wrap_pybind_function(forward), "forward");
}