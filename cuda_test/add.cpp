#include <torch/extension.h>

torch::Tensor add_cuda_forward(torch::Tensor a, torch::Tensor b);

torch::Tensor add_forward(torch::Tensor a, torch::Tensor b) {
    return add_cuda_forward(a, b);
}
