#include <torch/extension.h>
#include <vector>

// CUDA forward declarations
std::vector<torch::Tensor> neighbor_search_cuda(
    torch::Tensor positions,
    float cutoff,
    float box_length
);

// C++ interface
std::vector<torch::Tensor> neighbor_search(
    torch::Tensor positions,
    float cutoff,
    float box_length
) {
    return neighbor_search_cuda(positions, cutoff, box_length);
}

// PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
//     m.def("neighbor_search_cuda", &neighbor_search, "Neighbor search with CUDA");
// }
