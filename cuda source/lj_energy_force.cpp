#include <torch/extension.h>

void launch_lj_energy_force_kernel(
    torch::Tensor pos, torch::Tensor edge_index,
    torch::Tensor sigma, torch::Tensor epsilon,
    float box_length,
    torch::Tensor forces_out, torch::Tensor energy_out
);

std::vector<torch::Tensor> lj_energy_force_cuda(
    torch::Tensor pos, torch::Tensor edge_index,
    torch::Tensor sigma, torch::Tensor epsilon,
    float box_length
) {
    const auto N = pos.size(0);
    auto forces_out = torch::zeros({N, 3}, pos.options().dtype(torch::kFloat));
    auto energy_out = torch::zeros({1}, pos.options().dtype(torch::kFloat));
    launch_lj_energy_force_kernel(pos, edge_index, sigma, epsilon, box_length, forces_out, energy_out);
    return {energy_out, forces_out};
}

// PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
//     m.def("lj_energy_force_cuda", &lj_energy_force_cuda, "LJ Energy + Force CUDA");
// }
