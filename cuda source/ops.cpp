#include <torch/extension.h>
#include <vector>

// Forward declarations from LJ force (implemented in lj_energy_force.cpp)
std::vector<torch::Tensor> lj_energy_force_cuda(
    torch::Tensor pos,
    torch::Tensor edge_index,
    torch::Tensor sigma,
    torch::Tensor epsilon,
    float box_length
);

std::vector<torch::Tensor> neighbor_search_cuda(torch::Tensor positions, float cutoff, float box_length);

// Forward declarations for EAM (density & force passes)
void density_pass(
    torch::Tensor distances,
    torch::Tensor row_index,
    torch::Tensor col_index,
    torch::Tensor atom_types,
    torch::Tensor density_table,
    double inv_dr,
    int64_t n_r,
    torch::Tensor rho_out);

void force_pass(
    torch::Tensor distances,
    torch::Tensor dist_vec,
    torch::Tensor row_index,
    torch::Tensor col_index,
    torch::Tensor atom_types,
    torch::Tensor density_deriv_table,
    torch::Tensor pair_deriv_table,
    torch::Tensor dF_drho,
    double inv_dr,
    int64_t n_r,
    torch::Tensor forces_out);

// From kernel.cu
void set_lj_smoothing_cuda(int mode, float rc, float rs);

// Configure smoothing wrapper exposed to Python
void configure_lj_smoothing(int mode, float rc, float rs) {
    set_lj_smoothing_cuda(mode, rc, rs);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("lj_energy_force_cuda", &lj_energy_force_cuda, "LJ Energy + Force CUDA");
    m.def("neighbor_search_cuda", &neighbor_search_cuda, "Neighbor search with CUDA");
    m.def("density_pass", &density_pass, "EAM density accumulation (CUDA)");
    m.def("force_pass", &force_pass, "EAM force computation (CUDA)");
    m.def("configure_lj_smoothing", &configure_lj_smoothing, "Configure LJ smoothing (mode, rc, rs)");
}
