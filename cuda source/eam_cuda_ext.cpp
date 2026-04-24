#include <torch/extension.h>
#include <vector>

void density_pass_cuda(
    const torch::Tensor distances,
    const torch::Tensor row_index,
    const torch::Tensor col_index,
    const torch::Tensor atom_types,
    const torch::Tensor density_table, // [host E, neighbor E, n_r]
    const float inv_dr,
    const int n_r,
    torch::Tensor rho_out
);

void force_pass_cuda(
    const torch::Tensor distances,
    const torch::Tensor dist_vec,      // [n_edges,3]
    const torch::Tensor row_index,
    const torch::Tensor col_index,
    const torch::Tensor atom_types,
    const torch::Tensor density_deriv_table, // [host E, neighbor E, n_r]
    const torch::Tensor pair_deriv_table,    // [E, E, n_r]
    const torch::Tensor dF_drho,             // [N]
    const float inv_dr,
    const int n_r,
    torch::Tensor forces_out
);

// C++ 接口封装
void density_pass(
    torch::Tensor distances,
    torch::Tensor row_index,
    torch::Tensor col_index,
    torch::Tensor atom_types,
    torch::Tensor density_table,
    double inv_dr,
    int64_t n_r,
    torch::Tensor rho_out) {
  TORCH_CHECK(distances.is_cuda(), "distances must be CUDA tensor");
  density_pass_cuda(distances, row_index, col_index, atom_types, density_table,
                    static_cast<float>(inv_dr), static_cast<int>(n_r), rho_out);
}

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
    torch::Tensor forces_out) {
  TORCH_CHECK(distances.is_cuda(), "distances must be CUDA tensor");
  force_pass_cuda(distances, dist_vec, row_index, col_index, atom_types,
                  density_deriv_table, pair_deriv_table, dF_drho,
                  static_cast<float>(inv_dr), static_cast<int>(n_r), forces_out);
}
