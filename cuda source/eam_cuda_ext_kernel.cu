#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

__device__ __forceinline__ int clamp_int(int x, int hi){
    return x < 0 ? 0 : (x > hi ? hi : x);
}

__device__ __forceinline__ float lerp_table(float r, float inv_dr, int n_r, const float* __restrict__ table){
    float x = r * inv_dr;
    int i = (int)floorf(x);
    if(i < 0) i = 0;
    if(i > n_r - 2) i = n_r - 2;
    float t = x - i;
    float v0 = table[i];
    float v1 = table[i+1];
    return v0 + t * (v1 - v0);
}

// density pass kernel
// 修复：half-list (row=i < col=j) 下原版只做了 rho_i += f_j(r_ij)，
// 漏掉了 rho_j += f_i(r_ij)，导致索引较大原子的密度严重偏低。
// 正确做法：对每条边同时累加 i 侧与 j 侧的密度贡献。
__global__ void density_pass_kernel(
    const float* __restrict__ distances,
    const int64_t* __restrict__ row_index,
    const int64_t* __restrict__ col_index,
    const int64_t* __restrict__ atom_types,
    const float* __restrict__ density_table, // [host E, neighbor E, n_r]
    int E,
    float inv_dr,
    int n_r,
    int n_pairs,
    float* __restrict__ rho_out
){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= n_pairs) return;
    float r = distances[idx];
    int i_atom = row_index[idx];
    int j_atom = col_index[idx];
    int i_type = (int)atom_types[i_atom];
    int j_type = (int)atom_types[j_atom];

    // rho_i += f_{type_i,type_j}(r_ij)
    const float* table_j = density_table + (i_type * E + j_type) * n_r;
    float val_j_to_i = lerp_table(r, inv_dr, n_r, table_j);
    atomicAdd(rho_out + i_atom, val_j_to_i);

    // rho_j += f_{type_j,type_i}(r_ij)
    const float* table_i = density_table + (j_type * E + i_type) * n_r;
    float val_i_to_j = lerp_table(r, inv_dr, n_r, table_i);
    atomicAdd(rho_out + j_atom, val_i_to_j);
}

// 修复：力公式多元素合金正确性
// 原版：dF_sum * d_rho_dr (单一 df_j/dr)
//   - 对 single-element 系统，df_i = df_j 恰好正确
//   - 对 multi-element 合金 (df_i ≠ df_j) 会产生系统性偏差
// 正确公式：
//   f_scalar = dF_i/drho_i · df_{type_i,type_j}/dr
//            + dF_j/drho_j · df_{type_j,type_i}/dr
//            + dphi_{ij}/dr
__global__ void force_pass_kernel(
    const float* __restrict__ distances,
    const float* __restrict__ dist_vec, // [n_pairs,3]
    const int64_t* __restrict__ row_index,
    const int64_t* __restrict__ col_index,
    const int64_t* __restrict__ atom_types,
    const float* __restrict__ density_deriv_table, // [host E, neighbor E, n_r]
    const float* __restrict__ pair_deriv_table,    // [E,E,n_r]
    const float* __restrict__ dF_drho,             // [N]
    float inv_dr,
    int n_r,
    int E,
    int n_pairs,
    float* __restrict__ forces // [N,3]
){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= n_pairs) return;
    float r = distances[idx];
    if(r < 1e-8f) return;
    int i_atom = row_index[idx];
    int j_atom = col_index[idx];
    int i_type = (int)atom_types[i_atom];
    int j_type = (int)atom_types[j_atom];

    // df_{type_i,type_j}(r)/dr —— j neighbor contribution to rho_i
    const float* dens_deriv_j_to_i = density_deriv_table + (i_type * E + j_type) * n_r;
    float df_j_to_i_dr = lerp_table(r, inv_dr, n_r, dens_deriv_j_to_i);

    // df_{type_j,type_i}(r)/dr —— i neighbor contribution to rho_j
    const float* dens_deriv_i_to_j = density_deriv_table + (j_type * E + i_type) * n_r;
    float df_i_to_j_dr = lerp_table(r, inv_dr, n_r, dens_deriv_i_to_j);

    // dphi_{ij}(r)/dr
    const float* pair_deriv_ij = pair_deriv_table + ( (i_type * E + j_type) * n_r );
    float dphi_dr = lerp_table(r, inv_dr, n_r, pair_deriv_ij);

    float scalar = dF_drho[i_atom] * df_j_to_i_dr
                 + dF_drho[j_atom] * df_i_to_j_dr
                 + dphi_dr;

    const float* rij = dist_vec + 3*idx;
    float fx = -scalar * rij[0] / r;
    float fy = -scalar * rij[1] / r;
    float fz = -scalar * rij[2] / r;

    atomicAdd(forces + 3*i_atom + 0, fx);
    atomicAdd(forces + 3*i_atom + 1, fy);
    atomicAdd(forces + 3*i_atom + 2, fz);
    atomicAdd(forces + 3*j_atom + 0, -fx);
    atomicAdd(forces + 3*j_atom + 1, -fy);
    atomicAdd(forces + 3*j_atom + 2, -fz);
}

void density_pass_cuda(
    const torch::Tensor distances,
    const torch::Tensor row_index,
    const torch::Tensor col_index,
    const torch::Tensor atom_types,
    const torch::Tensor density_table,
    const float inv_dr,
    const int n_r,
    torch::Tensor rho_out){

    int n_pairs = distances.size(0);
    int E = density_table.size(0);
    int threads = 256;
    int blocks = (n_pairs + threads - 1)/threads;
    density_pass_kernel<<<blocks, threads>>>(
        distances.data_ptr<float>(),
        row_index.data_ptr<int64_t>(),
        col_index.data_ptr<int64_t>(),
        atom_types.data_ptr<int64_t>(),
        density_table.data_ptr<float>(),
        E,
        inv_dr,
        n_r,
        n_pairs,
        rho_out.data_ptr<float>()
    );
}

void force_pass_cuda(
    const torch::Tensor distances,
    const torch::Tensor dist_vec,
    const torch::Tensor row_index,
    const torch::Tensor col_index,
    const torch::Tensor atom_types,
    const torch::Tensor density_deriv_table,
    const torch::Tensor pair_deriv_table,
    const torch::Tensor dF_drho,
    const float inv_dr,
    const int n_r,
    torch::Tensor forces_out){
    int n_pairs = distances.size(0);
    int E = density_deriv_table.size(0);
    int threads = 256;
    int blocks = (n_pairs + threads - 1)/threads;
    force_pass_kernel<<<blocks, threads>>>(
        distances.data_ptr<float>(),
        dist_vec.data_ptr<float>(),
        row_index.data_ptr<int64_t>(),
        col_index.data_ptr<int64_t>(),
        atom_types.data_ptr<int64_t>(),
        density_deriv_table.data_ptr<float>(),
        pair_deriv_table.data_ptr<float>(),
        dF_drho.data_ptr<float>(),
        inv_dr,
        n_r,
        E,
        n_pairs,
        forces_out.data_ptr<float>()
    );
}
