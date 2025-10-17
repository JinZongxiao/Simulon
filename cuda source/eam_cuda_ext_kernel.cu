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
__global__ void density_pass_kernel(
    const float* __restrict__ distances,
    const int64_t* __restrict__ row_index,
    const int64_t* __restrict__ col_index,
    const int64_t* __restrict__ atom_types,
    const float* __restrict__ density_table, // [E, n_r]
    int E,
    float inv_dr,
    int n_r,
    int n_pairs,
    float* __restrict__ rho_out
){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= n_pairs) return;
    float r = distances[idx];
    int j_atom = col_index[idx];
    int j_type = (int)atom_types[j_atom];
    const float* table_j = density_table + j_type * n_r;
    float val = lerp_table(r, inv_dr, n_r, table_j);
    int i_atom = row_index[idx];
    atomicAdd(rho_out + i_atom, val);
}

__global__ void force_pass_kernel(
    const float* __restrict__ distances,
    const float* __restrict__ dist_vec, // [n_pairs,3]
    const int64_t* __restrict__ row_index,
    const int64_t* __restrict__ col_index,
    const int64_t* __restrict__ atom_types,
    const float* __restrict__ density_deriv_table, // [E,n_r]
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

    const float* dens_deriv_j = density_deriv_table + j_type * n_r;
    const float* pair_deriv_ij = pair_deriv_table + ( (i_type * E + j_type) * n_r );

    float d_rho_dr = lerp_table(r, inv_dr, n_r, dens_deriv_j);
    float dphi_dr = lerp_table(r, inv_dr, n_r, pair_deriv_ij);
    float dF_sum = dF_drho[i_atom] + dF_drho[j_atom];
    float scalar = dF_sum * d_rho_dr + dphi_dr;

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
