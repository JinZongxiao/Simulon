#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Device-side global controls for smoothing
__device__ int d_lj_mode = 0;        // 0: raw, 1: force-shift, 2: switch
__device__ float d_lj_rc = 0.0f;     // cutoff radius (Å)
__device__ float d_lj_rs = 0.0f;     // switch start radius (Å) when mode==2

__global__ void lj_energy_force_kernel(
    const float* pos,            // [N, 3]
    const int64_t* edge_i,       // [E]
    const int64_t* edge_j,       // [E]
    const float* sigma,          // [E]
    const float* epsilon,        // [E]
    const float box_length,
    float* forces,               // [N, 3]
    float* energy_out,           // [1]
    int E
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= E) return;

    int i = edge_i[idx];
    int j = edge_j[idx];
    if (i==j) return;

    float3 ri = make_float3(pos[3 * i + 0], pos[3 * i + 1], pos[3 * i + 2]);
    float3 rj = make_float3(pos[3 * j + 0], pos[3 * j + 1], pos[3 * j + 2]);

    float3 rij;
    // 与 CPU 版本一致：rij = ri - rj
    rij.x = ri.x - rj.x;
    rij.y = ri.y - rj.y;
    rij.z = ri.z - rj.z;

    // Minimum image convention
    rij.x -= box_length * roundf(rij.x / box_length);
    rij.y -= box_length * roundf(rij.y / box_length);
    rij.z -= box_length * roundf(rij.z / box_length);

    float r2 = rij.x * rij.x + rij.y * rij.y + rij.z * rij.z;
    if (r2 < 1e-12f) return;
    float r = sqrtf(r2);
    float inv_r = 1.0f / r;
    float inv_r2 = 1.0f / r2;
    float inv_r6 = inv_r2 * inv_r2 * inv_r2;

    float sig = sigma[idx];
    float eps = epsilon[idx];

    // Base LJ
    float s6 = powf(sig, 6.0f);
    float s12 = s6 * s6;
    float U_raw = 4.0f * eps * (s12 * inv_r6 * inv_r6 - s6 * inv_r6);
    float Fmag_raw = 24.0f * eps * (2.0f * s12 * inv_r6 * inv_r6 - s6 * inv_r6) * inv_r;

    // Read smoothing controls
    int mode;
    float rc, rs;
    // Copy device globals to registers
    mode = d_lj_mode;
    rc = d_lj_rc;
    rs = d_lj_rs;

    // Effective energy and force magnitude after smoothing
    float U_eff = U_raw;
    float Fmag = Fmag_raw;

    if (mode == 1) { // force-shift
        if (rc > 0.0f && r >= rc) return; // safety beyond cutoff
        float sr_rc = sig / rc;
        float sr6_rc = powf(sr_rc, 6.0f);
        float sr12_rc = sr6_rc * sr6_rc;
        float U_rc = 4.0f * eps * (sr12_rc - sr6_rc);
        float dUdr_rc = 24.0f * eps * (-2.0f * powf(sig,12.0f) / powf(rc,13.0f) + powf(sig,6.0f) / powf(rc,7.0f));
        U_eff = U_raw - U_rc - (r - rc) * dUdr_rc;
        float Fmag_rc = 24.0f * eps * (2.0f * sr12_rc - sr6_rc) / rc; // |F(rc)|
        Fmag = Fmag_raw - Fmag_rc;
    } else if (mode == 2) { // switch
        if (rc > 0.0f && r >= rc) return; // outside cutoff -> zero
        float S = 1.0f;
        float dSdr = 0.0f;
        if (r <= rs) {
            S = 1.0f; dSdr = 0.0f;
        } else if (r < rc) {
            float x = (r - rs) / (rc - rs);
            float x2 = x * x, x3 = x2 * x, x4 = x3 * x, x5 = x4 * x;
            S = 1.0f - 10.0f * x3 + 15.0f * x4 - 6.0f * x5;
            float dSdx = -30.0f * x2 + 60.0f * x3 - 30.0f * x4;
            dSdr = dSdx / (rc - rs);
        } else {
            S = 0.0f; dSdr = 0.0f;
        }
        // F = -dU_eff/dr = -(dU_raw/dr · S + U_raw · dS/dr)
        //                = Fmag_raw·S - U_raw·dSdr  (因 Fmag_raw ≡ -dU_raw/dr)
        U_eff = U_raw * S;
        Fmag = Fmag_raw * S - U_raw * dSdr;
    }

    // Force vector
    float scale = Fmag * inv_r; // since Fmag defined along radial direction
    float3 fij;
    fij.x = scale * rij.x;
    fij.y = scale * rij.y;
    fij.z = scale * rij.z;

    // Accumulate
    atomicAdd(&forces[3 * i + 0], fij.x);
    atomicAdd(&forces[3 * i + 1], fij.y);
    atomicAdd(&forces[3 * i + 2], fij.z);
    atomicAdd(&forces[3 * j + 0], -fij.x);
    atomicAdd(&forces[3 * j + 1], -fij.y);
    atomicAdd(&forces[3 * j + 2], -fij.z);

    atomicAdd(energy_out, U_eff);
}

void launch_lj_energy_force_kernel(
    torch::Tensor pos, torch::Tensor edge_index,
    torch::Tensor sigma, torch::Tensor epsilon,
    float box_length,
    torch::Tensor forces_out, torch::Tensor energy_out
) {
    const int E = edge_index.size(1);
    const int threads = 256;
    const int blocks = (E + threads - 1) / threads;

    lj_energy_force_kernel<<<blocks, threads>>>(
        pos.data_ptr<float>(),
        edge_index[0].data_ptr<int64_t>(),
        edge_index[1].data_ptr<int64_t>(),
        sigma.data_ptr<float>(),
        epsilon.data_ptr<float>(),
        box_length,
        forces_out.data_ptr<float>(),
        energy_out.data_ptr<float>(),
        E
    );
}

// Host setter to configure smoothing mode and radii
void set_lj_smoothing_cuda(int mode, float rc, float rs) {
    cudaMemcpyToSymbol(d_lj_mode, &mode, sizeof(int));
    cudaMemcpyToSymbol(d_lj_rc, &rc, sizeof(float));
    cudaMemcpyToSymbol(d_lj_rs, &rs, sizeof(float));
}
