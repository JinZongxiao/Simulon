#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

__global__ void neighbor_search_kernel(
    const float* positions,      // [N, 3]
    const int N,
    const float cutoff,
    const float box_length,
    int* neighbor_count,         // [N]
    int* neighbor_list,          // [N * max_neighbors]
    const int max_neighbors
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    int count = 0;
    float3 pos_i = make_float3(positions[3*i], positions[3*i+1], positions[3*i+2]);

    for (int j = i + 1; j < N; j++) {   // upper-triangle only (j > i)
        float3 pos_j = make_float3(positions[3*j], positions[3*j+1], positions[3*j+2]);

        float3 rij;
        rij.x = pos_j.x - pos_i.x;
        rij.y = pos_j.y - pos_i.y;
        rij.z = pos_j.z - pos_i.z;

        rij.x -= box_length * roundf(rij.x / box_length);
        rij.y -= box_length * roundf(rij.y / box_length);
        rij.z -= box_length * roundf(rij.z / box_length);

        float dist = sqrtf(rij.x*rij.x + rij.y*rij.y + rij.z*rij.z);

        if (dist < cutoff && count < max_neighbors) {
            neighbor_list[i * max_neighbors + count] = j;
            count++;
        }
    }

    neighbor_count[i] = count;
}

/*
 * 修复：原版 build_edge_list_kernel 在每个线程内串行求前缀和
 *   for (int k = 0; k < i; k++) start_idx += neighbor_count[k];
 * 这是 O(N²) GPU 工作量，对大系统极慢。
 *
 * 修复方案：在 C++ host 代码中用 torch::cumsum 预先计算 start_idx 张量，
 * 并作为参数传入 kernel，使每个线程直接读取自己的起始偏移量 → O(1) per thread。
 */
__global__ void build_edge_list_kernel(
    const int* neighbor_count,
    const int* start_idx,        // [N] 由 host 预计算（exclusive prefix sum）
    const int* neighbor_list,
    const int N,
    const int max_neighbors,
    int* edge_i,
    int* edge_j
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    int base = start_idx[i];    // O(1) 直接读，无串行循环

    for (int n = 0; n < neighbor_count[i]; n++) {
        int idx = base + n;
        edge_i[idx] = i;
        edge_j[idx] = neighbor_list[i * max_neighbors + n];
    }
}

std::vector<torch::Tensor> neighbor_search_cuda(
    torch::Tensor positions,
    float cutoff,
    float box_length
) {
    const int N = positions.size(0);
    const int max_neighbors = 500;

    auto opts_i32 = torch::dtype(torch::kInt32).device(positions.device());
    auto neighbor_count = torch::zeros({N},              opts_i32);
    auto neighbor_list  = torch::zeros({N, max_neighbors}, opts_i32);

    const int threads = 256;
    const int blocks  = (N + threads - 1) / threads;

    neighbor_search_kernel<<<blocks, threads>>>(
        positions.data_ptr<float>(),
        N, cutoff, box_length,
        neighbor_count.data_ptr<int>(),
        neighbor_list.data_ptr<int>(),
        max_neighbors
    );
    cudaDeviceSynchronize();

    int total_edges = neighbor_count.sum().item<int>();
    if (total_edges == 0) {
        auto ei = torch::zeros({2, 0}, torch::dtype(torch::kInt64).device(positions.device()));
        auto ea = torch::zeros({0},    torch::dtype(torch::kFloat32).device(positions.device()));
        return {ei, ea};
    }

    // ── 关键修复：用 PyTorch cumsum 计算 exclusive prefix sum（host side） ──
    // cumsum[i] = sum(neighbor_count[0..i])（inclusive）
    // exclusive start: start_idx[0] = 0, start_idx[i] = cumsum[i-1]
    auto cumsum_all = torch::cumsum(neighbor_count.to(torch::kInt64), 0);  // [N]
    auto start_idx_full = torch::zeros({N}, torch::dtype(torch::kInt64).device(positions.device()));
    if (N > 1) {
        start_idx_full.slice(0, 1, N) = cumsum_all.slice(0, 0, N - 1);
    }
    auto start_idx = start_idx_full.to(torch::kInt32);

    auto edge_i_int = torch::zeros({total_edges}, opts_i32);
    auto edge_j_int = torch::zeros({total_edges}, opts_i32);

    build_edge_list_kernel<<<blocks, threads>>>(
        neighbor_count.data_ptr<int>(),
        start_idx.data_ptr<int>(),
        neighbor_list.data_ptr<int>(),
        N, max_neighbors,
        edge_i_int.data_ptr<int>(),
        edge_j_int.data_ptr<int>()
    );
    cudaDeviceSynchronize();

    auto edge_i = edge_i_int.to(torch::kInt64);
    auto edge_j = edge_j_int.to(torch::kInt64);

    auto edge_index = torch::stack({edge_i, edge_j}, 0);
    auto pos_i = positions.index_select(0, edge_i);
    auto pos_j = positions.index_select(0, edge_j);
    auto rij = pos_j - pos_i;
    rij = rij - box_length * torch::round(rij / box_length);
    auto distances = torch::norm(rij, 2, 1);

    return {edge_index, distances};
}
