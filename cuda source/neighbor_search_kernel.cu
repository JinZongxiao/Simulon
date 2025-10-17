#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

__global__ void neighbor_search_kernel(
    const float* positions,      // [N, 3]
    const int N,
    const float cutoff,
    const float box_length,
    int* neighbor_count,         // [N] - number of neighbors for each atom
    int* neighbor_list,          // [N * max_neighbors] - neighbor indices
    const int max_neighbors
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    int count = 0;
    float3 pos_i = make_float3(positions[3*i], positions[3*i+1], positions[3*i+2]);
    
    for (int j = 0; j < N; j++) {
        if (i >= j) continue; // Only consider j > i to avoid duplicates
        
        float3 pos_j = make_float3(positions[3*j], positions[3*j+1], positions[3*j+2]);
        
        // Calculate distance with PBC
        float3 rij;
        rij.x = pos_j.x - pos_i.x;
        rij.y = pos_j.y - pos_i.y;
        rij.z = pos_j.z - pos_i.z;
        
        // Apply periodic boundary conditions
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

__global__ void build_edge_list_kernel(
    const int* neighbor_count,
    const int* neighbor_list,
    const int N,
    const int max_neighbors,
    int* edge_i,
    int* edge_j,
    int* total_edges
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    
    int start_idx = 0;
    for (int k = 0; k < i; k++) {
        start_idx += neighbor_count[k];
    }
    
    for (int n = 0; n < neighbor_count[i]; n++) {
        int idx = start_idx + n;
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
    const int max_neighbors = 500; // Adjust based on your system
    
    // Allocate device memory for neighbor list
    auto neighbor_count = torch::zeros({N}, torch::dtype(torch::kInt32).device(positions.device()));
    auto neighbor_list = torch::zeros({N, max_neighbors}, torch::dtype(torch::kInt32).device(positions.device()));
    
    const int threads = 256;
    const int blocks = (N + threads - 1) / threads;
    
    // Launch neighbor search kernel
    neighbor_search_kernel<<<blocks, threads>>>(
        positions.data_ptr<float>(),
        N,
        cutoff,
        box_length,
        neighbor_count.data_ptr<int>(),
        neighbor_list.data_ptr<int>(),
        max_neighbors
    );
    
    cudaDeviceSynchronize();
    
    // Calculate total number of edges
    int total_edges = neighbor_count.sum().item<int>();
    
    if (total_edges == 0) {
        auto empty_edge_index = torch::zeros({2, 0}, torch::dtype(torch::kInt64).device(positions.device()));
        auto empty_distances = torch::zeros({0}, torch::dtype(torch::kFloat32).device(positions.device()));
        return {empty_edge_index, empty_distances};
    }
    
    // Allocate edge index tensors
    auto edge_i_int = torch::zeros({total_edges}, torch::dtype(torch::kInt32).device(positions.device()));
    auto edge_j_int = torch::zeros({total_edges}, torch::dtype(torch::kInt32).device(positions.device()));
    
    // Build edge list
    build_edge_list_kernel<<<blocks, threads>>>(
        neighbor_count.data_ptr<int>(),
        neighbor_list.data_ptr<int>(),
        N,
        max_neighbors,
        edge_i_int.data_ptr<int>(),
        edge_j_int.data_ptr<int>(),
        nullptr
    );
    
    cudaDeviceSynchronize();
    
    // Convert to int64 for PyTorch compatibility
    auto edge_i = edge_i_int.to(torch::kInt64);
    auto edge_j = edge_j_int.to(torch::kInt64);
    
    // Calculate distances
    auto edge_index = torch::stack({edge_i, edge_j}, 0);
    auto pos_i = positions.index_select(0, edge_i);
    auto pos_j = positions.index_select(0, edge_j);
    
    auto rij = pos_j - pos_i;
    rij = rij - box_length * torch::round(rij / box_length);
    auto distances = torch::norm(rij, 2, 1);
    
    return {edge_index, distances};
}
