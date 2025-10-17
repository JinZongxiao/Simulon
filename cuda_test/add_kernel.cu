#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

template <typename scalar_t>
__global__ void add_kernel(const scalar_t* a, const scalar_t* b, scalar_t* out, int64_t size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] = a[idx] + b[idx];
    }
}

torch::Tensor add_cuda_forward(torch::Tensor a, torch::Tensor b) {
    TORCH_CHECK(a.device().is_cuda(), "a must be CUDA tensor");
    TORCH_CHECK(b.device().is_cuda(), "b must be CUDA tensor");
    TORCH_CHECK(a.sizes() == b.sizes(), "a and b must be same size");

    auto out = torch::empty_like(a);
    int64_t size = a.numel();
    int threads = 256;
    int blocks = (size + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(a.scalar_type(), "add_cuda_forward", ([&] {
        add_kernel<scalar_t><<<blocks, threads>>>(
            a.data_ptr<scalar_t>(),
            b.data_ptr<scalar_t>(),
            out.data_ptr<scalar_t>(),
            size);
    }));

    return out;
}
