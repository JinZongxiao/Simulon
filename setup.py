from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='simulon_cuda',
    ext_modules=[
        CUDAExtension(
            name='simulon_cuda',
            sources=[
                'cuda source/ops.cpp',
                'cuda source/lj_energy_force_kernel.cu',
                'cuda source/lj_energy_force.cpp',
                'cuda source/neighbor_search_kernel.cu',
                'cuda source/neighbor_search.cpp',
                # EAM merged sources
                'cuda source/eam_cuda_ext.cpp',
                'cuda source/eam_cuda_ext_kernel.cu',
            ],
            extra_compile_args={
                'cxx': ['-O3'],
                'nvcc': ['-O3']
            }
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
