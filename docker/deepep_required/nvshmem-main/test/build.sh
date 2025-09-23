hipcc -I ../nvshmem_src_rocm_wip/build/src/include nvshmemHelloWorld.cu.hip -o nvshmemHelloWorld.out -L ../nvshmem_src_rocm_wip/build/src/lib -lnvshmem_host -lnvshmem_device -fgpu-rdc
