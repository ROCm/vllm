mpirun --allow-run-as-root -np 2 -x NVSHMEM_DEBUG=INFO -x NVSHMEM_DEBUG_SUBSYS=ALL -x NVSHMEM_BOOTSTRAP="MPI" ../test/nvshmemHelloWorld.out
