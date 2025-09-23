mpirun --allow-run-as-root -np 2 -x NVSHMEM_BOOTSTRAP="MPI" bw.out
mpirun --allow-run-as-root -np 2 -x NVSHMEM_BOOTSTRAP="MPI" latency.out
mpirun --allow-run-as-root -np 2 -x NVSHMEM_BOOTSTRAP="MPI" stream_latency.out
