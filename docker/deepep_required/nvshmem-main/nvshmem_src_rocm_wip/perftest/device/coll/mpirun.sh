# succeed
mpirun --allow-run-as-root -np 2 -x NVSHMEM_BOOTSTRAP="MPI" alltoall_latency.out -n 100 -w 10 -b 1 -e 4M >logs/alltoall_latency.log
mpirun --allow-run-as-root -np 2 -x NVSHMEM_BOOTSTRAP="MPI" barrier_latency.out -n 1000 -w 10 >logs/barrier_latency.log
mpirun --allow-run-as-root -np 2 -x NVSHMEM_BOOTSTRAP="MPI" bcast_latency.out -n 100 -w 10 -b 1 -e 4M >logs/bcast_latency.log
mpirun --allow-run-as-root -np 2 -x NVSHMEM_BOOTSTRAP="MPI" sync_latency.out -n 1000 -w 10 >logs/sync_latency.log

# try reduce
mpirun --allow-run-as-root -np 2 -x NVSHMEM_BOOTSTRAP="MPI" reduction_latency.out -n 50 -w 10 -b 1 -e 4M >logs/reduction_latency.log
