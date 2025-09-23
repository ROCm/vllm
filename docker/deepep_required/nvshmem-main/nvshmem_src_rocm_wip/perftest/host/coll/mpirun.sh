mpirun --allow-run-as-root -np 2 -x NVSHMEM_BOOTSTRAP="MPI" barrier_all_on_stream.out >logs/barrier_all_on_stream.log
mpirun --allow-run-as-root -np 2 -x NVSHMEM_BOOTSTRAP="MPI" barrier_on_stream.out -n 1000 -w 10 >logs/barrier_on_stream.log
mpirun --allow-run-as-root -np 2 -x NVSHMEM_BOOTSTRAP="MPI" sync_all_on_stream.out >logs/sync_all_on_stream.log
mpirun --allow-run-as-root -np 2 -x NVSHMEM_BOOTSTRAP="MPI" sync_on_stream.out -n 1000 -w 10 >logs/sync_on_stream.log

# try reduce
mpirun --allow-run-as-root -np 2 -x NVSHMEM_BOOTSTRAP="MPI" reduction_on_stream.out -n 100 -w 10 -b 1 -e 4M >logs/reduction_on_stream.log
