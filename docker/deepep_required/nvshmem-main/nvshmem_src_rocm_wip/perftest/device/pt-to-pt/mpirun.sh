# succeed
mpirun --allow-run-as-root -np 2 -x NVSHMEM_BOOTSTRAP="MPI" shmem_atomic_bw.out -n 10 -w 10 -c 4 -e 65536 -t 1024 -a inc >logs/shmem_atomic_bw.log
mpirun --allow-run-as-root -np 2 -x NVSHMEM_BOOTSTRAP="MPI" shmem_atomic_latency.out >logs/shmem_atomic_latency.log

mpirun --allow-run-as-root -np 2 -x NVSHMEM_BOOTSTRAP="MPI" shmem_g_bw.out -n 100 -w 10 -t 1024 -c 8 -b 1024 -e 65536 -d double >logs/shmem_g_bw.log
mpirun --allow-run-as-root -np 2 -x NVSHMEM_BOOTSTRAP="MPI" shmem_g_latency.out -n 200 -w 20 -t 512 -e 64K >logs/shmem_g_latency.log
mpirun --allow-run-as-root -np 2 -x NVSHMEM_BOOTSTRAP="MPI" shmem_get_latency.out -n 200 -w 20 -t 1024 -e 64K >logs/shmem_get_latency.log
mpirun --allow-run-as-root -np 2 -x NVSHMEM_BOOTSTRAP="MPI" shmem_p_bw.out -n 10 -w 10 -t 1024 -c 4 -b 1024 -e 64K -s 1 >logs/shmem_p_bw.log
mpirun --allow-run-as-root -np 2 -x NVSHMEM_BOOTSTRAP="MPI" shmem_get_bw.out -n 200 -w 20 -b 1024 -e 32M -c 4 -t 1024 >logs/shmem_get_bw.log
mpirun --allow-run-as-root -np 2 -x NVSHMEM_BOOTSTRAP="MPI" shmem_p_latency.out -t 512 -e 64K -n 50 -w 5 >logs/shmem_p_latency.log
mpirun --allow-run-as-root -np 2 -x NVSHMEM_BOOTSTRAP="MPI" shmem_put_atomic_ping_pong_latency.out >logs/shmem_put_atomic_ping_pong_latency.log
mpirun --allow-run-as-root -np 2 -x NVSHMEM_BOOTSTRAP="MPI" shmem_put_bw.out -n 200 -w 20 -c 4 -t 1024 -e 32M >logs/shmem_put_bw.log
mpirun --allow-run-as-root -np 2 -x NVSHMEM_BOOTSTRAP="MPI" shmem_put_latency.out -e 64K -t 1024 -n 200 -w 20 >logs/shmem_put_latency.log
mpirun --allow-run-as-root -np 2 -x NVSHMEM_BOOTSTRAP="MPI" shmem_put_ping_pong_latency.out -e 1M -n 500 -w 50 >logs/shmem_put_ping_pong_latency.log
mpirun --allow-run-as-root -np 2 -x NVSHMEM_BOOTSTRAP="MPI" shmem_put_signal_ping_pong_latency.out -e 1M -n 500 -w 50 >logs/shmem_put_signal_ping_pong_latency.log
mpirun --allow-run-as-root -np 2 -x NVSHMEM_BOOTSTRAP="MPI" shmem_signal_ping_pong_latency.out -n 500 -w 50 >logs/shmem_signal_ping_pong_latency.log
mpirun --allow-run-as-root -np 2 -x NVSHMEM_BOOTSTRAP="MPI" shmem_st_bw.out -n 10 -w 10 -t 1024 -c 4 -e 32M >logs/shmem_st_bw.log

mpirun --allow-run-as-root -np 2 -x NVSHMEM_BOOTSTRAP="MPI" -x HSA_ENABLE_SDMA=0 shmem_p_ping_pong_latency.out -t 512 -e 16K -n 500 -w 50 >logs/shmem_p_ping_pong_latency.log

mpirun --allow-run-as-root -np 2 -x NVSHMEM_BOOTSTRAP="MPI" shmem_atomic_ping_pong_latency.out >logs/shmem_atomic_ping_pong_latency.log
