
# succeed
hipcc -I ../../common -I ../../../build/src/include shmem_atomic_bw.hip.cpp -o shmem_atomic_bw.out -L ../../../build/src/lib -lnvshmem_host -lnvshmem_device -L ../../../perftest/common -lutils -fgpu-rdc
hipcc -I ../../common -I ../../../build/src/include shmem_atomic_latency.hip.cpp -o shmem_atomic_latency.out -L ../../../build/src/lib -lnvshmem_host -lnvshmem_device -L ../../../perftest/common -lutils -fgpu-rdc

hipcc -I ../../common -I ../../../build/src/include shmem_g_bw.hip.cpp -o shmem_g_bw.out -L ../../../build/src/lib -lnvshmem_host -lnvshmem_device -L ../../../perftest/common -lutils -fgpu-rdc
hipcc -I ../../common -I ../../../build/src/include shmem_g_latency.hip.cpp -o shmem_g_latency.out -L ../../../build/src/lib -lnvshmem_host -lnvshmem_device -L ../../../perftest/common -lutils -fgpu-rdc
hipcc -I ../../common -I ../../../build/src/include shmem_get_latency.hip.cpp -o shmem_get_latency.out -L ../../../build/src/lib -lnvshmem_host -lnvshmem_device -L ../../../perftest/common -lutils -fgpu-rdc
hipcc -I ../../common -I ../../../build/src/include shmem_p_bw.hip.cpp -o shmem_p_bw.out -L ../../../build/src/lib -lnvshmem_host -lnvshmem_device -L ../../../perftest/common -lutils -fgpu-rdc
hipcc -I ../../common -I ../../../build/src/include shmem_get_bw.hip.cpp -o shmem_get_bw.out -L ../../../build/src/lib -lnvshmem_host -lnvshmem_device -L ../../../perftest/common -lutils -fgpu-rdc
hipcc -I ../../common -I ../../../build/src/include shmem_p_latency.hip.cpp -o shmem_p_latency.out -L ../../../build/src/lib -lnvshmem_host -lnvshmem_device -L ../../../perftest/common -lutils -fgpu-rdc
hipcc -I ../../common -I ../../../build/src/include shmem_put_atomic_ping_pong_latency.hip.cpp -o shmem_put_atomic_ping_pong_latency.out -L ../../../build/src/lib -lnvshmem_host -lnvshmem_device -L ../../../perftest/common -lutils -fgpu-rdc
hipcc -I ../../common -I ../../../build/src/include shmem_put_bw.hip.cpp -o shmem_put_bw.out -L ../../../build/src/lib -lnvshmem_host -lnvshmem_device -L ../../../perftest/common -lutils -fgpu-rdc
hipcc -I ../../common -I ../../../build/src/include shmem_put_latency.hip.cpp -o shmem_put_latency.out -L ../../../build/src/lib -lnvshmem_host -lnvshmem_device -L ../../../perftest/common -lutils -fgpu-rdc
hipcc -I ../../common -I ../../../build/src/include shmem_put_ping_pong_latency.hip.cpp -o shmem_put_ping_pong_latency.out -L ../../../build/src/lib -lnvshmem_host -lnvshmem_device -L ../../../perftest/common -lutils -fgpu-rdc
hipcc -I ../../common -I ../../../build/src/include shmem_put_signal_ping_pong_latency.hip.cpp -o shmem_put_signal_ping_pong_latency.out -L ../../../build/src/lib -lnvshmem_host -lnvshmem_device -L ../../../perftest/common -lutils -fgpu-rdc
hipcc -I ../../common -I ../../../build/src/include shmem_signal_ping_pong_latency.hip.cpp -o shmem_signal_ping_pong_latency.out -L ../../../build/src/lib -lnvshmem_host -lnvshmem_device -L ../../../perftest/common -lutils -fgpu-rdc
hipcc -I ../../common -I ../../../build/src/include shmem_st_bw.hip.cpp -o shmem_st_bw.out -L ../../../build/src/lib -lnvshmem_host -lnvshmem_device -L ../../../perftest/common -lutils -fgpu-rdc

hipcc -I ../../common -I ../../../build/src/include shmem_p_ping_pong_latency.hip.cpp -o shmem_p_ping_pong_latency.out -L ../../../build/src/lib -lnvshmem_host -lnvshmem_device -L ../../../perftest/common -lutils -fgpu-rdc

hipcc -I ../../common -I ../../../build/src/include shmem_atomic_ping_pong_latency.hip.cpp -o shmem_atomic_ping_pong_latency.out -L ../../../build/src/lib -lnvshmem_host -lnvshmem_device -L ../../../perftest/common -lutils -fgpu-rdc
