# succeed
hipcc -I ../../common -I ../../../build/src/include alltoall_latency.hip.cpp -o alltoall_latency.out -L ../../../build/src/lib -lnvshmem_host -lnvshmem_device -L ../../../perftest/common -lutils -fgpu-rdc
hipcc -I ../../common -I ../../../build/src/include barrier_latency.hip.cpp -o barrier_latency.out -L ../../../build/src/lib -lnvshmem_host -lnvshmem_device -L ../../../perftest/common -lutils -fgpu-rdc 
hipcc -I ../../common -I ../../../build/src/include bcast_latency.hip.cpp -o bcast_latency.out -L ../../../build/src/lib -lnvshmem_host -lnvshmem_device -L ../../../perftest/common -lutils -fgpu-rdc
hipcc -I ../../common -I ../../../build/src/include sync_latency.hip.cpp -o sync_latency.out -L ../../../build/src/lib -lnvshmem_host -lnvshmem_device -L ../../../perftest/common -lutils -fgpu-rdc

# try reduce
hipcc -I ../../common -I ../../../build/src/include reduction_latency.hip.cpp -o reduction_latency.out -L ../../../build/src/lib -lnvshmem_host -lnvshmem_device -L ../../../perftest/common -lutils -fgpu-rdc
