hipcc -I ../../common -I ../../../build/src/include bw.cpp -o bw.out -L ../../../build/src/lib -lnvshmem_host -lnvshmem_device -L ../../../perftest/common -lutils -fgpu-rdc
hipcc -I ../../common -I ../../../build/src/include latency.cpp -o latency.out -L ../../../build/src/lib -lnvshmem_host -lnvshmem_device -L ../../../perftest/common -lutils -fgpu-rdc
hipcc -I ../../common -I ../../../build/src/include stream_latency.hip.cpp -o stream_latency.out -L ../../../build/src/lib -lnvshmem_host -lnvshmem_device -L ../../../perftest/common -lutils -fgpu-rdc
