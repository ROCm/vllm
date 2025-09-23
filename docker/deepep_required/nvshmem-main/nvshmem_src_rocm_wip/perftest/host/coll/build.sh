hipcc -I ../../common -I ../../../build/src/include barrier_all_on_stream.cpp -o barrier_all_on_stream.out -L ../../../build/src/lib -lnvshmem_host -lnvshmem_device -L ../../../perftest/common -lutils -fgpu-rdc
hipcc -I ../../common -I ../../../build/src/include barrier_on_stream.cpp -o barrier_on_stream.out -L ../../../build/src/lib -lnvshmem_host -lnvshmem_device -L ../../../perftest/common -lutils -fgpu-rdc
hipcc -I ../../common -I ../../../build/src/include sync_all_on_stream.cpp -o sync_all_on_stream.out -L ../../../build/src/lib -lnvshmem_host -lnvshmem_device -L ../../../perftest/common -lutils -fgpu-rdc
hipcc -I ../../common -I ../../../build/src/include sync_on_stream.cpp -o sync_on_stream.out -L ../../../build/src/lib -lnvshmem_host -lnvshmem_device -L ../../../perftest/common -lutils -fgpu-rdc

# try reduce
hipcc -I ../../common -I ../../../build/src/include reduction_on_stream.cpp -o reduction_on_stream.out -L ../../../build/src/lib -lnvshmem_host -lnvshmem_device -L ../../../perftest/common -lutils -fgpu-rdc
