nvcc -rdc=true -ccbin g++ -gencode=$NVCC_GENCODE -I $NVSHMEM_HOME/include nvshmemHelloWorld.cu -o nvshmemHelloWorld.out -L $NVSHMEM_HOME/lib -lnvshmem_host -lnvshmem_device
