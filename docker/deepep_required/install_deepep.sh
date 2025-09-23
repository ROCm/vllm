#!/bin/bash
set -ex

# install nvshmem
cd /opt/nvshmem-main/nvshmem_src_rocm_wip
./configure && make -j -C build
echo "✅ Successfully built NVSHMEM"

# install deepep
cd /opt/deepEP_private
AITER_MOE=1 ROCM_HOME=/opt/rocm-6.4.3/ OMPI_DIR=/opt/mpich/install ROCSHMEM_DIR=/opt/nvshmem-main/nvshmem_src_rocm_wip/build/src/ python3 setup.py --variance rocm build develop --force-nvshmem-api
echo "✅ Successfully built and installed DeepEP"
