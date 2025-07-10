python3 -m pip install --upgrade pip

# Build & install AMD SMI
python3 -m pip install /opt/rocm/share/amd_smi

# Install dependencies
python3 -m pip install --upgrade numba \
    scipy \
    huggingface-hub[cli,hf_transfer] \
    setuptools_scm
python3 -m pip install "numpy<2"
python3 -m pip install -r requirements/rocm.txt

# Build vLLM for MI210/MI250/MI300.
export PYTORCH_ROCM_ARCH="gfx90a;gfx942"
python3 setup.py develop

