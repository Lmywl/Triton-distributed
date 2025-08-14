git submodule deinit --all -f # deinit previous submodules
rm -rf 3rdparty/triton # remove previous triton
git submodule update --init --recursive

pip3 install torch==2.4.1
pip3 install cuda-python==12.4 # need to align with your nvcc version
pip3 install ninja cmake wheel pybind11 numpy chardet pytest
pip3 install pynvml>=11.5.3

pip3 install nvidia-nvshmem-cu12==3.3.9 cuda.core==0.2.0 "Cython>=0.29.24"
CPPFLAGS="-I/usr/local/cuda/include" pip3 install https://developer.download.nvidia.com/compute/nvshmem/redist/nvshmem_python/source/nvshmem_python-source-0.1.0.36132199_cuda12-archive.tar.xz

# Remove triton installed with torch
pip uninstall triton -y
pip uninstall triton_dist -y # remove previous triton-dist
rm -rf /usr/local/lib/python3.12/dist-packages/triton -y
# Install Triton-distributed
export USE_TRITON_DISTRIBUTED_AOT=0
pip3 install -e python --verbose --no-build-isolation --use-pep517 &> install.log
