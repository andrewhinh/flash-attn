# flash-attn

Flash Attention in Cuda

## setup

```bash
# install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# install dependencies
uv sync

# for me, CUDA 12 (run `nvcc --version`) running on Linux x86_64 Ubuntu 22.04
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get -y install libcudnn9-dev-cuda-12
```

## usage

Test in Python:

```bash
uv run src/forward.py
```

Compile and test in C++:

```bash
nvcc -O3 src/forward.cu -o dist/forward -L/usr/lib
./dist/forward
```
