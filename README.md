# flash-attn

Flash Attention in Cuda

## setup

```bash
# install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# install dependencies
uv sync

# install opencv for saving images in C++
sudo apt-get -y install libopencv-dev

# install cudnn so we can use FlashAttention and run fast (optional)
# https://developer.nvidia.com/cudnn-downloads
# for me, CUDA 12 (run `nvcc --version`) running on Linux x86_64 Ubuntu 22.04
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get -y install libcudnn9-dev-cuda-12 cudnn cudnn-cuda-12

# "install" cudnn-frontend to ~/
git clone https://github.com/NVIDIA/cudnn-frontend.git

# install MPI (optional, if you intend to use multiple GPUs)
sudo apt install openmpi-bin openmpi-doc libopenmpi-dev
```

## usage

Test in Python:

```bash
uv run src/forward.py
```

Compile the test with cuDNN:

```bash
nvcc -I./cudnn-frontend/include -DENABLE_CUDNN -O3 -lcublas -lcublasLt --use_fast_math -lcudnn src/forward.cu -o dist/forward -I/usr/include/opencv4 -L/usr/lib -lopencv_core -lopencv_imgproc -lopencv_imgcodecs
```

Compile the test without cuDNN:

```bash
nvcc -O3 --use_fast_math -lcublas -lcublasLt src/forward.cu -o dist/forward -I/usr/include/opencv4 -L/usr/lib -lopencv_core -lopencv_imgproc -lopencv_imgcodecs
```

Test in C++:

```bash
./dist/forward
```
