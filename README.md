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

# (optional) install modal
uv add modal
uv run modal setup
```

## usage

Test C++ queue with Python:

```bash
uv run src/queue_test.py
```

Compile and test in C++:

```bash
nvcc -O3 -use_fast_math src/forward.cu -o dist/forward -L/usr/lib
./dist/forward
```

Test attn forward pass in Python:

```bash
uv run src/forward.py
```

or with Modal:

```bash
modal run src/forward.py
```

Run website locally:

```bash
uv run src/app.py
```

Serve on Modal:

```bash
modal serve src/app.py
```

Deploy on Modal:

```bash
modal deploy --env=main src/app.py
```
