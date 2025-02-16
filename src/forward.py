import time
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# layer + input
BS: int = 8
IN_SEQ_LEN: int = 1024
N_EMBD: int = 768
N_HEAD: int = 12
BLOCK_SIZE: int = 512

# img size
N_CHANNEL = 3
n = (BS * IN_SEQ_LEN * N_EMBD) // N_CHANNEL
factors = []
for i in range(1, int(n**0.5) + 1):
    if n % i == 0:
        factors.append(i)
        if i != n // i:
            factors.append(n // i)
factors = sorted(factors)
mid_index = len(factors) // 2
SHOW_W, SHOW_H = (
    factors[mid_index - 1],
    factors[mid_index] if len(factors) % 2 == 0 else factors[mid_index],
)

# files
PARENT_PATH = Path(__file__).parent.parent
ARTIFACTS_PATH = PARENT_PATH / "artifacts"
CU_FILE = PARENT_PATH / "src" / "forward.cu"


# layer
class CausalSelfAttention(nn.Module):
    def __init__(self, custom=False):
        super().__init__()
        assert N_EMBD % N_HEAD == 0
        self.c_attn = nn.Linear(N_EMBD, N_EMBD * 3)
        self.c_proj = nn.Linear(N_EMBD, N_EMBD)

        ## load the CUDA kernel as a python module
        self.custom = custom
        cuda_source = CU_FILE.read_text()
        cuda_source += """
        torch::Tensor forward(torch::Tensor out,
            torch::Tensor inp,
            int B, int T, int C, int NH,
            int block_size) {
            flash_attn(
                out.data_ptr<float>(),
                inp.data_ptr<float>(),
                B, T, C, NH,
                block_size);
            return out;
            }
        """
        self.flash_attn = load_inline(
            name="FlashAttention",
            cuda_sources=cuda_source,
            cpp_sources="torch::Tensor forward(torch::Tensor out, torch::Tensor inp, int B, int T, int C, int NH, int block_size);",
            functions=["forward"],
            with_cuda=True,
            extra_cuda_cflags=["-O3", "-use_fast_math"],
            build_directory=str(PARENT_PATH / "dist"),
            verbose=True,
        )

    def forward(self, x):
        B, T, C = (
            x.size()
        )  # batch size, sequence length, embedding dimensionality * 3(N_EMBD)
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        # nh is "number of heads", hs is "head size", and C (number of channels) = nh * hs
        # e.g. in GPT-2 (124M), N_HEAD=12, hs=64, so nh*hs=C=768 channels in the Transformer
        qkv = self.c_attn(x)

        if self.custom:
            y = torch.empty((B, T, C), device=x.device)
            y = self.flash_attn.forward(y, qkv, B, T, C, N_HEAD, BLOCK_SIZE)
        else:
            q, k, v = qkv.split(N_EMBD, dim=2)
            k = k.view(B, T, N_HEAD, C // N_HEAD).transpose(1, 2)  # (B, nh, T, hs)
            q = q.view(B, T, N_HEAD, C // N_HEAD).transpose(1, 2)  # (B, nh, T, hs)
            v = v.view(B, T, N_HEAD, C // N_HEAD).transpose(1, 2)  # (B, nh, T, hs)
            y = F.scaled_dot_product_attention(
                q, k, v, is_causal=True
            )  # flash attention
            y = (
                y.transpose(1, 2).contiguous().view(B, T, C)
            )  # re-assemble all head outputs side by side

        # output projection
        y = self.c_proj(y)
        return y


if __name__ == "__main__":
    x = 2 * torch.rand((BS, IN_SEQ_LEN, N_EMBD), device="cuda") - 1  # [-1, 1]
    x_save = x.clone().view(SHOW_W, SHOW_H, N_CHANNEL)
    x_save = (x_save - x_save.min()) / (x_save.max() - x_save.min())
    x_save = x_save.detach().cpu().numpy()
    print(f"Saving image of size: {SHOW_W} x {SHOW_H}")
    plt.imsave(ARTIFACTS_PATH / "in.png", x_save)

    # test python
    start = time.perf_counter()
    layer = CausalSelfAttention().to("cuda")
    with torch.no_grad():
        y = layer(x)
    y_save = y.clone().view(SHOW_W, SHOW_H, N_CHANNEL)
    y_save = (y_save - y_save.min()) / (y_save.max() - y_save.min())
    y_save = y_save.detach().cpu().numpy()
    print(f"Saving image of size: {SHOW_W} x {SHOW_H}")
    print(f"Python forward took {time.perf_counter() - start:.2f} seconds")
    plt.imsave(ARTIFACTS_PATH / "out.png", y_save)

    # test cu
    start = time.perf_counter()
    layer = CausalSelfAttention(custom=True).to("cuda")
    with torch.no_grad():
        y = layer(x)
    y_save = y.clone().view(SHOW_W, SHOW_H, N_CHANNEL)
    y_save = (y_save - y_save.min()) / (y_save.max() - y_save.min())
    y_save = y_save.detach().cpu().numpy()
    print(f"Saving image of size: {SHOW_W} x {SHOW_H}")
    print(f"cu forward took {time.perf_counter() - start:.2f} seconds")
    plt.imsave(ARTIFACTS_PATH / "out_cu.png", y_save)
