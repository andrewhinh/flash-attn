from pathlib import Path

import modal
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

from utils import (
    ALLOW_CONCURRENT_INPUTS,
    APP_NAME,
    ARTIFACTS_PATH,
    BLOCK_SIZE,
    CONTAINER_IDLE_TIMEOUT,
    DIST_PATH,
    GPU_CONFIG,
    IMAGE,
    N_EMBD,
    SRC_PATH,
    TIMEOUT,
    get_device,
    to_image,
)

# helpers
BS: int = 8
IN_SEQ_LEN: int = 1024
N_HEAD: int = 12


# layer
class CausalSelfAttention(nn.Module):
    def __init__(self, custom=False):
        super().__init__()
        assert N_EMBD % N_HEAD == 0
        self.c_attn = nn.Linear(N_EMBD, N_EMBD * 3)
        self.c_proj = nn.Linear(N_EMBD, N_EMBD)

        ## load the CUDA kernel as a python module
        self.custom = custom
        if modal.is_local():
            cuda_source = (SRC_PATH / "forward.cu").read_text()
        else:
            cuda_source = Path("/root/src/forward.cu").read_text()
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
            build_directory=DIST_PATH if modal.is_local() else "/root/dist",
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


# main
def main():
    device = get_device()
    torch.manual_seed(0)

    x = 2 * torch.rand((BS, IN_SEQ_LEN, N_EMBD), device=device) - 1  # [-1, 1]
    x_pil = to_image(x)
    if modal.is_local():
        x_pil.save(ARTIFACTS_PATH / "in.png", format="PNG")
    else:
        x_pil.save("/root/artifacts/in.png", format="PNG")

    # test python
    print("=== profiling PyTorch ===")
    layer = CausalSelfAttention().to(device)
    with torch.no_grad(), torch.autograd.profiler.profile(use_device=device) as prof:
        y_torch = layer(x)
    y_torch_pil = to_image(y_torch)
    if modal.is_local():
        y_torch_pil.save(ARTIFACTS_PATH / "out.png", format="PNG")
    else:
        y_torch_pil.save("/root/artifacts/out.png", format="PNG")
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

    # test cu
    print("=== profiling custom CUDA ===")
    layer = CausalSelfAttention(custom=True).to(device)
    with torch.no_grad(), torch.autograd.profiler.profile(use_device=device) as prof:
        y_cu = layer(x)
    y_cu_pil = to_image(y_cu)
    if modal.is_local():
        y_cu_pil.save(ARTIFACTS_PATH / "out_cu.png", format="PNG")
    else:
        y_cu_pil.save("/root/artifacts/out_cu.png", format="PNG")
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

    print(y_torch.min(), y_torch.max(), y_cu.min(), y_cu.max())

    print(
        "attn values sanity check:",
        torch.allclose(y_torch, y_cu, rtol=0, atol=1e1),
    )


# -----------------------------------------------------------------------------

app = modal.App(f"{APP_NAME}-test")


@app.function(
    image=IMAGE,
    gpu=GPU_CONFIG,
    timeout=TIMEOUT,
    container_idle_timeout=CONTAINER_IDLE_TIMEOUT,
    allow_concurrent_inputs=ALLOW_CONCURRENT_INPUTS,
)
def modal_fn():
    main()


@app.local_entrypoint()
def test():
    modal_fn.remote()


if __name__ == "__main__":
    main()
