import modal
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline
from transformers import GPT2LMHeadModel
from utils import (
    ALLOW_CONCURRENT_INPUTS,
    APP_NAME,
    ARTIFACTS_PATH,
    BLOCK_SIZE,
    DIST_PATH,
    GPU_CONFIG,
    IMAGE,
    MODEL_HF,
    N_EMBD,
    SCALEDOWN_WINDOW,
    SRC_PATH,
    TIMEOUT,
    VOLUME_CONFIG,
    get_device,
    interactive_plot,
)

# helpers
BS: int = 8
IN_SEQ_LEN: int = 1024  # always 1024 for GPT model checkpoints
N_HEAD: int = 12


# layer
class CausalSelfAttention(nn.Module):
    def __init__(self, custom=False):
        super().__init__()
        assert N_EMBD % N_HEAD == 0
        self.c_attn = nn.Linear(N_EMBD, N_EMBD * 3)
        self.c_proj = nn.Linear(N_EMBD, N_EMBD)

        ## load pretrained weights for linear layers
        ## basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        ## this means that we have to transpose these weights when we import them
        model_hf = GPT2LMHeadModel.from_pretrained(MODEL_HF)
        sd_hf = model_hf.state_dict()

        def load_weights(layer, key_suffix):
            key = next(k for k in sd_hf.keys() if k.endswith(key_suffix))
            assert sd_hf[key].shape[::-1] == layer.weight.shape
            with torch.no_grad():
                layer.weight.copy_(sd_hf[key].t())

        load_weights(self.c_attn, "attn.c_attn.weight")
        load_weights(self.c_proj, "attn.c_proj.weight")

        ## load the CUDA kernel as a python module
        self.custom = custom
        if modal.is_local():
            cuda_source = (SRC_PATH / "forward.cu").read_text()
        else:
            cuda_source = (SRC_PATH / "forward.cu").read_text()
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
            build_directory=DIST_PATH,
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

    # test python
    print("=== profiling PyTorch ===")
    layer = CausalSelfAttention().to(device)
    with torch.no_grad(), torch.autograd.profiler.profile(use_device=device) as prof:
        y_torch = layer(x)
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

    # For a single query, extract the sequence (shape: (T, N_EMBD))
    x_query = x[0]
    y_torch_query = y_torch[0]
    tokens = ["example"] * IN_SEQ_LEN
    interactive_plot(x_query, y_torch_query, tokens, ARTIFACTS_PATH / "torch.html")

    # test cu
    print("=== profiling custom CUDA ===")
    layer = CausalSelfAttention(custom=True).to(device)
    with torch.no_grad(), torch.autograd.profiler.profile(use_device=device) as prof:
        y_cu = layer(x)
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

    y_cu_query = y_cu[0]
    interactive_plot(x_query, y_cu_query, tokens, ARTIFACTS_PATH / "cu.html")

    # compare torch and custom CUDA
    print(
        "attn values sanity check:",
        torch.allclose(y_torch, y_cu, rtol=0, atol=1e1),
    )


# -----------------------------------------------------------------------------

app = modal.App(f"{APP_NAME}-test")


@app.function(
    image=IMAGE,
    gpu=GPU_CONFIG,
    volumes=VOLUME_CONFIG,
    timeout=TIMEOUT,
    scaledown_window=SCALEDOWN_WINDOW,
)
@modal.concurrent(max_inputs=ALLOW_CONCURRENT_INPUTS)
def modal_fn():
    main()


@app.local_entrypoint()
def test():
    modal_fn.remote()


if __name__ == "__main__":
    main()
