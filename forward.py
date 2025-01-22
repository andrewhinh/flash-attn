import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.nn import functional as F

N_EMBD: int = 768  # embedding dimension
N_HEAD: int = 12  # number of heads

BS: int = 1
IN_SEQ_LEN: int = 1000
SHOW_W: int = 800
SHOW_H: int = 960
IN_PNG: str = "in.png"
OUT_PNG: str = "out.png"


class CausalSelfAttention(nn.Module):
    def __init__(self):
        super().__init__()
        assert N_EMBD % N_HEAD == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(N_EMBD, 3 * N_EMBD)
        # output projection
        self.c_proj = nn.Linear(N_EMBD, N_EMBD)
        # regularization
        self.N_HEAD = N_HEAD
        self.N_EMBD = N_EMBD

    def forward(self, x):
        B, T, C = (
            x.size()
        )  # batch size, sequence length, embedding dimensionality (N_EMBD)
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        # nh is "number of heads", hs is "head size", and C (number of channels) = nh * hs
        # e.g. in GPT-2 (124M), N_HEAD=12, hs=64, so nh*hs=C=768 channels in the Transformer
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.N_EMBD, dim=2)
        k = k.view(B, T, self.N_HEAD, C // self.N_HEAD).transpose(
            1, 2
        )  # (B, nh, T, hs)
        q = q.view(B, T, self.N_HEAD, C // self.N_HEAD).transpose(
            1, 2
        )  # (B, nh, T, hs)
        v = v.view(B, T, self.N_HEAD, C // self.N_HEAD).transpose(
            1, 2
        )  # (B, nh, T, hs)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)  # flash attention
        y = (
            y.transpose(1, 2).contiguous().view(B, T, C)
        )  # re-assemble all head outputs side by side
        # output projection
        y = self.c_proj(y)
        return y


if __name__ == "__main__":
    layer = CausalSelfAttention()
    x = torch.randn((BS, IN_SEQ_LEN, N_EMBD))
    x = x.view(BS, SHOW_W, SHOW_H)
    plt.imsave(IN_PNG, x[0].detach().cpu().numpy(), cmap="gray")
    y = layer(x.view(1, IN_SEQ_LEN, N_EMBD))
    y = y.view(BS, SHOW_W, SHOW_H)
    plt.imsave(OUT_PNG, y[0].detach().cpu().numpy(), cmap="gray")
