from io import BytesIO
from pathlib import Path

import modal
import torch
from PIL import Image

APP_NAME = "flash-attn"
PARENT_PATH = Path(__file__).parent.parent
ARTIFACTS_PATH = PARENT_PATH / "artifacts"
SRC_PATH = PARENT_PATH / "src"
DIST_PATH = PARENT_PATH / "dist"

# layer
N_EMBD: int = 768
BLOCK_SIZE: int = 512
VOCAB_SIZE: int = 50257


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


# image
N_CHANNEL = 3  # RGB
MAX_SIZE = 512, 512


def to_image(t: torch.Tensor) -> Image:
    prod = torch.tensor(t.shape).prod().item()
    n = prod // N_CHANNEL
    factors = []
    for i in range(1, int(n**0.5) + 1):
        if n % i == 0:
            factors.append(i)
            if i != n // i:
                factors.append(n // i)
    factors = sorted(factors)
    mid_index = len(factors) // 2
    show_w, show_h = (
        factors[mid_index - 1],
        factors[mid_index] if len(factors) % 2 == 0 else factors[mid_index],
    )
    t = t.clone().view(show_w, show_h, N_CHANNEL)
    t = (t - t.min()) / (t.max() - t.min())
    t = (t * 255).byte()
    t = t.detach().cpu().numpy()
    t_pil = Image.fromarray(t, mode="RGB")
    t_pil.thumbnail(MAX_SIZE, Image.Resampling.LANCZOS)
    return t_pil


def to_bytes(pil: Image) -> bytes:
    buf = BytesIO()
    pil.save(buf, format="PNG")
    return buf.getvalue()


# Modal
IMAGE = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.0-cudnn-devel-ubuntu24.04", add_python="3.12"
    )
    .apt_install("git")
    .pip_install(  # add Python dependencies
        "pillow==11.0.0",
        "ninja==1.11.1.3",
        "torch==2.5.1",
        "python-fasthtml==0.6.10",
        "sqlite-minutils==4.0.3",  # needed for fasthtml
        "simpleicons==7.21.0",
        "requests==2.32.3",
        "beautifulsoup4==4.13.3",
        "tiktoken==0.9.0",
    )
    .add_local_dir(ARTIFACTS_PATH, "/root/artifacts")
    .add_local_dir(SRC_PATH, "/root/src")
    .add_local_dir(DIST_PATH, "/root/dist")
)

MINUTES = 60  # seconds
TIMEOUT = 5 * MINUTES
CONTAINER_IDLE_TIMEOUT = 15 * MINUTES
ALLOW_CONCURRENT_INPUTS = 1000  # max

GPU_TYPE = "t4"
if modal.is_local():
    GPU_COUNT = torch.cuda.device_count()
else:
    GPU_COUNT = 1
GPU_CONFIG = f"{GPU_TYPE}:{GPU_COUNT}"
