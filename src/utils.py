from pathlib import Path

import modal
import numpy as np
import plotly.graph_objects as go
import torch
from sklearn.decomposition import PCA

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


def interactive_plot(x, y, tokens, filename=None):
    """
    Creates an interactive 3D Plotly figure showing how a user query's token embeddings are transformed by the attention layer.
    - Uses PCA with 3 components to reduce the embeddings.
    - Blue markers denote the original embeddings; red markers show the updated embeddings.
    - Hover text reveals each token's index and change magnitude.
    - The top 5 tokens (by change magnitude) are highlighted with larger markers, and green arrows (lines + cone arrowheads) indicate their transformation.
    """
    T = x.shape[0]
    # Compute the per-token change magnitude.
    diff = y - x
    diff_norm = diff.norm(dim=-1)  # shape (T,)
    diff_norm_np = diff_norm.detach().cpu().numpy()

    # Perform PCA with 3 components on combined data.
    combined = torch.cat([x, y], dim=0)  # shape (2T, N_EMBD)
    combined_np = combined.detach().cpu().numpy()
    pca = PCA(n_components=3)
    combined_3d = pca.fit_transform(combined_np)
    x_3d = combined_3d[:T]
    y_3d = combined_3d[T:]

    # Identify top 5 tokens by change magnitude.
    top5_indices = np.argsort(diff_norm_np)[-5:][::-1]

    # Create hover text for each token.
    hover_text = [
        f"Token {i}: {tokens[i]}<br>Change: {diff_norm_np[i]:.3f}" for i in range(T)
    ]

    # Create 3D scatter traces for before and after.
    trace_before = go.Scatter3d(
        x=x_3d[:, 0],
        y=x_3d[:, 1],
        z=x_3d[:, 2],
        mode="markers",
        marker=dict(color="cyan", size=4),
        name="Before Attention",
        text=hover_text,
        hoverinfo="text",
    )

    trace_after = go.Scatter3d(
        x=y_3d[:, 0],
        y=y_3d[:, 1],
        z=y_3d[:, 2],
        mode="markers",
        marker=dict(color="magenta", size=4),
        name="After Attention",
        text=hover_text,
        hoverinfo="text",
    )

    # Extra traces to highlight the top 5 tokens with larger markers.
    trace_top_before = go.Scatter3d(
        x=x_3d[top5_indices, 0],
        y=x_3d[top5_indices, 1],
        z=x_3d[top5_indices, 2],
        mode="markers",
        marker=dict(color="gold", size=10, symbol="circle-open"),
        name="Top 5 (Before)",
        hoverinfo="skip",
    )

    trace_top_after = go.Scatter3d(
        x=y_3d[top5_indices, 0],
        y=y_3d[top5_indices, 1],
        z=y_3d[top5_indices, 2],
        mode="markers",
        marker=dict(color="gold", size=10, symbol="circle-open"),
        name="Top 5 (After)",
        hoverinfo="skip",
    )

    # Build the figure.
    fig = go.Figure(data=[trace_before, trace_after, trace_top_before, trace_top_after])

    # For each top token, add a line (arrow shaft) from before to after.
    for i in top5_indices:
        arrow_line = go.Scatter3d(
            x=[x_3d[i, 0], y_3d[i, 0]],
            y=[x_3d[i, 1], y_3d[i, 1]],
            z=[x_3d[i, 2], y_3d[i, 2]],
            mode="lines",
            line=dict(color="yellow", width=4),
            showlegend=False,
            hoverinfo="skip",
        )
        fig.add_trace(arrow_line)

    # Update layout with descriptive titles and axis labels.
    fig.update_layout(
        template="plotly_dark",
        scene=dict(
            xaxis=dict(
                backgroundcolor="black",
                gridcolor="gray",
                showbackground=True,
                zerolinecolor="gray",
            ),
            yaxis=dict(
                backgroundcolor="black",
                gridcolor="gray",
                showbackground=True,
                zerolinecolor="gray",
            ),
            zaxis=dict(
                backgroundcolor="black",
                gridcolor="gray",
                showbackground=True,
                zerolinecolor="gray",
            ),
        ),
        paper_bgcolor="black",  # outer paper
        plot_bgcolor="black",  # main plot background
        title=("Effect of Attention on Token Embeddings"),
        legend_title="Token States",
        hovermode="closest",
    )

    if filename:
        fig.write_html(filename)
        print(f"Interactive 3D plot saved as {filename}")

    return fig.to_json()


# Modal
IMAGE = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.0-cudnn-devel-ubuntu24.04", add_python="3.12"
    )
    .apt_install("git")
    .pip_install(  # add Python dependencies
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
