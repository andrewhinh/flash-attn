import os
from pathlib import Path

import modal
import requests
import tiktoken
import torch
import torch.nn as nn
from bs4 import BeautifulSoup
from fasthtml import common as fh
from pydantic import BaseModel
from simpleicons.icons import si_github

from forward import CausalSelfAttention
from utils import (
    ALLOW_CONCURRENT_INPUTS,
    APP_NAME,
    ARTIFACTS_PATH,
    BLOCK_SIZE,
    CONTAINER_IDLE_TIMEOUT,
    GPU_CONFIG,
    IMAGE,
    N_EMBD,
    TIMEOUT,
    VOCAB_SIZE,
    get_device,
    interactive_plot,
)

# -----------------------------------------------------------------------------


def get_app():  # noqa: C901
    # setup
    def _not_found(req, exc):
        message = "Page not found!"
        return (
            fh.Title(APP_NAME + " | 404"),
            fh.Div(
                nav(),
                fh.Main(
                    fh.P(
                        message,
                        cls="text-2xl text-red-300",
                    ),
                    cls="flex justify-center items-center grow p-8",
                ),
                toast_container(),
                footer(),
                cls="flex flex-col justify-between min-h-screen text-slate-100 bg-zinc-900 w-full",
            ),
        )

    f_app, _ = fh.fast_app(
        ws_hdr=True,
        exception_handlers={404: _not_found},
        hdrs=[
            fh.HighlightJS(langs=["python", "javascript", "html", "css"]),
            fh.Link(rel="icon", href="/favicon.ico", type="image/x-icon"),
            fh.Script(src="https://cdn.tailwindcss.com"),
            fh.Script(src="https://unpkg.com/htmx-ext-sse@2.2.1/sse.js"),
            fh.Script(src="https://cdn.plot.ly/plotly-2.32.0.min.js"),
        ],
        live=os.getenv("LIVE", False),
        debug=os.getenv("DEBUG", False),
        boost=True,
    )
    fh.setup_toasts(f_app)
    artifacts_path = ARTIFACTS_PATH if modal.is_local() else Path("/root/artifacts")

    # components
    global generations
    generations = []

    class Gen(BaseModel):
        id: int
        query: str
        data: str | None = None

    def gen_view(
        g: Gen,
        session,
    ):
        if g.data:
            return fh.Card(
                fh.P(
                    fh.A(
                        g.query,
                        href=f"https://duckduckgo.com/html/?q={g.query}",
                        target="_blank",
                        cls="text-blue-300 hover:text-blue-100",
                    ),
                    cls="w-2/3 text-lg flex justify-center items-center",
                ),
                fh.Div(
                    id=f"gen-{g.id}-plot",
                ),
                id=f"gen-{g.id}",
                cls="w-full flex flex-col justify-center items-center gap-4 p-4",
            ), fh.Script(
                f"var data = {g.data}; Plotly.newPlot('gen-{g.id}-plot', data);",
            )
        return fh.Card(
            fh.P(
                "Loading ...",
            ),
            hx_get=f"/gen/{g.id}",
            hx_trigger="every 1s",
            hx_swap="outerHTML",
            cls="w-full flex justify-center items-center p-4",
            id=f"gen-{g.id}",
        )

    # layout
    def nav():
        return fh.Nav(
            fh.A(
                f"{APP_NAME}",
                href="/",
                cls="text-xl text-blue-300 hover:text-blue-100 font-mono font-family:Consolas, Monaco, 'Lucida Console', 'Liberation Mono', 'DejaVu Sans Mono', 'Bitstream Vera Sans Mono', 'Courier New'",
            ),
            fh.Svg(
                fh.NotStr(
                    """<style>
                    .spinner_zWVm { animation: spinner_5QiW 1.2s linear infinite, spinner_PnZo 1.2s linear infinite; }
                    .spinner_gfyD { animation: spinner_5QiW 1.2s linear infinite, spinner_4j7o 1.2s linear infinite; animation-delay: .1s; }
                    .spinner_T5JJ { animation: spinner_5QiW 1.2s linear infinite, spinner_fLK4 1.2s linear infinite; animation-delay: .1s; }
                    .spinner_E3Wz { animation: spinner_5QiW 1.2s linear infinite, spinner_tDji 1.2s linear infinite; animation-delay: .2s; }
                    .spinner_g2vs { animation: spinner_5QiW 1.2s linear infinite, spinner_CMiT 1.2s linear infinite; animation-delay: .2s; }
                    .spinner_ctYB { animation: spinner_5QiW 1.2s linear infinite, spinner_cHKR 1.2s linear infinite; animation-delay: .2s; }
                    .spinner_BDNj { animation: spinner_5QiW 1.2s linear infinite, spinner_Re6e 1.2s linear infinite; animation-delay: .3s; }
                    .spinner_rCw3 { animation: spinner_5QiW 1.2s linear infinite, spinner_EJmJ 1.2s linear infinite; animation-delay: .3s; }
                    .spinner_Rszm { animation: spinner_5QiW 1.2s linear infinite, spinner_YJOP 1.2s linear infinite; animation-delay: .4s; }
                    @keyframes spinner_5QiW { 0%, 50% { width: 7.33px; height: 7.33px; } 25% { width: 1.33px; height: 1.33px; } }
                    @keyframes spinner_PnZo { 0%, 50% { x: 1px; y: 1px; } 25% { x: 4px; y: 4px; } }
                    @keyframes spinner_4j7o { 0%, 50% { x: 8.33px; y: 1px; } 25% { x: 11.33px; y: 4px; } }
                    @keyframes spinner_fLK4 { 0%, 50% { x: 1px; y: 8.33px; } 25% { x: 4px; y: 11.33px; } }
                    @keyframes spinner_tDji { 0%, 50% { x: 15.66px; y: 1px; } 25% { x: 18.66px; y: 4px; } }
                    @keyframes spinner_CMiT { 0%, 50% { x: 8.33px; y: 8.33px; } 25% { x: 11.33px; y: 11.33px; } }
                    @keyframes spinner_cHKR { 0%, 50% { x: 1px; y: 15.66px; } 25% { x: 4px; y: 18.66px; } }
                    @keyframes spinner_Re6e { 0%, 50% { x: 15.66px; y: 8.33px; } 25% { x: 18.66px; y: 11.33px; } }
                    @keyframes spinner_EJmJ { 0%, 50% { x: 8.33px; y: 15.66px; } 25% { x: 11.33px; y: 18.66px; } }
                    @keyframes spinner_YJOP { 0%, 50% { x: 15.66px; y: 15.66px; } 25% { x: 18.66px; y: 18.66px; } }
                </style>
                <rect class="spinner_zWVm" x="1" y="1" width="7.33" height="7.33"/>
                <rect class="spinner_gfyD" x="8.33" y="1" width="7.33" height="7.33"/>
                <rect class="spinner_T5JJ" x="1" y="8.33" width="7.33" height="7.33"/>
                <rect class="spinner_E3Wz" x="15.66" y="1" width="7.33" height="7.33"/>
                <rect class="spinner_g2vs" x="8.33" y="8.33" width="7.33" height="7.33"/>
                <rect class="spinner_ctYB" x="1" y="15.66" width="7.33" height="7.33"/>
                <rect class="spinner_BDNj" x="15.66" y="8.33" width="7.33" height="7.33"/>
                <rect class="spinner_rCw3" x="8.33" y="15.66" width="7.33" height="7.33"/>
                <rect class="spinner_Rszm" x="15.66" y="15.66" width="7.33" height="7.33"/>
                """
                ),
                id="spinner",
                cls="htmx-indicator w-8 h-8 absolute top-12 md:top-6 left-1/2 transform -translate-x-1/2 fill-blue-300",
            ),
            fh.A(
                fh.Svg(
                    fh.NotStr(
                        si_github.svg,
                    ),
                    cls="w-8 h-8 text-blue-300 hover:text-blue-100 cursor-pointer",
                ),
                href="https://github.com/andrewhinh/flash-attn",
                target="_blank",
            ),
            cls="flex justify-between items-center p-4 relative",
        )

    def main_content(
        session,
    ):
        return fh.Main(
            fh.Form(
                fh.Group(
                    fh.Input(
                        id="new-query",
                        name="query",
                    ),
                    fh.Button(
                        "Generate",
                        type="submit",
                        cls="text-blue-300 hover:text-blue-100 p-2 border-blue-300 border-2 hover:border-blue-100",
                    ),
                ),
                hx_post="/gen",
                hx_indicator="#spinner",
                hx_target="#gen-list",
                hx_swap="afterbegin",
                id="gen-form",
                cls="w-full md:w-2/3",
            ),
            fh.Div(
                id="gen-list",
                cls="w-full md:w-2/3 flex flex-col gap-2 justify-center items-center",
            ),
            cls="flex flex-col justify-start items-center grow gap-4 p-8",
        )

    def toast_container():
        return fh.Div(id="toast-container", cls="hidden")

    def footer():
        return fh.Footer(
            fh.Div(
                fh.P("Made by"),
                fh.A(
                    "Andrew Hinh",
                    href="https://ajhinh.com/",
                    cls="font-bold text-blue-300 hover:text-blue-100",
                ),
                cls="flex flex-col text-right gap-0.5",
            ),
            cls="flex justify-end items-center p-4 text-lg",
        )

    # threaded fns
    @fh.threaded
    def generate_and_save(g: Gen):
        # given query, return txt from web scrape
        res = requests.get(
            f"https://duckduckgo.com/html/?q={g.query}",
            headers={"User-Agent": "Mozilla/5.0"},
        )
        soup = BeautifulSoup(res.text, "html.parser")
        txt = soup.get_text()

        # get device
        device = get_device()

        # tokenize with tiktoken
        enc = tiktoken.get_encoding("gpt2")
        tokens = enc.encode(txt)
        xgen = torch.tensor([tokens], dtype=torch.long).to(device)  # shape (B, T)
        xgen = xgen[:, :BLOCK_SIZE]
        B, T = xgen.size()

        with torch.no_grad():
            # forward the token and position embeddings
            pos = torch.arange(0, T, dtype=torch.long, device=xgen.device)  # shape (T)
            pos_emb = nn.Embedding(BLOCK_SIZE, N_EMBD).to(device)(
                pos
            )  # position embeddings of shape (B, T, N_EMBD)
            tok_emb = nn.Embedding(VOCAB_SIZE, N_EMBD).to(device)(
                xgen
            )  # token embeddings of shape (B, T, N_EMBD)
            x = tok_emb + pos_emb
            # forward the layernorm and attn
            y = CausalSelfAttention(custom=True).to(device)(
                nn.LayerNorm(N_EMBD).to(device)(x)
            )

        g.data = interactive_plot(
            x[0],  # shape (T, N_EMBD)
            y[0],  # shape (T, N_EMBD)
            [enc.decode([token]) for token in tokens],
        )

        global generations
        generations[g.id] = g

    # routes
    ## for images, CSS, etc.
    @f_app.get("/{fname:path}.{ext:static}")
    def static_files(fname: str, ext: str):
        static_file_path = artifacts_path / f"{fname}.{ext}"
        if static_file_path.exists():
            return fh.FileResponse(static_file_path)

    ## toasts without target
    @f_app.post("/toast")
    def toast(session, message: str, type: str):
        fh.add_toast(session, message, type)
        return toast_container()

    ## pages
    @f_app.get("/")
    def home(
        session,
    ):
        return (
            fh.Title(APP_NAME),
            fh.Div(
                nav(),
                main_content(session),
                toast_container(),
                footer(),
                cls="flex flex-col justify-between min-h-screen text-slate-100 bg-zinc-900 w-full",
            ),
        )

    ## generation routes
    @f_app.post("/gen")
    def generate(
        session,
        query: str,
    ):
        # validation
        if not query:
            fh.add_toast(session, "No query provided", "error")
            return None

        # Clear input
        clear_img_input = fh.Input(
            id="new-query",
            name="query",
            hx_swap_oob="true",
        )

        # Generate
        global generations
        g = Gen(id=len(generations), query=query)
        generations.append(g)
        generate_and_save(g)

        return (
            gen_view(g, session),
            clear_img_input,
        )

    @f_app.get("/gen/{id}")
    def get_gen(
        session,
        id: int,
    ):
        g = generations[id]
        return gen_view(g, session)

    return f_app


f_app = get_app()

# -------------------------------------------------------------

app = modal.App(f"{APP_NAME}-frontend")


@app.function(
    image=IMAGE,
    gpu=GPU_CONFIG,
    timeout=TIMEOUT,
    container_idle_timeout=CONTAINER_IDLE_TIMEOUT,
    allow_concurrent_inputs=ALLOW_CONCURRENT_INPUTS,
)
@modal.asgi_app()
def modal_get():
    return f_app


if __name__ == "__main__":
    fh.serve(app="f_app")
