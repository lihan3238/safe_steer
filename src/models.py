"""Model registry + loading for the SafeSteer reproduction (single source).

Both scripts/test_load.py and the extraction/eval scripts import from here, so
model ids, layer maps, chat-template quirks, and the load/cleanup helpers live
in exactly one place.

LAYERS: the paper uses {14,16,20,25,31} on 32-layer Llama (fractional positions
~{0.44,0.50,0.625,0.78,0.97}). Other depths use the proportional mapping
(README §1). The first listed layer is the main one (paper layer 14 analogue).
"""
from __future__ import annotations

import gc
import os
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HF_HUB = Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface")) / "hub"

CONFIGS = {
    "qwen3": {
        "hub_id": "Qwen/Qwen3-8B",
        "local_dir": HF_HUB / "Qwen3-8B",
        "expected_layers": 36,
        "expected_hidden": 4096,
        "paper_layers": [18, 16, 23, 28, 35],   # main first (paper layer-14 analogue)
        "uses_chat_template": True,
        "extra_template_kwargs": {"enable_thinking": False},  # disable <think>
    },
    "llama": {
        "hub_id": "meta-llama/Llama-3.1-8B-Instruct",
        "local_dir": HF_HUB / "Llama-3.1-8B-Instruct",
        "expected_layers": 32,
        "expected_hidden": 4096,
        "paper_layers": [14, 16, 20, 25, 31],   # paper's exact set
        "uses_chat_template": True,
        "extra_template_kwargs": {},
    },
    "gemma": {
        "hub_id": "google/gemma-2-9b-it",
        "local_dir": HF_HUB / "gemma-2-9b-it",
        "expected_layers": 42,
        "expected_hidden": 3584,
        "paper_layers": [21, 18, 26, 33, 41],   # main first
        "uses_chat_template": True,
        "extra_template_kwargs": {},
    },
}


def resolve_source(cfg: dict) -> str:
    """Prefer the local --local-dir download over the Hub id (avoids re-download)."""
    local = cfg["local_dir"]
    if (local / "config.json").is_file():
        print(f"  Source: local dir {local}")
        return str(local)
    print(f"  Source: Hub id {cfg['hub_id']} (local dir not found at {local})")
    return cfg["hub_id"]


def load_model(name: str, device: str = "cuda:0"):
    """Load (tokenizer, model, cfg) for a registered model name, in bf16.

    Asserts the layer/hidden dims match the registry, so a wrong checkpoint
    surfaces immediately instead of producing bad vectors.
    """
    if name not in CONFIGS:
        raise KeyError(f"unknown model {name!r}; known: {list(CONFIGS)}")
    cfg = CONFIGS[name]
    source = resolve_source(cfg)

    tok = AutoTokenizer.from_pretrained(source)
    model = AutoModelForCausalLM.from_pretrained(
        source,
        dtype=torch.bfloat16,          # transformers 5.x spelling (torch_dtype also works)
        device_map=device,
        low_cpu_mem_usage=True,
    )
    model.eval()

    n_layers = model.config.num_hidden_layers
    hidden = model.config.hidden_size
    assert n_layers == cfg["expected_layers"], (
        f"{name}: layers {n_layers} != expected {cfg['expected_layers']}"
    )
    assert hidden == cfg["expected_hidden"], (
        f"{name}: hidden {hidden} != expected {cfg['expected_hidden']}"
    )
    return tok, model, cfg


def release_memory() -> tuple[float, float]:
    """Best-effort release of Python, CPU allocator, and CUDA cache memory."""
    gc.collect()

    if sys.platform.startswith("linux"):
        try:
            import ctypes

            ctypes.CDLL("libc.so.6").malloc_trim(0)
        except (AttributeError, OSError):
            pass

    if not torch.cuda.is_available() or not torch.cuda.is_initialized():
        return 0.0, 0.0

    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    return (
        torch.cuda.memory_allocated() / 1e9,
        torch.cuda.memory_reserved() / 1e9,
    )
