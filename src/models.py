"""Model registry + loading for the SafeSteer reproduction (single source).

test_load.py and the extraction/eval scripts all import from here, so model ids,
layer maps, chat-template quirks, and load/cleanup helpers live in one place.

LAYERS: the paper uses {14,16,20,25,31} on 32-layer Llama (fractional positions
~{0.44,0.50,0.625,0.78,0.97}). Other depths reuse those positions via
`paper_layers_for(n)`; the first entry is the main layer (paper layer-14 analogue).

CUSTOM MODELS: every model already downloaded under the HF cache is registered
below. For anything else, pass a path or "<dir-name>" and load_model() infers
layers/hidden from the checkpoint's config.json (see resolve + auto-config).
Base (non-chat) models set uses_chat_template=False.
"""
from __future__ import annotations

import gc
import json
import os
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HF_HUB = Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface")) / "hub"

# paper's fractional layer positions on 32-layer Llama, main layer (14) first
_PAPER_FRACTIONS = [14 / 32, 16 / 32, 20 / 32, 25 / 32, 31 / 32]


def paper_layers_for(n_layers: int) -> list[int]:
    """Map the paper's layer positions to a model of `n_layers` depth."""
    return [min(round(f * n_layers), n_layers - 1) for f in _PAPER_FRACTIONS]


def _entry(hub_id, local, layers, hidden, chat, extra=None):
    return {
        "hub_id": hub_id,
        "local_dir": HF_HUB / local,
        "expected_layers": layers,
        "expected_hidden": hidden,
        "paper_layers": paper_layers_for(layers),
        "uses_chat_template": chat,
        "extra_template_kwargs": extra or {},
    }


CONFIGS = {
    # --- Qwen3 (36 layers, hidden 4096) ---
    "qwen3":        _entry("Qwen/Qwen3-8B",      "Qwen3-8B",      36, 4096, True,
                           {"enable_thinking": False}),  # disable <think>
    "qwen3-base":   _entry("Qwen/Qwen3-8B-Base", "Qwen3-8B-Base", 36, 4096, False),
    # --- Qwen3-1.7B (28 layers, hidden 2048): fits 16GB comfortably, same family
    #     as the 8B so steering findings transfer. Smoke/main on a 16GB GPU. ---
    "qwen3-1.7b":      _entry("Qwen/Qwen3-1.7B",      "Qwen3-1.7B",      28, 2048, True,
                              {"enable_thinking": False}),
    "qwen3-1.7b-base": _entry("Qwen/Qwen3-1.7B-Base", "Qwen3-1.7B-Base", 28, 2048, False),
    # --- Llama-3.1 (32 layers, hidden 4096) ---
    "llama":        _entry("meta-llama/Llama-3.1-8B-Instruct", "Llama-3.1-8B-Instruct", 32, 4096, True),
    "llama-base":   _entry("meta-llama/Llama-3.1-8B",          "Llama-3.1-8B",          32, 4096, False),
    # --- Gemma-2 (42 layers, hidden 3584) ---
    "gemma":        _entry("google/gemma-2-9b-it", "gemma-2-9b-it", 42, 3584, True),
    "gemma-base":   _entry("google/gemma-2-9b",    "gemma-2-9b",    42, 3584, False),
}

# canonical layer aliases for the paper's primary model (kept for reference)
CONFIGS["llama"]["paper_layers"] = [14, 16, 20, 25, 31]


def _auto_config(name: str) -> dict:
    """Build a config for an unregistered model given a local dir or path.

    `name` may be an absolute path or a directory name under the HF hub cache.
    Reads config.json to fill layers/hidden; assumes a chat template if the
    tokenizer provides one (resolved at load time). Base detection is heuristic:
    a dir name ending in -base / containing 'base' -> non-chat default.
    """
    cand = Path(name)
    local = cand if cand.is_absolute() else HF_HUB / name
    cfg_path = local / "config.json"
    if not cfg_path.is_file():
        raise KeyError(
            f"unknown model {name!r}; not in CONFIGS {list(CONFIGS)} and no "
            f"config.json at {local}"
        )
    c = json.loads(cfg_path.read_text())
    n = c["num_hidden_layers"]
    # chat only if the dir name signals it (instruct/chat/-it); a base model
    # like "Llama-3.1-8B" has no chat template, so default to non-chat. The
    # generation path also double-checks tokenizer.chat_template at runtime.
    name_l = local.name.lower()
    is_chat = any(k in name_l for k in ("instruct", "chat", "-it"))
    return {
        "hub_id": name,
        "local_dir": local,
        "expected_layers": n,
        "expected_hidden": c["hidden_size"],
        "paper_layers": paper_layers_for(n),
        "uses_chat_template": is_chat,
        "extra_template_kwargs": {},
    }


def get_config(name: str) -> dict:
    """Registered config by name, else infer one from a local dir/path."""
    return CONFIGS[name] if name in CONFIGS else _auto_config(name)


def resolve_source(cfg: dict) -> str:
    """Prefer the local --local-dir download over the Hub id (avoids re-download)."""
    local = cfg["local_dir"]
    if (local / "config.json").is_file():
        print(f"  Source: local dir {local}")
        return str(local)
    print(f"  Source: Hub id {cfg['hub_id']} (local dir not found at {local})")
    return cfg["hub_id"]


def load_model(name: str, device: str = "cuda:0"):
    """Load (tokenizer, model, cfg) for a model name/path, in bf16.

    Asserts the layer/hidden dims match the (registered or inferred) config, so a
    wrong checkpoint surfaces immediately instead of producing bad vectors.
    """
    cfg = get_config(name)
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
