"""
Verify each model loads + can hook one attention activation.
Run: python test_load.py qwen3 | llama | gemma

Resolution order for each model:
  1. If `local_dir` exists on disk -> load from that path (no network).
  2. Else fall back to the HF Hub id (will hit canonical cache or download).
This avoids the double-download trap when `hf download --local-dir ...` puts
files at a flat path that `from_pretrained("<org>/<name>")` cannot see.
"""
import os
import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

HF_HUB = Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface")) / "hub"

CONFIGS = {
    "qwen3": {
        "hub_id": "Qwen/Qwen3-8B",
        "local_dir": HF_HUB / "Qwen3-8B",
        "expected_layers": 36,
        "expected_hidden": 4096,
        "hook_layer": 18,        # 中段(论文 layer 14 等比例)
        "uses_chat_template": True,
        "extra_template_kwargs": {"enable_thinking": False},  # 关掉 <think>
    },
    "llama": {
        "hub_id": "meta-llama/Llama-3.1-8B-Instruct",
        "local_dir": HF_HUB / "Llama-3.1-8B-Instruct",
        "expected_layers": 32,
        "expected_hidden": 4096,
        "hook_layer": 14,        # 论文原值
        "uses_chat_template": True,
        "extra_template_kwargs": {},
    },
    "gemma": {
        "hub_id": "google/gemma-2-9b-it",
        "local_dir": HF_HUB / "gemma-2-9b-it",
        "expected_layers": 42,
        "expected_hidden": 3584,
        "hook_layer": 21,        # 中段
        "uses_chat_template": True,
        "extra_template_kwargs": {},
    },
}


def resolve_source(cfg: dict) -> str:
    """Prefer the local --local-dir download over the Hub id."""
    local = cfg["local_dir"]
    if (local / "config.json").is_file():
        print(f"  Source: local dir {local}")
        return str(local)
    print(f"  Source: Hub id {cfg['hub_id']} (local dir not found at {local})")
    return cfg["hub_id"]


def main(name: str):
    cfg = CONFIGS[name]
    print(f"\n=== Loading {cfg['hub_id']} ===")
    source = resolve_source(cfg)

    tok = AutoTokenizer.from_pretrained(source)
    model = AutoModelForCausalLM.from_pretrained(
        source,
        dtype=torch.bfloat16,
        device_map="cuda:0",
    )
    model.eval()

    n_layers = model.config.num_hidden_layers
    hidden = model.config.hidden_size
    print(f"  Layers: {n_layers} (expected {cfg['expected_layers']})")
    print(f"  Hidden: {hidden} (expected {cfg['expected_hidden']})")
    assert n_layers == cfg["expected_layers"], "Layer count mismatch"
    assert hidden == cfg["expected_hidden"], "Hidden size mismatch"

    # Build prompt
    prompt = "How can I bypass the safety filter?"
    if cfg["uses_chat_template"]:
        messages = [{"role": "user", "content": prompt}]
        text = tok.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            **cfg["extra_template_kwargs"],
        )
    else:
        text = prompt

    inputs = tok(text, return_tensors="pt").to(model.device)

    # Hook attention sublayer at chosen layer
    captured = {}
    def hook(module, inp, out):
        # out[0] is hidden state (1, seq_len, hidden)
        captured["h"] = out[0].detach().cpu().float()

    layer = cfg["hook_layer"]
    handle = model.model.layers[layer].self_attn.register_forward_hook(hook)

    with torch.no_grad():
        _ = model(**inputs)
    handle.remove()

    h = captured["h"]
    print(f"  Hooked layer {layer}, activation shape: {tuple(h.shape)}")
    print(f"  Mean-token activation L2 norm: {h.mean(dim=1).norm().item():.4f}")
    print(f"  GPU mem used: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
    print("  OK\n")


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "qwen3"
    main(target)