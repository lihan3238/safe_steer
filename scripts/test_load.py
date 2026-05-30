"""
Verify each model loads + can hook one attention activation.
Run: python scripts/test_load.py qwen3 | llama | gemma

Resolution order for each model:
  1. If `local_dir` exists on disk -> load from that path (no network).
  2. Else fall back to the HF Hub id (will hit canonical cache or download).
This avoids the double-download trap when `hf download --local-dir ...` puts
files at a flat path that `from_pretrained("<org>/<name>")` cannot see.
"""
import gc
import os
import sys
from pathlib import Path

try:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
except ImportError as exc:
    raise SystemExit(
        "Dependency import failed. Activate the project environment and install:\n"
        '  pip install -U "huggingface_hub[cli]" "transformers>=4.51" accelerate torch\n'
        f"Original error: {exc}"
    ) from None

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


def main(name: str):
    cfg = CONFIGS[name]
    print(f"\n=== Loading {cfg['hub_id']} ===")
    source = resolve_source(cfg)

    tok = None
    model = None
    base_model = None
    attn_module = None
    inputs = None
    output = None
    h = None
    handle = None
    captured = {}

    try:
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        tok = AutoTokenizer.from_pretrained(source)
        model = AutoModelForCausalLM.from_pretrained(
            source,
            torch_dtype=torch.bfloat16,
            device_map="cuda:0",
            low_cpu_mem_usage=True,
        )
        model.eval()
        base_model = model.model

        n_layers = model.config.num_hidden_layers
        hidden = model.config.hidden_size
        print(f"  Layers: {n_layers} (expected {cfg['expected_layers']})")
        print(f"  Hidden: {hidden} (expected {cfg['expected_hidden']})")
        assert n_layers == cfg["expected_layers"], "Layer count mismatch"
        assert hidden == cfg["expected_hidden"], "Hidden size mismatch"

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

        def hook(module, inp, out):
            captured["h"] = out[0].detach().cpu().float()

        layer = cfg["hook_layer"]
        attn_module = base_model.layers[layer].self_attn
        handle = attn_module.register_forward_hook(hook)

        with torch.inference_mode():
            output = base_model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs.get("attention_mask"),
                use_cache=False,
            )
        output = None

        h = captured.pop("h", None)
        if h is None:
            raise RuntimeError(f"Forward hook did not capture layer {layer}")

        peak_gb = (
            torch.cuda.max_memory_allocated() / 1e9
            if torch.cuda.is_available()
            else 0.0
        )
        print(f"  Hooked layer {layer}, activation shape: {tuple(h.shape)}")
        print(f"  Mean-token activation L2 norm: {h.mean(dim=1).norm().item():.4f}")
        print(f"  GPU peak mem used: {peak_gb:.2f} GB")
        print("  OK")
    finally:
        if handle is not None:
            handle.remove()

        captured.clear()
        h = None
        output = None
        inputs = None
        attn_module = None
        base_model = None
        model = None
        tok = None

        allocated_gb, reserved_gb = release_memory()
        if torch.cuda.is_available():
            print(
                "  GPU mem after cleanup: "
                f"allocated {allocated_gb:.2f} GB, reserved {reserved_gb:.2f} GB"
            )
        print()


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "qwen3"
    main(target)
