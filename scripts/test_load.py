"""
Verify each model loads + can hook one attention activation.
Run: python scripts/test_load.py qwen3 | llama | gemma

Model registry, loading, and memory cleanup live in src/models.py (single
source). This script only exercises a forward pass + one attention hook to
confirm the layer structure is what the steering pipeline assumes.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root -> import src

try:
    import torch
    from src.models import CONFIGS, load_model, release_memory
except ImportError as exc:
    raise SystemExit(
        "Dependency import failed. Activate the project environment and install:\n"
        '  pip install -U "huggingface_hub[cli]" "transformers>=4.51" accelerate torch\n'
        f"Original error: {exc}"
    ) from None


def main(name: str):
    cfg = CONFIGS[name]
    print(f"\n=== Loading {cfg['hub_id']} ===")

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

        tok, model, cfg = load_model(name)   # asserts layer/hidden dims
        base_model = model.model
        print(f"  Layers: {model.config.num_hidden_layers} (expected {cfg['expected_layers']})")
        print(f"  Hidden: {model.config.hidden_size} (expected {cfg['expected_hidden']})")

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

        layer = cfg["paper_layers"][0]   # main layer (paper layer-14 analogue)
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
