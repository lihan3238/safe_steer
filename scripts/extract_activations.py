"""Extract attention activations for one (model, dataset, category) and save them.

Pipeline step 1->2 (README §2): load a CategorySplit (src/data.py), run the
model, mean-pool attention activations per layer (src/hooks.py), and save the
harmful-side and safe-side activation matrices for the next stage
(extract_vectors.py) to turn into steering vectors.

Example (Qwen3-8B, paper 5 layers, BeaverTails child_abuse, Alpaca generic-safe):
    python scripts/extract_activations.py --model qwen3 \
        --dataset BeaverTails --category child_abuse --safe-source alpaca

    python scripts/extract_activations.py --model qwen3 \
        --dataset CatQA --category adult_content          # defaults: paper layers, alpaca

Output:
    activations/<model>/<dataset>/<category>/harmful.pt
    activations/<model>/<dataset>/<category>/safe.pt
Each .pt holds {"acts": {layer: (N, hidden)}, "meta": {...}} (CPU float32).

--dry-run validates data/paths/layers WITHOUT loading the model (zero GPU cost).
"""
import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import torch

from src.data import load_category_split
from src.models import CONFIGS, get_config


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, metavar="NAME_OR_PATH",
                   help="registered model (%s) or a local dir/path" % "|".join(CONFIGS))
    p.add_argument("--dataset", required=True, choices=["CatQA", "BeaverTails"])
    p.add_argument("--category", required=True, help="category slug (see manifest)")
    p.add_argument("--safe-source", default="alpaca", choices=["alpaca", "beavertails"])
    p.add_argument("--layers", type=int, nargs="+", default=None,
                   help="layers to hook (default: the model's paper_layers)")
    p.add_argument("--use-response", action="store_true",
                   help="include response in the activation input (default prompt-only)")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--limit", type=int, default=None,
                   help="cap each side to first N examples (smoke runs)")
    p.add_argument("--out-root", default=str(REPO_ROOT / "activations"))
    p.add_argument("--dry-run", action="store_true",
                   help="validate data/paths/layers without loading the model")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = get_config(args.model)
    layers = args.layers if args.layers is not None else cfg["paper_layers"]

    # validate layers against the model's depth before doing anything expensive
    bad = [l for l in layers if not 0 <= l < cfg["expected_layers"]]
    if bad:
        raise SystemExit(f"layers {bad} out of range for {args.model} "
                         f"(0..{cfg['expected_layers'] - 1})")

    split = load_category_split(args.dataset, args.category, safe_source=args.safe_source)
    harmful = split.harmful[: args.limit] if args.limit else split.harmful
    safe = split.generic_safe[: args.limit] if args.limit else split.generic_safe

    print(f"model={args.model}  layers={layers}  use_response={args.use_response}")
    print(f"  {split.summary()}")
    print(f"  using harmful={len(harmful)}  safe={len(safe)}"
          + (f"  (limit={args.limit})" if args.limit else ""))

    out_dir = Path(args.out_root) / args.model / args.dataset / args.category
    print(f"  out: {out_dir}")

    if args.dry_run:
        print("  dry-run OK (model not loaded)")
        return

    from src.hooks import extract_activations
    from src.models import load_model, release_memory

    tok = model = None
    try:
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        tok, model, cfg = load_model(args.model)

        def run(examples, label):
            texts = [e.text for e in examples]
            resp = [e.response for e in examples] if args.use_response else None
            t0 = time.time()

            def progress(done, total):
                elapsed = time.time() - t0
                rate = done / elapsed if elapsed > 0 else 0.0
                eta = (total - done) / rate if rate > 0 else 0.0
                print(f"\r    {label}: {done}/{total}  "
                      f"{elapsed:5.1f}s  eta {eta:4.0f}s", end="", flush=True)

            acts = extract_activations(
                model, tok, texts, layers,
                responses=resp, use_response=args.use_response,
                batch_size=args.batch_size, progress=progress,
            )
            print(f"\r    {label}: {len(texts)}/{len(texts)} done in "
                  f"{time.time() - t0:.1f}s" + " " * 16)
            return acts

        harmful_acts = run(harmful, "harmful")
        safe_acts = run(safe, "safe")

        meta_common = {
            "model": args.model, "hub_id": cfg["hub_id"], "dataset": args.dataset,
            "category": args.category, "safe_source": args.safe_source,
            "layers": layers, "use_response": args.use_response,
        }
        out_dir.mkdir(parents=True, exist_ok=True)
        torch.save({"acts": harmful_acts,
                    "meta": {**meta_common, "role": "harmful", "n": len(harmful)}},
                   out_dir / "harmful.pt")
        torch.save({"acts": safe_acts,
                    "meta": {**meta_common, "role": "safe", "n": len(safe)}},
                   out_dir / "safe.pt")

        ref = layers[0]
        print(f"  harmful acts: layer {ref} -> {tuple(harmful_acts[ref].shape)}, "
              f"mean L2 {harmful_acts[ref].norm(dim=1).mean():.3f}")
        print(f"  safe    acts: layer {ref} -> {tuple(safe_acts[ref].shape)}, "
              f"mean L2 {safe_acts[ref].norm(dim=1).mean():.3f}")
        if torch.cuda.is_available():
            print(f"  GPU peak: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
        print("  saved harmful.pt + safe.pt")
    finally:
        tok = model = None
        release_memory()


if __name__ == "__main__":
    main()
