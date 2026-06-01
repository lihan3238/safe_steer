"""Compute steering vectors from saved activations (pipeline step 2->3, README §2).

Reads the {harmful,safe}.pt produced by extract_activations.py and turns them
into per-layer steering vectors omega via src/vectors.py:
    vanilla  : omega = mean(safe) - mean(harmful)             (Eq. 1)
    pruned   : keep the high-norm half of (mu_safe - harmful_i), average  (Sec. 3.3)

Pure CPU (just tensor math) -- no model load.

Example:
    python scripts/extract_vectors.py --model qwen3 --dataset CatQA --category adult_content
    python scripts/extract_vectors.py --model qwen3 --dataset CatQA --category adult_content \
        --prune --keep-fraction 0.5 --report

Output:
    vectors/<model>/<dataset>/<category>/vanilla.pt   (always)
    vectors/<model>/<dataset>/<category>/pruned.pt    (when --prune)
Each is a SteeringVectors blob (see src/vectors.py): {layer: (hidden,)} + meta.
"""
import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import torch

from src.vectors import compute_steering_vectors


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--category", required=True)
    p.add_argument("--prune", action="store_true", help="also compute the Sec. 3.3 pruned vector")
    p.add_argument("--keep-fraction", type=float, default=0.5)
    p.add_argument("--act-root", default=str(REPO_ROOT / "activations"))
    p.add_argument("--out-root", default=str(REPO_ROOT / "vectors"))
    p.add_argument("--report", action="store_true",
                   help="print per-layer norms and vanilla-vs-pruned comparison")
    return p.parse_args()


def _load_side(act_dir: Path, role: str):
    path = act_dir / f"{role}.pt"
    if not path.exists():
        raise SystemExit(f"missing {path}\nRun scripts/extract_activations.py first.")
    blob = torch.load(path, map_location="cpu", weights_only=False)
    return blob["acts"], blob["meta"]


def main():
    args = parse_args()
    act_dir = Path(args.act_root) / args.model / args.dataset / args.category
    out_dir = Path(args.out_root) / args.model / args.dataset / args.category

    harmful_acts, h_meta = _load_side(act_dir, "harmful")
    safe_acts, s_meta = _load_side(act_dir, "safe")

    layers = sorted(set(harmful_acts) & set(safe_acts))
    print(f"{args.model}/{args.dataset}/{args.category}  layers={layers}")
    print(f"  harmful n={h_meta['n']}  safe n={s_meta['n']}  safe_source={s_meta.get('safe_source')}")

    base_meta = {
        "model": args.model, "dataset": args.dataset, "category": args.category,
        "safe_source": s_meta.get("safe_source"), "layers": layers,
        "n_safe": s_meta["n"], "n_harmful": h_meta["n"],
        "use_response": h_meta.get("use_response"),
    }

    out_dir.mkdir(parents=True, exist_ok=True)

    vanilla = compute_steering_vectors(safe_acts, harmful_acts, prune=False,
                                       meta={**base_meta, "variant": "vanilla"})
    vanilla.save(out_dir / "vanilla.pt")
    print("  vanilla omega:")
    for l in layers:
        print(f"    layer {l:2d}: ||omega|| = {vanilla.vectors[l].norm():.4f}")
    print(f"  saved {out_dir / 'vanilla.pt'}")

    pruned = None
    if args.prune:
        pruned = compute_steering_vectors(safe_acts, harmful_acts, prune=True,
                                          keep_fraction=args.keep_fraction,
                                          meta={**base_meta, "variant": "pruned",
                                                "keep_fraction": args.keep_fraction})
        pruned.save(out_dir / "pruned.pt")
        print(f"  pruned omega (keep_fraction={args.keep_fraction}):")
        for l in layers:
            print(f"    layer {l:2d}: ||omega|| = {pruned.vectors[l].norm():.4f}")
        print(f"  saved {out_dir / 'pruned.pt'}")

    if args.report:
        print("\n  report:")
        for l in layers:
            v = vanilla.vectors[l]
            line = f"    layer {l:2d}: ||vanilla||={v.norm():.3f}"
            if pruned is not None:
                p = pruned.vectors[l]
                cos = torch.nn.functional.cosine_similarity(v, p, dim=0).item()
                line += f"  ||pruned||={p.norm():.3f}  cos(vanilla,pruned)={cos:+.4f}"
            print(line)


if __name__ == "__main__":
    main()
