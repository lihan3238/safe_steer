"""Quality axis (paper §4.4 / Table 3): score eval outputs with a reward model.

Reads an eval_steering.py output JSON (naive + per-multiplier steered text),
scores every response with QRM-Llama3.1-8B-v2 on the 5 HelpSteer attributes
(helpfulness/correctness/coherence/complexity/verbosity), and writes a
quality summary. This is the quality counterpart to eval_steering.py's safety
axis (%UR); together they reproduce the paper's safety-vs-quality tradeoff.

Why QRM (not the paper's Nemotron-340B): 340B is infeasible here. QRM-v2 outputs
the same 5 HelpSteer attributes (out.rewards is shape [batch, 5]) and, unlike
ArmoRM, can be made to load on transformers 5.x with a one-line patch to its
modeling_custom.py (guard the removed LLAMA_INPUTS_DOCSTRING import). Documented
deviation: different reward model than the paper, same 5 attributes + intent.
The paper deliberately avoids a generic LLM judge for quality (bias toward
"gives a solution" over refusals), so we use a reward model, per that rationale.

This runs on a GPU server (QRM-8B needs ~16GB). Pin the card with
CUDA_VISIBLE_DEVICES; never touch a card another user occupies.

Usage:
    CUDA_VISIBLE_DEVICES=7 python score_quality.py \
        --eval-json eval/qwen3-1.7b-base/BeaverTails/hate_speech_offensive/pruned_sweep_m0.5_1.0_2.0.json \
        --qrm /home/dell/.workplace/QRM-Llama3.1-8B-v2

Output: writes <eval-json stem>.quality.json next to the input, and prints a
naive-vs-steered table of the 5 attributes per multiplier.
"""
import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ATTRS = ["helpfulness", "correctness", "coherence", "complexity", "verbosity"]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--eval-json", required=True, help="eval_steering.py output JSON")
    p.add_argument("--qrm", default="/home/dell/.workplace/QRM-Llama3.1-8B-v2",
                   help="path to the (patched) QRM-Llama3.1-8B-v2 local dir")
    p.add_argument("--out", default=None, help="output path (default: <eval-json>.quality.json)")
    p.add_argument("--max-chars", type=int, default=4000,
                   help="truncate long responses before scoring (keeps QRM context bounded)")
    return p.parse_args()


def load_qrm(path):
    model = AutoModelForSequenceClassification.from_pretrained(
        path, device_map="cuda", trust_remote_code=True, dtype=torch.bfloat16)
    model.eval()
    tok = AutoTokenizer.from_pretrained(path, use_fast=True)
    return model, tok


@torch.no_grad()
def score(model, tok, prompt, response, max_chars):
    """Return the 5 HelpSteer attribute scores for one (prompt, response)."""
    response = (response or "")[:max_chars]
    if not response.strip():
        return [0.0] * 5
    msgs = [{"role": "user", "content": prompt},
            {"role": "assistant", "content": response}]
    enc = tok.apply_chat_template(msgs, return_tensors="pt", return_dict=True).to("cuda")
    out = model(input_ids=enc["input_ids"], attention_mask=enc.get("attention_mask"))
    return out.rewards.cpu().float()[0][:5].tolist()


def mean_attrs(score_rows):
    """Average each of the 5 attributes over a list of [5]-score lists."""
    if not score_rows:
        return [0.0] * 5
    t = torch.tensor(score_rows)
    return t.mean(dim=0).tolist()


def main():
    args = parse_args()
    eval_path = Path(args.eval_json)
    data = json.loads(eval_path.read_text(encoding="utf-8"))
    summ = data["summary"]
    print(f"{summ['model']}/{summ['dataset']}/{summ['category']} "
          f"variant={summ['variant']} layers={summ['layers']}")

    model, tok = load_qrm(args.qrm)
    print(f"QRM loaded from {args.qrm}\n")

    # naive side (steering-independent) — score once
    naive_scores = [score(model, tok, r["prompt"], r["naive"], args.max_chars)
                    for r in data["naive_rows"]]
    naive_mean = mean_attrs(naive_scores)

    def fmt(v):
        return "  ".join(f"{a[:4]}={s:.3f}" for a, s in zip(ATTRS, v))

    print(f"  naive        : {fmt(naive_mean)}")

    # each multiplier's steered side
    sweep_quality = []
    for s in data["sweep"]:
        m = s["multiplier"]
        steered_scores = [score(model, tok, r["prompt"], r["steered"], args.max_chars)
                          for r in s["rows"]]
        steered_mean = mean_attrs(steered_scores)
        delta = [st - na for st, na in zip(steered_mean, naive_mean)]
        print(f"  steered m={m:<4}: {fmt(steered_mean)}")
        print(f"    Δ vs naive : {fmt(delta)}")
        sweep_quality.append({
            "multiplier": m,
            "quality_mean": dict(zip(ATTRS, steered_mean)),
            "quality_delta_vs_naive": dict(zip(ATTRS, delta)),
            "per_row": steered_scores,
        })

    out = {
        "summary": {k: summ[k] for k in ("model", "dataset", "category", "variant",
                                         "layers", "n") if k in summ},
        "reward_model": args.qrm,
        "attributes": ATTRS,
        "naive_quality_mean": dict(zip(ATTRS, naive_mean)),
        "naive_per_row": naive_scores,
        "sweep_quality": sweep_quality,
    }
    out_path = Path(args.out) if args.out else eval_path.with_suffix(".quality.json")
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n  saved {out_path}")


if __name__ == "__main__":
    main()
