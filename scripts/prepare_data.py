"""Build minimal, deterministic dataset subsets for the SafeSteer reproduction.

WHAT THIS DOES (read-only on the HF cache, writes ONLY under ./data):
  - Reads the three Phase-1 datasets straight from the local HF cache.
  - Extracts the paper's representative categories (Table 1/2) PER DATASET
    (CatQA and BeaverTails are NOT mapped onto a shared category set).
  - Normalises every record to {id, text, response, category, role, source}.
  - Writes data/processed/<dataset>/*.jsonl and a data/manifest.json
    (metadata only -- no harmful text -- so the manifest is safe to commit).

WHY (maps to the paper):
  - role="harmful"      -> D^c_unsafe  : the per-category harmful side (Eq. 1).
  - role="generic_safe" -> D^generic_safe : the unpaired safe pool used by
    SafeSteer when no paired safe twin exists (paper Sec. 3.3 fallback, and
    BeaverTails' own `safe` records).
  - `text` is the prompt/question; `response` is the model answer when the
    source has one (BeaverTails, Alpaca) else null. Sec. 3.1 notes the
    activation input may be a prompt alone OR a {prompt, response} pair, so we
    keep both and let the extractor choose.

DETERMINISM (constitution Principle III):
  - Fixed SEED; sampling uses random.Random(SEED) over a stable read order.
  - data/ is gitignored; rebuild anytime with one command:
        conda run -n safesteer313 python scripts/prepare_data.py

DEPENDENCIES:
  - stdlib only for CatQA (json lines) and BeaverTails (gzip jsonl).
  - pyarrow ONLY for Alpaca (parquet). Install with: pip install pyarrow
    (we deliberately do NOT pull in the full `datasets` library).
"""
from __future__ import annotations

import argparse
import glob
import gzip
import json
import os
import random
from pathlib import Path

# --- locations -------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_ROOT = PROJECT_ROOT / "data" / "processed"
MANIFEST_PATH = PROJECT_ROOT / "data" / "manifest.json"

# HF datasets cache (read-only). Override with SAFESTEER_HF_DATASETS if needed.
HF_DATASETS = Path(
    os.environ.get("SAFESTEER_HF_DATASETS", "~/.cache/huggingface/datasets")
).expanduser()

CATQA_FILE = HF_DATASETS / "CategoricalHarmfulQA" / "data" / "catqa_english.json"
BEAVERTAILS_FILE = HF_DATASETS / "BeaverTails" / "round0" / "30k" / "train.jsonl.gz"
ALPACA_GLOB = str(HF_DATASETS / "alpaca" / "data" / "train-*.parquet")

# --- knobs (subset sizes); paper uses 1500/cat for extraction, 150-200 test --
SEED = 0
N_HARMFUL_PER_CAT = 200   # CatQA only has 50/cat -> capped to what's available
N_BEAVERTAILS_SAFE = 200
N_ALPACA = 500

# --- the paper's representative categories, PER DATASET --------------------
# CatQA (Table 1 / Table 2 right): Adult Content / Hate-Harass-Violence / Physical Harm
CATQA_CATEGORIES = {
    "adult_content": "Adult Content",
    "hate_harass_violence": "Hate/Harass/Violence",
    "physical_harm": "Physical Harm",
}
# BeaverTails (Table 2 left): Child Abuse / Terrorism-Org-Crime / Hate-Speech-Offensive
BEAVERTAILS_CATEGORIES = {
    "child_abuse": "child_abuse",
    "terrorism_organized_crime": "terrorism,organized_crime",
    "hate_speech_offensive": "hate_speech,offensive_language",
}

CATQA_SOURCE = "declare-lab/CategoricalHarmfulQA"
BEAVERTAILS_SOURCE = "PKU-Alignment/BeaverTails"
ALPACA_SOURCE = "tatsu-lab/alpaca"


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _sample(pool: list[dict], n: int, rng: random.Random) -> list[dict]:
    """Deterministic subsample (or the whole pool if it's smaller than n)."""
    if len(pool) <= n:
        return pool
    return rng.sample(pool, n)


def _dedup_by_text(rows: list[dict]) -> list[dict]:
    """Drop duplicate prompts so repeated samples don't bias the mean vector."""
    seen, out = set(), []
    for r in rows:
        if r["text"] not in seen:
            seen.add(r["text"])
            out.append(r)
    return out


# --- per-dataset extractors ------------------------------------------------
def prepare_catqa(rng: random.Random) -> list[dict]:
    """CatQA: harmful-only. One JSON object per line: {Category, Subcategory, Question}."""
    if not CATQA_FILE.exists():
        raise FileNotFoundError(f"CatQA not found: {CATQA_FILE}")
    rows = [json.loads(line) for line in CATQA_FILE.read_text(encoding="utf-8").splitlines() if line.strip()]

    artifacts = []
    for slug, native in CATQA_CATEGORIES.items():
        pool = [
            {
                "text": r["Question"],
                "response": None,
                "category": slug,
                "role": "harmful",
                "source": CATQA_SOURCE,
            }
            for r in rows
            if r.get("Category") == native
        ]
        pool = _dedup_by_text(pool)
        picked = _sample(pool, N_HARMFUL_PER_CAT, rng)
        for i, r in enumerate(picked):
            r["id"] = f"catqa/{slug}/{i:04d}"
        out = OUT_ROOT / "catqa" / f"{slug}.harmful.jsonl"
        _write_jsonl(out, picked)
        artifacts.append({
            "path": str(out.relative_to(PROJECT_ROOT)), "dataset": "CatQA",
            "source": CATQA_SOURCE, "source_file": str(CATQA_FILE),
            "category_slug": slug, "category_native": native, "role": "harmful",
            "n_requested": N_HARMFUL_PER_CAT, "n_available": len(pool), "n_actual": len(picked),
        })
        print(f"  CatQA  {slug:28s} harmful: {len(picked):4d} (avail {len(pool)})")
    return artifacts


def _read_beavertails() -> list[dict]:
    if not BEAVERTAILS_FILE.exists():
        raise FileNotFoundError(f"BeaverTails not found: {BEAVERTAILS_FILE}")
    rows = []
    with gzip.open(BEAVERTAILS_FILE, "rt", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def prepare_beavertails(rng: random.Random) -> list[dict]:
    """BeaverTails: {prompt, response, is_safe, category{14 bool}}.

    Harmful = is_safe False AND the category flag is True (multi-label, so a
    record can land in more than one category). generic_safe = is_safe True
    (BeaverTails' own safe pool, used as the unpaired safe side).
    """
    rows = _read_beavertails()
    artifacts = []

    for slug, native in BEAVERTAILS_CATEGORIES.items():
        pool = [
            {
                "text": r["prompt"],
                "response": r.get("response"),
                "category": slug,
                "role": "harmful",
                "source": BEAVERTAILS_SOURCE,
            }
            for r in rows
            if (not r.get("is_safe")) and r.get("category", {}).get(native)
        ]
        pool = _dedup_by_text(pool)
        picked = _sample(pool, N_HARMFUL_PER_CAT, rng)
        for i, r in enumerate(picked):
            r["id"] = f"beavertails/{slug}/{i:04d}"
        out = OUT_ROOT / "beavertails" / f"{slug}.harmful.jsonl"
        _write_jsonl(out, picked)
        artifacts.append({
            "path": str(out.relative_to(PROJECT_ROOT)), "dataset": "BeaverTails",
            "source": BEAVERTAILS_SOURCE, "source_file": str(BEAVERTAILS_FILE),
            "category_slug": slug, "category_native": native, "role": "harmful",
            "n_requested": N_HARMFUL_PER_CAT, "n_available": len(pool), "n_actual": len(picked),
        })
        print(f"  Beaver {slug:28s} harmful: {len(picked):4d} (avail {len(pool)})")

    # generic safe pool (not category-specific)
    safe_pool = _dedup_by_text([
        {
            "text": r["prompt"], "response": r.get("response"),
            "category": "generic", "role": "generic_safe", "source": BEAVERTAILS_SOURCE,
        }
        for r in rows if r.get("is_safe")
    ])
    picked = _sample(safe_pool, N_BEAVERTAILS_SAFE, rng)
    for i, r in enumerate(picked):
        r["id"] = f"beavertails/generic_safe/{i:04d}"
    out = OUT_ROOT / "beavertails" / "generic_safe.jsonl"
    _write_jsonl(out, picked)
    artifacts.append({
        "path": str(out.relative_to(PROJECT_ROOT)), "dataset": "BeaverTails",
        "source": BEAVERTAILS_SOURCE, "source_file": str(BEAVERTAILS_FILE),
        "category_slug": "generic", "category_native": "safe", "role": "generic_safe",
        "n_requested": N_BEAVERTAILS_SAFE, "n_available": len(safe_pool), "n_actual": len(picked),
    })
    print(f"  Beaver {'(safe pool)':28s} generic_safe: {len(picked):4d} (avail {len(safe_pool)})")
    return artifacts


def prepare_alpaca(rng: random.Random) -> list[dict]:
    """Alpaca: generic harmless pool. Parquet with instruction/input/output."""
    try:
        import pyarrow.parquet as pq
    except ImportError as e:
        raise SystemExit(
            "Alpaca is parquet -> needs pyarrow. Run: pip install pyarrow"
        ) from e

    matches = sorted(glob.glob(ALPACA_GLOB))
    if not matches:
        raise FileNotFoundError(f"Alpaca parquet not found: {ALPACA_GLOB}")
    table = pq.read_table(matches[0], columns=["instruction", "input", "output"])
    instr = table.column("instruction").to_pylist()
    inp = table.column("input").to_pylist()
    out_col = table.column("output").to_pylist()

    pool = []
    for ins, extra, ans in zip(instr, inp, out_col):
        text = ins if not (extra and extra.strip()) else f"{ins}\n\n{extra}"
        pool.append({
            "text": text, "response": ans, "category": "generic",
            "role": "generic_safe", "source": ALPACA_SOURCE,
        })
    pool = _dedup_by_text(pool)
    picked = _sample(pool, N_ALPACA, rng)
    for i, r in enumerate(picked):
        r["id"] = f"alpaca/generic_safe/{i:04d}"
    out = OUT_ROOT / "alpaca" / "generic_safe.jsonl"
    _write_jsonl(out, picked)
    print(f"  Alpaca {'(instructions)':28s} generic_safe: {len(picked):4d} (avail {len(pool)})")
    return [{
        "path": str(out.relative_to(PROJECT_ROOT)), "dataset": "Alpaca",
        "source": ALPACA_SOURCE, "source_file": matches[0],
        "category_slug": "generic", "category_native": "n/a", "role": "generic_safe",
        "n_requested": N_ALPACA, "n_available": len(pool), "n_actual": len(picked),
    }]


def main() -> None:
    global N_HARMFUL_PER_CAT, N_BEAVERTAILS_SAFE, N_ALPACA

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--only", choices=["catqa", "beavertails", "alpaca"], action="append",
        help="prepare only these datasets (default: all)",
    )
    parser.add_argument("--n-harmful", type=int, default=None,
                        help="override harmful samples per category (default %d)" % N_HARMFUL_PER_CAT)
    parser.add_argument("--n-safe", type=int, default=None,
                        help="override BeaverTails generic_safe size (default %d)" % N_BEAVERTAILS_SAFE)
    parser.add_argument("--n-alpaca", type=int, default=None,
                        help="override Alpaca generic_safe size (default %d)" % N_ALPACA)
    args = parser.parse_args()
    targets = args.only or ["catqa", "beavertails", "alpaca"]

    # CLI overrides for sample sizes (defaults are the smoke config; bump these
    # toward the paper's 1500/category for fuller runs where data allows).
    if args.n_harmful is not None:
        N_HARMFUL_PER_CAT = args.n_harmful
    if args.n_safe is not None:
        N_BEAVERTAILS_SAFE = args.n_safe
    if args.n_alpaca is not None:
        N_ALPACA = args.n_alpaca

    print(f"HF datasets root: {HF_DATASETS}")
    print(f"Output root:      {OUT_ROOT}")
    print(f"Seed: {SEED}  sizes: harmful/cat={N_HARMFUL_PER_CAT} "
          f"bt_safe={N_BEAVERTAILS_SAFE} alpaca={N_ALPACA}\n")

    artifacts = []
    # one RNG per dataset so adding/removing a dataset doesn't shift the others
    if "catqa" in targets:
        artifacts += prepare_catqa(random.Random(SEED))
    if "beavertails" in targets:
        artifacts += prepare_beavertails(random.Random(SEED))
    if "alpaca" in targets:
        artifacts += prepare_alpaca(random.Random(SEED))

    manifest = {
        "generated_with": {
            "script": "scripts/prepare_data.py", "seed": SEED,
            "hf_datasets_root": str(HF_DATASETS),
        },
        "schema": {
            "fields": ["id", "text", "response", "category", "role", "source"],
            "roles": ["harmful", "generic_safe"],
            "note": "text=prompt/question; response=answer or null (paper Sec. 3.1).",
        },
        "paper_note": (
            "CatQA and BeaverTails each use their OWN 3 representative categories "
            "(Table 1/2); categories are NOT mapped across datasets. Only 'Hate' "
            "overlaps and may be compared across datasets."
        ),
        "artifacts": artifacts,
    }
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nWrote manifest: {MANIFEST_PATH.relative_to(PROJECT_ROOT)} ({len(artifacts)} artifacts)")


if __name__ == "__main__":
    main()
