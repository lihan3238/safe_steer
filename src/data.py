"""Unified loader for the prepared SafeSteer datasets.

This is the ONLY entry point the vector-extraction pipeline uses to read data.
It reads data/processed/*.jsonl (produced by scripts/prepare_data.py) via the
manifest as the single source of truth -- it never touches the HF cache and
never samples/shuffles/truncates (that is prepare_data.py's job, with a fixed
seed, so determinism has exactly one source).

Core objects:
  Example        -- one normalised record {id, text, response, category, role, source}.
  CategorySplit  -- the input unit for SafeSteer Eq. 1: a category's harmful
                    side (D^c_unsafe) paired with a generic-safe pool
                    (D^generic_safe).

Paper note (Sec. 4.1): the generic-safe pool differs by setup -- CatQA uses
Alpaca as generic safe, BeaverTails uses its own `safe` split. The loader does
not hardcode this; the caller picks via `safe_source`.

Run `python src/data.py` for a self-test (no pytest dependency).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = PROJECT_ROOT / "data" / "manifest.json"

REQUIRED_FIELDS = ("id", "text", "response", "category", "role", "source")
VALID_ROLES = ("harmful", "generic_safe")

# Paper Sec. 3.3, second refinement: for the chat->base transfer, refine the
# generic-safe pool to ONLY {prompt, response} pairs whose response is a refusal,
# so omega points toward "refuse" rather than "safe topic". We detect refusals
# by substring match against common refusal openers (refusal-direction style,
# Arditi & Obeso 2023). Deterministic; no API. Used by extract_activations.py
# via --safe-filter refusal. Requires --use-response (the refusal lives in the
# response text, not the prompt).
REFUSAL_MARKERS = (
    "i cannot", "i can't", "i can not", "i won't", "i will not", "i'm sorry",
    "i am sorry", "i apologize", "i'm not able to", "i am not able to",
    "i'm unable", "i am unable", "as an ai", "as a responsible", "i do not",
    "it is not appropriate", "it's not appropriate", "i must decline",
    "i'm not going to", "cannot provide", "can't provide", "cannot help",
    "can't help", "not able to provide", "i refuse", "it is illegal",
    "it's illegal", "please do not", "i would advise against",
    "i cannot assist", "i can't assist", "i'm just an ai", "no, i cannot",
    "no, i can't",
)


def is_refusal(text: str | None) -> bool:
    """True if `text` reads as a refusal (Sec. 3.3 safe-set refinement)."""
    tl = (text or "").lower()
    return any(m in tl for m in REFUSAL_MARKERS)


@dataclass(frozen=True)
class Example:
    """One normalised record. `response` is None when the source has no answer."""

    id: str
    text: str
    response: str | None
    category: str
    role: str
    source: str

    @classmethod
    def from_dict(cls, d: dict, *, where: str) -> "Example":
        missing = [k for k in REQUIRED_FIELDS if k not in d]
        if missing:
            raise ValueError(f"{where}: record missing fields {missing}: {d!r}")
        if not isinstance(d["text"], str) or not d["text"].strip():
            raise ValueError(f"{where}: empty/non-str `text` in {d.get('id')!r}")
        if d["role"] not in VALID_ROLES:
            raise ValueError(f"{where}: bad role {d['role']!r} (expect {VALID_ROLES})")
        return cls(
            id=d["id"], text=d["text"], response=d["response"],
            category=d["category"], role=d["role"], source=d["source"],
        )


@dataclass(frozen=True)
class CategorySplit:
    """SafeSteer Eq. 1 input: harmful side + generic-safe pool for one category."""

    dataset: str          # "CatQA" | "BeaverTails"
    category: str         # native slug, e.g. "child_abuse"
    harmful: list[Example]        # D^c_unsafe
    generic_safe: list[Example]   # D^generic_safe
    safe_source: str      # "alpaca" | "beavertails"

    def __post_init__(self) -> None:
        if not self.harmful:
            raise ValueError(f"{self.dataset}/{self.category}: empty harmful side")
        if not self.generic_safe:
            raise ValueError(f"{self.dataset}/{self.category}: empty generic_safe pool")

    def summary(self) -> str:
        return (
            f"{self.dataset}/{self.category}: "
            f"{len(self.harmful)} harmful + {len(self.generic_safe)} generic_safe "
            f"(safe_source={self.safe_source})"
        )


# --- manifest + jsonl ------------------------------------------------------
def load_manifest(path: Path = MANIFEST_PATH) -> dict:
    if not path.exists():
        raise FileNotFoundError(
            f"manifest not found: {path}\nRun: python scripts/prepare_data.py"
        )
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path | str) -> list[Example]:
    """Read one processed jsonl into validated Examples."""
    path = Path(path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    if not path.exists():
        raise FileNotFoundError(
            f"processed file not found: {path}\nRun: python scripts/prepare_data.py"
        )
    out = []
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
        if line.strip():
            out.append(Example.from_dict(json.loads(line), where=f"{path.name}:{i}"))
    return out


# --- artifact lookups (manifest-driven) ------------------------------------
def _artifacts(manifest: dict | None = None) -> list[dict]:
    return (manifest or load_manifest())["artifacts"]


def load_examples(
    dataset: str | None = None,
    role: str | None = None,
    category: str | None = None,
    manifest: dict | None = None,
) -> list[Example]:
    """Flat, filtered view across all artifacts -- for exploration.

    Filters match manifest fields: `dataset` (e.g. "CatQA"), `role`
    ("harmful"|"generic_safe"), `category` (the manifest category_slug).
    """
    manifest = manifest or load_manifest()
    examples: list[Example] = []
    for art in _artifacts(manifest):
        if dataset is not None and art["dataset"] != dataset:
            continue
        if role is not None and art["role"] != role:
            continue
        if category is not None and art["category_slug"] != category:
            continue
        examples.extend(load_jsonl(art["path"]))
    return examples


def available_categories(dataset: str, manifest: dict | None = None) -> list[str]:
    """Harmful category slugs available for a dataset, in manifest order."""
    seen: list[str] = []
    for art in _artifacts(manifest):
        if art["dataset"] == dataset and art["role"] == "harmful":
            if art["category_slug"] not in seen:
                seen.append(art["category_slug"])
    return seen


def _generic_safe_artifact(safe_source: str, manifest: dict) -> dict:
    """Pick the generic-safe artifact by source dataset name."""
    wanted = {"alpaca": "Alpaca", "beavertails": "BeaverTails"}.get(safe_source)
    if wanted is None:
        raise ValueError(f"safe_source must be 'alpaca' or 'beavertails', got {safe_source!r}")
    for art in _artifacts(manifest):
        if art["dataset"] == wanted and art["role"] == "generic_safe":
            return art
    raise ValueError(f"no generic_safe artifact for safe_source={safe_source!r}")


def load_category_split(
    dataset: str,
    category: str,
    safe_source: str = "alpaca",
    manifest: dict | None = None,
) -> CategorySplit:
    """Assemble the Eq. 1 input for one (dataset, category).

    `dataset`/`category` select the harmful side (manifest category_slug).
    `safe_source` selects the generic-safe pool: "alpaca" (CatQA setup, default)
    or "beavertails" (BeaverTails' own safe split).
    """
    manifest = manifest or load_manifest()

    harmful_art = next(
        (a for a in _artifacts(manifest)
         if a["dataset"] == dataset and a["role"] == "harmful"
         and a["category_slug"] == category),
        None,
    )
    if harmful_art is None:
        avail = available_categories(dataset, manifest)
        raise ValueError(
            f"no harmful artifact for {dataset}/{category}; available: {avail}"
        )

    safe_art = _generic_safe_artifact(safe_source, manifest)
    return CategorySplit(
        dataset=dataset,
        category=category,
        harmful=load_jsonl(harmful_art["path"]),
        generic_safe=load_jsonl(safe_art["path"]),
        safe_source=safe_source,
    )


# --- self-test (Principle VI: cover only what would corrupt a result) ------
def _self_test() -> None:
    manifest = load_manifest()
    arts = _artifacts(manifest)
    print(f"manifest: {len(arts)} artifacts, seed={manifest['generated_with']['seed']}")

    # 1) every artifact loads, schema is valid, manifest n_actual matches disk
    total = 0
    for art in arts:
        rows = load_jsonl(art["path"])
        total += len(rows)
        assert len(rows) == art["n_actual"], (
            f"{art['path']}: disk {len(rows)} != manifest n_actual {art['n_actual']}"
        )
    print(f"  schema OK, {total} records, disk counts match manifest n_actual")

    # 2) category listings
    for ds in ("CatQA", "BeaverTails"):
        cats = available_categories(ds, manifest)
        assert cats, f"{ds}: no harmful categories found"
        print(f"  {ds} categories: {cats}")

    # 3) both generic-safe sources assemble a non-empty split (covers __post_init__)
    cat = available_categories("BeaverTails", manifest)[0]
    for src in ("alpaca", "beavertails"):
        split = load_category_split("BeaverTails", cat, safe_source=src, manifest=manifest)
        print("  " + split.summary())

    # 4) a CatQA split with the paper-default safe source
    catqa_cat = available_categories("CatQA", manifest)[0]
    split = load_category_split("CatQA", catqa_cat, manifest=manifest)
    print("  " + split.summary())
    assert split.harmful[0].response is None, "CatQA harmful should have null response"

    print("self-test OK")


if __name__ == "__main__":
    _self_test()
