"""Steering-vector computation for SafeSteer (paper Eq. 1 + Sec. 3.3 pruning).

Input is what src/hooks.py produces: per-layer activation matrices
{layer: (N, hidden)} for the safe side and the harmful side. Output is one
steering vector per layer, {layer: (hidden,)}.

Eq. 1 (vanilla, "difference of means"):
    omega_l = mean(safe_l) - mean(harmful_l)
This points from harmful toward safe; at inference we ADD it (Eq. 2) to push
generations toward the safe region.

Sec. 3.3 (pruned activations -- noise filtering):
    The paper, on PAIRED data: take pairwise mean differences harmful-vs-harmless,
    take the median of their L2 norms, keep only differences whose norm exceeds
    the median (top ~50%), and average those.

    We use UNPAIRED, unequal-sized generic-safe data, so there is no row-to-row
    pairing. Interpretation used here (documented deviation, constitution
    Principle I):
        d_i = mu_safe - harmful_i           # each harmful sample vs the safe mean
        keep i where ||d_i|| > median(||d||) # the most informative half
        omega = mean(kept d_i)
    Why this matches the paper's rationale: differences with small norm mean the
    model barely separates harmful from safe for that sample (entangled, low
    signal) -- exactly what Sec. 3.3 discards. It is also self-consistent:
    mean(ALL d_i) = mu_safe - mu_harmful, i.e. the vanilla Eq. 1 vector, so
    pruning just restricts that mean to the high-norm half. Deterministic, no
    random pairing.

Run `python src/vectors.py` for a self-test on synthetic tensors.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch


@dataclass
class SteeringVectors:
    """Per-layer steering vectors plus the metadata needed to reproduce them."""

    vectors: dict[int, torch.Tensor]     # {layer: (hidden,)}
    pruned: bool
    keep_fraction: float
    n_safe: int
    n_harmful: int
    meta: dict

    def layers(self) -> list[int]:
        return sorted(self.vectors)

    def save(self, path: Path | str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "vectors": {l: v.cpu() for l, v in self.vectors.items()},
                "pruned": self.pruned,
                "keep_fraction": self.keep_fraction,
                "n_safe": self.n_safe,
                "n_harmful": self.n_harmful,
                "meta": self.meta,
            },
            path,
        )

    @classmethod
    def load(cls, path: Path | str) -> "SteeringVectors":
        blob = torch.load(path, map_location="cpu", weights_only=False)
        return cls(**blob)


def _check(safe: torch.Tensor, harmful: torch.Tensor, layer: int) -> None:
    if safe.ndim != 2 or harmful.ndim != 2:
        raise ValueError(f"layer {layer}: expected 2D (N, hidden), got "
                         f"{tuple(safe.shape)} and {tuple(harmful.shape)}")
    if safe.shape[1] != harmful.shape[1]:
        raise ValueError(f"layer {layer}: hidden mismatch {safe.shape[1]} vs {harmful.shape[1]}")


def difference_of_means(safe: torch.Tensor, harmful: torch.Tensor) -> torch.Tensor:
    """Eq. 1: mean(safe) - mean(harmful) -> (hidden,)."""
    return safe.float().mean(dim=0) - harmful.float().mean(dim=0)


def pruned_difference(
    safe: torch.Tensor, harmful: torch.Tensor, keep_fraction: float = 0.5
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sec. 3.3: keep the highest-norm half of (mu_safe - harmful_i), average them.

    Returns (omega (hidden,), keep_mask (n_harmful,) bool).
    Keeps samples with norm STRICTLY above the (1-keep_fraction) quantile, so
    keep_fraction=0.5 keeps roughly the top half (those exceeding the median).
    """
    safe = safe.float()
    harmful = harmful.float()
    mu_safe = safe.mean(dim=0)                       # (hidden,)
    diffs = mu_safe.unsqueeze(0) - harmful           # (n_harmful, hidden)
    norms = diffs.norm(dim=1)                        # (n_harmful,)
    threshold = torch.quantile(norms, 1.0 - keep_fraction)
    keep_mask = norms > threshold
    if not keep_mask.any():                          # degenerate (all-equal norms)
        keep_mask = norms >= threshold
    omega = diffs[keep_mask].mean(dim=0)
    return omega, keep_mask


def compute_steering_vectors(
    safe_acts: dict[int, torch.Tensor],
    harmful_acts: dict[int, torch.Tensor],
    prune: bool = False,
    keep_fraction: float = 0.5,
    meta: dict | None = None,
) -> SteeringVectors:
    """Build per-layer steering vectors from hook activations.

    safe_acts/harmful_acts: {layer: (N, hidden)} from src.hooks.extract_activations.
    prune=False -> Eq. 1 vanilla; prune=True -> Sec. 3.3 pruned.
    """
    layers = sorted(set(safe_acts) & set(harmful_acts))
    if not layers:
        raise ValueError("no common layers between safe_acts and harmful_acts")

    vectors: dict[int, torch.Tensor] = {}
    n_safe = n_harmful = 0
    for layer in layers:
        safe, harmful = safe_acts[layer], harmful_acts[layer]
        _check(safe, harmful, layer)
        n_safe, n_harmful = safe.shape[0], harmful.shape[0]
        if prune:
            vectors[layer], _ = pruned_difference(safe, harmful, keep_fraction)
        else:
            vectors[layer] = difference_of_means(safe, harmful)

    return SteeringVectors(
        vectors=vectors, pruned=prune, keep_fraction=keep_fraction,
        n_safe=n_safe, n_harmful=n_harmful, meta=meta or {},
    )


# --- self-test: synthetic tensors, exact hand-checked values ----------------
def _self_test() -> None:
    torch.manual_seed(0)

    # 1) Eq. 1 vanilla: known means -> known difference
    safe = torch.tensor([[1.0, 1.0], [3.0, 3.0]])      # mean (2,2)
    harmful = torch.tensor([[0.0, 0.0], [2.0, 0.0]])   # mean (1,0)
    omega = difference_of_means(safe, harmful)
    assert torch.allclose(omega, torch.tensor([1.0, 2.0])), omega
    print(f"  difference_of_means -> {omega.tolist()} (expect [1.0, 2.0]): OK")

    # 2) Sec. 3.3 pruning: construct norms so the kept set is known.
    #    mu_safe = (0,0). harmful rows chosen so ||mu_safe - row|| = 1,2,3,4.
    safe2 = torch.zeros(4, 2)
    harmful2 = torch.tensor([[1.0, 0.0],   # norm 1
                             [2.0, 0.0],   # norm 2
                             [3.0, 0.0],   # norm 3
                             [4.0, 0.0]])  # norm 4
    omega2, keep = pruned_difference(safe2, harmful2, keep_fraction=0.5)
    # median of {1,2,3,4} = 2.5 -> keep norms 3,4 -> rows [3,0],[4,0]
    # diffs = mu_safe - row = -row ; mean of kept = -(3+4)/2 = -3.5 on dim0
    assert keep.tolist() == [False, False, True, True], keep.tolist()
    assert torch.allclose(omega2, torch.tensor([-3.5, 0.0])), omega2
    print(f"  pruned_difference keeps {keep.tolist()} -> {omega2.tolist()} "
          f"(expect [-3.5, 0.0]): OK")

    # 3) sanity: mean(ALL diffs) == vanilla Eq. 1 vector (omega without pruning)
    full = difference_of_means(safe2, harmful2)
    all_diffs_mean = (safe2.mean(0).unsqueeze(0) - harmful2).mean(0)
    assert torch.allclose(full, all_diffs_mean), (full, all_diffs_mean)
    print("  unpruned mean == vanilla Eq.1 vector: OK")

    # 4) end-to-end via compute_steering_vectors + save/load round-trip
    safe_acts = {14: torch.randn(20, 8), 18: torch.randn(20, 8)}
    harmful_acts = {14: torch.randn(15, 8), 18: torch.randn(15, 8)}
    sv = compute_steering_vectors(safe_acts, harmful_acts, prune=True, meta={"model": "test"})
    assert sv.layers() == [14, 18]
    assert all(sv.vectors[l].shape == (8,) for l in sv.layers())
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "sv.pt"
        sv.save(p)
        sv2 = SteeringVectors.load(p)
    assert sv2.pruned and sv2.n_harmful == 15 and sv2.meta["model"] == "test"
    assert torch.allclose(sv.vectors[14], sv2.vectors[14])
    print(f"  compute + save/load round-trip: layers {sv2.layers()}, "
          f"pruned={sv2.pruned}: OK")

    print("self-test OK")


if __name__ == "__main__":
    _self_test()
