"""Inference-time steering injection for SafeSteer (paper Eq. 2).

This module is MECHANISM ONLY: it adds steering vectors into the forward pass.
The generation loop (sampling, batching, %UR scoring) lives in
scripts/eval_steering.py.

Paper Eq. 2:  theta^attn_l  <-  theta^attn_l + m * omega_l
The paper phrases this as modifying the self-attention weights. We implement the
equivalent, cleaner intervention: a forward hook on the SAME self_attn module
that hooks.py reads from, rewriting its OUTPUT hidden state at every token
position:
        h_l  <-  h_l + m * omega_l
Adding a constant vector to every position of the attention output is equivalent
to adding it through the projection, but it touches no weights and detaches
instantly on context exit (Principle V: steering must toggle cleanly). This is
the standard activation-steering form used by refusal_direction / CAA.

`m` (multiplier) scales strength and may be negative (reverse steering), to
sweep the paper's multiplier range.

Run `python src/steering.py` for a self-test on a tiny random model (no GPU).
"""
from __future__ import annotations

import torch


def _split_attn_output(out):
    """Return (hidden_state, rebuild) so we can rewrite hidden and restore shape.

    self_attn forward returns either a tensor or a tuple (hidden, *rest).
    `rebuild(new_hidden)` puts a modified hidden back into the original form.
    """
    if isinstance(out, torch.Tensor):
        return out, (lambda new: new)
    if isinstance(out, (tuple, list)) and out and isinstance(out[0], torch.Tensor):
        rest = tuple(out[1:])
        return out[0], (lambda new: (new, *rest))
    raise TypeError(
        f"unexpected self_attn output type {type(out)!r}; "
        "transformers layout may have changed -- inspect before steering"
    )


class SteeringHook:
    """Context manager that injects m * omega_l into self_attn output per layer.

    Usage:
        with SteeringHook(model, {14: omega14}, multiplier=0.5):
            out = model.generate(...)     # steered
        # hooks auto-removed here -> model back to naive

    `vectors`: {layer: (hidden,)}. `base_model` defaults to model.model
    (Llama/Qwen/Gemma decoder stack: base_model.layers[i].self_attn).
    """

    def __init__(self, model, vectors: dict[int, torch.Tensor], multiplier: float,
                 base_model=None):
        self.model = model
        self.vectors = vectors
        self.multiplier = float(multiplier)
        self.base_model = base_model if base_model is not None else model.model
        self.hidden_size = model.config.hidden_size
        self._handles: list = []
        for layer, omega in vectors.items():
            if omega.shape[-1] != self.hidden_size:
                raise ValueError(
                    f"layer {layer}: omega dim {omega.shape[-1]} != hidden "
                    f"{self.hidden_size}"
                )

    def _make_hook(self, omega: torch.Tensor):
        def hook(_module, _inp, out):
            hidden, rebuild = _split_attn_output(out)
            # match dtype/device: omega is saved as fp32/cpu, model may be bf16/cuda
            delta = self.multiplier * omega.to(dtype=hidden.dtype, device=hidden.device)
            return rebuild(hidden + delta)   # broadcasts over (batch, seq, hidden)
        return hook

    def __enter__(self) -> "SteeringHook":
        for layer, omega in self.vectors.items():
            attn = self.base_model.layers[layer].self_attn
            self._handles.append(attn.register_forward_hook(self._make_hook(omega)))
        return self

    def __exit__(self, *exc) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()


def apply_steering(model, vectors: dict[int, torch.Tensor], multiplier: float,
                   base_model=None) -> SteeringHook:
    """Convenience constructor -- use as `with apply_steering(model, vecs, m):`."""
    return SteeringHook(model, vectors, multiplier, base_model=base_model)


# --- self-test: tiny random model, no download of weights, no GPU ----------
def _self_test() -> None:
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(0)
    cfg = LlamaConfig(
        vocab_size=128, hidden_size=16, intermediate_size=32,
        num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=4,
        max_position_embeddings=32,
    )
    model = LlamaForCausalLM(cfg).eval()
    ids = torch.randint(0, cfg.vocab_size, (2, 6))
    H = cfg.hidden_size

    def logits(vectors=None, m=0.0):
        if vectors is None:
            with torch.inference_mode():
                return model(input_ids=ids, use_cache=False).logits
        with SteeringHook(model, vectors, m), torch.inference_mode():
            return model(input_ids=ids, use_cache=False).logits

    base = logits()

    # 1) injection changes the output
    omega = {2: torch.randn(H)}
    steered = logits(omega, m=1.0)
    assert not torch.allclose(base, steered, atol=1e-5), "injection had no effect"
    print("  injection changes output: OK")

    # 2) clean teardown: after the `with` block, forward matches naive again
    after = logits()
    assert torch.allclose(base, after, atol=1e-6), "hook leaked after context exit"
    print("  hook removed on exit (no leak): OK")

    # 3) m=0 is identity
    assert torch.allclose(base, logits(omega, m=0.0), atol=1e-6), "m=0 not identity"
    print("  m=0 == naive: OK")

    # 4) the intervention is linear in m: h += m*omega. The map m -> logits is
    #    only LOCALLY linear (later layers add softmax/SiLU/RMSNorm), so we probe
    #    at small m where the first-order term dominates: +m and -m must give
    #    opposite perturbations (cosine -> -1), verified empirically at m=1e-4.
    d_pos = (logits(omega, m=1e-4) - base).flatten()
    d_neg = (logits(omega, m=-1e-4) - base).flatten()
    cos = torch.dot(d_pos, d_neg) / (d_pos.norm() * d_neg.norm() + 1e-12)
    assert cos < -0.999, f"+m/-m not opposite at small m (cos={cos:.4f})"
    # and the perturbation scales ~linearly with m (10x m -> ~10x norm)
    d_10 = (logits(omega, m=1e-3) - base).flatten()
    ratio = d_10.norm() / d_pos.norm()
    assert 9.0 < ratio < 11.0, f"perturbation not linear in m (ratio={ratio:.2f})"
    print(f"  linear in m (cos={cos:.4f}, 10x-scale ratio={ratio:.2f}): OK")

    # 5) multi-layer injection applies at every requested layer
    multi = {1: torch.randn(H), 3: torch.randn(H)}
    with SteeringHook(model, multi, 0.5) as hook:
        assert len(hook._handles) == 2
        out_multi = model(input_ids=ids, use_cache=False).logits
    assert not torch.allclose(base, out_multi, atol=1e-5), "multi-layer no effect"
    print("  multi-layer injection: OK")

    print("self-test OK")


if __name__ == "__main__":
    _self_test()
