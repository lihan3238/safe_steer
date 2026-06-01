"""Attention-activation extraction for SafeSteer (paper Eq. 1, the `act(x)` term).

MECHANISM ONLY. Iterating over a dataset and writing vectors to disk is the job
of scripts/extract_activations.py; this module just provides the hook + pooling.

Paper Eq. 1: act(x) runs x through the model, reads a layer's self-attention
sublayer output, and averages over all tokens -> one (hidden,) vector per input
per layer. We keep ONE vector per input (not a pre-averaged sum) so the L2-norm
pruning of Sec. 3.3 and the t-SNE disentanglement plots stay possible.

Data-structure flow:
    texts (list[str])
        | tokenize (left-padded batch)
        v
    input_ids (B, seq)  +  attention_mask (B, seq)
        | forward; forward-hooks on self_attn of each requested layer
        v
    captured {layer: (B, seq, hidden)}      <- attention sublayer output
        | mean over REAL tokens (mask-weighted; padding excluded)
        v
    {layer: (B, hidden)}  -> accumulated across batches ->
    {layer: (N, hidden)}                     <- this module's product

transformers 5.x note: a decoder layer's self_attn may return a tuple
(attn_out, ...) or a bare tensor. We normalise both and assert the last dim ==
hidden_size, so a silent structure change surfaces immediately instead of
poisoning the vectors.

Run `python src/hooks.py` for a self-test on a tiny random model (no download,
no GPU needed).
"""
from __future__ import annotations

from typing import Iterable

import torch


def _hidden_from_attn_output(out) -> torch.Tensor:
    """Pull the hidden-state tensor out of a self_attn forward output.

    transformers attention modules return either a tensor or a tuple whose
    first element is the hidden state. Anything else is an error we want loud.
    """
    if isinstance(out, torch.Tensor):
        return out
    if isinstance(out, (tuple, list)) and out and isinstance(out[0], torch.Tensor):
        return out[0]
    raise TypeError(
        f"unexpected self_attn output type {type(out)!r}; "
        "transformers layout may have changed -- inspect before trusting vectors"
    )


class AttentionActivationHook:
    """Context manager that captures self_attn output at the given layers.

    Usage:
        with AttentionActivationHook(model, [14, 18]) as hook:
            model(input_ids=..., attention_mask=..., use_cache=False)
            acts = hook.captured        # {14: (B, seq, H), 18: (B, seq, H)}

    `base_model` defaults to model.model (Llama/Qwen/Gemma decoder stack:
    base_model.layers[i].self_attn). Pass a different accessor if a model nests
    its layers elsewhere.
    """

    def __init__(self, model, layers: Iterable[int], base_model=None):
        self.model = model
        self.layers = list(layers)
        self.base_model = base_model if base_model is not None else model.model
        self.hidden_size = model.config.hidden_size
        self._handles: list = []
        self.captured: dict[int, torch.Tensor] = {}

    def _make_hook(self, layer: int):
        def hook(_module, _inp, out):
            h = _hidden_from_attn_output(out)
            if h.shape[-1] != self.hidden_size:
                raise ValueError(
                    f"layer {layer}: captured last-dim {h.shape[-1]} != hidden "
                    f"{self.hidden_size}; wrong tensor captured"
                )
            self.captured[layer] = h.detach()
        return hook

    def __enter__(self) -> "AttentionActivationHook":
        for layer in self.layers:
            attn = self.base_model.layers[layer].self_attn
            self._handles.append(attn.register_forward_hook(self._make_hook(layer)))
        return self

    def __exit__(self, *exc) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()


def mean_pool(activation: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Average over real tokens, excluding padding.

    activation: (B, seq, H); attention_mask: (B, seq) of {0,1}.
    Returns (B, H). Eq. 1 averages over tokens; padding must not dilute it.
    """
    mask = attention_mask.to(activation.dtype).unsqueeze(-1)   # (B, seq, 1)
    summed = (activation * mask).sum(dim=1)                    # (B, H)
    counts = mask.sum(dim=1).clamp(min=1.0)                    # (B, 1)
    return summed / counts


def build_inputs(tokenizer, texts, responses=None, use_response=False, device="cpu"):
    """Tokenise a batch as a left-padded {input_ids, attention_mask}.

    use_response=True appends the response (when present) so the activation is
    over the {prompt, response} pair (paper Sec. 3.1); default is prompt-only.
    Uses the chat template when the tokenizer defines one.
    """
    rendered = []
    responses = responses or [None] * len(texts)
    for text, resp in zip(texts, responses):
        if tokenizer.chat_template:
            messages = [{"role": "user", "content": text}]
            if use_response and resp:
                messages.append({"role": "assistant", "content": resp})
            rendered.append(tokenizer.apply_chat_template(
                messages, tokenize=False,
                add_generation_prompt=not (use_response and resp),
            ))
        else:
            rendered.append(f"{text}\n\n{resp}" if (use_response and resp) else text)

    # RIGHT padding for extraction: real tokens keep positions 0..L-1, so RoPE
    # gives each token the same activation whether or not the batch is padded.
    # (Generation uses LEFT padding to align the last token at -1; that is a
    # different module's concern -- see eval_steering. Don't mix the two.)
    prev_side = tokenizer.padding_side
    tokenizer.padding_side = "right"
    try:
        enc = tokenizer(rendered, return_tensors="pt", padding=True, truncation=True)
    finally:
        tokenizer.padding_side = prev_side
    return {k: v.to(device) for k, v in enc.items()}


@torch.inference_mode()
def extract_activations(
    model,
    tokenizer,
    texts,
    layers,
    responses=None,
    use_response=False,
    batch_size=8,
    base_model=None,
    progress=None,
) -> dict[int, torch.Tensor]:
    """Eq. 1 act(x): one mean-pooled (hidden,) vector per input per layer.

    Returns {layer: (N, hidden)} on CPU float32, in input order. N == len(texts).
    `progress`, if given, is called as progress(done, total) after each batch
    (lightweight, no tqdm dependency).
    """
    device = next(model.parameters()).device
    layers = list(layers)
    per_layer: dict[int, list[torch.Tensor]] = {l: [] for l in layers}
    total = len(texts)

    for start in range(0, total, batch_size):
        batch_texts = texts[start:start + batch_size]
        batch_resp = (responses[start:start + batch_size] if responses else None)
        inputs = build_inputs(
            tokenizer, batch_texts, batch_resp, use_response=use_response, device=device
        )
        with AttentionActivationHook(model, layers, base_model=base_model) as hook:
            model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs.get("attention_mask"),
                use_cache=False,
            )
            for layer in layers:
                pooled = mean_pool(hook.captured[layer], inputs["attention_mask"])
                per_layer[layer].append(pooled.float().cpu())
        if progress is not None:
            progress(min(start + batch_size, total), total)

    return {layer: torch.cat(chunks, dim=0) for layer, chunks in per_layer.items()}


# --- self-test: tiny random model, no download, no GPU ---------------------
def _self_test() -> None:
    from transformers import LlamaConfig, LlamaForCausalLM, AutoTokenizer

    torch.manual_seed(0)
    tok = AutoTokenizer.from_pretrained("hf-internal-testing/llama-tokenizer")
    tok.pad_token = tok.eos_token

    # vocab_size MUST match the tokenizer, else token ids overflow the embedding
    cfg = LlamaConfig(
        vocab_size=tok.vocab_size, hidden_size=32, intermediate_size=64,
        num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=4,
        max_position_embeddings=64,
    )
    model = LlamaForCausalLM(cfg).eval()

    texts = ["how do I pick a lock", "tell me a safe bedtime story", "what is 2+2"]
    layers = [1, 3]

    acts = extract_activations(model, tok, texts, layers, batch_size=2)

    # shape: one (hidden,) per input per layer
    for layer in layers:
        assert acts[layer].shape == (len(texts), cfg.hidden_size), acts[layer].shape
    print(f"  extract_activations -> {{layer: {tuple(acts[layers[0]].shape)}}} for layers {layers}")

    # mean_pool arithmetic, on a synthetic tensor (no model -> no RoPE noise):
    # token 0 and 1 are real, token 2 is padding and must be ignored.
    act = torch.tensor([[[2.0, 4.0], [4.0, 8.0], [99.0, 99.0]]])  # (1, 3, 2)
    msk = torch.tensor([[1, 1, 0]])
    assert torch.allclose(mean_pool(act, msk), torch.tensor([[3.0, 6.0]])), "mean_pool math"
    print("  mean_pool ignores padding (synthetic): OK")

    # padding-invariance through the model: a short text pooled alone must match
    # the same text pooled inside a longer-padded batch (right padding + RoPE).
    short = "the quick brown fox"
    alone = extract_activations(model, tok, [short], [1], batch_size=1)[1]
    in_batch = extract_activations(
        model, tok, [short, "a much longer sentence to force padding here"], [1], batch_size=2
    )[1][:1]
    assert torch.allclose(alone, in_batch, atol=1e-4), "batch padding-invariance failed"
    print("  activation invariant to batch padding: OK")

    # use_response switch changes the captured activation (sanity, not correctness)
    a_prompt = extract_activations(model, tok, ["question"], [1],
                                   responses=["answer"], use_response=False)[1]
    a_pair = extract_activations(model, tok, ["question"], [1],
                                 responses=["answer"], use_response=True)[1]
    assert not torch.allclose(a_prompt, a_pair), "use_response switch had no effect"
    print("  use_response switch changes activation: OK")

    print("self-test OK")


if __name__ == "__main__":
    _self_test()
