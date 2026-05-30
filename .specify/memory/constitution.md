<!--
SYNC IMPACT REPORT
==================
Version change: 1.0.0 → 1.1.0
Bump rationale: Added one principle (VI. Lean by Default) and materially revised
the Development Workflow & Quality Gates to remove over-engineering; refined the
runtime-guidance rule to make AGENTS.md the single cross-tool source. New
principle + expanded guidance = MINOR per semver. No principle removed/redefined.

Modified principles:
  - I–V unchanged in intent (wording stable)
  - Added: VI. Lean by Default (No Over-Engineering)

Added sections: none (Principle VI added under Core Principles)
Removed sections: none

Templates requiring updates:
  - .specify/templates/plan-template.md ✅ reviewed (dynamic Constitution Check
    gate reads this file; no edit required)
  - .specify/templates/spec-template.md ✅ reviewed (no conflict)
  - .specify/templates/tasks-template.md ✅ reviewed (no mandatory test-task
    category imposed; aligns with Principle VI)

Follow-up TODOs:
  - AGENTS.md ⚠ pending: currently a SPECKIT stub; populate at /speckit-plan with
    canonical stack/commands/structure, and reduce CLAUDE.md to the single line
    `@AGENTS.md` (remove the duplicated lead text). Tracked by Governance §
    Runtime guidance. Deferred until AGENTS.md has real content to import.
-->

# SafeSteer-From-Scratch Constitution

## Core Principles

### I. Faithful Core Reproduction

The steering algorithm is the object of reproduction and MUST match the paper
(Ghosh et al., "A Simple Yet Effective Method for Non-Refusing Context Relevant
Fine-grained Safety Steering in LLMs", EMNLP 2025) exactly in mechanism:

- Steering vectors MUST be computed as the mean activation difference between
  safe and unsafe inputs (Eq. 1), per layer, over attention activations.
- The intervention MUST add the scaled vector to the chosen layer's
  self-attention output at every token position during the forward pass
  (Eq. 2: `θ_l ← θ_l + m · ω_l`), with the SAME layer used for extraction and
  intervention.
- The "pruned activations" variant (keep diff components whose L2 norm exceeds
  the median) MUST be implemented and selectable.
- The method MUST remain gradient-free and inference-time only: no fine-tuning,
  no weight updates, no required contrastive pairs.

Any deviation from the paper (e.g., a different activation site, a substitute
model, an approximated metric) MUST be recorded explicitly in code comments and
in the run's notes, labelled as a deviation, with its rationale. Silent
divergence from the paper is prohibited. Rationale: a reproduction whose core
math drifts from the source teaches the wrong thing and cannot confirm or
falsify the paper's claim.

### II. Minimal & Fast-to-Run First

The project optimizes for a *complete but minimal* path that a newcomer can run
end-to-end quickly, then scale up:

- A "smoke" configuration MUST exist that runs the full pipeline (extract →
  steer → generate → evaluate) end-to-end on commodity hardware (single
  consumer GPU or CPU) in minutes, using a small model and a tiny data subset.
- Defaults MUST favor the smallest faithful setting that still exercises every
  stage; larger models (Llama-2-7B/-chat, Llama-3-8B), full categories, and full
  sample counts MUST be opt-in via config, never required to see the method work.
- New scope MUST NOT be added until the existing smoke path runs green. Breadth
  is earned by a working minimal loop, not assumed up front.

Rationale: the stated goal is to "quickly run it" to understand and explore
SafeSteer; a 7B-only, hours-long pipeline defeats that purpose.

### III. Reproducibility & Determinism (NON-NEGOTIABLE)

Every reported result MUST be regenerable from the repository alone:

- Dependencies MUST be pinned (lockfile or pinned `requirements`); Python and
  key library versions MUST be recorded.
- Every run MUST be driven by a versioned config file and a fixed random seed;
  given the same config + seed + environment, results MUST be reproducible.
- Each result artifact MUST log: model id + revision, dataset + split, layer,
  multiplier, category, seed, hardware, and wall-clock/cost.
- Model and dataset acquisition MUST use one consistent cache layout to avoid
  silent re-downloads (do not mix `hf download --local-dir` with
  `from_pretrained(hub_id)`); the chosen layout MUST be documented once.

Rationale: a reproduction that cannot be re-run deterministically is an anecdote,
not evidence. This is the one place rigor is mandatory — it is not redundant
overhead, it is what makes the work a reproduction at all.

### IV. Claim-Linked Verification

Work is "done" only when tied back to the paper's falsifiable claim, not when the
code merely runs:

- The core claim under test is: **category-specific steering reduces the
  percentage of unsafe responses (naive → steered) while preserving helpfulness
  and coherence and avoiding blanket refusals.**
- The evaluation harness MUST measure the paper's primary signal — a drop in
  %unsafe-responses from naive to steered generation — plus at least one utility
  signal (helpfulness/coherence) and a refusal check.
- A result MUST NOT be described as "reproduced" until the harness shows the
  expected directional effect (steered %unsafe < naive %unsafe with utility
  retained) on at least the smoke configuration, and the numbers are recorded.
- Evaluators that require external services (LLM-as-judge, reward models) MUST
  have a documented offline or lighter-weight fallback so verification is not
  blocked by API access.

Rationale: running ≠ reproducing. The claim is the contract.

### V. Readable, Modular & Explorable

The codebase is a learning and exploration instrument first:

- The steering intervention MUST be implemented as a clean, removable hook
  (attach/detach) so it can be toggled and inspected without editing model code.
- Model, layer(s), multiplier, category, dataset, and prune-on/off MUST be
  swappable from config or CLI without code changes, to support sweeps and
  exploration.
- Code MUST favor clarity over cleverness: small, named functions mapping to the
  paper's steps; comments link code to the paper's equations/sections.
- At least one runnable entry point (CLI and/or notebook) MUST let a user
  reproduce a single steered generation and a single sweep without reading the
  whole codebase.

Rationale: the deliverable is understanding; opaque code that "works" fails the
project's purpose.

### VI. Lean by Default (No Over-Engineering)

This is a paper reproduction for learning, not a product. Effort goes into a
clean, complete, faithful reproduction — and stops there:

- Build only what the reproduction and exploration of SafeSteer require. No
  speculative abstraction, no plugin frameworks, no service layers, no premature
  generalization "for later."
- Automated tests are written ONLY where a silent bug would invalidate a result:
  the steering-vector computation, the intervention hook, and the metric/eval
  scoring. Glue, I/O, and config plumbing are verified by running the smoke path,
  not by dedicated test scaffolding. No coverage targets, no test-for-test's-sake.
- Tooling stays minimal: prefer a small, standard stack over elaborate CI,
  containers, or orchestration unless a concrete, recurring pain demands it.
- "Lean" is not "incomplete": every step of the paper's core method MUST be
  present and correct. Cut redundancy and ceremony, never coverage of the method.

Rationale: the user asked for "干净利落、不遗漏" — clean and crisp without
omissions. Over-engineering and omission are both failures; this principle
guards against the first while Principles I and IV guard against the second.

## Technology Stack & Scope Constraints

- **Language/runtime**: Python 3.10+ with PyTorch and HuggingFace
  `transformers` for white-box activation access and hooking.
- **Reference models**: Llama-2-7B, Llama-2-7B-chat, Llama-3-8B (32-layer,
  intervention layer 14 as the paper's primary). A small open model MUST be
  usable as the smoke-test default (per Principle II); large models are opt-in.
- **Datasets**: CategoricalQA (CatQA), BeaverTails, and Alpaca Instructions as
  in the paper; the reproduction MAY ship tiny bundled/subsampled splits for the
  smoke path. Dataset provenance and licenses MUST be respected and recorded.
- **Steering math**: mean-difference vectors (Eq. 1), optional median-L2 pruning,
  additive self-attention intervention scaled by multiplier `m` (Eq. 2);
  multipliers in the paper's studied range (e.g., ~0.5) are the defaults.
- **Evaluation**: primary metric is the drop in %unsafe-responses (naive →
  steered) via a safety classifier; utility via helpfulness/coherence scoring.
  External-judge dependencies (GPT-class classifier, Nemotron-class reward model)
  MUST have a documented lighter or offline substitute (Principle IV).
- **Scope (in)**: a minimal, faithful, runnable reproduction of the core
  SafeSteer method and its primary safety/utility evaluation, with knobs for
  exploration.
- **Scope (out)**: production hardening, a general-purpose steering library,
  multimodal/VLM variants, novel method extensions, and large-scale
  benchmarking beyond what confirms the core claim. These are non-goals unless
  promoted by an amendment.
- **Content handling**: the work involves harmful-content categories for safety
  evaluation; harmful examples are used only as evaluation inputs and MUST NOT be
  surfaced as usable instructions. Outputs and notes follow defensive-research
  norms.

## Development Workflow & Quality Gates

Gates are deliberately light (Principle VI). The bar is "the reproduction runs
and the claim holds", not a heavyweight engineering process.

- **Spec-driven flow**: features proceed through the Spec Kit lifecycle
  (constitution → specify → plan → tasks → implement); design artifacts cite the
  principles they satisfy.
- **Smoke gate**: before any full-scale run, the smoke configuration (Principle
  II) MUST pass end-to-end. A change that breaks the smoke path is not mergeable.
  Running the smoke path is the primary integration check — it substitutes for
  broad test scaffolding on glue/I/O code.
- **Verification gate**: a result is reported only after the claim-linked
  verification of Principle IV is satisfied and the numbers + config are logged.
- **Core-correctness tests**: the steering-vector math, the intervention hook,
  and the eval scoring MUST have focused unit tests (Principle VI); nothing else
  requires dedicated tests.
- **Reproduction-fidelity review**: changes touching the steering math, layer
  selection, metric definition, or evaluation protocol MUST be reviewed against
  Principle I, and any deviation from the paper documented.
- **Research notes**: substantive findings, deviations, and reproduced numbers
  are mirrored to the paper-notes vault (Obsidian `lihan3238`) so the
  reproduction and the reading notes stay in sync.

## Governance

This constitution supersedes ad-hoc practice for the SafeSteer reproduction. When
a principle conflicts with convenience, the principle wins or the constitution is
amended first.

- **Amendments**: proposed as a documented change to this file, with rationale
  and a version bump. Amendments that materially change scope or method fidelity
  MUST note the impact on existing reproduced results.
- **Versioning**: semantic versioning of the constitution. MAJOR = principle
  removal/redefinition or backward-incompatible governance change; MINOR = new
  principle/section or materially expanded guidance; PATCH = clarifications and
  wording.
- **Compliance**: every plan and PR is checked against these principles. Added
  complexity MUST be justified against the minimal-reproduction goal (Principles
  II & VI); unjustified complexity is rejected.
- **Runtime guidance**: `AGENTS.md` at the repo root is the SINGLE canonical,
  cross-tool source of concrete stack, commands, and project structure, readable
  by any AGENTS.md-aware tool (Claude Code, Codex, Cursor, Gemini CLI). `CLAUDE.md`
  MUST be reduced to the single line `@AGENTS.md` — edit `AGENTS.md` only, never
  mirror rules into two files. Tool-specific files (e.g., Claude Code skills) may
  hold genuinely tool-specific bits; the shared core lives in `AGENTS.md`.
  `AGENTS.md` MUST stay consistent with this constitution.

**Version**: 1.1.0 | **Ratified**: 2026-05-30 | **Last Amended**: 2026-05-30
