## Coder Agent Language

- Coder agents MUST use the user's input language if and only if they are communicating directly with the user 、writing code comments or presenting options to the user.
- In all other contexts, keep the existing language and conventions unchanged; do not translate or otherwise alter reasoning, planning, tool use, delegated-agent instructions, code, comments, commit messages, commands, paths, schema fields, logs, or generated artifacts merely because of the user's input language.

## Documentation Updates

When completing any work, update `README.md` and all relevant existing documentation as needed. Documentation must remain accurate, consistent with the implemented changes, and free of stale instructions.

### README structure (learning-focused note — keep it this way)

`README.md` is a **learning note**, not an ops manual. It has exactly these sections; do not add ops/process content (env setup, tokens, proxies, download commands, hardware tables, troubleshooting, progress checklists, run logs):

1. **原理流(论文方法)** — the paper's method in prose + formulas (Eq.1/Eq.2). Theory only, no code.
2. **数据流(代码实现)** — the step↔formula↔module↔data-structure(in→out) table + the end-to-end pipeline diagram.
3. **关键文件导览** — one row per `src/*.py` / `scripts/*.py`: 职责 · 核心接口 · in→out.
4. **各模块实现要点(对照论文)** — per-module notes: implementation choices, deviations from the paper (mark as documented deviation), and how to self-test.
5. **潜在改进点(复现中发现)** — research-valuable weaknesses / directions found while reproducing (things the paper glosses over). Insight only, no ops content.
6. **快速上手** — a few commands.
7. **参考** — paper + reusable repos.

### Rule: after finishing any module, update README in the SAME change

For every new/changed `src/*.py` or `scripts/*.py`, before committing:

- **§2 table**: set the module's status ✅/⬜ and keep its `in → out` data structures accurate.
- **§3 file guide**: add/update its row (职责 · 核心接口 · in→out). **A new code file MUST get a §3 row** — this is the step most easily forgotten.
- **§4 notes**: add a short "模块 N" block when the module embodies a paper step or a non-obvious choice (formula link, deviation rationale, self-test summary).
- **§6 快速上手**: add the run command if it is a new entry point.

Keep rows terse and aligned with the code's real signatures. If a file is deleted/renamed, fix every section that references it (no stale rows).

## Compute / GPU Servers

Two pre-configured remote servers are available for GPU runs (the local WSL box
has only a single RTX 5070 Ti, 16GB — fine for ≤1.7B models, not for 8B+).

| Host | SSH | Usable GPUs |
|---|---|---|
| `hello` | `hello@10.77.0.101` | **1× RTX 5090** (the only card; use it) |
| `dell` | `dell@10.77.0.102` | **only cards 6 and 7** of 8 — set `CUDA_VISIBLE_DEVICES=6` or `7` (or `6,7`). Cards 0–5 are off-limits. |

### Working directory (servers)

On BOTH servers, do ALL work strictly under `~/.workplace/`. Create it if it
does not exist (`mkdir -p ~/.workplace`) and keep every clone, download, output,
and scratch file inside it. Do not write project files anywhere else in the
server's home or filesystem.

### Conda environments (servers)

Use only project-owned envs. Never install into, modify, or run jobs from a
pre-existing env unless the user explicitly designates it.

- **`dell`**: project env exists at `~/miniconda3/envs/safesteer`. Run with
  `~/miniconda3/bin/conda run -n safesteer ...`.
- **`hello`**: project env exists at `~/.workplace/conda/envs/safesteer`. Run
  with `~/.workplace/conda/envs/safesteer/bin/python ...`.

### Hard rule: shared machines — never disturb other users' jobs

These servers are SHARED. Before launching anything on a GPU you MUST:

1. **Check occupancy first**: run `nvidia-smi` (or `nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv`) and inspect running processes. On `dell`, check cards 6 and 7 specifically.
2. **Only use a card that is free** (no other user's process, ample free memory). If your allowed card(s) are busy, WAIT or ask — do NOT preempt, kill, or crowd another job.
3. **Pin your device explicitly** with `CUDA_VISIBLE_DEVICES` so a run can never spill onto a card you are not allowed to use (especially the 0–5 range on `dell`).
4. **Never** kill, suspend, throttle, or reduce the memory headroom of any process you did not start. Causing even slight interference with another user's run is not acceptable.
5. Size your own job (batch size, model size, parallelism) to fit comfortably within the free memory on your allowed card(s), leaving margin.

When in doubt about whether a card is free or whether an action could affect someone else, stop and ask the user rather than risk interference.

### Network proxy (temporary use only, leave no trace)

Outbound proxy for network calls: `http://10.77.0.11:10808`.

On the shared servers it MUST be used **only transiently, per-command** — never
persisted, never leaving a trace that could leak privacy:

- Apply it **per-command only**, e.g.
  `https_proxy=http://10.77.0.11:10808 http_proxy=http://10.77.0.11:10808 hf download ...`
- Do NOT `export` it into the shell session beyond the single command, and do NOT
  write it into `~/.bashrc`, `~/.profile`, `git config`, `~/.config`, pip/conda
  config, `/etc/environment`, or any dotfile on the server.
- After use, ensure no proxy setting lingers in shell history or config. Prefer
  one-shot inline env vars that vanish when the command returns.
- Rationale: these are shared machines; a persisted proxy address is both a
  privacy leak and a footprint other users should not inherit.

### Inter-machine file transfer (local <-> servers, server <-> server)

For moving models / data / results between machines on the LAN (`10.77.0.x`):

- **Prefer `rsync` over `scp`**: it resumes partial transfers (`--partial`/
  `--inplace`), skips already-identical files, and is far less painful when a
  16GB model transfer is interrupted.
- **Do NOT route LAN transfers through the proxy.** The proxy is for *outbound
  internet* only; LAN-to-LAN must go direct. Explicitly clear proxy vars for the
  transfer, e.g.:
  `env -i bash -c 'rsync -a --partial SRC/ host:DST/'` or `unset *_proxy` first.
  A proxy-polluted shell can silently tunnel a local copy through the proxy and
  make it crawl.
- Example (resume a model copy):
  `rsync -a --partial --inplace ~/.cache/huggingface/hub/<Model>/ hello@10.77.0.101:~/.workplace/models/<Model>/`
- Reuse what a machine already has before copying/redownloading (省流量): check
  each host's HF cache and `~/.workplace` first. Note a cached dir can be a
  gated *empty shell* (only metadata, ~KB) — verify real weight size, not just
  that the directory exists.

<!-- SPECKIT START -->
For additional context about technologies to be used, project structure,
shell commands, and other important information, read the current plan
<!-- SPECKIT END -->
