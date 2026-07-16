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

## Runtime Environment

When this repo is nested under `science_ai`, use the enclosing workspace as the
shared authority:

- Follow `../../experiments/runbooks/research-claims.md` for
  reproduction-derived hypotheses, benchmark-gated follow-up ideas, and paper
  readiness.
- Follow `../../experiments/runbooks/research-notes.md` for literature,
  source/code lookup, and novelty checks.
- Follow `../../experiments/runbooks/research-compute.md` plus the active
  research profile for paths, remotes, GPU permissions, environments,
  proxy/download routing, transfers, and judge APIs.

The former shared science skill shims are retired. Use only project-owned
environments, never install into `base`, and ask when a required environment
path is not documented or conflicts with live state.

<!-- SPECKIT START -->
For additional context about technologies to be used, project structure,
shell commands, and other important information, read the current plan
<!-- SPECKIT END -->
