# SafeSteer 复现资源清单 (2026-06)

本文件详列了 `safesteer_from_scratch` 复现所依赖的模型与数据集资源状态。

## 1. 模型 — `~/.cache/huggingface/hub/`

HF cache 在 WSL 内部盘 `~/.cache/huggingface/`（不放 `/mnt/c`，跨文件系统 IO 慢 5-10×）。

| HF ID | 角色 | License | 体积 / 文件 | 状态 |
|---|---|---|---|---|
| `Qwen/Qwen3-8B` | post-trained chat | Apache 2.0 | 16 GB / 5 safetensors | ✅ |
| `Qwen/Qwen3-8B-Base` | base | Apache 2.0 | 16 GB / 5 safetensors | ✅ |
| `meta-llama/Llama-3.1-8B` | base | Llama 3.1（gated） | 18 GB / 3 safetensors | ✅ |
| `meta-llama/Llama-3.1-8B-Instruct` | RLHF chat | Llama 3.1（gated） | — | ⏳ 下载中 |
| `google/gemma-2-9b` | base | Gemma | 35 GB / 8 safetensors | ✅ |
| `google/gemma-2-9b-it` | instruction-tuned | Gemma | 18 GB / 4 safetensors | ✅ |

> **布局**：`hf download ... --local-dir ~/.cache/huggingface/hub/<NAME>`（flat layout），加载脚本走 local path 而非 hub id，避开 double-download 陷阱。详见 lihan-cards `hf-download-local-dir-double-download-trap`。

## 2. 数据集 — `~/.cache/huggingface/datasets/`

### Phase 1 — paper-faithful

| HF ID | 出处 | 角色 | 体积 / 文件 | 状态 |
|---|---|---|---|---|
| `declare-lab/CategoricalHarmfulQA` | [Bhardwaj 2024 (arXiv 2402.11746)](https://arxiv.org/abs/2402.11746) | CatQA · 11 cats × 5 sub × 10 Q × 3 langs · unsafe-only | 400 KB · 3 JSON | ✅ |
| `PKU-Alignment/BeaverTails` | [Ji NeurIPS 2023 (arXiv 2307.04657)](https://arxiv.org/abs/2307.04657) | 14 cats unsafe + safe split（round0/30k + 330k） | 38 MB · jsonl.gz/xz | ✅ |
| `tatsu-lab/alpaca` | [Taori 2023](https://github.com/tatsu-lab/stanford_alpaca) | generic safe pool · 52K | 24 MB · parquet | ✅ |

### Phase 2 — 2025-26 canonical

| HF ID | 出处 | 角色 | 体积 / 文件 | 状态 |
|---|---|---|---|---|
| `walledai/HarmBench` | [Mazeika ICML 2024 (arXiv 2402.04249)](https://arxiv.org/abs/2402.04249) | 510 行为列表镜像 | 192 KB · 3 parquet | ✅ |
| `JailbreakBench/JBB-Behaviors` | [Chao NeurIPS D&B 2024 (arXiv 2404.01318)](https://arxiv.org/abs/2404.01318) | 100 harm + 100 paired benign | 1.1 MB · 3 CSV | ✅ |
| `walledai/StrongREJECT` | [Souly NeurIPS 2024 (arXiv 2402.10260)](https://arxiv.org/abs/2402.10260) | 313 prompts + 连续 judge | 40 KB · 1 CSV | ✅ 走 GitHub raw 绕过 gated |
| `allenai/wildguardmix` | [Han NeurIPS D&B 2024 (arXiv 2406.18495)](https://arxiv.org/abs/2406.18495) | 92K paired refusal/compliance | 54 MB · 2 parquet | ✅ |

### Phase 3 — extensions (待 Phase 1+2 跑通再下)

| HF ID | 角色 | 对应 idea |
|---|---|---|
| `ai-safety-institute/AgentHarm` | 110 task × 11 cats agent action（gated） | Y2 |
| `JailbreakV-28K/JailBreakV-28k` | OOD novel-attack 泛化 | Y1 |
| `stanford-crfm/air-bench-2024` | 314 policy-clause 细粒度类别 | Y3（可选） |

## 3. Gated 数据集的下载坑

- `hf download <gated repo>` 在未通过 license accept 时**不会报错**，只拿到能访问的文件（通常只剩 README + .gitattributes）。脚本里要校验 data 文件数而不是 dir 存在。
- HF gated 模式：`auto` = 即时通过（点一下就行），`manual` = 人工审核。
- 推荐校验：`find ~/.cache/huggingface/datasets/<NAME> -name '*.parquet' -o -name '*.csv' -o -name '*.json*' | wc -l` 应 > 0。
