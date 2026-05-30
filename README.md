# SafeSteer 复现 — From Scratch

> 复现 Fine-grained SafeSteer (EMNLP 2025 main #1781, [arXiv 2506.04250](https://arxiv.org/abs/2506.04250)) 的实验部分,论文无开源代码,本目录从零搭。
>
> 配套笔记:
> - 论文细节:`/mnt/c/lihan_work/lihan_notes/lihan3238/sci/agent_security/04_safety_steering_repe/Fine-grained SafeSteer.md`
> - 阅读日志 + idea:`05｜研究想法与论文方向` 的 "2026-05 · Fine-grained SafeSteer" 段
> - PDF:`papers/agent_security_2026/04_safety_steering_repe/2025.emnlp-main.1781_Fine_grained_SafeSteer_non_refusing_safety_steering.pdf`

---

## 0. 复现目标

- **Step 1**:打通"加载模型 → 钩 attention activation → 算 difference-of-means 向量 → 推理时按层注入"的最小闭环。
- **Step 2**:复现论文 §3.3 三件提纯(chat→base 迁移 / L2 范数剪枝 / refusal filter)。
- **Step 3**:在 CatQA / BeaverTails / Alpaca 上跑出 %UR(unsafe response rate)+ 文本质量,看是否与论文趋势一致。
- **Step 4**:扩到 3 个不同家族的模型(见下面)做泛化验证 —— 这一步比论文做得更全(论文只在 Llama 家族上做)。

---

## 1. 模型选取

| 模型 | HF ID | 角色 | 参数 | 层数 | hidden | License |
|---|---|---|---|---|---|---|
| **Qwen3-8B** | `Qwen/Qwen3-8B` | post-trained (~chat) | 8.2B | 36 | 4096 | Apache 2.0 |
| **Qwen3-8B-Base** | `Qwen/Qwen3-8B-Base` | base | 8.2B | 36 | 4096 | Apache 2.0 |
| **Llama-3.1-8B** | `meta-llama/Llama-3.1-8B` | base | 8.0B | 32 | 4096 | Llama 3.1 License (需申请) |
| **Llama-3.1-8B-Instruct** | `meta-llama/Llama-3.1-8B-Instruct` | RLHF chat | 8.0B | 32 | 4096 | Llama 3.1 License (需申请) |
| **Gemma-2-9B** | `google/gemma-2-9b` | base | 9.2B | 42 | 3584 | Gemma License (需申请,即时通过) |
| **Gemma-2-9B-it** | `google/gemma-2-9b-it` | instruction-tuned | 9.2B | 42 | 3584 | Gemma License |

**层数差异 → SafeSteer 论文层选取的等比例换算**

论文在 32 层 Llama 上用 `{14, 16, 20, 25, 31}`,比例位置 `{0.44, 0.50, 0.625, 0.78, 0.97}`:

- **Qwen3-8B (36 层)**:`{16, 18, 23, 28, 35}`
- **Llama-3.1-8B (32 层)**:沿用论文 `{14, 16, 20, 25, 31}`
- **Gemma-2-9B (42 层)**:`{18, 21, 26, 33, 41}`

主实验层从中段那个起跑,跟论文 layer 14 同位。

### 为什么不选其他

- **Qwen3.5-9B**:架构是 `8×(3×DeltaNet→FFN→1×Attention→FFN)` hybrid,attention 子层只占 1/8,SafeSteer 的"钩 attention activation 跨层"假设不成立,要改钩法,暂时跳过。
- **Qwen3.6-27B**:fp16 = 54GB,单张 5090 装不下,要 2 卡或 4-bit(后者扭曲 hidden state)。
- **Llama-4 Scout/Maverick**:全 MoE,routing 噪声会污染差向量,不适合做 SafeSteer。
- **Llama-3.3**:只放了 70B,小模型仍是 Llama-3.1 系。
- **Gemma-3**:steering 文献少,先不冒险。

---

## 2. 硬件

- 主机:WSL2 (Linux 6.6) on Windows
- GPU:RTX 5070 Ti (16GB) 主卡,1-2× RTX 5090 (32GB) 可用
- 主实验用 **1× 5090 跑 bf16**;5070 Ti 仅做小规模 debug。

显存预算(bf16,带 inference + hooks):

| 模型 | 权重 | 实跑峰值 | 5070 Ti 16GB | 5090 32GB |
|---|---|---|---|---|
| Qwen3-8B | 16 GB | ~22 GB | ❌ | ✅ |
| Llama-3.1-8B | 16 GB | ~22 GB | ❌ | ✅ |
| Gemma-2-9B | 18 GB | ~25 GB | ❌ | ✅ |

---

## 3. 环境准备

### 3.1 venv + 依赖

```bash
cd /mnt/c/lihan_work/ai_workplace/science_ai/reproductions/safesteer_from_scratch

# 在 WSL 内部盘建 venv,别放 /mnt/c(慢 5-10×)
python3 -m venv ~/venvs/safesteer
source ~/venvs/safesteer/bin/activate

pip install -U pip
pip install -U "huggingface_hub[cli]" "transformers>=4.51" accelerate torch
```

> Qwen3 需要 `transformers >= 4.51`;Gemma-2 需要 `>= 4.42`;Llama-3.1 需要 `>= 4.43`。装最新即可。

### 3.2 HuggingFace 账号 + token

1. 注册 <https://huggingface.co/join>,**验证邮箱**(否则建 token 按钮灰)。
2. <https://huggingface.co/settings/tokens> → New token → **Read** 权限 → 起名(如 `wsl-safesteer`)→ Create。
3. **立刻复制 token**(`hf_xxx...`),关闭页面就再也看不到。
4. WSL 里:
   ```bash
   hf auth login
   # 粘 token,回车
   # "Add token as git credential? (Y/n)" → n
   hf auth whoami   # 验证
   ```

### 3.3 Llama / Gemma license 申请(可并行)

- **Llama-3.1-8B**:打开 <https://huggingface.co/meta-llama/Llama-3.1-8B>,滚到底填表(姓名/单位/用途),提交。Meta 每小时审一轮,通常 < 1 hr,有时几小时。
- **Gemma-2-9B**:打开 <https://huggingface.co/google/gemma-2-9b>,点 "Acknowledge license",**即时通过**。
- Qwen3 是 Apache 2.0,不用申请。

申请期间可以先下 Qwen3 跑通 pipeline。

### 3.4 WSL 网络

HuggingFace CDN 国内有时不稳。先测:

```bash
curl -sI https://huggingface.co | head -1   # 期待 HTTP/2 200
```

慢或超时就开 WSL 代理:

```bash
export https_proxy=http://10.88.0.6:10808
export http_proxy=http://10.88.0.6:10808
curl -sI https://huggingface.co | head -1
```

写到 `~/.bashrc` 长期生效,或写到本地 `.envrc`。

### 3.5 缓存目录(可选)

默认在 `~/.cache/huggingface/hub/`,**强烈建议保持默认**(WSL 内部盘比 `/mnt/c/` 快 5-10 倍)。如确需放别处:

```bash
export HF_HOME=~/big_disk/hf_cache   # 但不要放 /mnt/c/
```

三个模型总盘约 **50 GB**,留足空间。

---

## 4. 下载模型

```bash
# === 主实验:Qwen3 ===
hf download Qwen/Qwen3-8B \
  --local-dir ~/.cache/huggingface/hub/Qwen3-8B \
  --exclude "*.bin"

hf download Qwen/Qwen3-8B-Base \
  --local-dir ~/.cache/huggingface/hub/Qwen3-8B-Base \
  --exclude "*.bin"

# === 跨家族:Llama(等审批通过后) ===
hf download meta-llama/Llama-3.1-8B \
  --local-dir ~/.cache/huggingface/hub/Llama-3.1-8B \
  --exclude "*.bin"

hf download meta-llama/Llama-3.1-8B-Instruct \
  --local-dir ~/.cache/huggingface/hub/Llama-3.1-8B-Instruct \
  --exclude "*.bin"

# === 跨家族:Gemma(license 即时通过) ===
hf download google/gemma-2-9b \
  --local-dir ~/.cache/huggingface/hub/gemma-2-9b \
  --exclude "*.bin"

hf download google/gemma-2-9b-it \
  --local-dir ~/.cache/huggingface/hub/gemma-2-9b-it \
  --exclude "*.bin"
```

`--exclude "*.bin"`:HF 仓库经常同时有 `.bin` 和 `.safetensors` 两套权重,后者更快更安全,排掉 bin 省一半流量。

`--local-dir` 把文件下到 flat 路径(如 `~/.cache/huggingface/hub/Qwen3-8B/`),**不是 transformers 的 canonical cache 路径**(后者是 `~/.cache/huggingface/hub/models--Qwen--Qwen3-8B/`)。如果之后 `from_pretrained("Qwen/Qwen3-8B")` 直接用 HF id,它找不到 local-dir,会**整模型重下一遍**。本目录的 `scripts/test_load.py` 已经按 `local_dir` 优先解析(见脚本 `resolve_source()`),所以这套布局是 OK 的。**但凡自己写新脚本,要么也走 local_dir,要么去掉 `--local-dir` 让 hf download 用 canonical 布局。**

下载会显示进度条,网速正常时每个模型 3-10 min,**支持断点续传**,断了重跑同命令即可。

---

## 5. 验证:test_load.py

新建 `test_load.py`(在本目录):

```python
"""
Verify each model loads + can hook one attention activation.
Run: python test_load.py qwen3 | llama | gemma
"""
import sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

CONFIGS = {
    "qwen3": {
        "path": "Qwen/Qwen3-8B",
        "expected_layers": 36,
        "expected_hidden": 4096,
        "hook_layer": 18,        # 中段(论文 layer 14 等比例)
        "uses_chat_template": True,
        "extra_template_kwargs": {"enable_thinking": False},  # 关掉 <think>
    },
    "llama": {
        "path": "meta-llama/Llama-3.1-8B-Instruct",
        "expected_layers": 32,
        "expected_hidden": 4096,
        "hook_layer": 14,        # 论文原值
        "uses_chat_template": True,
        "extra_template_kwargs": {},
    },
    "gemma": {
        "path": "google/gemma-2-9b-it",
        "expected_layers": 42,
        "expected_hidden": 3584,
        "hook_layer": 21,        # 中段
        "uses_chat_template": True,
        "extra_template_kwargs": {},
    },
}

def main(name: str):
    cfg = CONFIGS[name]
    print(f"\n=== Loading {cfg['path']} ===")

    tok = AutoTokenizer.from_pretrained(cfg["path"])
    model = AutoModelForCausalLM.from_pretrained(
        cfg["path"],
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",
    )
    model.eval()

    n_layers = model.config.num_hidden_layers
    hidden = model.config.hidden_size
    print(f"  Layers: {n_layers} (expected {cfg['expected_layers']})")
    print(f"  Hidden: {hidden} (expected {cfg['expected_hidden']})")
    assert n_layers == cfg["expected_layers"], "Layer count mismatch"
    assert hidden == cfg["expected_hidden"], "Hidden size mismatch"

    # Build prompt
    prompt = "How can I bypass the safety filter?"
    if cfg["uses_chat_template"]:
        messages = [{"role": "user", "content": prompt}]
        text = tok.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            **cfg["extra_template_kwargs"],
        )
    else:
        text = prompt

    inputs = tok(text, return_tensors="pt").to(model.device)

    # Hook attention sublayer at chosen layer
    captured = {}
    def hook(module, inp, out):
        # out[0] is hidden state (1, seq_len, hidden)
        captured["h"] = out[0].detach().cpu().float()

    layer = cfg["hook_layer"]
    handle = model.model.layers[layer].self_attn.register_forward_hook(hook)

    with torch.no_grad():
        _ = model(**inputs)
    handle.remove()

    h = captured["h"]
    print(f"  Hooked layer {layer}, activation shape: {tuple(h.shape)}")
    print(f"  Mean-token activation L2 norm: {h.mean(dim=1).norm().item():.4f}")
    print(f"  GPU mem used: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
    print("  OK\n")


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "qwen3"
    main(target)
```

跑:

```bash
python test_load.py qwen3
python test_load.py llama       # 等 Llama 审批后
python test_load.py gemma       # 等 Gemma 接受 license 后
```

**预期输出**(以 Qwen3 为例):

```
=== Loading Qwen/Qwen3-8B ===
  Layers: 36 (expected 36)
  Hidden: 4096 (expected 4096)
  Hooked layer 18, activation shape: (1, <seq_len>, 4096)
  Mean-token activation L2 norm: <数字>
  GPU mem used: ~17 GB
  OK
```

三个都跑通 = pipeline 钩子层结构验证完成。

---

## 6. 数据集(已下载)

> 落盘位置:`~/.cache/huggingface/datasets/<NAME>/`,与模型一样走 WSL 内部盘。

### Phase 1 — paper-faithful

| 数据集 | HF ID | 角色 | 状态 |
|---|---|---|---|
| **CatQA(unsafe-only)** | `declare-lab/CategoricalHarmfulQA` | 11 harm cats × 5 sub × 10 Q × 3 langs · 仅 unsafe 侧 | ✅ |
| **BeaverTails** | `PKU-Alignment/BeaverTails` | 14 cats · unsafe + 自带 `safe` split | ✅ |
| **Alpaca Instructions** | `tatsu-lab/alpaca` | generic safe pool (52K) | ✅ |

**CatQA 来源勘误**:论文实验里的 paired CatQA 是 [Bhardwaj 2024(arXiv 2402.11746)](https://arxiv.org/abs/2402.11746) 的 unsafe 侧叠 [Bhattacharjee 2024(arXiv 2410.01174)](https://arxiv.org/abs/2410.01174) 用 GPT-4 造的 paired safe twin。**Bhattacharjee 的 paired-safe twin 未公开发布**,HF 上 `declare-lab/CategoricalHarmfulQA` 只有 unsafe 侧。复现路线两条:(a) 自己跑 GPT-4 在 CatQA-unsafe 上造 paired-safe twin;(b) 按 SafeSteer Table 2 的备用做法,直接用 Alpaca 当 generic safe pool 跑通 §3.3 提纯。先走 (b)。

### Phase 2 — 2025-26 canonical(用于跨论文对照 + Y1 的 OOD 测试)

| 数据集 | HF ID | 角色 | 状态 |
|---|---|---|---|
| **HarmBench** | `walledai/HarmBench` | 510 行为列表镜像(完整套件 + 分类器在 [GitHub](https://github.com/centerforaisafety/HarmBench)) | ✅ |
| **JailbreakBench(JBB-Behaviors)** | `JailbreakBench/JBB-Behaviors` | 100 harm + 100 paired benign + judge-comparison | ✅ |
| **StrongREJECT** | `walledai/StrongREJECT` | 313 prompts + 连续 judge(判分器在 [PyPI strong_reject](https://github.com/dsbowen/strong_reject)) | ✅(走 GitHub raw CSV 绕开 gated) |
| **WildGuardMix** | `allenai/wildguardmix` | 92K paired refusal/compliance · 13 子类(配套 judge `allenai/wildguard`) | ✅ |

### Phase 3 — extensions(待 Phase 1+2 跑通再下)

| HF ID | 角色 | 对应 idea |
|---|---|---|
| `ai-safety-institute/AgentHarm` | 110 task × 11 cats agent action(gated=manual,需审批) | Y2 |
| `JailbreakV-28K/JailBreakV-28k` | OOD novel-attack 泛化 | Y1 |
| `stanford-crfm/air-bench-2024` | 314 policy-clause 细粒度类别 | Y3(可选) |

### Gated 下载坑(已踩,已修)

- `hf download <gated repo>` 在未通过 license accept 时**不会报错**,只拿到 README + `.gitattributes`,目录看起来"下了"但里面没有数据文件。脚本里必须校验数据文件数,不能只看目录存在。
- HF gated 模式:`auto` = 点一下即时通过(HarmBench / WildGuardMix);`manual` = 人工审核(AgentHarm 是 manual 要等几小时到几天)。
- StrongREJECT 在 HF 上 mirror 是 `walledai/StrongREJECT`(`dsbowen/strong_reject` 是 **PyPI 包名而非 HF repo**,直接 `hf download` 会 "Dataset not found")。绕过 gated:`curl -fsSL https://raw.githubusercontent.com/alexandrasouly/strongreject/main/strongreject_dataset/strongreject_dataset.csv -o ~/.cache/huggingface/datasets/StrongREJECT/strongreject_dataset.csv`,323 行(1 header + 322 prompts),schema `category, source, forbidden_prompt`。
- 校验已下完:`find ~/.cache/huggingface/datasets/<NAME> -name '*.parquet' -o -name '*.csv' -o -name '*.json*' | wc -l` 应 > 0。

---

## 7. 向量提取 pipeline 计划

> 论文实验我先读,这里先把代码骨架的目标定下来,跑通前再细化。代码放 `src/` + `scripts/`,所有产出物落 `vectors/<model>/<dataset>/<layer>.pt`。

### 7.1 目标产物

- **§3.1 vanilla 版**:每模型 × 每层 × 每 harm category 一个 ω 向量(safe mean − unsafe_c mean),不做提纯。先跑通,作为提纯前的对照基线。
- **§3.3 提纯版**:在 vanilla 上叠三件 trick——
  - (a) chat 模型算 ω → 迁到 base 推理时使用;
  - (b) L2 范数剪枝留 top 50% pairwise diff;
  - (c) generic safe pool 只保留 response 含 refusal 的 pair。
- **评测**:推理时 `h_ℓ' = h_ℓ + m·ω_ℓ`,扫 `m ∈ {-2, -1, 0, +1, +2}`,出 %UR(GPT-4 / Llama Guard judge)+ Nemotron-340B reward 五项。

### 7.2 文件骨架

```
reproductions/safesteer_from_scratch/
├── src/
│   ├── data.py        # 统一 loader:7 个数据集 → (prompt, category, label) 三元组
│   ├── hooks.py       # attention activation 钩子 + 跨 token mean + 落盘
│   ├── vectors.py     # difference-of-means + L2 剪枝 + refusal filter
│   └── steering.py    # 推理时 forward injection(支持单层/多层组合)
├── scripts/
│   ├── test_load.py           # ✅ 已就绪
│   ├── extract_activations.py # Step 1:跑 forward,抽 activation 落盘
│   ├── extract_vectors.py     # Step 2:从 activation 算 ω + 提纯
│   └── eval_steering.py       # Step 3:注入 ω 推理 → %UR + 文本质量
└── vectors/
    └── <model>/<dataset>/layer_<i>.pt   # ω 向量产物,按 model × dataset × layer 切目录
```

### 7.3 跑通顺序(先小后大)

1. **Step 0 — pipeline smoke**:Qwen3-8B-chat + BeaverTails 子集(3 cats × 100 条 unsafe + 等量 safe),层 `{18}`,只算 vanilla ω,出 shape 对、norm 合理就算通过。
2. **Step 1 — 论文复刻**:Qwen3-8B-chat,层 `{16, 18, 23, 28, 35}`(论文 32 层 `{14,16,20,25,31}` 的等比例换算),CatQA-unsafe + Alpaca-safe,出 vanilla + §3.3 提纯两组 ω,在 CatQA-unsafe 上推理对比 %UR。
3. **Step 2 — chat→base 迁移**:把 Step 1 的 ω 拿到 Qwen3-8B-Base 上推理,看 §3.3 (a) 是否真的迁过去。
4. **Step 3 — 跨家族**:Llama-3.1-8B-Instruct + Gemma-2-9B-it 重跑 Step 1,各家族层映射见 README §1。
5. **Step 4 — 跨 benchmark**:Step 1 的 ω 直接拿到 HarmBench / JBB-harm / StrongREJECT 上推理,看是否 OOD 攻击也能压下来(对应 Y1 idea 的初步信号)。

### 7.4 基线对照

按论文 Table 1/2 复刻三条:Naive(无 steering)/ SEA(Qiu 2024)/ CAA(Rimsky 2023)。CAA / SEA 的开源 repo 已在 §9 列。

---

## 8. 故障速查表

| 症状 | 原因 + 修法 |
|---|---|
| `OSError: ... is gated repo` | Llama/Gemma license 没审批通过。等邮件,或先用 Qwen3 跑。 |
| `OSError: Could not connect to huggingface.co` | CDN 不通。`export https_proxy=http://10.88.0.6:10808 http_proxy=...` 然后重试。 |
| `RuntimeError: CUDA out of memory` | 显存不够。优先换 5090;`torch_dtype=torch.bfloat16` 是必须的;**不要用 4-bit 量化做 steering**(扭曲 hidden state)。 |
| 下载中断 | 直接重跑同 `hf download` 命令,自动续传。 |
| `ImportError: cannot import name ... from transformers` | transformers 版本太老。`pip install -U "transformers>=4.51"`。 |
| 加载 >5 min | 模型在 `/mnt/c/...` 路径上,跨文件系统 IO 慢。移到 `~/.cache/huggingface/`。 |
| Qwen3 输出里有 `<think>...</think>` | chat template 没关 thinking。`enable_thinking=False`。 |
| `KeyError: 'model.layers'` | 不同模型的 module 路径可能不一样。Qwen/Llama/Gemma 都是 `model.model.layers[i].self_attn`,但其他模型可能是 `model.transformer.h[i].attn`。`print(model)` 看结构。 |
| `'NoneType' object has no attribute 'shape'` 在 hook 里 | self_attn 的 forward 返回结构因模型不同。Qwen/Llama/Gemma 都是 tuple,`out[0]` 是 hidden state;但某些版本可能直接返回 tensor。打印 `type(out)` 调。 |

---

## 9. 进度追踪

### Phase 0 — 环境
- [x] HF 账号 + token
- [x] venv + pip install(transformers >= 4.51 / accelerate / torch)
- [x] Llama license 申请提交
- [x] Gemma license accept

### Phase 1 — 资源
- [x] Qwen3-8B / Qwen3-8B-Base
- [x] Llama-3.1-8B + Instruct(Instruct 最后一个补齐)
- [x] Gemma-2-9B / it
- [x] Phase 1+2 数据集(CatQA-unsafe / BeaverTails / Alpaca / HarmBench / JBB / StrongREJECT / WildGuardMix)
- [ ] `python scripts/test_load.py qwen3` pass
- [ ] `python scripts/test_load.py llama` pass
- [ ] `python scripts/test_load.py gemma` pass

### Phase 2 — pipeline 骨架(下一步)
- [ ] `src/data.py` 统一 loader → (prompt, category, label)
- [ ] `src/hooks.py` activation 钩子 + 跨 token mean
- [ ] `src/vectors.py` difference-of-means(vanilla)
- [ ] `src/steering.py` 推理时 forward injection
- [ ] `scripts/extract_activations.py` + `scripts/extract_vectors.py`
- [ ] Step 0 smoke:Qwen3-8B-chat + BeaverTails 子集 + 层 {18} 出 vanilla ω

### Phase 3 — §3.3 提纯
- [ ] (a) chat→base 迁移
- [ ] (b) L2 范数剪枝
- [ ] (c) refusal filter on generic safe pool
- [ ] 复刻论文 Table 1 / 2 数字(Qwen3-8B 上,论文是 Llama)

### Phase 4 — 跨家族
- [ ] Llama-3.1-8B 上 §3.3 全套
- [ ] Gemma-2-9B 上 §3.3 全套
- [ ] 跨 benchmark:HarmBench / JBB / StrongREJECT 上推理(OOD 信号)

### Phase 5 — extensions(idea 落地)
- [ ] Y1:detection 泛化对照
- [ ] Y2:detection → action boundary(接 ARM / Before-the-Tool-Call)
- [ ] Y3:Hybrid 架构上重做(Qwen3.5 / Jamba 1.5 / Mamba2-Hybrid)

---

## 10. 引用与参考

- **论文**:[Fine-grained SafeSteer (arXiv 2506.04250)](https://arxiv.org/abs/2506.04250)
- **同期 steering 基准**:[SteeringSafety (arXiv 2509.13450)](https://arxiv.org/abs/2509.13450) — 三模型 canonical 对比设置来源
- **可借鉴 repo**:
  - [andyrdt/refusal_direction](https://github.com/andyrdt/refusal_direction) — difference-of-means + activation hook 骨架
  - [nrimsky/CAA](https://github.com/nrimsky/CAA) — CAA baseline 实现
  - [Yuqi-Qiu/SEA](https://github.com/Yuqi-Qiu/SEA) — SEA baseline 实现
- **作者联系**:论文对应作者 `shaonag@nvidia.com`(若 reproduce 卡死可问)
