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

脚本已放在 `scripts/test_load.py`。当前版本会优先使用本地 `--local-dir`
模型目录,并在测试结束后做显式清理:

- forward 只跑 `model.model(...)`,避免额外生成 `lm_head` logits。
- `use_cache=False`,避免保留 KV cache。
- `finally` 中移除 hook,断开模型/输入/输出引用,并调用 `gc.collect()`、
  `malloc_trim()`、`torch.cuda.empty_cache()`、`torch.cuda.ipc_collect()`。

跑:

```bash
python scripts/test_load.py qwen3
python scripts/test_load.py llama       # 等 Llama 审批后
python scripts/test_load.py gemma       # 等 Gemma 接受 license 后
```

**预期输出**(以 Qwen3 为例):

```
=== Loading Qwen/Qwen3-8B ===
  Source: local dir ~/.cache/huggingface/hub/Qwen3-8B
  Layers: 36 (expected 36)
  Hidden: 4096 (expected 4096)
  Hooked layer 18, activation shape: (1, <seq_len>, 4096)
  Mean-token activation L2 norm: <数字>
  GPU peak mem used: ~17 GB
  OK
  GPU mem after cleanup: allocated 0.00 GB, reserved 0.00 GB
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

### Phase 1.5 — 子集处理与归一(`scripts/prepare_data.py`)

> 目标:从 HF 缓存**只读**取数,按论文设置抽**最小确定性子集**,归一成统一 schema 喂下游。`data/` 不入库(`.gitignore` 忽略 `data/`,仅保留 `manifest.json`),靠脚本 + 固定 seed 随时重建。

**防"污染"怎么做**:不整份 copy 原始数据集(BeaverTails 330k/Alpaca 52K 又大又冗余)。脚本只用只读模式打开 HF 缓存、只往 `data/` 写,原始缓存一个字节都不动 —— 这就避免了污染,又比整份 copy 小得多、可版本化(靠 manifest+seed,不是靠拷数据)。

**论文怎么用这三个数据集**(§4.1 + Table 1/2,供对照):

| 数据集 | 论文角色 | 论文规模 | 本复现子集(可调) |
|---|---|---|---|
| CatQA `declare-lab/CategoricalHarmfulQA` | `D^c_unsafe` 有害侧(仅 unsafe,无 twin) | 11 类 | 取论文 3 类,每类全量 50 |
| BeaverTails `PKU-Alignment/BeaverTails` | 有害侧 + 自带 `safe` 当通用 safe | 提取 1500/类(train)、生成 150-200/类(test) | 取论文 3 类,每类有害 ≤200 + 通用 safe 200(走 30k train) |
| Alpaca `tatsu-lab/alpaca` | `D^generic_safe` 通用无害池 | 全量 | 抽 500 |

> 子集规模(200/类)比论文提取用的 1500/类小,只为 smoke 快速跑通;复现论文数字时把 `scripts/prepare_data.py` 顶部的 `N_HARMFUL_PER_CAT` 调大即可。manifest 会记录实际抽到的条数。

**关键:两个数据集各用各的 3 类,不跨集映射**(忠于论文,避免张冠李戴)。论文的"3 代表类别"在两边本就不是同一组:

| | CatQA(Table 1 / 2 右) | BeaverTails(Table 2 左) |
|---|---|---|
| 类 1 | `Adult Content` | `child_abuse` |
| 类 2 | `Hate/Harass/Violence` | `hate_speech,offensive_language` |
| 类 3 | `Physical Harm` | `terrorism,organized_crime` |

> CatQA 没有 terrorism 类。**不要**拿 CatQA 的 `Physical Harm` 去顶 BeaverTails 的 `terrorism` —— 那是两个不同有害主题,混为一类会让转向向量 `ω_c` 语义错配、跨集对照失真。脚本里 `category` 字段直接用各数据集原生类名,向量永远在"同数据集+同类别"内算。唯一天然重叠的是 **Hate** 类(两边都有),这一类可做跨集对照。

**归一 schema**(每行一条 JSON):

```json
{"id":"catqa/adult_content/0007","text":"<prompt/question>","response":null,
 "category":"adult_content","role":"harmful","source":"declare-lab/CategoricalHarmfulQA"}
```

- `text` = prompt / 有害问题;`response` = 回答(BeaverTails/Alpaca 有,CatQA 为 `null`)。论文 §3.1 说 activation 输入可以是「prompt 单独」或「prompt+response 配对」,所以两者都留,下游自己选。
- `role ∈ {harmful, generic_safe}`(`paired_safe` 暂无 —— CatQA twin 未公开,走 Alpaca generic safe)。

**产物布局**:

```
data/                              # 整个 gitignore
├── processed/
│   ├── catqa/        adult_content.harmful.jsonl  hate_harass_violence.harmful.jsonl  physical_harm.harmful.jsonl
│   ├── beavertails/  <slug>.harmful.jsonl × 3 + generic_safe.jsonl
│   └── alpaca/       generic_safe.jsonl
└── manifest.json                  # ✅ 入库:纯元数据,无有害文本
```

**操作(你来跑)**:

```bash
# 前置:Alpaca 是 parquet,只需装 pyarrow(不装整个 datasets 库)
conda run -n safesteer313 pip install pyarrow

# 一条命令确定性重建(seed=0);也可 --only catqa / --only beavertails / --only alpaca 单独跑
conda run -n safesteer313 python scripts/prepare_data.py
```

**实跑输出**(seed=0,30k train;条数因数据分布而异,以脚本打印为准):

```
HF datasets root: /home/lihan/.cache/huggingface/datasets
Output root:      .../data/processed
Seed: 0

  CatQA  adult_content                harmful:   50 (avail 50)
  CatQA  hate_harass_violence         harmful:   46 (avail 46)
  CatQA  physical_harm                harmful:   50 (avail 50)
  Beaver child_abuse                  harmful:   81 (avail 81)
  Beaver terrorism_organized_crime    harmful:  188 (avail 188)
  Beaver hate_speech_offensive        harmful:  200 (avail 1226)
  Beaver (safe pool)                  generic_safe:  200 (avail 4555)
  Alpaca (instructions)               generic_safe:  500 (avail 52002)

Wrote manifest: data/manifest.json (8 artifacts)
```

**manifest 字段**:`source`(HF id)/`source_file`(实际读的文件)/`category_slug`+`category_native`(归一名↔原生名)/`role`/`n_requested`/`n_available`/`n_actual`(实际抽到) —— 便于以后核对子集与论文规模差异。

**重建说明**:`data/` 不入库,改动 seed 或子集大小后重跑上面那条命令即可;manifest 入库,记录了每次抽样的参数与实际条数。

**已知数据现实**(看到非整数条数不要以为脚本出错):

- **CatQA `hate_harass_violence` 只有 46 条**(非 50):该数据集 `Hate/Harass/Violence` 类原生去重后就是 46 条,manifest `n_available=46` 如实记录。
- **BeaverTails `child_abuse` 只有 81 条,且语义会"漂移"**:BeaverTails 是**多标签**标注(一条 prompt 的 `category` 是 14 个 bool,可同时为真),`child_abuse` 在 30k 子集里既少又混入相邻主题(如偷拍/隐私)。这会稀释该类转向向量 `ω` 的纯度。
  → **复现论文数字时**改用 **330k 全量**(`round0/330k/train.jsonl.xz`)并把 `N_HARMFUL_PER_CAT` 调到论文的 **1500**,给稀疏类凑够干净样本。smoke 阶段用 30k 子集够跑通流程。

---

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

## 6.5 实验流程 ↔ 代码数据流(主笔记)

> 这是本仓的"主笔记"板块:把**理论步骤 → 论文公式 → 代码位置 → 数据结构 in→out**钉在一起,每完成一个模块就增量补一行,让笔记和代码同步生长。✅=已实现,⬜=待实现。

| # | 理论步骤 | 论文 | 代码 | 数据结构 in → out | 状态 |
|---|---|---|---|---|---|
| 1 | 取数据、按数据集×类别组装"有害侧 + 通用安全侧" | §4.1 | `src/data.py` | `data/processed/*.jsonl` → `CategorySplit{harmful, generic_safe}` | ✅ |
| 2 | 抽 attention 子层 activation,跨 token 求均值,每输入每层一个向量 | Eq.1 的 `act(x)` | `src/hooks.py` | `texts:list[str]` → `{layer: (N, hidden)}` | ✅ |
| 3 | 差分均值得转向向量 ω = mean(safe) − mean(unsafe);可选 L2 范数剪枝 | Eq.1 + §3.3 | `src/vectors.py` | `safe (N,h)`, `unsafe (N,h)` → `{layer: (hidden,)}` | ⬜ |
| 4 | 推理时把 ω 按层注入 self-attention 输出:`h_l += m·ω_l` | Eq.2 | `src/steering.py` | `ω{layer:(hidden,)}` + `m` → 钩住前向、改写 `h_l` | ⬜ |
| 5 | 评测:steered vs naive 的 %UR(unsafe rate)下降 + 文本质量 | §4.4 | `scripts/eval_steering.py` | 生成文本 → `%UR↓`, helpfulness/coherence | ⬜ |

**端到端数据流管线**:

```
CategorySplit            {layer:(N,hidden)}        {layer:(hidden,)}      改写前向
harmful / generic_safe ──► [hooks] mean-pool ──► [vectors] 差分+剪枝 ──► [steering] h+=m·ω ──► [eval] %UR↓
   (src/data.py)            (src/hooks.py)         (src/vectors.py)        (src/steering.py)   (eval_steering.py)
```

**模块 2 关键点(`src/hooks.py`,对照 Eq.1)**:

- `act(x)` 的实现 = 在 `model.model.layers[l].self_attn` 注册 forward hook,抓子层输出 `(B, seq, hidden)`,再 `mean_pool` **按 attention_mask 跨真实 token 求均值**(排除 padding)→ `(B, hidden)`。
- **每样本保留一个向量**(`(N, hidden)`,不预先平均):§3.3 的 L2 剪枝和 t-SNE 解缠图都需要逐样本数据。
- 输入格式开关 `use_response`:默认 **prompt-only**(CatQA 无 response,保证有害/安全侧格式统一);置 True 则把 `{prompt, response}` 拼进去(论文 §3.1 允许两种)。
- transformers 5.x 健壮性:`self_attn` 输出可能是 tuple 或 tensor,统一取 hidden 并**断言最后一维 == hidden_size**,结构若变即刻报错而非污染向量。

**自测(不需 GPU/大模型)**:`python src/hooks.py` 用一个微型随机权重 Llama 验证 ① 输出形状 `(N, hidden)`;② `mean_pool` 对 padding 不敏感(padded == unpadded);③ `use_response` 开关确实改变 activation。

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
- [x] `python scripts/test_load.py qwen3` pass
- [x] `python scripts/test_load.py llama` pass
- [x] `python scripts/test_load.py gemma` pass

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
