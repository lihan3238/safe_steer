# SafeSteer 复现 — From Scratch

> 从零复现 Fine-grained SafeSteer (EMNLP 2025 main #1781, [arXiv 2506.04250](https://arxiv.org/abs/2506.04250))。论文无开源代码。本仓是**学习笔记 + 最小可跑复现**:把论文方法的**原理流**和代码的**数据流**两条线对齐,帮助理解 SafeSteer。
>
> 配套:论文细节笔记 `lihan_notes/.../04_safety_steering_repe/Fine-grained SafeSteer.md` · PDF 在 `papers/agent_security_2026/04_safety_steering_repe/`。

---

## 1. 原理流(论文方法)

SafeSteer 是**推理时**的细粒度安全转向:无需微调、无需梯度、无需对比数据对。一次前向即可把"有害"激活推向"安全"区。核心两步:

**Step 1 — 算每类的转向向量 ω(离线一次)**

对某个有害类别 `c`,取一批**有害文本**和一批**安全文本**,各自过模型、在某层 attention 子层取激活、**跨所有 token 求均值**,记作 `act(x)`。转向向量是两侧均值之差:

$$\omega^{c}_{\ell} \;=\; \frac{1}{|D^{c}_{safe}|}\sum_{x}\mathrm{act}(x^{safe}_{\ell}) \;-\; \frac{1}{|D^{c}_{unsafe}|}\sum_{x}\mathrm{act}(x^{unsafe}_{\ell}) \tag{Eq.1}$$

它指向"有害 → 安全"的方向。**可选 L2 剪枝**(§3.3):只保留 `‖差向量‖` 较大(信息量高)的那一半样本再求均值,过滤噪声。

**Step 2 — 推理时按层注入(每次生成)**

在选定层 `ℓ`,把 ω 乘以一个标量 `m`(multiplier,控制强度,可正可负)加到 self-attention 输出的**每个 token 位置**:

$$\theta^{attn}_{\ell} \;\leftarrow\; \theta^{attn}_{\ell} + m\cdot\omega^{c}_{\ell} \tag{Eq.2}$$

**提取与注入用同一层**(论文明确要求)。论文在 32 层 Llama 上主用 layer 14;跨模型时按比例位置换算(如 36 层 Qwen 用 18、42 层 Gemma 用 21)。

**目标(论文核心主张)**:steered 生成相比 naive,**%UR(unsafe response rate)显著下降**,同时**保住 helpfulness/coherence、避免一刀切拒答**。

---

## 2. 数据流(代码实现)

原理流的每一步对应一个代码单元。下表把**理论步骤 → 论文公式 → 代码 → 数据结构 in→out**钉在一起(✅ 已实现 / ⬜ 待实现):

| # | 理论步骤 | 论文 | 代码 | 数据结构 in → out | 状态 |
|---|---|---|---|---|---|
| 1 | 按数据集×类别取"有害侧 + 安全侧" | §4.1 | `src/data.py` | `data/processed/*.jsonl` → `CategorySplit{harmful, generic_safe}` | ✅ |
| 2 | 抽 attention 激活,跨 token 求均值 | Eq.1 `act(x)` | `src/hooks.py` | `texts:list[str]` → `{layer:(N,hidden)}` | ✅ |
| 3 | 差分得 ω(+L2 剪枝) | Eq.1 + §3.3 | `src/vectors.py` | `safe,(harmful) (N,h)` → `{layer:(hidden,)}` | ✅ |
| 4 | 推理时按层注入 `h+=m·ω` | Eq.2 | `src/steering.py` | `ω{layer:(hidden,)}` + `m` → 改写前向 | ⬜ |
| 5 | 评测 %UR↓ + 文本质量 | §4.4 | `scripts/eval_steering.py` | 生成文本 → `%UR↓`, help/coherence | ⬜ |

**端到端管线**:

```
CategorySplit            {layer:(N,hidden)}        {layer:(hidden,)}      改写前向        分类器
harmful / generic_safe ─► [hooks] mean-pool ─► [vectors] 差分+剪枝 ─► [steering] h+=m·ω ─► [eval] %UR↓
  (src/data.py)            (src/hooks.py)         (src/vectors.py)        (src/steering.py)  (eval_steering.py)
```

---

## 3. 关键文件导览

| 文件 | 职责 | 核心接口 | in → out |
|---|---|---|---|
| `src/data.py` | 统一 loader,manifest 驱动,只读不采样 | `load_category_split(dataset, category, safe_source)` | jsonl → `CategorySplit` |
| `src/hooks.py` | Eq.1 的 `act(x)`:钩激活 + 跨 token 均值 | `extract_activations(model, tok, texts, layers, use_response=)` | `texts` → `{layer:(N,hidden)}` |
| `src/vectors.py` | Eq.1 差分 + §3.3 剪枝 | `compute_steering_vectors(safe, harmful, prune=)` | 激活 → `SteeringVectors{layer:(hidden,)}` |
| `src/steering.py` ⬜ | Eq.2 注入(可加/可拆 hook) | _待实现_ | ω + m → steered 前向 |
| `scripts/test_load.py` | 验证模型加载 + 单层钩子结构 | `python scripts/test_load.py qwen3\|llama\|gemma` | — |
| `scripts/prepare_data.py` | 从 HF 缓存抽确定性子集(seed=0) | `python scripts/prepare_data.py` | HF 缓存 → `data/processed/` + `manifest.json` |

> `data/` 不入库(`.gitignore`),仅 `data/manifest.json` 入库;一条 `prepare_data.py` 即可确定性重建。每个 `src/*.py` 末尾有 `_self_test()`,`python src/<name>.py` 直接自测(纯 CPU,无需大模型)。

---

## 4. 各模块实现要点(对照论文)

**模块 1 `src/data.py`** — 论文 §4.1 的数据设定有两处必须理解:
- **CatQA 无 paired safe twin**:论文的 paired CatQA 用 GPT-4 造 safe 孪生(Bhattacharjee 2024),**未公开**;HF 上 `declare-lab/CategoricalHarmfulQA` 只有 unsafe 侧。故走论文 Table 2 备用路线:**用 Alpaca 当 generic safe**(`safe_source="alpaca"`)。
- **两数据集各用各的 3 类,不跨集映射**:CatQA 用 `Adult Content / Hate/Harass/Violence / Physical Harm`;BeaverTails 用 `child_abuse / hate_speech / terrorism`。强行对齐(如拿 CatQA 的 Physical Harm 顶 terrorism)会让 ω 语义错配。仅 Hate 类两边天然重叠,可跨集对照。
- **BeaverTails 多标签**:一条 prompt 可同时挂多个有害类,某些类样本少且主题漂移。论文靠**大样本(1500/类)+ L2 剪枝**冲淡噪声,而非清洗标签。

**模块 2 `src/hooks.py`** — Eq.1 的 `act(x)`:
- 在 `model.model.layers[ℓ].self_attn` 注册 forward hook 抓 `(B,seq,hidden)`,再按 `attention_mask` **跨真实 token 求均值**(排除 padding)→ `(B,hidden)`。
- **每样本保留一个向量**(不预先平均):§3.3 剪枝和 t-SNE 解缠图都需逐样本数据。
- **右填充**:提取场景必须右填充,否则 left-pad + RoPE 会让真实 token 位置随 batch 漂移、激活被污染(self-test 已验证批不变性)。
- `use_response` 开关:默认 prompt-only(CatQA 无 response,两侧格式统一);论文 §3.1 也允许 prompt+response。
- transformers 5.x:`self_attn` 输出可能是 tuple/tensor,统一取 hidden 并**断言末维==hidden_size**,结构变即报错而非静默污染。

**模块 3 `src/vectors.py`** — Eq.1 + §3.3:
- **vanilla**:`ω = mean(safe) − mean(harmful)`,每层一个 `(hidden,)`。
- **L2 剪枝复现取舍**:论文为**配对数据**写"pairwise 差→取范数中位数→留 top-50%→均值"。我们用**非配对**generic safe,改取 `d_i = mu_safe − harmful_i` → 留 `‖d_i‖ > median` 的一半 → 均值。依据:① 与论文 rationale 吻合(差异小=低信息=丢弃);② 自洽(不剪枝时全体均值 == vanilla ω);③ 确定性。已在 docstring 标为 documented deviation。

---

## 5. 快速上手

```bash
conda activate safesteer313                 # py3.13 / torch2.12+cu130 / transformers5.9
pip install pyarrow                          # 仅为读 Alpaca parquet(首次)

python scripts/prepare_data.py               # 生成 data/processed/ + manifest(seed=0)
python src/data.py                           # 各模块自测,纯 CPU
python src/hooks.py
python src/vectors.py
python scripts/test_load.py qwen3            # 验证真实模型加载+钩子(需 GPU)
```

---

## 6. 参考

- **论文**:[Fine-grained SafeSteer (arXiv 2506.04250)](https://arxiv.org/abs/2506.04250)
- **可借鉴 repo**:[andyrdt/refusal_direction](https://github.com/andyrdt/refusal_direction)(diff-of-means + hook 骨架) · [nrimsky/CAA](https://github.com/nrimsky/CAA)(CAA 基线) · [Yuqi-Qiu/SEA](https://github.com/Yuqi-Qiu/SEA)(SEA 基线)
- **作者**:`shaonag@nvidia.com`
