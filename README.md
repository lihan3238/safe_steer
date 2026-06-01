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
| 4 | 推理时按层注入 `h+=m·ω` | Eq.2 | `src/steering.py` | `ω{layer:(hidden,)}` + `m` → 改写前向 | ✅ |
| 5 | 评测 %UR↓ + 文本质量 | §4.4 | `scripts/eval_steering.py` | 生成文本 → `%UR↓`, help/coherence | ⬜ |

**端到端管线**:

```
CategorySplit            {layer:(N,hidden)}        {layer:(hidden,)}      改写前向        分类器
harmful / generic_safe ─► [hooks] mean-pool ─► [vectors] 差分+剪枝 ─► [steering] h+=m·ω ─► [eval] %UR↓
  (src/data.py)            (src/hooks.py)         (src/vectors.py)        (src/steering.py)  (eval_steering.py)
```

---

## 3. 关键文件导览

> 数据形态贯穿全程:**文本 → 激活 `(N,hidden)` → 转向向量 ω `(hidden,)` → 改写前向**。下面按这条线讲每个文件。

**`src/models.py`** — 模型"户口本",单一配置来源。
- 记录每个模型(`qwen3`/`llama`/`gemma`)的 HF id、本地路径、层数、hidden、`paper_layers`(层选择见 §4)、聊天模板怪癖。
- `load_model(name)` → `(tok, model, cfg)`,bf16 加载并断言层/维度对得上(错 checkpoint 立刻报错)。`release_memory()` 收尾清显存。
- **存在意义**:test_load / extract / 将来的 eval 都从这里取配置,避免各抄一份导致漂移。

**`src/data.py`** — 数据"统一取货口",manifest 驱动、只读、不采样。
- `load_category_split(dataset, category, safe_source)` → `CategorySplit{harmful, generic_safe}`(一个类别的有害侧 + 安全侧,都是 `Example` 文本记录)。
- 采样/打乱只发生在 `prepare_data.py`(固定 seed),loader 只读,保证确定性来源唯一。
- `python src/data.py` 自测(纯 CPU)。

**`src/hooks.py`** — 论文 Eq.1 的 `act(x)`:把文本变成激活。
- `extract_activations(model, tok, texts, layers, use_response=, batch_size=, progress=)` → `{layer:(N,hidden)}`(CPU float32,顺序同输入)。
- 内部:在指定层 `self_attn` 挂 forward hook 抓 `(B,seq,hidden)` → 按 mask **跨真实 token 求均值** → `(B,hidden)`,逐 batch 堆叠。
- **每样本一个向量**(不预先平均),供 §3.3 剪枝 / t-SNE 用。`progress` 回调打印进度,无 tqdm 依赖。
- `python src/hooks.py` 自测(微型随机模型,纯 CPU)。

**`src/vectors.py`** — 论文 Eq.1 差分 + §3.3 剪枝:激活变转向向量。
- `compute_steering_vectors(safe_acts, harmful_acts, prune=)` → `SteeringVectors`,内含 `{layer:(hidden,)}`,带 `save/load`。
- vanilla:`ω = mean(safe) − mean(harmful)`;剪枝:只用高范数的一半样本(取舍见 §4)。
- `python src/vectors.py` 自测(合成张量精确对拍,纯 CPU)。

**`src/steering.py`** — 论文 Eq.2 注入:推理时把 ω 加进前向。
- `with SteeringHook(model, vectors, multiplier):` 块内生成即被 steer,退出自动拆除。
- 改 `self_attn` **输出** `h_l += m·ω_l`(每 token 位置),不动权重、可 toggle。
- `python src/steering.py` 自测(纯 CPU)。

**`scripts/prepare_data.py`** — 离线一次性:HF 缓存 → `data/processed/*.jsonl` + `manifest.json`(确定性子集,seed=0)。
**`scripts/test_load.py`** — 冒烟:`python scripts/test_load.py qwen3|llama|gemma`,验证模型加载 + 单层钩子结构。
**`scripts/extract_activations.py`** — 编排 data→hooks,真实模型激活落盘(详见下)。

### `extract_activations.py` 与它的 `.pt` 产物

```bash
python scripts/extract_activations.py --model qwen3 --dataset CatQA --category adult_content \
    [--layers 18 16 ...] [--safe-source alpaca|beavertails] [--use-response] \
    [--batch-size 8] [--limit N] [--dry-run]
```

做的事:取 `CategorySplit`(有害 + 安全文本)→ 加载模型 → 两侧各跑 `extract_activations` → 落盘。`--dry-run` 不加载模型,只验证数据/路径/层(零显存)。

产物 `activations/<model>/<dataset>/<category>/{harmful,safe}.pt`,每个是:

```python
{
  "acts": {18: Tensor(N, hidden), 16: ..., ...},  # 每层一个激活矩阵;float32/CPU
  "meta": {model, hub_id, dataset, category, safe_source, layers, use_response, role, n},
}
```

例:CatQA/adult_content 出 `harmful.pt`(acts 为 `{layer:(50,4096)}`,4MB)和 `safe.pt`(`{layer:(500,4096)}`,41MB)。

- **它是半成品**:文本已变激活,但还没算成 ω。下一步 `extract_vectors.py` 读这两个文件做 `mean(safe)−mean(harmful)`。
- **为何落盘**:加载大模型最贵(~13s/16GB);激活算一次,可反复试不同剪枝/multiplier,不必重跑模型。
- **`acts` 为何按层存 dict**:论文要多层对比,按层存最自然,直接喂 `vectors.py`。

> `data/`、`activations/`、`*.pt` 均不入库(`.gitignore`),靠脚本 + 固定 seed 重建;仅 `data/manifest.json` 入库。每个 `src/*.py` 末尾有 `_self_test()`,纯 CPU 可跑。

## 4. 各模块实现要点(对照论文)

**层选择(`paper_layers`,贯穿提取/注入)** — 为什么是这几层、要不要改:
- **"比例位置" = 层号 / 总层数**,即该层处于网络深度的百分之几。论文在 32 层 Llama 上扫了 `{14,16,20,25,31}`(位置 `≈{0.44,0.50,0.625,0.78,0.97}`,覆盖浅→深),主结果用 **layer 14**。
- 我按相同比例位置迁到 Qwen3(36 层)`≈{16,18,22,28,35}`、Gemma(42 层)`≈{18,21,26,33,41}`,中段那层放第一个当主力。依据:RepE 共识是 LLM **按深度分层**处理信息(浅层词法、中层语义),"相对深度"比"绝对层号"更可能可迁移。
- **但这只是起点猜测,不是真理**:不同模型语义涌现的层是经验事实、未必按比例分布;论文这 5 层只在 Llama 上验证过。**Qwen3 的最佳层应由实验定**——等 `eval_steering.py` 跑通后扫层、看哪层 %UR 降最多。所以 `--layers` 是命令行可覆盖参数,不必信手算值(差 1 层如 22/23 无关紧要)。

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

**模块 4 `src/steering.py`** — Eq.2 注入:
- **改输出而非改权重**:Eq.2 字面是改 `θ^attn`(权重),我们用 forward hook 改 self_attn **输出** `h_l += m·ω_l`(每 token 位置加同一向量)。两者等价,但 hook 不动权重、退出 `with` 即刻拆除(Principle V:可 toggle)。是 refusal_direction/CAA 的标准做法。
- **同一子层**:注入点与 `hooks.py` 提取点是同一个 `self_attn`(论文要求 same layer)。
- **dtype/device 对齐**:ω 落盘是 fp32/CPU,注入时转成激活的 dtype+device(模型多为 bf16/cuda)。
- **m 可正可负**:复现论文 multiplier 范围;负 m = 反向转向。
- **self-test 要点**:`python src/steering.py` 验证 ① 注入改变输出;② 退出 `with` 后无残留(可拆);③ m=0 恒等;④ **线性性**——小 m 下 +m/−m 扰动余弦 = −1.0000、10× m → 10× 扰动幅度(直接证明 `h+=m·ω` 的线性结构);⑤ 多层注入。纯 CPU。

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
