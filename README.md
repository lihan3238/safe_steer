# SafeSteer 复现 — From Scratch

> 从零复现 Fine-grained SafeSteer (EMNLP 2025 main #1781, [arXiv 2506.04250](https://arxiv.org/abs/2506.04250))。论文无开源代码。本仓是**学习笔记 + 最小可跑复现**:把论文方法的**原理流**和代码的**数据流**两条线对齐,帮助理解 SafeSteer。
>
> **复现状态:初步复现完成** ✅ — 核心机制(Eq.1/Eq.2/§3.3)+ 安全轴 %UR + 质量轴 5 属性全部打通,并在 qwen3-1.7b-base 与 qwen3-8b-base 两个规模上验证了论文核心主张(steered %UR 下降、真转向、文本连贯;详见 §4 实测表)。
>
> **数值对齐论文的深挖结论**(在论文确切模型 Meta-Llama-3-8B + 确切设定上):论文头条 87.5→**0** 是 **self-steer + CatQA**(非 chat→base 迁移)。最关键的复现修正:**ω 必须在残差流(decoder layer 输出)上提取与注入,而非注意力子层输出**——后者范数小 ~15×,注回残差流仅占 ~3%、几乎不 steer。修正后 Meta-Llama-3-8B CatQA(adult_content, layer 14, m=1.5)**naive 98% → steered 56%(降 +42)**,文本基本连贯;旧的注意力输出版本同判官下只降 +2~+5。剩余到字面 →0 的差距是次要因素(GPT-4 判官更宽容 + 把强度推到轻度退化),详见 §5 第 8-11 条。Baseline 对比(CAA/SEA)本轮搁置(原因见 §4)。
>
> 配套:论文细节笔记 `lihan_notes/.../04_safety_steering_repe/Fine-grained SafeSteer.md` · PDF 在 `papers/agent_security_2026/04_safety_steering_repe/`。

---

## 1. 原理流(论文方法)

SafeSteer 是**推理时**的细粒度安全转向:无需微调、无需梯度、无需对比数据对。一次前向即可把"有害"激活推向"安全"区。核心两步:

**Step 1 — 算每类的转向向量 ω(离线一次)**

对某个有害类别 `c`,取一批**有害文本**和一批**安全文本**,各自过模型、在某层取激活(论文原文称 "attention activations",但忠实复现需取**残差流**,见下方实现关键 + §5)、**跨所有 token 求均值**,记作 `act(x)`。转向向量是两侧均值之差:

$$\omega^{c}_{\ell} \;=\; \frac{1}{|D^{c}_{safe}|}\sum_{x}\mathrm{act}(x^{safe}_{\ell}) \;-\; \frac{1}{|D^{c}_{unsafe}|}\sum_{x}\mathrm{act}(x^{unsafe}_{\ell}) \tag{Eq.1}$$

它指向"有害 → 安全"的方向。**可选 L2 剪枝**(§3.3):只保留 `‖差向量‖` 较大(信息量高)的那一半样本再求均值,过滤噪声。

**Step 2 — 推理时按层注入(每次生成)**

在选定层 `ℓ`,把 ω 乘以一个标量 `m`(multiplier,控制强度,可正可负)加到该层**残差流(decoder layer 输出)的每个 token 位置**:

$$\theta^{attn}_{\ell} \;\leftarrow\; \theta^{attn}_{\ell} + m\cdot\omega^{c}_{\ell} \tag{Eq.2}$$

> **实现关键(复现踩坑,见 §5)**:论文 Eq.1/Eq.2 字面说"attention activations / self-attention 输出",但按字面取**注意力子层输出**(范数小)算 ω、再注回残差流,扰动只占残差流 ~3%,几乎不 steer。论文真正引用的 CAA/refusal_direction 是在**残差流**上提取与注入。**ω 的提取层与注入层必须是残差流、且为同一层**(论文要求 same layer)。

**提取与注入用同一层**(论文明确要求)。论文在 32 层 Llama 上主用 layer 14;跨模型时按比例位置换算(如 36 层 Qwen 用 18、42 层 Gemma 用 21)。

**目标(论文核心主张)**:steered 生成相比 naive,**%UR(unsafe response rate)显著下降**,同时**保住 helpfulness/coherence、避免一刀切拒答**。

---

## 2. 数据流(代码实现)

原理流的每一步对应一个代码单元。下表把**理论步骤 → 论文公式 → 代码 → 数据结构 in→out**钉在一起(✅ 已实现 / ⬜ 待实现):

| # | 理论步骤 | 论文 | 代码 | 数据结构 in → out | 状态 |
|---|---|---|---|---|---|
| 1 | 按数据集×类别取"有害侧 + 安全侧" | §4.1 | `src/data.py` | `data/processed/*.jsonl` → `CategorySplit{harmful, generic_safe}` | ✅ |
| 2 | 抽残差流激活(decoder layer 输出),跨 token 求均值 | Eq.1 `act(x)` | `src/hooks.py` | `texts:list[str]` → `{layer:(N,hidden)}` | ✅ |
| 3 | 差分得 ω(+L2 剪枝) | Eq.1 + §3.3 | `src/vectors.py` | `safe,(harmful) (N,h)` → `{layer:(hidden,)}` | ✅ |
| 4 | 推理时按层注入 `h+=m·ω` | Eq.2 | `src/steering.py` | `ω{layer:(hidden,)}` + `m` → 改写前向 | ✅ |
| 5 | 安全轴:steered vs naive 的 %UR 下降 | §4.4 | `scripts/eval_steering.py` | 生成文本 → `%UR↓` (judge 二分类) | ✅ |
| 6 | 质量轴:5 个 HelpSteer 属性 | §4.4 Table 3 | `scripts/score_quality.py` | eval json → 5 属性分(naive vs steered) | ✅ |

**端到端管线**:

```
CategorySplit            {layer:(N,hidden)}        {layer:(hidden,)}      改写前向        安全:judge→%UR↓
harmful / generic_safe ─► [hooks] mean-pool ─► [vectors] 差分+剪枝 ─► [steering] h+=m·ω ─► [eval]
  (src/data.py)            (src/hooks.py)         (src/vectors.py)        (src/steering.py)  └► 质量:QRM→5属性
                                                                                              (eval_steering + score_quality)
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
- `is_refusal(text)` + `REFUSAL_MARKERS`:子串匹配判断回答是否为拒答(论文 §3.3 技巧二的 safe 集精炼用,供 `extract_activations.py --safe-filter refusal`)。
- 采样/打乱只发生在 `prepare_data.py`(固定 seed),loader 只读,保证确定性来源唯一。
- `python src/data.py` 自测(纯 CPU)。

**`src/hooks.py`** — 论文 Eq.1 的 `act(x)`:把文本变成激活。
- `extract_activations(model, tok, texts, layers, use_response=, batch_size=, progress=)` → `{layer:(N,hidden)}`(CPU float32,顺序同输入)。
- 内部:在指定**decoder layer**(残差流输出)挂 forward hook 抓 `(B,seq,hidden)` → 按 mask **跨真实 token 求均值** → `(B,hidden)`,逐 batch 堆叠。
- **每样本一个向量**(不预先平均),供 §3.3 剪枝 / t-SNE 用。`progress` 回调打印进度,无 tqdm 依赖。
- `python src/hooks.py` 自测(微型随机模型,纯 CPU)。

**`src/vectors.py`** — 论文 Eq.1 差分 + §3.3 剪枝:激活变转向向量。
- `compute_steering_vectors(safe_acts, harmful_acts, prune=)` → `SteeringVectors`,内含 `{layer:(hidden,)}`,带 `save/load`。
- vanilla:`ω = mean(safe) − mean(harmful)`;剪枝:只用高范数的一半样本(取舍见 §4)。
- `python src/vectors.py` 自测(合成张量精确对拍,纯 CPU)。

**`src/steering.py`** — 论文 Eq.2 注入:推理时把 ω 加进前向。
- `with SteeringHook(model, vectors, multiplier):` 块内生成即被 steer,退出自动拆除。
- 改 **decoder layer 输出(残差流)** `h_l += m·ω_l`(每 token 位置),不动权重、可 toggle。
- `python src/steering.py` 自测(纯 CPU)。

**`scripts/prepare_data.py`** — 离线一次性:HF 缓存 → `data/processed/*.jsonl` + `manifest.json`(确定性子集,seed=0)。
**`scripts/test_load.py`** — 冒烟:`python scripts/test_load.py qwen3|llama|gemma`,验证模型加载 + 单层钩子结构。
**`scripts/extract_activations.py`** — 编排 data→hooks,真实模型激活落盘(详见下)。
**`scripts/extract_vectors.py`** — 编排 hooks→vectors:读 `{harmful,safe}.pt` → `compute_steering_vectors` → 存 ω 到 `vectors/<m>/<ds>/<cat>/{vanilla,pruned}.pt`。纯 CPU。`--prune --report` 打印每层 ‖ω‖ 及 vanilla↔pruned 余弦。
**`scripts/eval_steering.py`** — 编排 vectors→steering→评测:naive vs steered 生成 → judge 判 unsafe → %UR 下降。`--multiplier 0.5 1 2` 扫强度(一次加载、naive 只算一次,自动报最佳 m);`--layers` 指定注入层(单层=该层,多层=同时注入)。`--vec-model` 解耦 ω 来源(跨模型迁移,如 chat→base)。**解码**:默认 greedy,base 模型务必加 `--do-sample [--temperature 0.7 --top-p 0.9]`(否则退化成复读,见 §5)。**判官**:`--classifier llm`(默认,Anthropic/OpenAI 双路,凭据走 `.env`)或 `keyword`(离线);`--judge-workers 16` 并行判官(网关慢时把判官阶段从 ~60min 压到 ~5min);判官调用带 5 次退避重试(一次超时不再作废整轮)。产物 `<variant>_sweep_*.json`(含每条 naive/steered 文本,供事后换判官重判)。
**`scripts/score_quality.py`** — 质量轴(论文 Table 3):读 eval json 的 naive/steered 文本 → QRM-Llama3.1-8B-v2 打 5 个 HelpSteer 属性分 → 写 `*.quality.json`。需 GPU 服务器(QRM ~16GB);`CUDA_VISIBLE_DEVICES` 锁定空闲卡。`--eval-json --qrm <path>`。产物含每个 multiplier 的 naive↔steered 属性均值与 Δ。

### `extract_activations.py` 与它的 `.pt` 产物

```bash
python scripts/extract_activations.py --model qwen3 --dataset CatQA --category adult_content \
    [--layers 18 16 ...] [--safe-source alpaca|beavertails] [--use-response] \
    [--safe-filter none|refusal] [--batch-size 8] [--limit N] [--dry-run]
```

做的事:取 `CategorySplit`(有害 + 安全文本)→ 加载模型 → 两侧各跑 `extract_activations` → 落盘。`--dry-run` 不加载模型,只验证数据/路径/层(零显存)。`--safe-filter refusal`(论文 §3.3 技巧二):把 safe 集筛成 response 含拒答的样本(`src/data.py:is_refusal` 子串匹配),令 ω 指向"拒答方向"而非"安全话题"(隐含开启 `--use-response`)。

产物 `activations/<model>/<dataset>/<category>/{harmful,safe}.pt`,每个是:

```python
{
  "acts": {18: Tensor(N, hidden), 16: ..., ...},  # 每层一个激活矩阵;float32/CPU
  "meta": {model, hub_id, dataset, category, safe_source, layers, use_response, safe_filter, role, n},
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
- 在 `model.model.layers[ℓ]`(**decoder layer 输出 = 残差流**)注册 forward hook 抓 `(B,seq,hidden)`,再按 `attention_mask` **跨真实 token 求均值**(排除 padding)→ `(B,hidden)`。**不是 `self_attn` 子层输出**——后者范数小 ~15×,据其算的 ω 注回残差流几乎无效(复现关键,见 §5)。
- **每样本保留一个向量**(不预先平均):§3.3 剪枝和 t-SNE 解缠图都需逐样本数据。
- **右填充**:提取场景必须右填充,否则 left-pad + RoPE 会让真实 token 位置随 batch 漂移、激活被污染(self-test 已验证批不变性)。
- `use_response` 开关:默认 prompt-only(CatQA 无 response,两侧格式统一);论文 §3.1 也允许 prompt+response。
- transformers 5.x:decoder layer 输出可能是 tuple/tensor,统一取 hidden 并**断言末维==hidden_size**,结构变即报错而非静默污染。

**模块 3 `src/vectors.py`** — Eq.1 + §3.3:
- **vanilla**:`ω = mean(safe) − mean(harmful)`,每层一个 `(hidden,)`。
- **L2 剪枝复现取舍**:论文为**配对数据**写"pairwise 差→取范数中位数→留 top-50%→均值"。我们用**非配对**generic safe,改取 `d_i = mu_safe − harmful_i` → 留 `‖d_i‖ > median` 的一半 → 均值。依据:① 与论文 rationale 吻合(差异小=低信息=丢弃);② 自洽(不剪枝时全体均值 == vanilla ω);③ 确定性。已在 docstring 标为 documented deviation。

**模块 4 `src/steering.py`** — Eq.2 注入:
- **注入残差流、不动权重**:Eq.2 字面是改 `θ^attn`(权重),我们用 forward hook 把 `m·ω_l` 加到 **decoder layer 输出(残差流)** 的每个 token 位置 `h_l += m·ω_l`。这是 refusal_direction/CAA 的标准做法;hook 不动权重、退出 `with` 即刻拆除(Principle V:可 toggle)。**注入残差流(范数大、行为线性编码处)才真正 steer**——加到 `self_attn` 子层输出(范数小 ~15×)几乎无效(见 §5)。
- **同一层**:注入点与 `hooks.py` 提取点是同一个 decoder layer 残差流(论文要求 same layer)。
- **dtype/device 对齐**:ω 落盘是 fp32/CPU,注入时转成激活的 dtype+device(模型多为 bf16/cuda)。
- **m 可正可负**:复现论文 multiplier 范围;负 m = 反向转向。
- **self-test 要点**:`python src/steering.py` 验证 ① 注入改变输出;② 退出 `with` 后无残留(可拆);③ m=0 恒等;④ **线性性**——小 m 下 +m/−m 扰动余弦 = −1.0000、10× m → 10× 扰动幅度(直接证明 `h+=m·ω` 的线性结构);⑤ 多层注入。纯 CPU。

**模块 5 `scripts/eval_steering.py`** — §4.4 安全轴,验证论文核心主张:
- **安全轴 %UR**:naive(无 steering)vs steered(注入 m·ω)各生成,judge 判 unsafe,看 %UR 下降。质量轴在模块 6 单独做(论文明确**不用通用 LLM 评质量** —— judge 偏好"给方案"胜过"拒答",会低估安全回答,故质量轴用 reward 模型)。
- **judge = LLM 二分类**:论文用 GPT-4;我们用 Claude(被测是 Qwen,无自评偏差)。`--judge-protocol` 双路(anthropic/openai),凭据走 `.env`。这是与论文的 documented deviation(judge 模型不同,但方法论一致:LLM 仅做安全二分类)。
- **multiplier 扫描**:`--multiplier 0.5 1 2 ...` 一次加载扫多值,naive 只算一次,自动报告最佳 m。

**关键实证发现(样本量 → ω 尺度 → 有效 m)**:
- 论文 Eq.2 **不归一化** ω(实测确认),且论文用 **1500 样本/类**、m≈0.5–1。
- 我们先用 CatQA adult_content **仅 50 样本**算 ω → 范数失控(深层 ‖ω‖≈99 ≫ 激活尺度 ~13)→ 逼得 **m=4** 才见效、且文本崩坏(复读机乱码,judge 误判 safe)。
- 改用 BeaverTails hate_speech **1226 样本** 重算 → ω 范数回落(深层 18、浅层 2.3,接近激活尺度)→ **m=1.0 即最佳**(%UR 40%→10%,真转向、文本连贯)、回到论文 multiplier 范围。
- **结论**:mean-difference 的 ω 尺度对**样本量**敏感;样本太少 → ω 噪声大、范数虚高 → 需异常大的 m 补偿,反而破坏文本。**复现论文趋势必须用足够样本(贴近 1500/类)**,这也是 §3.3 剪枝去噪的同源动机。

**两个规模的实测验证(均 BeaverTails hate_speech,pruned,m=1.0 最佳)**:

| 模型 | naive %UR | steered %UR | drop | 质量轴(QRM 5 属性) | 备注 |
|---|---|---|---|---|---|
| **qwen3-1.7b-base** | 40% | **10%** | +30 | help/corr/cohe 不降反升(Δ>0) | limit=10,降幅大、零质量代价 |
| **qwen3-8b-base** | 50% | **40%** | +10 | help/corr 轻微降(Δ≈−0.03)、cohe 基本平 | limit=10,方向对但温和、有轻微 trade-off |

- 两个规模**方向一致**(m=1.0 时 %UR 下降、抽检确认真转向、文本连贯),核心主张复现成立。
- 8B 降幅(+10)比 1.7B(+30)温和,且质量出现**轻微 trade-off**(help/corr 降 ~0.03):① `limit=10` 统计噪声大(50%/40% 仅差 1 个样本);② 8B base 激活更扩散、ω 范数更小、更难 steer。这与论文一致 —— 论文也指出维持质量在某些模型上更难、安全与质量本是 trade-off。**要论文级强结果需放大 test 规模(论文用 150/类)**,这是后续工作,不影响"机制有效"的结论。

**chat→base 迁移(论文 §3.3 技巧一)— 复现了机制,但 Qwen3 上效果为负**:
- 论文做法:从 **chat 模型**提 ω(safe/unsafe 概念更分离、方向更纯)→ 注入 **base 模型**生成,声称比 base 自提取质量更高、拒答更少。我们用 `eval_steering.py --vec-model` 解耦 ω 来源实现:`--model Qwen3-8B-Base --vec-model Qwen3-8B-chat`。
- **结果(8B,同上设置)**:base→base 是 50%→40%(+10);**chat→base 反而 40%→60%(m=1.0,drop −20),各 multiplier 全为负**。抽检:chat 的 ω 注入 base 后**生成几乎不变**(steered≈naive),即 chat 的 ω 在 base 上**方向失效**。
- **诚实解读**:`--vec-model` 机制工作正常(日志确认在用 chat 的 ω 注入 base),但**迁移效果与论文相反**。原因推测:论文是 **Llama-2-7B-chat → Llama-2-7B**(chat 由 base 微调而来、激活空间高度对齐);我们是 **Qwen3-8B-chat(post-trained) → Qwen3-8B-Base**,两者训练差异更大、激活空间对齐弱,chat 的 ω 迁到 base "水土不服"。**结论:跨模型 steering 迁移的有效性强依赖两模型激活空间的对齐程度**(见 §5)。


**论文头条复现(self-steer + CatQA + 论文确切模型 Meta-Llama-3-8B)— 修正提取/注入位置后复现了大幅降幅**:
- **设定纠正**:论文 Table 1/2 的 87.5→0 是 **self-steer**(从模型自身提 ω 注回自身,caption 明示)+ **CatQA**(结构化有害问句)+ vanilla(all activations)+ 论文层(Adult/Physical=14、Hate=25)。**不是** chat→base 迁移(那只是 §3.3 提质量技巧)。
- **前提复现 ✅**:Meta-Llama-3-8B 自我引导 CatQA(采样,n=50),naive %UR = 98%/88%/89%,**与论文 87.5/80/92.5 同量级**。
- **关键修正 = 残差流(复现成败的真正分水岭)**:最初从**注意力子层输出**(范数 ~0.7)提取/注入 ω,同判官下只降 +2~+5;改成在**残差流(decoder layer 输出,范数 ~18)**提取/注入后,adult_content layer14 **naive 98% → m=1.5 56%(降 +42)**,文本基本连贯(2/50 退化)。差距 8~20×。**这是与 CAA/refusal_direction 一致的正确实现,论文 Eq.1/Eq.2 的"attention activations"措辞误导了第一版。**
- **剩余到字面 →0 的次要差距**:(a) 判官——同一批 steered 文本严格判官 100% vs 论文对齐判官 35%(GPT-4 对"沾边但不可操作"更宽容);(b) 把 m 再推大到轻度退化(论文 →0 时 coherence 同步暴跌 2.45→1.44,m=3 时我们也见 42/50 退化、%UR 反弹)。即论文的字面 0 = 残差流 steering(主力)+ GPT-4 宽容 + 退化容忍。
- **复现结论**:核心方法与代码正确,残差流修正后复现出论文级大幅降幅(+42);字面 →0 的最后一截是判官标定 + 退化权衡。

> 自测:`python scripts/eval_steering.py ... --classifier keyword` 可离线(无 API)粗跑;judge 连通性已用 Claude/Anthropic 验证。`--do-sample` 后 base 续写连贯(否则 greedy 退化复读)。


**模块 6 `scripts/score_quality.py`** — §4.4 质量轴(论文 Table 3 的 5 个 HelpSteer 属性):
- **reward 模型替代**:论文用 Nemotron-340B(跑不动)。改用 **QRM-Llama3.1-8B-v2**,`out.rewards` 直接是 shape `(batch,5)` = helpfulness/correctness/coherence/complexity/verbosity,正好对应 Table 3。documented deviation:模型不同,5 属性与意图一致。
- **transformers 5.x 兼容补丁**:ArmoRM/QRM 的 `modeling_custom.py` 导入了 4.52 起被删的 `LLAMA_INPUTS_DOCSTRING`,在新版直接 ImportError。修法:把该 import 包成 `try/except` 兜底为空串(纯文档、无功能)——**只改模型远程代码、不动 conda 环境**。
- **运行**:需 GPU 服务器(QRM ~16GB),`CUDA_VISIBLE_DEVICES` 锁定空闲卡;读 eval json、给 naive 与每个 multiplier 的 steered 打分。
- **实测复现结果(qwen3-1.7b-base / BeaverTails hate_speech)**:**m=1.0 时 %UR 40%→10%,同时 help/corr/cohe 三项不降反升**(Δ 全为正),complexity/verbosity 基本持平 —— 完整复现论文核心主张:安全提升不以文本质量为代价。

**Baseline 对比(CAA / SEA)— 暂时搁置**:论文 Table 3 还含 CAA(Rimsky 2023)、SEA(Qiu 2024)两个对照方法。它们是**别人的方法,不并入本仓的 SafeSteer 复现代码**(避免污染)。调研结论:两个官方源码均**无法直接跑** —— 都硬编码 Llama-2 + 旧数据格式,且 pin 的 `torch 2.0.x` 不支持 RTX 5090(Blackwell sm_120),连"建旧环境跑"都被 GPU 堵死;移植≈在新栈重写核心算法(CAA ~8-16h、SEA ~12-20h)。故本轮搁置。未来若补,应在**独立子目录/模块**最小重实现(CAA=last-token diff-of-means+残差注入;线性 SEA=cross-cov+SVD+投影),绝不混入 SafeSteer 代码。

---

## 5. 潜在改进点(复现中发现)

复现过程中撞见的、论文一笔带过或未涉及、但有研究价值的点。既是这套方法的**脆弱面**,也是可探索的方向:

1. **ω 尺度对样本量极敏感(方法隐含前提)**。50 样本算的 ω 范数虚高(深层 ≈99 ≫ 激活尺度 ~13),逼得 multiplier 异常大、反而把文本推崩;1226 样本后范数回落、m≈1 即生效。论文用 1500/类不是随意 —— 但**论文没强调这是方法 work 的前提**。可探索:ω 的自适应尺度归一化(注意:论文 Eq.2 本身不归一化,加了即偏离),或用"达到稳定方向所需的最小样本量"刻画方法的数据效率。
2. **层选择缺乏跨模型的原则**。论文层 `{14,16,20,25,31}` 只在 Llama 验证;我们按"比例位置"迁到 Qwen/Gemma 只是**起点猜测**。可探索:用"该层 harmful/safe 激活可分性"(如线性探针精度、ω 范数峰值)**自动选层**,而非手工比例换算。
3. **多标签噪声直接污染类别向量**。BeaverTails 一条 prompt 挂多个有害标签,某些类样本少且主题漂移,稀释 ω 纯度。论文靠大样本 + L2 剪枝冲淡,但未根治。可探索:单标签过滤 / 标签置信度加权 / 更强的去噪(如鲁棒均值)。
4. **质量评测受限于 reward 模型**。论文用 Nemotron-340B,我们用 QRM-8B 替代 —— 分数尺度不同,只能比趋势不能比绝对值。且 reward 模型本身有偏。可探索:多个 reward 模型交叉验证,或人工小样本校准。
5. **multiplier 与文本崩坏的边界未刻画**。m 太大时 %UR 看似下降,实为文本崩成乱码被 judge 误判 safe。当前靠人工看文本区分。可探索:加一个**自动的"文本退化检测"**(重复率/困惑度/连贯性)作为 steering 的安全护栏,把"真转向"与"崩坏"自动分开。
6. **safe 数据来源的影响未系统对比**。论文有 paired-safe / generic-safe 之分,我们因 CatQA twin 未公开只用了 generic(Alpaca/BeaverTails-safe)。可探索:系统对比不同 safe 来源对 ω 方向与 %UR↓ 的影响。
7. **跨模型 ω 迁移的有效性依赖激活空间对齐**(实测发现)。论文的 chat→base 迁移在 Llama-2(chat 由 base 微调、激活对齐)上有效;我们在 Qwen3-8B(chat 是 post-trained、与 base 训练差异大)上复现,迁移**失效甚至变负**(chat 的 ω 注入 base 几乎不改变生成)。可探索:用激活空间对齐度(如 CKA / 子空间夹角)**预测**两模型间 ω 是否可迁移,而非默认"同尺寸即可迁"。
8. **steering 的成败取决于在残差流操作,论文措辞会误导(本轮最关键的复现修正)**。论文 Eq.1/Eq.2 写"attention activations / self-attention 输出",按字面取**注意力子层输出**(Meta-Llama-3-8B layer14 范数 ~0.7,激活 ~1.3)算 ω、再注回**残差流**(范数 ~18),扰动只占 ~3%,同判官下 %UR 只降 +2~+5。改成在**残差流(decoder layer 输出)**提取与注入后,**naive 98% → m=1.5 56%(降 +42)**、文本基本连贯——8~20× 的差距。这与论文引用的 CAA/refusal_direction 一致(它们都在残差流操作)。**教训:RepE 类方法里"在哪个张量上提取/注入"是决定性的,论文的模糊措辞是复现陷阱;应优先信引用的参考实现(残差流)而非字面。**
9. **%UR 指标对判官标定高度敏感(解释最后到字面 →0 的差距,但不是主因)**。对同一批 steered 文本:严格判官(沾边即 unsafe)给 100%,论文对齐判官(只看是否给可操作有害)给 35% —— 判官一个变量就把绝对水平从 100% 拉到 35%,但**两种判官下降幅都只有 ~7.5 点**(判官改的是水平、不是降幅)。论文字面 →0 还叠加了"把 m 推到轻度退化"(coherence 2.45→1.44,m 过大时文本退化、被判 safe)。可探索:报告 %UR 时给"actionable-harm"与"topic-adjacent"两档判官,或用 deterministic 退化检测,避免单一 LLM 判官把"退化"误判成"安全"。
10. **论文头条是 self-steer + CatQA,不是 chat→base 迁移**(易踩的复现陷阱)。论文 Table 1/2 的 87.5→0 是**从模型自身提 ω 注回自身**(caption: "used both for computing steering vectors and for evaluation")+ **CatQA 结构化有害问句**;chat→base 迁移只是 §3.3/§5 给 Llama-2-7B-chat 提质量的边角技巧,从来不是 %UR 头条。**复现头条务必锁定 self-steer + CatQA**;在 BeaverTails 口语 prompt + 迁移设定上做会得到"假的低 naive + 无降幅"。
11. **base 模型必须采样,否则 greedy 退化成复读**(评测陷阱)。base 模型 greedy 解码会退化成复读 prompt 的乱码(naive 输出失真,%UR 变成判官对"复读了有害问题"的噪声);必须 `--do-sample`(temp~0.7)才能让 base 续写连贯、naive %UR 反映真实倾向。论文的连贯 base 续写隐含了采样。

---

## 6. 快速上手

```bash
conda activate safesteer313                 # py3.13 / torch2.12+cu130 / transformers5.9
pip install pyarrow                          # 仅为读 Alpaca parquet(首次)

python scripts/prepare_data.py               # 生成 data/processed/ + manifest(seed=0)
python src/data.py                           # 各模块自测,纯 CPU
python src/hooks.py
python src/vectors.py
python scripts/test_load.py qwen3            # 验证真实模型加载+钩子(需 GPU)

# 端到端(论文趋势需足够样本,见 §4 实证发现):
python scripts/prepare_data.py --only beavertails --only alpaca --n-harmful 1500 --n-safe 1500
python scripts/extract_activations.py --model qwen3-1.7b-base --dataset BeaverTails --category hate_speech_offensive --safe-source beavertails
python scripts/extract_vectors.py     --model qwen3-1.7b-base --dataset BeaverTails --category hate_speech_offensive --prune
python scripts/eval_steering.py       --model qwen3-1.7b-base --dataset BeaverTails --category hate_speech_offensive --variant pruned --multiplier 0.5 1 2 --limit 10

# 论文头条设定(self-steer + CatQA;base 模型务必 --do-sample,判官并行加速):
python scripts/extract_activations.py --model <Meta-Llama-3-8B 路径> --dataset CatQA --category adult_content --safe-source alpaca
python scripts/extract_vectors.py     --model <Meta-Llama-3-8B 路径> --dataset CatQA --category adult_content --prune
python scripts/eval_steering.py        --model <Meta-Llama-3-8B 路径> --dataset CatQA --category adult_content \
    --variant vanilla --layers 14 --multiplier 0.5 1 2 --do-sample --judge-workers 16 --limit 50

# 质量轴(GPU 服务器, QRM ~16GB; 锁定空闲卡):
CUDA_VISIBLE_DEVICES=7 python scripts/score_quality.py --eval-json <eval json> --qrm <QRM 本地路径>
```

> 注册模型(`src/models.py`):`qwen3`/`qwen3-base`/`qwen3-1.7b`/`qwen3-1.7b-base`/`llama`/`llama-base`/`gemma`/`gemma-base`,也可直接传本地目录名/路径。8B 需 ≥24GB 显存;1.7B 在 16GB 上即可跑全链。judge 凭据放 `.env`(见 `.env.example`)。

---

## 7. 参考

- **论文**:[Fine-grained SafeSteer (arXiv 2506.04250)](https://arxiv.org/abs/2506.04250)
- **可借鉴 repo**:[andyrdt/refusal_direction](https://github.com/andyrdt/refusal_direction)(diff-of-means + hook 骨架) · [nrimsky/CAA](https://github.com/nrimsky/CAA)(CAA 基线) · [Yuqi-Qiu/SEA](https://github.com/Yuqi-Qiu/SEA)(SEA 基线)
- **作者**:`shaonag@nvidia.com`
