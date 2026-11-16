# Fine-grained SafeSteer 复现日志 (2026-06)

本复现基于论文 Meta-Llama-3-8B 模型，在 CatQA 数据集上进行 self-steer 实验。

## 复现结论与实证分析

- **证明（方法有效）**：核心机制（per-category difference-of-means + 残差流注入）在论文确切设定上确实大幅降 %UR——Meta-Llama-3-8B / CatQA adult_content / layer 14 / m=1.5：**naive 98% → 56%（−42）**，文本基本连贯。高 naive 前提也复现（98/88/89% ≈ 论文 87.5/80/92.5）。§3.3 "星型结构"哲学（generic Alpaca safe 当共享 anchor）成立。
- **反驳一（论文措辞误导）**：Eq.1/Eq.2 字面的 "attention activations / self-attention 输出" **按字面做几乎不 work**——注意力子层输出范数 ~0.7，算的 ω 注回残差流（范数 ~18）只占 ~3%，同判官只降 +2~+5。**必须在残差流（decoder layer 输出）提取与注入**才有 +42（与论文引用的 CAA/refusal_direction 一致；论文实际用的应是残差流，措辞不精确）。这直接坐实了 attention 差 vs residual 差中 residual 才对的结论。
- **反驳二（头条 →0 非纯 steering 之力）**：判官标定只改绝对水平、不改降幅（同批文本严格判官 100% vs actionable 判官 35%，但两者降幅都仅 ~7.5）；字面 0 还叠加 "把 m 推到轻度退化"（m 过大→文本退化→判 safe，正对应论文 Table 1 自己的 coherence 2.45→1.44 暴跌）。使用 Claude-haiku（无 GPT-4 access），落不到字面 0 属预期。

## 新发现

1. **"在哪个张量上提取/注入"是 RepE 复现的决定性变量**，论文模糊措辞是陷阱——连纯 attention 模型上 attention-子层 vs residual 都差 8-20×，强化了在 hybrid 上"信号从哪抽"的研究价值与必要性。
2. **%UR 高度依赖判官标定**（100% vs 35% 同文本）→ 报告 %UR 须固定判官 + 配 deterministic 退化检测。
3. **复现陷阱清单**：headline 是 self-steer（非 chat→base 迁移，后者只是 §3.3 提质量技巧）；base 模型必须采样（greedy 退化复读）；CatQA paired-safe 未公开须用 Alpaca 备用。
