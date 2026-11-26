# Qwen 系安全对齐数据集调研

调研日期：2026-11-17。目标是为 Qwen3-8B 的后续安全对齐微调选择数据；“Qwen 生成”与“用 Qwen 作为被微调模型”分开记录。以下数字和许可来自当时公开数据卡或本机文件，使用前应再次核验版本及上游数据条件。

## 首选：Qwen3-8B 自生成安全 SFT

### ThinkSafe-Qwen3-8B-WildGuard

- 数据：[Seanie-lee/ThinkSafe-Qwen3-8B-WildGuard](https://huggingface.co/datasets/Seanie-lee/ThinkSafe-Qwen3-8B-WildGuard)，版本 `1c5f2ed6676ad737d895cc5592a333837598a1e3`。
- 数据卡：39,883 条 train，字段 `instruction`、`response`、`prompt_label`、`response_label`。数据卡称它使用 Qwen3-8B 和 WildGuard，构造方法见 [ThinkSafe 论文](https://arxiv.org/abs/2601.23143)。论文使用 SafeChain 的 prompt；论文注明 WildJailbreak 是 SafeChain 的来源之一。
- 用途：最贴近“Qwen3-8B 生成的安全对齐 SFT”。不是天然的 chosen/rejected 配对；应核实 `response_label`、抽查质量，明确 target 是仅 final answer 还是包含 `<think>` 的完整回答。
- 风险：数据集卡没有明确声明 license；公开可下载不等于明确的微调及再分发授权。检查上游许可、训练/评测重叠及模型自己生成数据的偏差。仅公开 train，没有独立验证集。

### ThinkSafe-Qwen3-8B-Activation-data

- 数据：[Sangsang/ThinkSafe-Qwen3-8B-Activation-data](https://huggingface.co/datasets/Sangsang/ThinkSafe-Qwen3-8B-Activation-data)，版本 `2e1e16b7344a39edb0d0784c0fcf2a1de658df6c`。
- 数据卡：38,752 条 train，字段同上；由 `Qwen/Qwen3-8B` 生成，危险请求采用 activation steering，良性请求不使用该干预；基于 `UWNSL/SafeChain` prompt，使用 Llama-Guard-3-8B 与结构检查筛选。响应含推理和最终回答。
- 用途：适合与上一数据集做两种安全数据生成方式的对照，不建议直接合并当作互不相关数据。
- 风险：数据卡指出 guard 筛选不等于人工验证，且下游评测尚未完成；未声明数据集许可。与 WildJailbreak、SafeChain 及本项目数据存在潜在重叠，应以 prompt 级去重和上游 ID 核对。

### 其他相关候选

- [Seanie-lee/ThinkSafe-Qwen3-4B-WildGuard](https://huggingface.co/datasets/Seanie-lee/ThinkSafe-Qwen3-4B-WildGuard)：39,887 条，Qwen3-4B 生成，字段与 8B 版相同，未声明许可。可用于教师规模对照，但不是 8B 同模型生成。对应 revision `f8ea72b9c71b4c1b2480a69d6baf331ed8e1f231`。
- [neurips2026-10977/ThinkSafe-Qwen3-8B](https://huggingface.co/datasets/neurips2026-10977/ThinkSafe-Qwen3-8B)：39,808 条同结构数据，但数据卡除自动生成的 schema 外为空，生成过程和许可不能独立核实；暂作线索，不作为首选。revision `3c26faecbb1efab6e839f95e60b5bd8cfcf7fdf3`。

## 本机已有：Qwen3-8B 的标注响应

[Alibaba-AAIG/StreamGuardBench](https://huggingface.co/datasets/Alibaba-AAIG/StreamGuardBench) 包含 Qwen3-8B 等模型生成的安全和不安全回答；文本字段为 `prompt`、`response`、`label`、`reason`，其中 `0` 表示 safe、`1` 表示 unsafe。数据卡声明 Apache-2.0，但仍需核对各上游来源和模型输出许可；`main` 和 `v1` 的标签版本不同。本项目配置固定 revision `7fa1203b2d9be5671d3b40f2c1df0035deeb098c`。

本机直接读取该 revision 的 train Arrow 得到：

| 子集 | 本地路径前缀（相对项目根目录） | 记录数 | label=0 | label=1 |
|---|---|---:|---:|---:|
| WildGuard / Qwen3-8B | `data-preparation/data/raw/streamguard/wildguard/qwen3_8b/train/` | 37,900 | 28,265 | 9,635 |
| S-Eval / Qwen3-8B | `data-preparation/data/raw/streamguard/s_eval/qwen3_8b/train/` | 9,000 | 6,074 | 2,926 |

这些是 response-level 的机器标签，可用于 safety probe 和评测研究；做安全 SFT 时应复核后仅把期望的安全回答作为目标。做 DPO 还需要同一 prompt 的可靠 chosen/rejected，不能把不同 prompt 随意配对。原数据也是 benchmark；S-Eval 和 WildGuard 训练数据混入某个评测后，不能再把同源或重复样本当作独立测试集。

之前全量 `02_prepare.py --sources streamguard` 因图文子集引用缺失图片而中断；上述**纯文本 Arrow 文件已经在本机**，但不代表整个 StreamGuard 已导入项目 SQLite。

## 偏好数据补充

- [puwaer/cvalues_rlhf_en_cot](https://huggingface.co/datasets/puwaer/cvalues_rlhf_en_cot)：96,339 条英文 DPO 数据。数据卡注明 Qwen3-235B-A22B-Instruct-2507 只生成 chosen 的思考链，chosen 最终回答由 GPT-OSS 生成；不能误记为“Qwen 生成全部正例”。许可 CC-BY-NC-SA-4.0；revision `956224b055de54cb3e44b1fc319a448548bed3d5`。考虑推理链处理和 judge 标注误差后再用。
- [SINAI/ALIA-es-Safety-DPO](https://huggingface.co/datasets/SINAI/ALIA-es-Safety-DPO)：47,455 条西班牙语偏好数据，候选回答来自多个模型，包含 Qwen3.5-122B；应按 metadata 检查具体行的模型来源。Kimi k2 担任 judge，且 chosen/rejected 可能都是安全回答而非安全/不安全对。许可 CC-BY-SA-4.0；revision `a73a3db2d2a9e3d4130280f5e34df0ef7c14ad9a`。适合西语偏好研究，不是本项目中英文安全目标的首选。

## 留作评测，不作为普通生成式 SFT

[Qwen/Qwen3GuardTest](https://huggingface.co/datasets/Qwen/Qwen3GuardTest) 是 Qwen 官方安全分类与流式定位 benchmark。响应由 Qwen3-8B、其他模型和部分 abliterated 模型生成，prompt 源自 BeaverTails test；有人工标注和 token 区间。许可 CC-BY-NC-4.0。适合保留为 guard / probe 的评测集，不要混入训练。

## 建议的初步实验

1. **确定起点**：本机同时有 `/workspace/models/Qwen3-8B` 和 `/workspace/models/Qwen3-8B-Base`；记录完整路径、revision、chat template 和 thinking 开关。ThinkSafe 数据主要面向含推理的 Qwen3，应明确目标是复现 thinking 还是训练 no-thinking 的安全回答。
2. **小规模筛选**：在许可明确后，对 ThinkSafe-Qwen3-8B 取按风险与良性/有害 prompt 分层的样本，检查 `response_label`、拒绝是否切题、正文是否安全、截断与 `<think>` 格式；针对中文目标单独检查语言覆盖。
3. **构建训练与验证**：SFT 只用经复核的期望回答，同时保留适量良性请求与正常回答以检查过度拒绝。按 prompt 及来源分组去重，所有同 prompt 响应留在同一 split；评测 benchmark 和近重复内容在训练前剔除。
4. **对照**：原始 Qwen3-8B、ThinkSafe 安全 SFT、等数据量的 SFT 对照；评估有害请求拒绝、良性请求正常回答、通用能力。judge 不是 ground truth，记录模型、rubric、版本并做人工分层抽检。
5. **后续 DPO**：只对同 prompt 下语义和上下文一致的候选构建 preference；chosen 应为安全且对请求有用的回答。拒绝过度的良性回答可以单独标注，不要简单将所有拒绝都设为 chosen。
