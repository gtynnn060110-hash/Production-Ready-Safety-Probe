## 数据

训练数据&ID数据: BeaverTails(多数投票, 平票算safe)
OOD数据: WildGuardTest

类别: Privacy, Misinformation, Harmful Language, Malicious Uses, overall

映射表

| 输出头 | BeaverTails 类别 |
| --- | --- |
| Privacy | `privacy_violation` |
| Misinformation | `misinformation_regarding_ethics,laws_and_safety` |
| Harmful Language | `discrimination,stereotype,injustice`；`hate_speech,offensive_language`；`sexually_explicit,adult_content` |
| Malicious Uses | `drug_abuse,weapons,banned_substance`；`financial_crime,property_crime,theft`；`self_harm`；`violence,aiding_and_abetting,incitement` |

划分: Train / Validation / ID test = 8:1:1

**正例 / 负例表**

| 集合 | 样本数 | Privacy | Misinformation | Harmful Language | Malicious Uses | Overall |
| --- | --- | --- | --- | --- | --- | --- |
| Train | 69,852 | 3,933 / 65,919 | 304 / 69,548 | 2,661 / 67,191 | 22,251 / 47,601 | 30,003 / 39,849 |
| Validation | 8,723 | 453 / 8,270 | 32 / 8,691 | 292 / 8,431 | 2,703 / 6,020 | 3,571 / 5,152 |
| ID test | 8,702 | 496 / 8,206 | 42 / 8,660 | 355 / 8,347 | 2,896 / 5,806 | 3,897 / 4,805 |
| OOD：WildGuardTest | 1,709 | — | — | — | — | 284 / 1,425 |

## Probe

| Probe | 结构 | 参数量 |
| --- | --- | --- |
| Linear | 单层线性分类器 | 20,485 |
| SingProbe | 因果 MQA → 输出投影 → Q 残差 → RMSNorm → 分类器；4 个 Query heads，共享 1 个 KV head，head dim=64 | 1,639,941 |
| Gemini / RMAttn | 两层 MLP（100、100）→ 窗口 attention pooling → 内部 heads 求和；每类 10 个内部 heads | 429,805 |

## Loss

| Loss | 计算方式 |
| --- | --- |
| BCE | 全部回答 token 的 BCE 均值 |
| SingProbe Weighted | `Σ_t w_t · BCE(z_t,y)`；`w_t=softmax_t(β·sigmoid(z_t)/T) β=clip(std(p)/τ,0,1)` |
| PlugGuard ATC | 首尾约束 + 平滑 + 单调性 |

## 参数

### 超参扫描

范围:

| 参数 | 扫描范围 |
| --- | --- |
| layer | 7、15、19、23、31 |
| lr | 5e-5、1e-4、3e-4 |
| singprobe window | 64、256、2048 |
| RMAttn window | 10、20、40 |
| weighted t | 1.0、0.3、0.1 |
| ATC N | 1、5、10 |
| ATC（λ_TV, λ_mono） | (0,0)、(0.01,0.01)、(0.1,0.1) |

结果:
layer=block 15

| probe | loss | lr | window | t | ATC N | ATC（λ_TV, λ_mono） |
| --- | --- | --- | --- | --- | --- | --- |
| Linear | BCE | 3e-4 | — | — | — | — |
| Linear | Weighted | 3e-4 | — | 0.1 | — | — |
| Linear | ATC | 3e-4 | — | — | 5 | (0,0) |
| SingProbe | BCE | 1e-4 | 64 | — | — | — |
| SingProbe | Weighted | 3e-4 | 64 | 0.1 | — | — |
| SingProbe | ATC | 3e-4 | 256 | — | 5 | (0.1,0.1) |
| RMAttn | BCE | 3e-4 | 40 | — | — | — |
| RMAttn | Weighted | 3e-4 | 40 | 0.1 | — | — |
| RMAttn | ATC | 3e-4 | 40 | — | 1 | (0,0) |

### 训练

| 项目 | 设置 |
| --- | --- |
| Backbone | Qwen3.5-9B, max length=4096，BF16 |
| Epoch / Seed | 4 / 42 |
| Microbatch * 梯度累积 | 8*4=32 |
| 优化器 | AdamW，weight decay=0.01，betas=(0.9,0.999) |
| 学习率调度  | warmup 5% + cosine；max grad norm=1.0 |

## 5. 结果

### Overall：Max, F1校准

| 配置 | 阈值 | ID AP | ID F1 | OOD AP | OOD F1 | OOD Recall | OOD FPR |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Linear + BCE | 2.046875 | 94.23 | 86.26 | 71.28 | 55.29 | 88.38 | 26.18 |
| Linear + Weighted | 0.929688 | 94.91 | 87.20 | 73.39 | 61.14 | 88.38 | 20.07 |
| Linear + ATC | 1.664062 | 93.66 | 85.46 | 72.50 | 51.95 | 89.08 | 30.67 |
| SingProbe + BCE | 1.296875 | 95.15 | 87.47 | 77.85 | 59.95 | 91.20 | 22.53 |
| SingProbe + Weighted | -0.384766 | 95.73 | **88.34** | 77.31 | 64.29 | 88.73 | 17.40 |
| SingProbe + ATC | 0.008667 | 95.69 | 88.00 | 74.51 | 62.03 | 89.44 | 19.72 |
| RMAttn + BCE | 1.158920 | 95.31 | 87.58 | 75.86 | 60.73 | 91.20 | 21.75 |
| RMAttn + Weighted | 0.293586 | **95.79** | 88.16 | **78.54** | **64.96** | 89.08 | **16.98** |
| RMAttn + ATC | 0.722599 | 95.43 | 87.35 | 76.78 | 58.05 | 90.14 | 24.00 |

### Overall：Max, FPR 5%校准

| 配置 | 阈值 | Val FPR | ID Recall | ID F1 | ID FPR | OOD Recall | OOD F1 | OOD FPR |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Linear + BCE | 2.671875 | 4.95 | 79.37 | 85.38 | 5.31 | 80.28 | 60.16 | 17.26 |
| Linear + Weighted | 1.320313 | 4.99 | 81.29 | 86.52 | 5.37 | 81.34 | 62.77 | 15.51 |
| Linear + ATC | 2.265625 | 4.95 | 76.91 | 83.97 | 5.08 | 84.15 | 57.18 | 21.96 |
| SingProbe + BCE | 1.992188 | 4.99 | 81.63 | 86.89 | 5.08 | 85.56 | 65.41 | 15.16 |
| SingProbe + Weighted | 0.464844 | 4.99 | 83.14 | 87.78 | 5.10 | 79.93 | 68.68 | 10.53 |
| SingProbe + ATC | 0.835938 | 4.99 | 83.32 | 87.80 | 5.24 | 81.34 | 65.81 | 13.12 |
| RMAttn + BCE | 1.713331 | 4.99 | 82.09 | 86.94 | 5.47 | 85.21 | 63.94 | 16.21 |
| RMAttn + Weighted | 0.899600 | 4.99 | 84.01 | **87.94** | 5.72 | 83.45 | **67.91** | 12.42 |
| RMAttn + ATC | 1.510334 | 4.99 | 81.60 | 86.58 | 5.60 | 85.56 | 64.71 | 15.72 |

### Overall：Max, FPR 1%校准

| 配置 | 阈值 | Val FPR | ID Recall | ID F1 | ID FPR | OOD Recall | OOD F1 | OOD FPR |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Linear + BCE | 4.625000 | 0.99 | 58.89 | 73.64 | 0.85 | 45.77 | 57.40 | 2.74 |
| Linear + Weighted | 3.281250 | 0.97 | 62.59 | 76.46 | 0.92 | 47.89 | 59.65 | 2.53 |
| Linear + ATC | 4.062500 | 0.99 | 55.48 | 70.83 | 0.96 | 55.99 | 62.35 | 4.70 |
| SingProbe + BCE | 3.640625 | 0.97 | 68.28 | 80.31 | 1.44 | 63.03 | 68.98 | 3.93 |
| SingProbe + Weighted | 2.640625 | 0.99 | 69.26 | 81.12 | 1.21 | 52.11 | 62.71 | 2.81 |
| SingProbe + ATC | 2.859375 | 0.99 | 68.69 | 80.86 | 0.98 | 54.23 | 61.85 | 4.21 |
| RMAttn + BCE | 3.630421 | 0.99 | 67.98 | 80.11 | 1.39 | 64.79 | 68.02 | 5.12 |
| RMAttn + Weighted | 3.516486 | 0.99 | 69.59 | **81.37** | 1.19 | 55.63 | 64.49 | 3.37 |
| RMAttn + ATC | 3.450614 | 0.99 | 67.33 | 79.74 | 1.25 | 64.44 | 68.28 | 4.84 |

### Mean / Last, F1校准

| 配置 | ID Mean | ID Last | OOD Mean | OOD Last |
| --- | --- | --- | --- | --- |
| Linear + BCE | 94.93 / 87.20 | 93.88 / 86.12 | 76.35 / 66.26 | 64.92 / 58.65 |
| Linear + Weighted | 94.93 / 87.37 | 93.67 / 85.25 | 75.85 / 66.28 | 59.94 / 55.59 |
| Linear + ATC | 93.48 / 85.38 | 94.17 / 86.05 | 73.69 / 53.13 | 64.48 / 59.17 |
| SingProbe + BCE | 95.71 / 88.28 | 95.49 / 87.71 | 81.48 / 72.12 | 70.60 / 61.10 |
| SingProbe + Weighted | 95.79 / 88.27 | 95.26 / 87.33 | 80.95 / **73.26** | 69.77 / 60.20 |
| SingProbe + ATC | 94.81 / 86.35 | **95.83 / 88.32** | 79.80 / 67.76 | 75.63 / 68.72 |
| RMAttn + BCE | **95.83 / 88.29** | 95.66 / 87.96 | 80.64 / 70.23 | 71.05 / 65.07 |
| RMAttn + Weighted | 95.65 / 87.54 | 95.07 / 86.77 | **82.02** / 73.15 | 66.84 / 54.55 |
| RMAttn + ATC | 94.70 / 86.02 | 95.59 / 87.53 | 80.30 / 65.70 | 74.62 / 66.86 |

### ID：Max, 四个类别

AP / F1。

| 配置 | Privacy | Misinformation | Harmful Language | Malicious Uses |
| --- | --- | --- | --- | --- |
| ID 正例数 | 496 | 42 | 355 | 2,896 |
| Linear + BCE | 93.15 / 82.39 | 9.50 / 8.47 | 52.54 / 48.82 | 96.26 / 89.66 |
| Linear + Weighted | 94.45 / 85.17 | 9.35 / 8.55 | 54.62 / 49.78 | 96.83 / 90.84 |
| Linear + ATC | 92.63 / 82.80 | 6.41 / 8.00 | 44.71 / 47.37 | 95.47 / 89.03 |
| SingProbe + BCE | 94.61 / 87.12 | 9.36 / 16.00 | 59.31 / 54.45 | 97.36 / 91.05 |
| SingProbe + Weighted | 94.56 / 86.98 | 10.48 / 14.81 | 59.32 / 53.93 | 97.58 / 91.80 |
| SingProbe + ATC | 95.11 / 87.01 | 13.26 / 21.74 | 58.54 / 55.31 | 97.44 / 91.62 |
| RMAttn + BCE | 93.89 / 85.29 | 10.66 / 14.58 | 57.62 / 53.99 | 97.38 / 91.45 |
| RMAttn + Weighted | 94.86 / 86.54 | 10.70 / 20.74 | 60.45 / 54.71 | 97.70 / 91.90 |
| RMAttn + ATC | 94.75 / 86.81 | 9.42 / 16.28 | 57.62 / 55.26 | 97.43 / 91.32 |

结论:

选RMAttn + Weighted这个方案, 然后可以考虑用mean读出, 数据集问题明显、尤其是类别不均衡.