# 正式全量实验配置

使用`configs/base.yaml`叠加本目录的一份组合配置，不叠加`pilot.yaml`，也不再额外叠加`configs/probes/`或`configs/losses/`。后两者是原始默认组件；本目录保存的是九组分别调优后的完整组件参数。

## 共用设置

- 全量问答按head多数投票、平票safe，完成筛选后按prompt分组80/10/10划分，split seed42；不限制各split样本数，不合并验证集到训练集。实际数量见prepare输出的manifest。WildGuardTest有效回答标签1,709条，仅用于外部测试。
- 冻结Qwen3.5-9B，取零基block15完整输出、最终norm之前；每token沿hidden维做无仿射RMS归一化。仍为单层输入，SingProbe不是原论文默认三层输入。
- AdamW，betas=(0.9,0.999)，weight decay0.01，cosine/warmup5%，clip1.0，BF16计算、FP32 loss，实际microbatch8、累积4、有效batch32。
- 所有组共同使用4 epochs上限，并按验证集Overall AUPRC、固定max读出保存最佳checkpoint。4不是已经证明的全量最优epoch数；不使用pilot的最佳epoch直接指定正式训练终点。
- 全部prepared split均提取特征；九组复用同一份数据和单层特征缓存，不重复存九份。提取前根据实际token数估算磁盘需求。
- 阈值只由验证集确定，再分别报告ID测试和WildGuardTest。保存last/mean/max三种读出，不按测试分数挑读出。WildGuardTest仅Overall有标签。

## 九组参数

| 文件 | Probe | Loss | LR | Window | Loss额外参数 |
|---|---|---|---|---|---|
| linear_bce.yaml | Linear | uniform token BCE | 3e-4 | — | — |
| linear_weighted.yaml | Linear | confidence weighted | 3e-4 | — | T=0.1,tau=0.35,detach=true |
| linear_atc.yaml | Linear | ATC | 3e-4 | — | N=5,TV=0,mono=0 |
| singprobe_bce.yaml | SingProbe | uniform token BCE | 1e-4 | 64 | — |
| singprobe_weighted.yaml | SingProbe | confidence weighted | 3e-4 | 64 | T=0.1,tau=0.35,detach=true |
| singprobe_atc.yaml | SingProbe | ATC | 3e-4 | 256 | N=5,TV=0.1,mono=0.1 |
| gemini_bce.yaml | Gemini RMAttn | uniform token BCE | 3e-4 | 40 | — |
| gemini_weighted.yaml | Gemini RMAttn | confidence weighted | 3e-4 | 40 | T=0.1,tau=0.35,detach=true |
| gemini_atc.yaml | Gemini RMAttn | ATC | 3e-4 | 40 | N=1,TV=0,mono=0 |

这是每组调优后的对比，不是固定Probe和学习率只替换loss的纯消融。ATC某些组的零正则是搜索结果，不应改成默认0.01后仍称为该组最优配置。

## 运行

以SingProbe + weighted为例，从项目根目录运行：

```bash
conda activate probe
probe prepare --config configs/base.yaml --config configs/experiments/singprobe_weighted.yaml
probe extract --config configs/base.yaml --config configs/experiments/singprobe_weighted.yaml
probe train --config configs/base.yaml --config configs/experiments/singprobe_weighted.yaml
probe evaluate --config configs/base.yaml --config configs/experiments/singprobe_weighted.yaml
```

其他组合替换最后一个配置文件。数据准备和特征提取只需实际执行一次，其他组合会复用已校验的产物。先用`--dry-run`查看路径。

默认训练seed42；正式比较建议也跑43、44，每条train/evaluate命令一致追加`--set seed=43`或`--set seed=44`。数据和特征与训练seed独立，不会因此重新划分或提取。不能只挑测试分数最好的seed报告。

## 来源与当前状态

参数来源于库外pilot搜索`/workplace/yht/probe-search-y3vVK5i2/summary.json`：8,192条训练、完整验证、114次成功试验，九组各做seed42/43/44验证；没有用测试分数选参数。原始搜索结果和脚本仍留在库外。

沿用这九组超参数作为多数票数据训练的起点。按验证集重新选择checkpoint和校准阈值。
