# NBS 钢材表面缺陷分割实验

本目录整理了 U-Net、BiSeNetV2、BiSeNetV2+ScConv 的数据、配置、训练脚本和模型架构图。NEU-Seg 固定划分为：训练集 2904 张、验证集 726 张、测试集 840 张。训练集与验证集来自原训练池，测试集使用官方 test 子集；验证集用于选最佳模型，测试集只在训练结束后评估。

## 目录说明

- `data/NEU_Seg-main/`：图像、掩膜及固定划分清单 `splits/{train,val,test}.txt`。
- `config/`：基础配置、U-Net 配置及 6 组 BiSeNet/ScConv 参数方案。
- `models/`：U-Net、BiSeNetV2 及 ScConv 模型结构实现。
- `scripts/`：各模型训练程序、数据读取、指标、绘图和划分校验代码。
- `docs/architecture/`：各模型的 Visio 架构图（VSDX）。
- `results2中断`：每个实验独立保存训练日志、最佳权重、配置快照、曲线图和测试指标。
- `.venv/`：项目虚拟环境。

## 运行

在 PowerShell 中进入 `G:\NBS`，运行下列命令。默认数据集是 NEU-Seg。

```powershell
.\.venv\Scripts\python.exe .\train.py --model unet
.\.venv\Scripts\python.exe .\train.py --model bisenet
.\.venv\Scripts\python.exe .\train.py --model scconv
```

选择参数方案：

```powershell
.\.venv\Scripts\python.exe .\train.py --model bisenet --config config1
.\.venv\Scripts\python.exe .\train.py --model bisenet --config config2
.\.venv\Scripts\python.exe .\train.py --model bisenet --config config3
.\.venv\Scripts\python.exe .\train.py --model scconv --config configScConv1
.\.venv\Scripts\python.exe .\train.py --model scconv --config configScConv2
.\.venv\Scripts\python.exe .\train.py --model scconv --config configScConv3
```

各配置及数据划分可通过训练脚本原有参数覆盖，例如 `--epochs 1` 适用于检查启动流程。不要把官方测试集用作逐轮验证集，否则会造成测试集信息泄漏。

## 划分清单与结果

`data/NEU_Seg-main/splits/` 中的三个清单保存样本文件名，训练脚本会按清单读取并检查数量/交叉情况。更换数据后请重新生成并核对清单，不要手工在训练/验证/测试间重复样本。

每次运行按配置分别写入 `results2中断` 下的目录，实验目录名为 `[数据集]_[epoch]_[编号]`，编号从 0 开始且已有目录不会被覆盖。BiSeNet 与 ScConv 保存 `best.pth`、`config.json`、`result.txt`、训练曲线、PR/ROC/F1 曲线和混淆矩阵；U-Net 保存 `best.pth`、`config.json`、`result.txt`、`history.jsonl`、独立测试集指标 JSON、训练总览图 `results.png`、测试集 PR/ROC/F1 曲线、混淆矩阵和预测可视化。测试集曲线按像素级 one-vs-rest 概率统计，指标标注在对应实验目录。

## ScConv 消融实验

`results[归档]` 保留早期 7 组实验（BiSeNetV2 / U-Net / 旧版 ScConv）。

**论文统一协议（当前）**：所有臂共用 `configScConvAblation.py` 这一份基座——
`cosine + 5 轮 warmup`、**300 轮跑满、不早停**（`EARLY_STOPPING_PATIENCE=100000`）、
`torch.use_deterministic_algorithms(True)`、无梯度裁剪。不早停是刻意的：余弦退火的
lr 曲线按总轮数排定，配早停会在 lr 还高的时候把模型掐掉，反而压低 mIoU；"最好结果"
由最佳验证检查点来取。

**90 轮协议的旧结果已归档到 `results[归档]`**（22 个 run + 当时的
`ablation_summary.md/.csv` 快照）。它们与 300 轮协议不可混表，汇总脚本会把跨协议的
差值标成"不可作为结论"。要看旧数据：
`.\.venv\Scripts\python.exe .\scripts\run_metrics.py --root results\archive_90ep`

单次运行约 4.2 小时（300 轮 × 约 50 s/轮）。

**一键运行（PyCharm）**：打开 `run_experiment.py`，改顶部的 `EXPERIMENT` 与 `SEEDS`，直接右键 Run。它会为每个种子起独立进程串行训练，结束后自动刷新汇总。

命令行等价用法：

```powershell
.\.venv\Scripts\python.exe .\run_experiment.py                  # 用文件里的默认设置
.\.venv\Scripts\python.exe .\run_experiment.py --config configScConvR2 --seeds 42
```

**汇总**：

```powershell
.\.venv\Scripts\python.exe .\scripts\summarize_ablation.py --with_cost
```

输出 `results2中断` 与 `.csv`（含每类 IoU、前景召回、mean±std、参数量与 FLOPs）。

**成本测量**：`.\.venv\Scripts\python.exe .\scripts\measure_cost.py`

**单模型诊断**（残差尺度、SRU 门控动态范围）：

```powershell
.\.venv\Scripts\python.exe .\scripts\diagnose_scconv.py --run_dir results\archive_90ep\bisenetv2_repl_R1_detail\NEU_90_0 --with_data --max_batches 8
```

### 归因对照臂（C2a / C2b / C2c）

D2 的 +0.50% 是 ScConv 的功劳，还是"在融合特征后多加一个块"的功劳？三个对照臂把
fused 位置的块换成普通块，其余逐项不变（同位置、同无残差、同 seed、同协议）：

| 臂 | fused 位置的块 | 参数量 | GFLOPs | 说明 |
| --- | --- | --- | --- | --- |
| C2a | `Conv1x1+BN+ReLU` | 3,639,460 | 3.78 | 同位置、参数只有 ScConv 的 55% |
| C2b | `Conv3x3+BN+ReLU` | 3,770,532 | 3.95 | 同位置、参数是 ScConv 的 491% |
| C2c | `ReLU`（零参数） | 3,622,820 | 3.76 | 只多一层非线性，参数量与 B0 完全相同 |

```powershell
.\.venv\Scripts\python.exe .\run_experiment.py --config configScConvC2a --seeds 42,123,2024
.\.venv\Scripts\python.exe .\run_experiment.py --config configScConvC2b --seeds 42,123,2024
.\.venv\Scripts\python.exe .\run_experiment.py --config configScConvC2c --seeds 42
```

### 全量重跑（22 个臂 × 3 种子）

```powershell
# 每个 arm 建议单独一条命令，便于中断续跑（已有结果文件会自动跳过）
.\.venv\Scripts\python.exe .\run_experiment.py --config configScConvB0 --seeds 42,123,2024
.\.venv\Scripts\python.exe .\run_experiment.py --config configScConvD2 --seeds 42,123,2024
# …… S0/D1/P1/P2/G1/R1/R2/R1D2 同理
```

### U-Net 基线（与主表同一套 recipe）

`train_unet.py` 走 `scripts/dataset.py` 的 `build_loaders`，与 BiSeNetV2 共用同一份固定
划分清单、同一套增强、同一个 shuffle 生成器；`configUnetP300` 再把调度对齐成
cosine + 5 warmup + 300 轮不早停 + 关闭梯度裁剪。

两处刻意保留的差异（论文实验设置里要写明）：

| 项 | BiSeNetV2 / ScConv 各臂 | U-Net |
| --- | --- | --- |
| batch | 32 | **16**（batch 32 峰值 4.93 GB > 4.3 GB 显存，会溢出到系统内存） |
| 确定性 | 开 | **关**（开启后每轮 1,981 ms vs 312 ms，300 轮差 10 h vs 1.6 h） |

```powershell
.\.venv\Scripts\python.exe .\run_experiment.py --config configUnetP300 --seeds 42,123,2024
.\.venv\Scripts\python.exe .\train.py --model unet --config configUnetP300 --dataset neu --epochs 1  # 冒烟
```

> U-Net 实测：16.95 GFLOPs / 7.85M 参数（BiSeNetV2 是 3.76 GFLOPs / 3.62M），
> **54 s/轮（含验证）→ 300 轮约 4.5 小时/种子**。数据来自 `results2中断`
> 那次真实运行（plateau + 早停，264 轮，test mIoU 0.8755）——它属于旧协议，
> 已作为独立的 `UNet` 臂保留，不与 `UNetP300` 混表。

> 注意 `--dataset` 已统一成 `dataset.py` 的写法（`neu` / `sd-saliency` / `magnetic-tile`），
> 这条路径不再支持 Severstal。

**结果读取**：两种落盘格式（`metrics.json` 与 `test_metrics.json`）由
`scripts/run_metrics.py` 统一读取，`summarize_ablation.py` 因此能把 U-Net 并入同一张表。
汇总表会按 run 列出实际轮数与调度，跨协议的差值会被标注为"不可作为结论"。

```powershell
.\.venv\Scripts\python.exe .\scripts\run_metrics.py --root results
```


