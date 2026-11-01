# BiSeNetV2 工业缺陷分割消融实验

本实验使用独立模型实现、配置包、运行入口和结果目录。现有 BiSeNetV2、ScConv 实验入口、数据划分和历史结果目录不作为写入目标。

## 实验组合

| 实验 | 多尺度上下文 | 缺陷感知融合 | 边界监督 |
|---|---:|---:|---:|
| Baseline | × | × | × |
| Exp-1 | ✓ | × | × |
| Exp-2 | × | ✓ | × |
| Exp-3 | × | × | ✓ |
| Exp-4 | ✓ | ✓ | × |
| Exp-5 | × | ✓ | ✓ |
| Exp-6 | ✓ | ✓ | ✓ |

Baseline 通过独立模型包装器运行原 BiSeNetV2 路径。多尺度上下文插入语义分支输出与融合模块之间；缺陷感知融合替换 BGA；边界头读取 Detail Branch 的步长 4 特征，只在训练阶段增加边界 BCE 监督。Boundary F1 从最终分割掩码计算，容差为 1 像素。

## PyCharm 运行

打开 run_experiment_bisenetv2_defect.py，修改文件顶部的 DATASETS、EXPERIMENT、SEEDS 等参数，然后右键选择 Run。设置 EXPERIMENT = "ALL" 可按顺序运行七组。DATASETS 可设为 ("neu",)、("sd-saliency",) 或同时包含两个数据集。

默认保留配置中的训练轮数和批大小。必要时也可以通过命令行覆盖：

    .\.venv\Scripts\python.exe .\run_experiment_bisenetv2_defect.py --dataset both --experiment ALL --seeds 42,123,2024

SD900 使用项目固定划分适配器；NEU 使用项目已有固定 train/validation/test 清单。两个数据集分别训练和汇总。

## 结果位置

所有产物写入：

    results/bisenetv2_sdneu_defect_ablation/<NEU|SD900>/<Baseline|Exp-1...Exp-6>/seed_<seed>/

训练器会在每个 seed 目录下创建唯一运行子目录，包含 best.pth、metrics.json、config.json、训练历史、图表和预测可视化。独立汇总文件保存在根目录的 summary.csv 与 summary.md。
