"""U-Net 的论文协议配置：cosine 300 轮跑满、不早停、batch 16、确定性关。

与 ``configScConvAblation``（BiSeNetV2 / ScConv 各臂用的那份）逐项对齐的意图，是
让主表三行只差架构：

| 项 | BiSeNetV2 / ScConv 各臂 | U-Net（本配置） |
| --- | --- | --- |
| 调度 | cosine + 5 warmup | 同 |
| 轮数 | 300，不早停 | 同 |
| 确定性 | 开 | **关**（见下） |
| 梯度裁剪 | 无 | 无（``MAX_GRAD_NORM=0``） |
| 数据管线 | ``dataset.build_loaders`` | 同（同一份固定划分、同一套增强） |
| batch | 32 | **16**（见下） |

两处刻意的例外，都必须在论文的实验设置里写明：

1. **batch 32 → 16**。U-Net 在 batch 32 下峰值显存 4.93 GB，超过这张卡的 4.3 GB，
   WDDM 会溢出到系统内存，每轮从 3 分钟变成 14.5 分钟（实测 1 epoch 花 924 s）。
   batch 16 的峰值是 2.52 GB。这个值写在 ``config_unet`` 的
   ``DATASET_DEFAULTS["neu"]["batch_size"]`` 里，不在本文件中重复定义。
2. **确定性关**。batch 16 下单批耗时 1,981 ms（确定性开）vs 312 ms（关），差 6.3 倍；
   300 轮大致是 15 小时 vs 4.5 小时。代价是同种子重跑不再逐位复现——所以**不要用重跑
   来"修"一个不好看的数字**，3 个种子的均值才是结果。核心对比（BiSeNetV2 vs +ScConv）
   仍全部开着确定性，逐种子配对的性质不受影响；U-Net 只是参考行。

实测成本（RTX 3050 Laptop，4.3 GB，batch 16，确定性关）：**54 s/轮（含验证）**，
数据来自 ``results2中断/unet/NEU_300_0`` 那次 264 轮的真实运行（training_seconds 14,258）。
300 轮约 4.5 小时/种子，3 个种子约 13.5 小时。

用法::

    python train.py --model unet --config configUnetP300 --seed 42
    python run_experiment.py --config configUnetP300 --seeds 42,123,2024
"""
from pathlib import Path

from config_unet import *  # noqa: F401,F403

PROJECT_DIR = Path(__file__).resolve().parents[1]

EPOCHS = 300
SCHEDULER_NAME = "cosine"
WARMUP_EPOCHS = 5
# 确定性关：见模块说明里的成本对比。
DETERMINISTIC_ALGORITHMS = False
# 与 BiSeNetV2 那条路径一致：不裁剪梯度。
MAX_GRAD_NORM = 0.0
# 余弦退火负责收尾，不早停。用极大值而不是 300，是为了 --epochs 覆盖时也不打断。
EARLY_STOPPING_PATIENCE = 100000
SAVE_DIR = str(PROJECT_DIR / "results2中断" / "unet_p300")

# batch 16 不是在这里设的：`config_unet` 的 DATASET_DEFAULTS["neu"]["batch_size"] 已经是
# 16（U-Net 在 batch 32 下峰值 4.93 GB 超过 4.3 GB 显存）。这里刻意不再复制一份，
# 避免两处定义不一致；要改就改 config_unet 那一处。
