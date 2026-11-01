"""ScConv 消融实验的共享基座（同时就是 S0 臂）。

与 configScConv.py 的区别只有"训练协议"，架构参数完全一致：
1. DETERMINISTIC_ALGORITHMS=True：开启 torch 全确定性算法，消除同配置两次运行
   之间约 1% 的不可复现波动（旧实验里 bisenetv2_scconv 与 _config1 配置逐字节
   相同、test mIoU 却差 1.13%，就是这个问题）。副作用是 33.7 → 约 50 s/epoch，
   这个代价必须付，否则任何对比都在比较随机噪声。
2. SCHEDULER_NAME="cosine" + WARMUP_EPOCHS=5：替换 ReduceLROnPlateau。旧协议里
   lr 在 ~120 轮就塌到 min_lr=1e-6，后面 100+ 轮几乎不学习。
3. EPOCHS=300，且**不早停**：给每个模型足够的优化步数，让"谁的上限更高"这个
   问题尽量不受训练预算的干扰。余弦退火的 lr 曲线是按总轮数排定的，第 300 轮才
   降到最低——如果配早停，会在 lr 还很高的时候把模型掐掉，反而系统性压低 mIoU。
   所以这里 patience 设成一个极大值：跑满全程，由"最佳验证检查点"来取最好结果。

300 轮 × 约 50 s/epoch ≈ 4.2 小时/次。22 个已有臂 + 7 个归因对照臂 ≈ 120 小时，
U-Net（batch 16、确定性关）另算约 3.5 小时/种子。

本文件的默认架构就是 S0（现状配置：只插 fused、带残差、scale=0.1），
其余消融臂（B0/D1/D2/C2a/C2b/C2c/P1/P2/P3/G1/R1/R2/R1D2）都以本模块为父配置，
只覆盖一个变量。
"""
from pathlib import Path

from configScConv import *  # noqa: F401,F403

PROJECT_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_DIR / "results2中断" / "bisenetv2_ablation_S0_scconv_fused"
EXPERIMENT_NAME = "S0-scconv-fused"
EXPERIMENT_NOTE = "消融基座/S0：ScConv 只作用于 BGA 融合特征，带残差，初始尺度 0.1。"

# ---- 训练协议（论文统一协议：cosine 300 轮跑满，不早停）----
DETERMINISTIC_ALGORITHMS = True
SCHEDULER_NAME = "cosine"
WARMUP_EPOCHS = 5
EPOCHS = 300
# 余弦退火负责收尾，早停不该介入。用极大值而不是 300，是为了 `--epochs` 覆盖时
# （比如临时试 400 轮）也不会被这条耐心值提前打断。
EARLY_STOPPING_PATIENCE = 100000

# ---- 架构（与 configScConv 保持一致，S0 = 现状）----
USE_SC_CONV = True
SC_CONV_PLACEMENT = ("fused",)
SC_CONV_RESIDUAL = True
SC_CONV_RESIDUAL_SCALE = 0.1
# True = 官方写法 γ/Σγ（实测门控会塌到 0.5 附近，见 G1 臂说明）
SC_CONV_GAMMA_NORMALISE = True
# 替换式应用：把已有卷积整块换成 ScConv（空元组 = 不替换，维持附加式）
SC_CONV_REPLACE = ()
SC_CONV_REPLACE_KEEP_RELU = True
# fused 位置实际放什么块。默认 "scconv"；C2a/C2b/C2c 归因对照臂把它换成
# 普通 1x1 卷积 / 普通 3x3 卷积 / 纯 ReLU，用来判断 D2 的增益属于算子还是位置。
SC_CONV_FUSED_BLOCK = "scconv"


def to_dict():
    return {name: str(value) if isinstance(value, Path) else value
            for name, value in globals().copy().items() if name.isupper()}
