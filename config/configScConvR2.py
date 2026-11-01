"""R2 臂：把语义分支 GELayerS1 里的 3x3 卷积替换成 ScConv（替换式）。

## 替换了什么

语义分支 stage3/4/5 由 ``GELayerS2 + N×GELayerS1`` 组成，每个 GELayerS1 形如::

    conv1 = ConvBNReLU(C -> C, 3x3)          # <- 替换
    dwconv = depthwise 3x3 (C -> C*expansion)  # 保留（负责扩展，ScConv 不做升维）
    project = 1x1 (C*expansion -> C)           # 保留

替换共 **5 个**（stage3 有 1 个 GELayerS1、stage4 有 1 个、stage5 有 3 个），
覆盖 stride 8 / 16 / 32 三个尺度：

    semantic.stage3[1].conv1 (32ch)
    semantic.stage4[1].conv1 (64ch)
    semantic.stage5[1..3].conv1 (128ch)

## 与 R1 的区别

* R1 在细节分支（高分辨率、浅层、类内重复 3x3），R2 在语义分支（低分辨率、
  深层、expansion=6 的高冗余结构）。R2 更贴近 ScConv 论文的实验场景
  （深层高通道），但替换掉的容量也更多，掉点风险更高。
* R2 的通道数跨 32/64/128 三档，ScConv 在这些通道数下都能满足 GroupNorm 分组
  与 CRU 的整除约束（已由 measure_cost.py 验证）。
* 与 R1 一样：**不含** BGA 之后的附加 ScConv，保证变量唯一。

⚠️ 预期它比 R1 更激进：参数量下降幅度更大（约 -10%），是"省算力"叙事的
最强版本，但如果掉点，说明语义分支的这部分容量是必要的。
"""
from pathlib import Path

from configScConvAblation import *  # noqa: F401,F403

OUTPUT_DIR = PROJECT_DIR / "results2中断" / "bisenetv2_repl_R2_semantic"
EXPERIMENT_NAME = "R2-replace-semantic"
EXPERIMENT_NOTE = "替换式 R2：semantic 分支 5 个 GELayerS1.conv1 → ScConv。"

USE_SC_CONV = True
SC_CONV_PLACEMENT = ()
SC_CONV_REPLACE = ("semantic.ge_conv1",)
SC_CONV_REPLACE_KEEP_RELU = True


def to_dict():
    return {name: str(value) if isinstance(value, Path) else value
            for name, value in globals().copy().items() if name.isupper()}
