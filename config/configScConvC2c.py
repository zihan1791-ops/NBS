"""C2c 臂：fused 位置**只放一个 ReLU**，零参数对照。

这一臂存在的唯一目的是排除最便宜的伪解释：D2 的增益是不是仅仅因为"融合特征
之后又多过了一次非线性"。输出 = ``ReLU(BGA(detail, semantic))``，参数量与 B0
完全相同（3,622,820），因为 ReLU 没有参数。

判定：
  * C2c ≈ B0 → 多一层非线性本身无效，D2/C2a/C2b 的差异必须由块本身解释；
  * C2c ≈ D2 → 说明"加任何东西都行"，前面的归因结论要推翻重来（正常情况下
    不会出现，出现了就先查初始化与 BN 是否引入了伪影）。

因为要与 D2 保持逐项可比，它继承 D2（无残差）并保持 ``USE_SC_CONV=True``：
模型里确实存在 fused 位置的块，只是这个块是 ReLU。
"""
from pathlib import Path

from configScConvD2 import *  # noqa: F401,F403

OUTPUT_DIR = PROJECT_DIR / "results2中断" / "bisenetv2_control_C2c_relu"
EXPERIMENT_NAME = "C2c-relu-control"
EXPERIMENT_NOTE = "D2 归因对照：fused 位置只放 ReLU（零参数），其余不变。"

SC_CONV_FUSED_BLOCK = "relu"


def to_dict():
    return {name: str(value) if isinstance(value, Path) else value
            for name, value in globals().copy().items() if name.isupper()}
