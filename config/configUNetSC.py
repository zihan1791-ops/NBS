"""U-Net + ScConv 位置消融配置。

数据、增强、类别定义和训练设置继承 configUnetP300，确保所有消融臂采用相同协议。
通过 UNETSC_ABLATION 设置模型结构；启动脚本会逐臂传入对应值。
"""
from pathlib import Path
import os

# 根据启动脚本选择数据集配置；SD900 使用项目已有的独立数据根目录与 batch 设置。
if os.environ.get("UNETSC_DATASET", "NEU").upper() == "SD900":
    from configSD900Unet import *  # noqa: F401,F403
else:
    from configUnetP300 import *  # noqa: F401,F403

PROJECT_DIR = Path(__file__).resolve().parents[1]
UNETSC_ABLATION = "baseline"
# 实际训练输出目录由启动脚本按消融臂覆盖到 G:/NBS/UNetSC/<arm>。
SAVE_DIR = str(PROJECT_DIR / "UNetSC" / UNETSC_ABLATION)

# train.py 在加载训练脚本前会导入本配置；替换导入别名，使既有训练器使用本模型。
import sys
from UNetSC import build_unet as _build_unet
import types
_unet_module = types.ModuleType("UNet")
_unet_module.build_unet = _build_unet
sys.modules["UNet"] = _unet_module

