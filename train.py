"""NBS 统一训练入口。通过 --model 选择架构，--config 选择对应实验配置。"""
from __future__ import annotations

import argparse
import importlib
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CONFIG_DIR = ROOT / "config"
MODELS_DIR = ROOT / "models"
SCRIPTS_DIR = ROOT / "scripts"
for directory in (CONFIG_DIR, MODELS_DIR, SCRIPTS_DIR):
    sys.path.insert(0, str(directory))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Steel defect segmentation experiments with fixed NEU 2904/726/840 splits",
        add_help=False,
    )
    parser.add_argument("--model", choices=("unet", "bisenet", "scconv")
                        , default="unet"
                        , help="模型架构：unet、bisenet 或 scconv")
    parser.add_argument("--config"
                        , default="config_unet"
                        , help="配置模块名；默认为 config_unet / config / configScConv")
    known, remaining = parser.parse_known_args()

    defaults = {"unet": "config_unet", "bisenet": "config", "scconv": "configScConv"}
    config_name = known.config or defaults[known.model]

    if known.model == "unet":
        importlib.import_module("config")
        selected = importlib.import_module(config_name)
        # train_unet.py 里写的是 `import config_unet` / `from config_unet import ...`，
        # 所以要覆盖的键是 "config_unet"。覆盖 "config" 对它不起作用——这正是
        # 之前 `--model unet --config 任意值` 一直被静默忽略的原因。
        sys.modules["config_unet"] = selected
        script = SCRIPTS_DIR / "train_unet.py"
    elif known.model == "bisenet":
        importlib.import_module("config")
        selected = importlib.import_module(config_name)
        sys.modules["config"] = selected
        script = SCRIPTS_DIR / "train_bisenet.py"
    else:
        importlib.import_module("config")
        importlib.import_module("configScConv")
        selected = importlib.import_module(config_name)
        sys.modules["configScConv"] = selected
        script = SCRIPTS_DIR / "train_scconv.py"

    sys.argv = [str(script), *remaining]
    runpy.run_path(str(script), run_name="__main__")


if __name__ == "__main__":
    main()
