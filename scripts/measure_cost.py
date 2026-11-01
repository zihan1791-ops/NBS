"""测量各消融臂的参数量与推理 FLOPs（用 torch 自带算子计数器，无需额外依赖）。

用法：
    python scripts/measure_cost.py
    python scripts/measure_cost.py --configs configScConvB0 configScConvD2 --height 200 --width 200

只在推理模式（eval）下统计，即部署时的单次前向；训练模式下辅助头会额外计算。
"""
from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
for directory in (PROJECT_DIR / "config", PROJECT_DIR / "models", SCRIPT_DIR):
    sys.path.insert(0, str(directory))

import torch  # noqa: E402
from torch.utils.flop_counter import FlopCounterMode  # noqa: E402

import config as base_config  # noqa: E402

DEFAULT_CONFIGS = (
    "configScConvB0",
    "configScConvAblation",
    "configScConvD2",
    "configScConvC2a",
    "configScConvC2b",
    "configScConvC2c",
    "configScConvG1",
    "configScConvP2",
)


def build(config_name: str):
    module = importlib.import_module(config_name)
    sys.modules["configScConv"] = module
    for cached in ("bisenetv2_scconv", "bisenetv2"):
        sys.modules.pop(cached, None)
    scconv = importlib.import_module("bisenetv2_scconv")
    return scconv.build_bisenetv2_scconv(
        scconv_enabled=module.USE_SC_CONV,
        scconv_placement=module.SC_CONV_PLACEMENT,
        scconv_residual=module.SC_CONV_RESIDUAL,
        scconv_residual_scale=module.SC_CONV_RESIDUAL_SCALE,
        scconv_gamma_normalise=getattr(module, "SC_CONV_GAMMA_NORMALISE", True),
        scconv_group_num=module.SC_CONV_GROUP_NUM,
        scconv_gate_threshold=module.SC_CONV_GATE_THRESHOLD,
        scconv_alpha=module.SC_CONV_ALPHA,
        scconv_squeeze_ratio=module.SC_CONV_SQUEEZE_RATIO,
        scconv_group_size=module.SC_CONV_GROUP_SIZE,
        scconv_group_kernel_size=module.SC_CONV_GROUP_KERNEL_SIZE,
        scconv_replace=getattr(module, "SC_CONV_REPLACE", ()),
        scconv_replace_keep_relu=getattr(module, "SC_CONV_REPLACE_KEEP_RELU", True),
        scconv_fused_block=getattr(module, "SC_CONV_FUSED_BLOCK", "scconv"),
        dropout=module.HEAD_DROPOUT,
    ), module


def main() -> None:
    parser = argparse.ArgumentParser(description="测量参数量与推理 FLOPs")
    parser.add_argument("--configs", nargs="+", default=list(DEFAULT_CONFIGS))
    parser.add_argument("--height", type=int, default=base_config.IMAGE_SIZE)
    parser.add_argument("--width", type=int, default=base_config.IMAGE_SIZE)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    args = parser.parse_args()

    if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()):
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print(f"input = 1x{base_config.INPUT_CHANNELS}x{args.height}x{args.width}, device={device}, mode=eval")
    header = f"{'config':24s} {'params':>10s} {'Δparams':>10s} {'GFLOPs':>9s} {'ΔGFLOPs':>10s}"
    print(header)
    print("-" * len(header))

    reference = None
    for config_name in args.configs:
        model, module = build(config_name)
        model = model.to(device).eval()
        params = sum(parameter.numel() for parameter in model.parameters())

        with torch.no_grad():
            inputs = torch.randn(1, base_config.INPUT_CHANNELS, args.height, args.width, device=device)
            counter = FlopCounterMode(display=False)
            with counter:
                model(inputs)
        total_flops = counter.get_total_flops()

        if reference is None:
            reference = (params, total_flops)
        delta_params = params - reference[0]
        delta_flops = total_flops - reference[1]
        print(f"{config_name:24s} {params:10,d} {delta_params:+10,d} "
              f"{total_flops / 1e9:9.2f} {delta_flops / 1e9:+10.2f}")

    print("\nΔ 以第一个 config 为基准。FLOPs 为乘加计 2 次的约定（torch 计数器口径）。")


if __name__ == "__main__":
    main()
