"""PyCharm entry point for the complete, isolated SD900 experiment set."""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from run_experiment import finished_seeds

# ============================ 实验设置（PyCharm 运行时改这里）============================
# Set one key to run one experiment; use "ALL" to run all registered experiments in order.
EXPERIMENT = "R1"
SEEDS = (42, 123, 2024)
BATCH_SIZE = 8
SKIP_FINISHED = True

# SD900 保持与项目实验表相同的已注册配置。
# UNetP300 通过 configSD900Unet 进行路由，它封装了 configUnetP300 并仅更改 SD900 数据集别名、批量大小和独立输出目录。

EXPERIMENTS = {
    "ORTH": ("bisenetv2_orth_config", "bisenet", "BiSeNetV2 + 跨分支正交去相关损失"),
    # 论文统一协议：cosine 300 轮跑满 + 不早停，全部臂共用一份基座配置
    "B0": ("configScConvB0", "scconv", "基线：关闭 ScConv"),
    "S0": ("configScConvAblation", "scconv", "现状：ScConv 挂在 BGA 之后 + 残差 0.1"),
    "D1": ("configScConvD1", "scconv", "残差初值 0.1 → 1.0（模块完全激活）"),
    "D2": ("configScConvD2", "scconv", "去掉残差，fused 位置直插"),
    # ---- 归因对照（E1）：fused 位置换成普通块 ----
    "C2a": ("configScConvC2a", "scconv", "fused 位置换 Conv1x1+BN+ReLU（同位置、零 ScConv）"),
    "C2b": ("configScConvC2b", "scconv", "fused 位置换 Conv3x3+BN+ReLU（同位置、零 ScConv）"),
    "C2c": ("configScConvC2c", "scconv", "fused 位置只放 ReLU（零参数）"),
    # ---- 位置消融 ----
    "P1": ("configScConvP1", "scconv", "只作用于细节分支输出"),
    "P2": ("configScConvP2", "scconv", "只作用于语义分支上下文输出"),
    "G1": ("configScConvG1", "scconv", "SRU 门控去掉 γ/Σγ 归一化"),
    # ---- 替换式主线 ----
    "R1": ("configScConvR1", "scconv", "替换 detail 分支 4 个重复 3x3 卷积（-8.1% 参数, -15.7% FLOPs）"),
    "R2": ("configScConvR2", "scconv", "替换 semantic 分支 5 个 GELayerS1.conv1（-10.8% 参数）"),
    "R1D2": ("configScConvR1D2", "scconv", "R1 + D2 组合"),
    # ---- 基线（E3）----
    "UNetP300": ("configUnetP300", "unet", "U-Net，论文协议（300 轮 cosine，SD900 batch 8）"),
}
EXPERIMENT_ORDER = tuple(EXPERIMENTS)
SC_CONV_ARMS = tuple(key for key, (_, model, _) in EXPERIMENTS.items() if model == "scconv")
# =========================================================================================

PROJECT_DIR = Path(__file__).resolve().parent
DATA_ROOT = PROJECT_DIR / "data" / "SD-saliency"
RESULTS_ROOT = PROJECT_DIR / "results2中断" / "SD900"
TRAIN_ADAPTER = PROJECT_DIR / "train_SD900.py"


def parse_args():
    parser = argparse.ArgumentParser(description="Run isolated SD900 experiments")
    parser.add_argument("--experiment", choices=(*EXPERIMENTS, "ALL"), default=None,
                        help="select one registered experiment or ALL")
    parser.add_argument("--all-scconv", action="store_true", help="run all 13 ScConv/BiSeNet experiment arms")
    parser.add_argument("--model", choices=("unet", "bisenet", "scconv"), default=None,
                        help="optional one-off model override")
    parser.add_argument("--config", default=None, help="optional config override")
    parser.add_argument("--seeds", default=",".join(map(str, SEEDS)))
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--device", choices=("cuda", "cpu"), default=None)
    parser.add_argument("--smoke", action="store_true",
                        help="check one real batch only; no training or saving")
    parser.add_argument("--no-skip-finished", action="store_true")
    return parser.parse_args()


def resolve_sd_config(tag: str, config_name: str) -> str:
    """Use the SD900 wrapper while retaining the registered U-Net P300 protocol."""
    if tag == "UNetP300" and config_name == "configUnetP300":
        return "configSD900Unet"
    return config_name


def main() -> int:
    args = parse_args()
    if args.batch_size < 1:
        raise SystemExit("--batch-size 必须大于 0")
    if args.all_scconv and (args.model or args.config or args.experiment):
        raise SystemExit("--all-scconv 不能和 --model、--config 或 --experiment 同时使用")
    seeds = tuple(int(item.strip()) for item in args.seeds.split(",") if item.strip())
    if not seeds:
        raise SystemExit("至少需要一个训练 seed")

    selected = args.experiment or EXPERIMENT
    if args.all_scconv:
        jobs = [(tag, *EXPERIMENTS[tag][:2]) for tag in SC_CONV_ARMS]
    elif selected == "ALL":
        jobs = [(tag, *EXPERIMENTS[tag][:2]) for tag in EXPERIMENT_ORDER]
    elif args.model:
        model = args.model
        config_name = args.config or ("configSD900Unet" if model == "unet" else "config")
        tag = next((name for name, (cfg, registered_model, _) in EXPERIMENTS.items()
                    if cfg == config_name and registered_model == model), model)
        jobs = [(tag, config_name, model)]
    elif args.config:
        if "unet" in args.config.lower():
            model = "unet"
        elif "bisenet" in args.config.lower() or "orth" in args.config.lower():
            model = "bisenet"
        else:
            model = "scconv"
        tag = next((name for name, (cfg, registered_model, _) in EXPERIMENTS.items()
                    if cfg == args.config and registered_model == model), args.config)
        jobs = [(tag, args.config, model)]
    elif selected in EXPERIMENTS:
        config_name, model, _description = EXPERIMENTS[selected]
        jobs = [(selected, config_name, model)]
    else:
        raise SystemExit(f"未知实验 {selected!r}；可选：{', '.join(EXPERIMENT_ORDER)} 或 ALL")

    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUNBUFFERED"] = "1"
    failed = []
    for tag, registered_config, model in jobs:
        config_name = resolve_sd_config(tag, args.config or registered_config)
        if model == "scconv":
            output_dir = RESULTS_ROOT / "scconv" / tag
        elif model == "unet":
            output_dir = RESULTS_ROOT / "unet"
        else:
            output_dir = RESULTS_ROOT / "bisenet" / tag

        if args.smoke:
            command = [sys.executable, str(TRAIN_ADAPTER), "--smoke",
                       "--model", model, "--config", config_name,
                       "--batch_size", str(args.batch_size)]
            if args.device:
                command.extend(("--device", args.device))
            print("\n>>> " + " ".join(command), flush=True)
            result = subprocess.run(command, cwd=PROJECT_DIR, env=env, stderr=subprocess.STDOUT)
            if result.returncode:
                failed.append((tag, result.returncode))
            continue

        done = finished_seeds(output_dir) if (SKIP_FINISHED and not args.no_skip_finished) else set()
        for seed in seeds:
            if seed in done:
                print(f"[skip] {tag} seed={seed} 已完成")
                continue
            command = [
                sys.executable, str(TRAIN_ADAPTER),
                "--model", model, "--config", config_name,
                "--dataset", "sd-saliency", "--data_root", str(DATA_ROOT),
                "--batch_size", str(args.batch_size), "--seed", str(seed),
            ]
            command.extend(("--save_dir" if model == "unet" else "--output_dir", str(output_dir)))
            if args.epochs is not None:
                command.extend(("--epochs", str(args.epochs)))
            if args.device:
                command.extend(("--device", args.device))
            print(f"\n>>> {tag} seed={seed}\n>>> {' '.join(command)}\n", flush=True)
            result = subprocess.run(command, cwd=PROJECT_DIR, env=env, stderr=subprocess.STDOUT)
            if result.returncode:
                failed.append((tag, seed, result.returncode))

    if args.smoke:
        return 1 if failed else 0
    if failed:
        print(f"失败任务：{failed}", file=sys.stderr)
        return 1
    metrics = PROJECT_DIR / "scripts" / "run_metrics.py"
    if metrics.is_file():
        print("\nSD900 已有结果（只读汇总，不触碰 NEU 汇总文件）：", flush=True)
        subprocess.run([sys.executable, str(metrics), "--root", str(RESULTS_ROOT)],
                       cwd=PROJECT_DIR, env=env, check=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
