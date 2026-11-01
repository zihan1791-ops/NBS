"""PyCharm 一键实验入口。

用法：改下面「实验设置」里的 EXPERIMENT / SEEDS，然后在 PyCharm 里**右键 Run**
这个文件即可。不需要命令行参数。每个种子会单独起一个 python 进程串行跑，
PyCharm 的 Run 窗口里能实时看到训练日志。

跑完会自动调用 scripts/summarize_ablation.py 刷新
results2中断/ablation_summary.md 与 .csv。

也可以临时用命令行覆盖：
    python run_experiment.py --config configScConvR2 --seeds 42
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

# ============================ 实验设置（改这里）============================

EXPERIMENT = "S0"          # 取值见下面的 EXPERIMENTS
SEEDS = (42, 123, 2024)    # 想只跑一次就写 (42,)；确定性已开，同种子重跑结果逐位相同

# 追加给 train.py 的额外参数，例如 ["--epochs", "120"]
EXTRA_ARGS: list[str] = []

# 跳过已完成的种子（结果目录里已有 metrics.json 时）
SKIP_FINISHED = True

# 训练完自动刷新汇总
SUMMARIZE = True

# =========================================================================

PROJECT_DIR = Path(__file__).resolve().parent
CONFIG_DIR = PROJECT_DIR / "config"
MODELS_DIR = PROJECT_DIR / "models"

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
    "UNetP300": ("configUnetP300", "unet", "U-Net，论文协议（300 轮 cosine，batch 16）"),
}

ORDER = ("ORTH", "B0", "D2", "C2a", "C2b", "C2c", "S0", "D1", "P1", "P2", "G1",
         "R1", "R2", "R1D2", "UNetP300")


def parse_overrides() -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument("--config", default=None, help="直接指定配置模块名，覆盖 EXPERIMENT")
    parser.add_argument("--model", default=None, choices=("scconv", "unet", "bisenet"),
                        help="覆盖模型架构；默认取 EXPERIMENTS 里登记的那一个")
    parser.add_argument("--seeds", default=None, help="逗号分隔的种子，覆盖 SEEDS")
    parser.add_argument("--extra", nargs=argparse.REMAINDER, default=None,
                        help="其余参数原样传给 train.py")
    return parser.parse_known_args()[0]


def finished_seeds(output_dir: Path) -> set[int]:
    """扫描结果目录，返回已经产出结果文件的种子集合。

    两种落盘格式都要认：BiSeNet/ScConv 写 ``metrics.json``，U-Net 写
    ``test_metrics.json``；配置快照的层数也不一样（前者有 ``effective_arguments``）。
    """
    done: set[int] = set()
    if not output_dir.is_dir():
        return done
    for run_dir in output_dir.iterdir():
        if not run_dir.is_dir():
            continue
        config_path = run_dir / "config.json"
        if not config_path.is_file():
            continue
        if not (run_dir / "metrics.json").is_file() and not (run_dir / "test_metrics.json").is_file():
            continue
        try:
            with config_path.open("r", encoding="utf-8") as handle:
                document = json.load(handle)
        except Exception:
            continue
        arguments = document.get("effective_arguments", document)
        seed = arguments.get("seed")
        if seed is None:
            continue
        done.add(int(seed))
    return done


def resolve_output_dir(config_module, model: str) -> Path:
    """取出这次实验真正会写入的目录。

    U-Net 用的是 ``SAVE_DIR``，其余用 ``OUTPUT_DIR``。不能只读 ``OUTPUT_DIR``：
    ``config_unet`` 现在会从 ``config.py`` 继承一份 ``OUTPUT_DIR = results2中断/bisenetv2``，
    照着它扫"已完成种子"会扫错目录，甚至把 U-Net 的种子里已有 BiSeNet 结果的那些
    误判成"已完成"而跳过。
    """
    if model == "unet":
        return Path(config_module.SAVE_DIR)
    return Path(config_module.OUTPUT_DIR)


def main() -> int:
    overrides = parse_overrides()

    if overrides.config:
        config_name = overrides.config
        if overrides.model:
            model = overrides.model
        elif "unet" in config_name.lower():
            model = "unet"
        elif "bisenet" in config_name.lower() or "orth" in config_name.lower():
            model = "bisenet"
        else:
            model = "scconv"
        label = config_name
    else:
        if EXPERIMENT not in EXPERIMENTS:
            print(f"未知 EXPERIMENT={EXPERIMENT!r}；可选：{', '.join(ORDER)}", file=sys.stderr)
            return 2
        config_name, experiment_model, label = EXPERIMENTS[EXPERIMENT]
        model = overrides.model or experiment_model
        label = f"{EXPERIMENT}（{label}）"

    seeds = SEEDS
    if overrides.seeds:
        seeds = tuple(int(item) for item in overrides.seeds.split(",") if item.strip())
    extra_args = list(EXTRA_ARGS)
    if overrides.extra:
        extra_args.extend(item for item in overrides.extra if item and item != "--")

    sys.path.insert(0, str(CONFIG_DIR))
    sys.path.insert(0, str(MODELS_DIR))
    config_module = __import__(config_name)
    output_dir = resolve_output_dir(config_module, model)

    print("=" * 78)
    print(f"实验      : {label}")
    print(f"模型      : {model}")
    print(f"配置模块  : {config_name}")
    print(f"结果目录  : {output_dir}")
    print(f"种子      : {list(seeds)}")
    print(f"轮数      : {getattr(config_module, 'EPOCHS', '?')}"
          f"（调度 {getattr(config_module, 'SCHEDULER_NAME', 'plateau')}，"
          f"确定性 {getattr(config_module, 'DETERMINISTIC_ALGORITHMS', False)}）")
    if extra_args:
        print(f"额外参数  : {extra_args}")
    print("=" * 78)

    done = finished_seeds(output_dir) if SKIP_FINISHED else set()
    if done:
        print(f"已完成且将跳过的种子：{sorted(done)}（改 SKIP_FINISHED=False 可强制重跑）\n")

    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUNBUFFERED"] = "1"
    failures = []

    for seed in seeds:
        if seed in done:
            print(f"[skip] seed={seed} 已有结果文件")
            continue
        argv = [
            sys.executable,
            str(PROJECT_DIR / "train.py"),
            "--model", model,
            "--config", config_name,
            "--seed", str(seed),
            *extra_args,
        ]
        print(f"\n>>> seed={seed}\n>>> {' '.join(argv)}\n", flush=True)
        completed = subprocess.run(argv, cwd=str(PROJECT_DIR), env=env, stderr=subprocess.STDOUT)
        if completed.returncode != 0:
            failures.append((seed, completed.returncode))
            print(f"!!! seed={seed} 失败，退出码 {completed.returncode}", file=sys.stderr)

    if SUMMARIZE:
        print("\n刷新汇总 …")
        subprocess.run(
            [sys.executable, str(PROJECT_DIR / "scripts" / "summarize_ablation.py"), "--with_cost"],
            cwd=str(PROJECT_DIR), env=env,
        )

    if failures:
        print(f"\n失败的种子：{failures}", file=sys.stderr)
        return 1
    print("\n全部完成。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
