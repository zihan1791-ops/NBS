"""PyCharm 一键运行 U-Net + ScConv 位置消融实验。

先在 DATASET 中选择 NEU 或 SD900，再运行本脚本。每个数据集下依次训练六个消融臂，
每臂对三个随机种子串行训练。结果保存在 UNetSC/<数据集>/<消融臂>/。
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

# ============================ 实验设置（改这里）============================
DATASET = "SD900"  # 可选："NEU" / "SD900"
SEEDS = (42, 123, 2024)  # 单次试跑可改为 (42, 123, 2024)
ABLATIONS = (
    "baseline",            # U-Net 基线，不使用 ScConv
    "encoder_deep",        # 仅 down3 / down4
    "bottleneck",          # 仅最深层 down4
    "decoder_deep",        # 仅 up1 / up2
    "encoder_bottleneck",  # down3 / down4
    "all",                 # 所有编码器、解码器双卷积块
)
SKIP_FINISHED = True
EXTRA_ARGS: list[str] = ["--epochs", "150"]  # 例如 ["--epochs", "120"]
# ==========================================================================

PROJECT_DIR = Path(__file__).resolve().parent
DATASET_OPTIONS = {"NEU": "neu", "SD900": "sd-saliency"}
if DATASET not in DATASET_OPTIONS:
    raise ValueError(f"未知数据集 {DATASET!r}；可选：{tuple(DATASET_OPTIONS)}")
DATASET_NAME = DATASET_OPTIONS[DATASET]
OUTPUT_ROOT = PROJECT_DIR / "UNetSC" / DATASET


def finished_seeds(output_dir: Path) -> set[int]:
    """返回已有测试指标文件的种子。"""
    done: set[int] = set()
    if not output_dir.is_dir():
        return done
    for run_dir in output_dir.iterdir():
        metrics = run_dir / "test_metrics.json"
        config_path = run_dir / "config.json"
        if not run_dir.is_dir() or not metrics.is_file() or not config_path.is_file():
            continue
        try:
            data = json.loads(config_path.read_text(encoding="utf-8"))
            args = data.get("effective_arguments", data)
            if args.get("seed") is not None:
                done.add(int(args["seed"]))
        except (OSError, ValueError, TypeError):
            continue
    return done


def main() -> int:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUNBUFFERED"] = "1"
    failures = []
    print("=" * 78)
    print("实验：U-Net + ScConv 位置消融")
    print(f"数据集：{DATASET} ({DATASET_NAME})")
    print(f"消融臂：{ABLATIONS}")
    print(f"种子：{SEEDS}")
    print(f"结果目录：{OUTPUT_ROOT}")
    print("=" * 78)

    for ablation in ABLATIONS:
        output_dir = OUTPUT_ROOT / ablation
        output_dir.mkdir(parents=True, exist_ok=True)
        done = finished_seeds(output_dir) if SKIP_FINISHED else set()
        if done:
            print(f"[{ablation}] 已完成并跳过种子：{sorted(done)}")
        for seed in SEEDS:
            if seed in done:
                continue
            argv = [
                sys.executable, str(PROJECT_DIR / "train.py"),
                "--model", "unet", "--config", "configUNetSC",
                "--seed", str(seed), "--dataset", DATASET_NAME,
                "--save_dir", str(output_dir), *EXTRA_ARGS,
            ]
            run_env = dict(env)
            run_env["UNETSC_ABLATION"] = ablation
            run_env["UNETSC_DATASET"] = DATASET
            print(f"\n>>> dataset={DATASET}, ablation={ablation}, seed={seed}\n"
                  f">>> {' '.join(argv)}\n", flush=True)
            result = subprocess.run(argv, cwd=str(PROJECT_DIR), env=run_env,
                                    stderr=subprocess.STDOUT)
            if result.returncode:
                failures.append((DATASET, ablation, seed, result.returncode))
                print(f"!!! {ablation} seed={seed} 失败，退出码 {result.returncode}",
                      file=sys.stderr)

    if failures:
        print(f"\n失败任务：{failures}", file=sys.stderr)
        return 1
    print("\n全部消融实验完成。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
