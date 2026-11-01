"""PyCharm-friendly isolated runner for BiSeNetV2 SD900/NEU-Seg ablations.

Edit the settings below, then right-click this file and choose Run in PyCharm.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from run_experiment import finished_seeds

# ============================ PyCharm settings ============================
DATASETS = ("sd-saliency",)  # "neu", "sd-saliency", or both
EXPERIMENT = "ALL"  # Baseline, Exp-1 ... Exp-6, or ALL
SEEDS = (42, 123, 2024)
BATCH_SIZE = None  # None keeps each dataset's configured default
EPOCHS = None
DEVICE = None  # "cuda", "cpu", or None
SKIP_FINISHED = True
# =========================================================================

PROJECT_DIR = Path(__file__).resolve().parent
RESULTS_ROOT = PROJECT_DIR / "results2中断" / "bisenetv2_sdneu_defect_ablation"
CONFIG_PACKAGE = "bisenetv2_sdneu_defect_ablation_configs"

EXPERIMENTS = {
    "Baseline": f"{CONFIG_PACKAGE}.baseline",
    "Baseline_CEfg2": f"{CONFIG_PACKAGE}.baseline",
    "Baseline_CEfg15": f"{CONFIG_PACKAGE}.baseline",
    "Exp-1": f"{CONFIG_PACKAGE}.exp1_multiscale_context",
    "Exp-2": f"{CONFIG_PACKAGE}.exp2_defect_aware_bga",
    "Exp-3": f"{CONFIG_PACKAGE}.exp3_boundary_supervision",
    "Exp-4": f"{CONFIG_PACKAGE}.exp4_multiscale_context_bga",
    "Exp-5": f"{CONFIG_PACKAGE}.exp5_bga_boundary",
    "Exp-6": f"{CONFIG_PACKAGE}.exp6_full",
}
# Keep weighted-loss arms opt-in; the historical ALL run remains unchanged.
WEIGHTED_CE_EXPERIMENTS = {"Baseline_CEfg2": 2.0, "Baseline_CEfg15": 1.5}
EXPERIMENT_ORDER = tuple(key for key in EXPERIMENTS if key not in WEIGHTED_CE_EXPERIMENTS)


def parse_args():
    parser = argparse.ArgumentParser(description="Run isolated BiSeNetV2 defect ablations")
    parser.add_argument("--dataset", choices=("neu", "sd-saliency", "both"), default=None)
    parser.add_argument("--experiment", choices=(*EXPERIMENTS, "ALL"), default=None)
    parser.add_argument("--seeds", default=None, help="comma-separated seeds")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--device", choices=("cuda", "cpu"), default=None)
    parser.add_argument("--no-skip-finished", action="store_true")
    return parser.parse_args()


def selected_datasets(value):
    if value == "both":
        return ("neu", "sd-saliency")
    if value:
        return (value,)
    return tuple(DATASETS)


def selected_experiments(value):
    tag = value or EXPERIMENT
    if tag == "ALL":
        return list(EXPERIMENT_ORDER)
    if tag not in EXPERIMENTS:
        raise SystemExit(f"Unknown experiment {tag!r}; choose one of {', '.join(EXPERIMENTS)} or ALL.")
    return [tag]


def command_for(dataset, tag, config_name, seed, output_dir, args):
    if dataset == "sd-saliency":
        command = [
            sys.executable,
            str(PROJECT_DIR / "train_SD900.py"),
            "--model", "bisenet",
            "--config", config_name,
            "--dataset", "sd-saliency",
            "--output_dir", str(output_dir),
            "--seed", str(seed),
        ]
        if tag in WEIGHTED_CE_EXPERIMENTS:
            command.extend(("--foreground_ce_weight", str(WEIGHTED_CE_EXPERIMENTS[tag])))
        if args.batch_size is not None:
            command.extend(("--batch_size", str(args.batch_size)))
        if args.device:
            command.extend(("--device", args.device))
    else:
        command = [
            sys.executable,
            str(PROJECT_DIR / "train.py"),
            "--model", "bisenet",
            "--config", config_name,
            "--dataset", "neu",
            "--output_dir", str(output_dir),
            "--seed", str(seed),
        ]
        if args.batch_size is not None:
            command.extend(("--batch_size", str(args.batch_size)))
        if args.device:
            command.extend(("--device", args.device))
    if args.epochs is not None:
        command.extend(("--epochs", str(args.epochs)))
    return command


def main() -> int:
    args = parse_args()
    if args.seeds:
        seeds = tuple(
            int(item.strip()) for item in args.seeds.split(",") if item.strip()
        )
    else:
        seeds = tuple(int(seed) for seed in SEEDS)
    if not seeds:
        raise SystemExit("At least one seed is required.")

    datasets = selected_datasets(args.dataset)
    experiments = selected_experiments(args.experiment)
    weighted_ce_experiments = set(WEIGHTED_CE_EXPERIMENTS)
    if weighted_ce_experiments.intersection(experiments) and datasets != ("sd-saliency",):
        raise SystemExit("Weighted-CE experiments are defined only for the SD-saliency dataset.")
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUNBUFFERED"] = "1"
    failures = []

    for dataset in datasets:
        dataset_dir = "NEU" if dataset == "neu" else "SD900"
        for tag in experiments:
            config_name = EXPERIMENTS[tag]
            for seed in seeds:
                output_dir = RESULTS_ROOT / dataset_dir / tag / f"seed_{seed}"
                done = (
                    finished_seeds(output_dir)
                    if SKIP_FINISHED and not args.no_skip_finished
                    else set()
                )
                if seed in done:
                    print(f"[skip] {dataset_dir}/{tag} seed={seed} already has metrics.json")
                    continue

                command = command_for(dataset, tag, config_name, seed, output_dir, args)
                print(f"\n>>> {dataset_dir} / {tag} / seed={seed}\n>>> {' '.join(command)}\n", flush=True)
                result = subprocess.run(command, cwd=PROJECT_DIR, env=env, stderr=subprocess.STDOUT)
                if result.returncode:
                    failures.append((dataset_dir, tag, seed, result.returncode))

    if failures:
        print(f"Failed jobs: {failures}", file=sys.stderr)
        return 1

    summary_script = PROJECT_DIR / "scripts" / "summarize_bisenetv2_defect.py"
    if summary_script.is_file():
        subprocess.run(
            [sys.executable, str(summary_script), "--root", str(RESULTS_ROOT)],
            cwd=PROJECT_DIR,
            env=env,
            check=False,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
