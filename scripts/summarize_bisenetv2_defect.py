"""Summarize only the isolated BiSeNetV2 SD900/NEU-Seg defect ablations."""
from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path


def read_runs(root: Path):
    rows = []
    for metrics_path in sorted(root.glob("*/*/seed_*/**/metrics.json")):
        try:
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            config = json.loads(
                (metrics_path.parent / "config.json").read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError):
            continue

        relative = metrics_path.relative_to(root)
        if len(relative.parts) < 5:
            continue
        effective = config.get("effective_arguments", {})
        test = metrics.get("test_metrics", {})
        rows.append({
            "dataset": relative.parts[0],
            "experiment": relative.parts[1],
            "seed": effective.get("seed", relative.parts[2].removeprefix("seed_")),
            "best_epoch": metrics.get("best_epoch"),
            "mIoU": test.get("mIoU"),
            "mDice": test.get("mDice"),
            "accuracy": test.get("accuracy"),
            "boundary_f1": test.get("boundary_f1"),
            "training_seconds": metrics.get("training_seconds"),
            "parameter_count": metrics.get("parameter_count"),
            "run_dir": str(metrics_path.parent),
        })
    return rows


def mean_std(values):
    numbers = [float(value) for value in values if value is not None]
    if not numbers:
        return "—"
    mean = statistics.mean(numbers)
    deviation = statistics.stdev(numbers) if len(numbers) > 1 else 0.0
    return f"{mean:.4f} ± {deviation:.4f}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "results2中断"
        / "bisenetv2_sdneu_defect_ablation",
    )
    args = parser.parse_args()
    rows = read_runs(args.root)
    args.root.mkdir(parents=True, exist_ok=True)

    columns = (
        "dataset", "experiment", "seed", "best_epoch", "mIoU", "mDice",
        "accuracy", "boundary_f1", "training_seconds", "parameter_count", "run_dir",
    )
    csv_path = args.root / "summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

    grouped = {}
    for row in rows:
        grouped.setdefault((row["dataset"], row["experiment"]), []).append(row)

    lines = [
        "# BiSeNetV2 SD900 / NEU-Seg defect ablation",
        "",
        "Metrics are mean ± standard deviation over completed seeds. Boundary F1 uses a one-pixel matching tolerance.",
        "",
        "| Dataset | Experiment | Seeds | mIoU | mDice | Accuracy | Boundary F1 |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for (dataset, experiment), items in sorted(grouped.items()):
        lines.append(
            f"| {dataset} | {experiment} | {len(items)} | "
            f"{mean_std(item['mIoU'] for item in items)} | "
            f"{mean_std(item['mDice'] for item in items)} | "
            f"{mean_std(item['accuracy'] for item in items)} | "
            f"{mean_std(item['boundary_f1'] for item in items)} |"
        )
    (args.root / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {csv_path} and {args.root / 'summary.md'}")


if __name__ == "__main__":
    main()
