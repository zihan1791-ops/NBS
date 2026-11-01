"""Paired, per-image attribution for the SD-saliency BiSeNetV2 ablation.

Run from G:\\NBS with its project Python environment. This script only reads
the existing data/checkpoints and writes reports under results2中断/.../结果归因.
"""
from __future__ import annotations

import argparse
import csv
import importlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


EXPERIMENTS = ("Baseline", "Baseline_CEfg2", "Baseline_CEfg15", "Exp-1", "Exp-2", "Exp-3", "Exp-4", "Exp-5", "Exp-6")
SEEDS = (42, 123, 2024)


def setup_imports(project: Path, experiment: str):
    for path in (project / "config", project / "models", project / "scripts"):
        sys.path.insert(0, str(path))
    config_map = {
        "Baseline": "baseline",
        "Baseline_CEfg2": "baseline",
        "Baseline_CEfg15": "baseline",
        "Exp-1": "exp1_multiscale_context",
        "Exp-2": "exp2_defect_aware_bga",
        "Exp-3": "exp3_boundary_supervision",
        "Exp-4": "exp4_multiscale_context_bga",
        "Exp-5": "exp5_bga_boundary",
        "Exp-6": "exp6_full",
    }
    cfg = importlib.import_module(
        f"bisenetv2_sdneu_defect_ablation_configs.{config_map[experiment]}"
    )
    sys.modules["config"] = cfg
    from bisenetv2_sdneu_defect_ablation import build_bisenetv2_sdneu_defect_ablation
    from dataset import build_loaders
    return cfg, build_bisenetv2_sdneu_defect_ablation, build_loaders


def load_model(project, experiment, run_dir, device):
    cfg, builder, _ = setup_imports(project, experiment)
    effective = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))["effective_arguments"]
    model = builder(
        num_classes=2,
        auxiliary=cfg.USE_AUXILIARY_HEADS,
        main_head_channels=int(effective["main_head_channels"]),
        auxiliary_head_channels=cfg.AUXILIARY_HEAD_CHANNELS,
        dropout=cfg.HEAD_DROPOUT,
        use_multiscale_context=cfg.USE_MULTI_SCALE_CONTEXT,
        use_defect_aware_bga=cfg.USE_DEFECT_AWARE_BGA,
        use_boundary_supervision=cfg.USE_BOUNDARY_SUPERVISION,
    ).to(device)
    checkpoint = torch.load(run_dir / "best.pth", map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return cfg, model, effective


def get_test_loader(project, cfg, seed, batch_size, device):
    _, _, build_loaders = setup_imports(project, "Baseline")
    from types import SimpleNamespace
    args = SimpleNamespace(
        dataset="sd-saliency", data_root=project / "data" / "SD-saliency",
        num_classes=None, image_size=200, batch_size=batch_size, num_workers=0,
        no_augment=True, seed=seed,
    )
    _, _, test_loader = build_loaders(args, device, cfg)
    return test_loader


def edge_map(binary):
    b = binary.astype(bool)
    edge = np.zeros_like(b)
    edge[1:, :] |= b[1:, :] != b[:-1, :]
    edge[:-1, :] |= b[:-1, :] != b[1:, :]
    edge[:, 1:] |= b[:, 1:] != b[:, :-1]
    edge[:, :-1] |= b[:, :-1] != b[:, 1:]
    return edge


def boundary_f1(pred, target, tol=1):
    pe, te = edge_map(pred), edge_map(target)
    def dilate(x):
        ten = torch.from_numpy(x.astype(np.float32))[None, None]
        return F.max_pool2d(ten, 2 * tol + 1, stride=1, padding=tol)[0, 0].numpy() > 0
    tp_p = int((pe & dilate(te)).sum())
    tp_t = int((te & dilate(pe)).sum())
    precision = tp_p / max(int(pe.sum()), 1)
    recall = tp_t / max(int(te.sum()), 1)
    return 2 * precision * recall / max(precision + recall, 1e-12)


@torch.inference_mode()
def infer_rows(project, exp, seed, run_dir, device, batch_size):
    cfg, model, effective = load_model(project, exp, run_dir, device)
    loader = get_test_loader(project, cfg, seed, batch_size, device)
    rows = []
    for batch in loader:
        logits = model(batch["image"].to(device))
        if isinstance(logits, dict):
            logits = logits["out"]
        pred = logits.argmax(dim=1).cpu().numpy().astype(bool)
        target = batch["mask"].numpy().astype(bool)
        for sid, p, y in zip(batch["id"], pred, target):
            tp = int((p & y).sum()); fp = int((p & ~y).sum()); fn = int((~p & y).sum())
            union = tp + fp + fn
            area = float(y.mean() * 100)
            size = "tiny" if area <= 1 else "small" if area <= 5 else "medium" if area <= 20 else "large"
            rows.append({
                "seed": seed, "experiment": exp, "sample_id": sid,
                "gt_area_pct": area, "size_bin": size,
                "iou_fg": tp / union if union else 1.0,
                "dice_fg": 2 * tp / max(2 * tp + fp + fn, 1),
                "precision_fg": tp / max(tp + fp, 1), "recall_fg": tp / max(tp + fn, 1),
                "fp_pixels": fp, "fn_pixels": fn,
                "fp_rate_bg_pct": 100 * fp / max(int((~y).sum()), 1),
                "fn_rate_fg_pct": 100 * fn / max(int(y.sum()), 1),
                "boundary_f1_1px": boundary_f1(p, y),
            })
    return rows


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--seeds", default="42,123,2024")
    parser.add_argument("--experiments", default=",".join(EXPERIMENTS))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    project = args.project.resolve()
    root = args.root or project / "results2中断" / "bisenetv2_sdneu_defect_ablation" / "SD900"
    out = args.out or root / "结果归因"
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    experiments = [x.strip() for x in args.experiments.split(",") if x.strip()]
    all_rows = []
    for seed in seeds:
        for exp in experiments:
            matches = sorted((root / exp / f"seed_{seed}").glob("**/best.pth"))
            if len(matches) != 1:
                print(f"SKIP {exp} seed={seed}: expected one best.pth, found {len(matches)}")
                continue
            run_dir = matches[0].parent
            rows = infer_rows(project, exp, seed, run_dir, device, args.batch_size)
            all_rows.extend(rows)
            print(f"Done {exp} seed={seed}: {len(rows)} test images")
    write_csv(out / "逐图指标.csv", all_rows)

    lookup = {(r["seed"], r["experiment"], r["sample_id"]): r for r in all_rows}
    paired = []
    for row in all_rows:
        if row["experiment"] == "Baseline":
            continue
        base = lookup.get((row["seed"], "Baseline", row["sample_id"]))
        if not base:
            continue
        paired.append({
            "seed": row["seed"], "experiment": row["experiment"], "sample_id": row["sample_id"],
            "size_bin": row["size_bin"], "gt_area_pct": row["gt_area_pct"],
            "delta_iou_fg": row["iou_fg"] - base["iou_fg"],
            "delta_dice_fg": row["dice_fg"] - base["dice_fg"],
            "delta_boundary_f1": row["boundary_f1_1px"] - base["boundary_f1_1px"],
            "delta_fp_rate_bg_pct": row["fp_rate_bg_pct"] - base["fp_rate_bg_pct"],
            "delta_fn_rate_fg_pct": row["fn_rate_fg_pct"] - base["fn_rate_fg_pct"],
            "improved_iou": row["iou_fg"] > base["iou_fg"],
            "worsened_iou": row["iou_fg"] < base["iou_fg"],
        })
    write_csv(out / "相对基线逐图变化.csv", paired)

    summary = []
    for exp in experiments:
        for size in ("all", "tiny", "small", "medium", "large"):
            subset = [r for r in paired if r["experiment"] == exp and (size == "all" or r["size_bin"] == size)]
            if not subset:
                continue
            summary.append({
                "experiment": exp, "size_bin": size, "paired_images": len(subset),
                "mean_delta_iou_fg": float(np.mean([r["delta_iou_fg"] for r in subset])),
                "median_delta_iou_fg": float(np.median([r["delta_iou_fg"] for r in subset])),
                "improved_images": sum(r["improved_iou"] for r in subset),
                "worsened_images": sum(r["worsened_iou"] for r in subset),
                "mean_delta_boundary_f1": float(np.mean([r["delta_boundary_f1"] for r in subset])),
                "mean_delta_fp_rate_bg_pct": float(np.mean([r["delta_fp_rate_bg_pct"] for r in subset])),
                "mean_delta_fn_rate_fg_pct": float(np.mean([r["delta_fn_rate_fg_pct"] for r in subset])),
            })
    write_csv(out / "分组归因汇总.csv", summary)

    examples = []
    for exp in experiments:
        if exp == "Baseline":
            continue
        subset = [r for r in paired if r["experiment"] == exp]
        examples += sorted(subset, key=lambda r: r["delta_iou_fg"])[:10]
        examples += sorted(subset, key=lambda r: r["delta_iou_fg"], reverse=True)[:10]
    write_csv(out / "优先复核样本.csv", examples)
    notes = [
        "# SD-saliency 逐图结果归因",
        "",
        "逐图指标使用每个实验保存的 best.pth；数据读取、尺寸和掩码阈值复用项目训练 Dataset；按训练 seed 构造相同测试划分，比较同 seed、同 sample_id 的相对变化。",
        "",
        "- 逐图指标.csv：各模型每张测试图的前景 IoU/Dice、Precision/Recall、FP/FN 和 1px Boundary F1。",
        "- 相对基线逐图变化.csv：改进组逐样本减去同 seed 的 Baseline。正的 delta_iou_fg 表示该样本前景 IoU 提升。",
        "- 分组归因汇总.csv：按缺陷面积粗分 tiny(<=1%)、small(1-5%)、medium(5-20%)、large(>20%) 汇总。此面积分组是分析用阈值，不是数据集官方类别。",
        "- 优先复核样本.csv：每组前景 IoU 下降最多及提升最多的样本，便于对照现有 visualization 图片。",
        "",
        "注意：按项目当前 Dataset 实现，SD-saliency 的随机划分由 seed 决定。因此同一个 seed 下各消融组可配对；不同 seed 的测试图集合不同。汇总按逐图配对观察趋势，正式结论仍应同时参考原三 seed 的全局测试指标。",
        "现有预览图仅保存前 20 个测试样本；完整逐图预测指标覆盖测试集全部样本。",
    ]
    (out / "阅读说明.md").write_text("\n".join(notes) + "\n", encoding="utf-8")
    print(f"Reports written to: {out}")


if __name__ == "__main__":
    main()
