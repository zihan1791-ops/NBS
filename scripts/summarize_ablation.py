"""汇总 ``results2中断/`` 下所有实验 run，输出 ``results2中断/ablation_summary.md`` 与 ``.csv``。

用法::

    python scripts/summarize_ablation.py
    python scripts/summarize_ablation.py --with_partial   # 连未完成的 run 也列出来
    python scripts/summarize_ablation.py --with_cost      # 额外测量参数量与 FLOPs

只读取各 run 目录里的 ``config.json`` / ``metrics.json`` / ``test_metrics.json``
/ ``history.jsonl``，不训练、不改结果。两种落盘格式的差异由 ``run_metrics.py``
吸收，因此 U-Net 的 run 也能直接进同一张表。

注意：不同 arm 的训练轮数可能不同（例如消融臂 90 轮、主表臂 200 轮）。汇总表
会按 run 如实列出轮数，并在跨协议时给出警告——**跨协议的 Δ 不能当作结论**。
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_DIR / "results2中断"
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import run_metrics  # noqa: E402

# arm 代号 -> (结果目录名, 中文说明)
ARMS = {
    "B0": ("bisenetv2_ablation_B0_baseline", "基线：关闭 ScConv（= 原版 BiSeNetV2）"),
    "S0": ("bisenetv2_ablation_S0_scconv_fused", "现状做法：ScConv 挂在 BGA 之后，带残差，scale=0.1"),
    "D1": ("bisenetv2_ablation_D1_scale1", "残差初值 0.1 → 1.0（模块完全激活）"),
    "D2": ("bisenetv2_ablation_D2_direct", "去掉残差，ScConv 直接替换该位置的卷积输出"),
    "C2a": ("bisenetv2_control_C2a_conv1x1", "归因对照：fused 位置换成 Conv1x1+BN+ReLU（无残差）"),
    "C2b": ("bisenetv2_control_C2b_conv3x3", "归因对照：fused 位置换成 Conv3x3+BN+ReLU（无残差）"),
    "C2c": ("bisenetv2_control_C2c_relu", "归因对照：fused 位置只放 ReLU（零参数）"),
    "P1": ("bisenetv2_ablation_P1_detail", "只作用于细节分支输出"),
    "P2": ("bisenetv2_ablation_P2_semantic", "只作用于语义分支上下文输出"),
    "P3": ("bisenetv2_ablation_P3_detail_semantic", "两路各自 ScConv 后再交给原版 BGA"),
    "G1": ("bisenetv2_ablation_G1_gate_fix", "SRU 门控去掉 γ/Σγ 归一化（门控动态范围 ×100）"),
    "R1": ("bisenetv2_repl_R1_detail", "替换式：detail 分支 4 个重复 3x3 卷积 → ScConv"),
    "R2": ("bisenetv2_repl_R2_semantic", "替换式：semantic 分支 5 个 GELayerS1.conv1 → ScConv"),
    "R1D2": ("bisenetv2_repl_R1_plus_D2", "组合：R1 替换 + D2（fused 无残差）"),
    "UNet": ("unet", "U-Net 基线（历史/自有协议；与主线不一定同 recipe）"),
    "UNetP300": ("unet_p300", "U-Net 基线（论文协议：300 轮 cosine，batch 16，确定性关）"),
}

FIELDS = [
    "arm", "seed", "run", "format", "scheduler", "epochs_planned", "epochs_trained",
    "params", "best_epoch", "val_miou", "test_miou", "test_mdice", "accuracy",
    "iou_bg", "iou_inclusion", "iou_patch", "iou_scratch",
    "recall_inclusion", "recall_patch", "recall_scratch",
    "fg_recall", "bg_false_positive", "scconv_scale", "training_minutes", "model_variant",
]

# arm -> 对应配置模块，供 --with_cost 测量参数量与 FLOPs。
# U-Net 臂不在此表：它是另一种架构，成本由 run_metrics 的参数量字段给出。
CONFIG_OF = {
    "B0": "configScConvB0",
    "S0": "configScConvAblation",
    "D1": "configScConvD1",
    "D2": "configScConvD2",
    "C2a": "configScConvC2a",
    "C2b": "configScConvC2b",
    "C2c": "configScConvC2c",
    "P1": "configScConvP1",
    "P2": "configScConvP2",
    "P3": "configScConvP3",
    "G1": "configScConvG1",
    "R1": "configScConvR1",
    "R2": "configScConvR2",
    "R1D2": "configScConvR1D2",
}


def measure_costs(arms):
    """测量各 arm 的参数量与推理 FLOPs；失败时返回空表而不是中断汇总。"""
    try:
        import measure_cost
        import torch
        from torch.utils.flop_counter import FlopCounterMode
    except Exception as error:  # pragma: no cover
        print(f"跳过成本测量：{error}", file=sys.stderr)
        return {}

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    costs = {}
    for arm in arms:
        config_name = CONFIG_OF.get(arm)
        if config_name is None:
            continue
        try:
            model, _module = measure_cost.build(config_name)
            model = model.to(device).eval()
            with torch.no_grad():
                inputs = torch.randn(1, 3, 200, 200, device=device)
                counter = FlopCounterMode(display=False)
                with counter:
                    model(inputs)
            costs[arm] = {
                "params": sum(parameter.numel() for parameter in model.parameters()),
                "flops": counter.get_total_flops(),
            }
            model.to(torch.device("cpu"))
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
        except Exception as error:  # pragma: no cover
            print(f"跳过 {arm} 的成本测量：{error}", file=sys.stderr)
    return costs


def collect(include_partial: bool):
    """读取所有 run；跳过类别数不是 4 的 run（主表只讨论 NEU-Seg 的 4 类）。"""
    rows, partial, skipped = [], [], []
    for arm, (folder, _note) in ARMS.items():
        arm_dir = RESULTS_DIR / folder
        if not arm_dir.is_dir():
            continue
        for run_dir in sorted(arm_dir.iterdir()):
            if not run_dir.is_dir():
                continue
            record = run_metrics.load_run(run_dir)
            if record is None:
                if include_partial:
                    partial.append({
                        "arm": arm,
                        "run": run_dir.name,
                        "epochs": run_metrics.count_history_epochs(run_dir),
                    })
                continue
            if record["num_classes"] != 4:
                skipped.append(f"{arm}/{run_dir.name}：{record['num_classes']} 类")
                continue
            confusion = record["confusion_matrix"]
            if not confusion:
                skipped.append(f"{arm}/{run_dir.name}：缺混淆矩阵")
                continue

            support = [sum(row) for row in confusion]
            recalls = [confusion[i][i] / max(support[i], 1) for i in range(len(confusion))]
            per_class_iou = record["per_class_iou"]

            rows.append({
                "arm": arm,
                "seed": record["seed"],
                "run": run_dir.name,
                "format": record["format"],
                "scheduler": record["scheduler"] or "",
                "epochs_planned": record["epochs_planned"],
                "epochs_trained": record["epochs_trained"],
                "params": record["params"],
                "best_epoch": record["best_epoch"],
                "val_miou": record["val_miou"],
                "test_miou": record["test_miou"],
                "test_mdice": record["test_mdice"],
                "accuracy": record["accuracy"],
                "iou_bg": per_class_iou[0],
                "iou_inclusion": per_class_iou[1],
                "iou_patch": per_class_iou[2],
                "iou_scratch": per_class_iou[3],
                "recall_inclusion": recalls[1],
                "recall_patch": recalls[2],
                "recall_scratch": recalls[3],
                "fg_recall": statistics.mean(recalls[1:]),
                "bg_false_positive": (support[0] - confusion[0][0]) / max(support[0], 1),
                "scconv_scale": record["scconv_scale"],
                "training_minutes": round((record["training_seconds"] or 0) / 60, 1),
                "model_variant": record["model_variant"],
                "deterministic": record["deterministic"],
            })

    order = list(ARMS)
    rows.sort(key=lambda row: (
        order.index(row["arm"]),
        row["seed"] if row["seed"] is not None else -1,
    ))
    return rows, partial, skipped


def aggregate(rows):
    groups = {}
    for row in rows:
        groups.setdefault(row["arm"], []).append(row)
    out = []
    for arm in ARMS:
        if arm not in groups:
            continue
        group = groups[arm]

        def cell(key, decimals=4):
            values = [row[key] for row in group if row[key] is not None]
            if not values:
                return "—"
            if len(values) == 1:
                return f"{values[0]:.{decimals}f}"
            return f"{statistics.mean(values):.{decimals}f} ± {statistics.stdev(values):.{decimals}f}"

        miou_values = [row["test_miou"] for row in group]
        out.append({
            "arm": arm,
            "n": len(group),
            "test_miou": statistics.mean(miou_values),
            "test_miou_std": statistics.stdev(miou_values) if len(miou_values) > 1 else 0.0,
            "params": group[0]["params"],
            "epochs": sorted({row["epochs_planned"] for row in group if row["epochs_planned"]}),
            "schedulers": sorted({row["scheduler"] for row in group if row["scheduler"]}),
            # 直接存"开/关/—"的渲染结果：混了 True/None 时也能如实并列
            "deterministic": sorted({_flag(row["deterministic"]) for row in group}),
            "cells": {
                "val_miou": cell("val_miou"),
                "test_miou": cell("test_miou"),
                "iou_inclusion": cell("iou_inclusion"),
                "iou_scratch": cell("iou_scratch"),
                "fg_recall": cell("fg_recall"),
                "bg_false_positive": cell("bg_false_positive"),
            },
        })
    return out


def _join(values) -> str:
    if not values:
        return "—"
    return "/".join(str(value) for value in values)


def _flag(value) -> str:
    if value is True:
        return "开"
    if value is False:
        return "关"
    return "—"


def write_csv(rows, path: Path) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in FIELDS})


def write_markdown(rows, groups, partial, skipped, path: Path, costs=None) -> None:
    baseline = next((group for group in groups if group["arm"] == "B0"), None)
    lines = [
        "# ScConv 消融实验汇总",
        "",
        "由 `scripts/summarize_ablation.py` 自动生成。数据来自各 run 目录的 `metrics.json`"
        "（BiSeNet/ScConv）或 `test_metrics.json`（U-Net），统一由 `scripts/run_metrics.py`"
        " 读取；测试集 840 张。",
        "",
        "## 各 arm 的训练协议",
        "",
        "| arm | n | 调度 | 计划轮数 | 确定性 | 参数量 |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for group in groups:
        params = f"{group['params']:,}" if group["params"] else "—"
        lines.append(
            f"| {group['arm']} | {group['n']} | {_join(group['schedulers'])} "
            f"| {_join(group['epochs'])} | {_join(group['deterministic'])} "
            f"| {params} |"
        )

    mixed = [group["arm"] for group in groups if len(group["epochs"]) > 1 or len(group["schedulers"]) > 1]
    lines += [
        "",
        "> 协议由各 run 的 `config.json` 读取；同一 arm 内出现多个取值时用 `/` 分隔（说明该臂混了不同协议，需分开看）。",
        "> 确定性：`torch.use_deterministic_algorithms(True)` + 模型初始化/数据增强走独立生成器。",
        "> 已验证同配置同种子两次运行 368 个权重张量逐位相同；关闭确定性时同样两次运行有 310/368 个张量不同。",
    ]
    if mixed:
        lines += [
            "",
            f"> ⚠ 协议不统一的臂：{', '.join(mixed)}。这些臂内部的结果不能直接平均。",
        ]

    lines += [
        "",
        "## 各 run 明细",
        "",
        "| arm | seed | 轮数 | test mIoU | mDice | Acc | 夹杂 IoU | 斑块 IoU | 划痕 IoU | 前景召回 | scale(best) | 耗时(min) |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row['arm']} | {row['seed']} | {row['epochs_trained']} "
            f"| {row['test_miou']:.4f} | {row['test_mdice']:.4f} "
            f"| {row['accuracy']:.4f} | {row['iou_inclusion']:.4f} | {row['iou_patch']:.4f} "
            f"| {row['iou_scratch']:.4f} | {row['fg_recall']:.4f} | {row['scconv_scale'] or '—'} "
            f"| {row['training_minutes']:.0f} |"
        )

    lines += [
        "",
        "## 按 arm 聚合（mean ± std）",
        "",
        "| arm | n | val mIoU | **test mIoU** | 夹杂 IoU | 划痕 IoU | 前景召回 | 背景误检 |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for group in groups:
        cells = group["cells"]
        lines.append(
            f"| {group['arm']} | {group['n']} | {cells['val_miou']} | **{cells['test_miou']}** "
            f"| {cells['iou_inclusion']} | {cells['iou_scratch']} | {cells['fg_recall']} "
            f"| {cells['bg_false_positive']} |"
        )

    if baseline:
        baseline_epochs = baseline["epochs"]
        lines += [
            "",
            "## 相对基线（B0）的 test mIoU 差值",
            "",
            "| arm | n | Δ test mIoU | 基线 σ | 本臂 σ | 轮数 | 判定 |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
        cross_protocol = []
        for group in groups:
            if group["arm"] == "B0":
                continue
            delta = group["test_miou"] - baseline["test_miou"]
            same_protocol = group["epochs"] == baseline_epochs and group["schedulers"] == baseline["schedulers"]
            if not same_protocol:
                cross_protocol.append(group["arm"])
            if group["n"] == 1:
                verdict = "单种子筛选结果，待复赛确认"
            else:
                pooled = ((group["test_miou_std"] ** 2 + baseline["test_miou_std"] ** 2) / 2) ** 0.5
                verdict = "超出合并 1σ" if abs(delta) > pooled else "在合并 1σ 内（视为无差异）"
            if not same_protocol:
                verdict = "⚠ 跨协议，差值不可作为结论"
            lines.append(
                f"| {group['arm']} | {group['n']} | {delta:+.4f} "
                f"| {baseline['test_miou_std']:.4f} | {group['test_miou_std']:.4f} "
                f"| {_join(group['epochs']) or '—'} vs {_join(baseline_epochs) or '—'} | {verdict} |"
            )
        lines += [
            "",
            "> 只有 n>1 的臂才能给出方差；单种子的臂仅用于筛选方向，结论需要补种子后才能写进论文。",
        ]
        if cross_protocol:
            lines += [
                f"> ⚠ 轮数或调度与 B0 不一致的臂：{', '.join(cross_protocol)}。"
                "跨协议的 Δ 只说明\"协议不同\"，不能说明算子好坏——要么补同协议 run，要么分开成表。",
            ]

    lines += ["", "## arm 说明", ""]
    for arm in ARMS:
        if not any(group["arm"] == arm for group in groups):
            continue
        lines.append(f"- **{arm}**（{ARMS[arm][1]}）：{ARMS[arm][0]}")

    if partial:
        lines += ["", "## 未完成的 run", ""]
        for item in partial:
            lines.append(f"- {item['arm']} / {item['run']}：已完成 {item['epochs']} 轮")

    if skipped:
        lines += ["", "## 已跳过（类别数不符 / 缺混淆矩阵）", ""]
        for item in skipped:
            lines.append(f"- {item}")

    if costs:
        baseline_arm = "B0" if "B0" in costs else next(iter(costs))
        base = costs[baseline_arm]
        lines += [
            "",
            f"## 计算成本（推理，1×3×200×200；Δ 以 {baseline_arm} 为基准）",
            "",
            "| arm | 参数量 | Δ参数量 | GFLOPs | ΔGFLOPs |",
            "| --- | --- | --- | --- | --- |",
        ]
        for arm, value in costs.items():
            lines.append(
                f"| {arm} | {value['params']:,} | {value['params'] - base['params']:+,} "
                f"| {value['flops'] / 1e9:.2f} | {(value['flops'] - base['flops']) / 1e9:+.2f} |"
            )
        lines += [
            "",
            "> 由 `scripts/measure_cost.py` 的同一套逻辑测得；FLOPs 采用 torch 计数器口径（乘加计 2）。",
        ]

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="汇总 ScConv 消融实验结果")
    parser.add_argument("--with_partial", action="store_true", help="同时列出未完成的 run")
    parser.add_argument("--with_cost", action="store_true",
                        help="额外测量各 arm 的参数量与推理 FLOPs（需构建模型，约十秒）")
    args = parser.parse_args()

    rows, partial, skipped = collect(args.with_partial)
    if not rows:
        print("没有找到任何完成的 run（需要 metrics.json 或 test_metrics.json）", file=sys.stderr)
        raise SystemExit(1)

    groups = aggregate(rows)
    costs = measure_costs([group["arm"] for group in groups]) if args.with_cost else {}
    csv_path = RESULTS_DIR / "ablation_summary.csv"
    md_path = RESULTS_DIR / "ablation_summary.md"
    write_csv(rows, csv_path)
    write_markdown(rows, groups, partial, skipped, md_path, costs)

    print(f"已写入：{csv_path}")
    print(f"已写入：{md_path}")
    print(f"完成的 run：{len(rows)} 个；arm：{', '.join(group['arm'] for group in groups)}")
    if partial:
        print(f"未完成：{len(partial)} 个")
    if skipped:
        print(f"已跳过：{len(skipped)} 个（类别数不符或缺混淆矩阵）")


if __name__ == "__main__":
    main()
