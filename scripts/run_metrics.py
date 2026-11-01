"""把不同训练脚本产出的结果目录归一化成同一套指标记录（只读，不训练、不改结果）。

项目里有两条训练路径，落盘格式并不一致：

* ``scripts/train_scconv.py``（BiSeNetV2 / ScConv / 对照臂）
  写 ``metrics.json``：测试指标**嵌在** ``test_metrics`` 里，字段名 ``accuracy``，
  验证指标在 ``best_val_metrics``，配置快照是 ``{"settings":…, "effective_arguments":…}``。
* ``scripts/train_unet.py``（U-Net）
  写 ``test_metrics.json``：测试指标**平铺在顶层**，字段名 ``Acc``，
  没有 ``best_val_metrics``（验证曲线只在 ``history.jsonl`` 里），
  配置快照就是扁平的 ``vars(args)``。

``load_run()`` 读任意一种，返回同一套字段，供 ``summarize_ablation.py`` 之类
的汇总脚本复用。这样"U-Net 进主表"不需要人工换算，也不需要在结果目录里补写
文件——汇总脚本直接读，原始产物保持不动。

命令行自检（列出所有能识别的 run，不训练）::

    python scripts/run_metrics.py --root results2中断
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# 两种落盘格式
BISENET_FORMAT = "bisenetv2"
UNET_FORMAT = "unet"

BISENET_METRICS = "metrics.json"
UNET_METRICS = "test_metrics.json"

_UNET_PARAMS: dict[int, int | None] = {}


def _load_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def detect_format(run_dir: Path) -> str | None:
    """按结果文件名判断这是哪条训练路径的产物；认不出就返回 None。"""
    run_dir = Path(run_dir)
    if (run_dir / BISENET_METRICS).is_file():
        return BISENET_FORMAT
    if (run_dir / UNET_METRICS).is_file():
        return UNET_FORMAT
    return None


def read_seed(run_dir: Path) -> int | None:
    """读 ``config.json`` 里的 seed。

    BiSeNet 侧存在 ``effective_arguments`` 下，U-Net 侧就是扁平 ``vars(args)``。
    """
    path = Path(run_dir) / "config.json"
    if not path.is_file():
        return None
    document = _load_json(path)
    arguments = document.get("effective_arguments", document)
    seed = arguments.get("seed")
    try:
        return None if seed is None else int(seed)
    except (TypeError, ValueError):
        return None


def read_protocol(run_dir: Path) -> dict:
    """从配置快照里提取训练协议，用于检查各 arm 是否真的同协议。"""
    path = Path(run_dir) / "config.json"
    if not path.is_file():
        return {}
    document = _load_json(path)
    arguments = document.get("effective_arguments", document)
    settings = document.get("settings", {}) or {}

    def pick(key, setting_key=None):
        value = arguments.get(key)
        if value is None and setting_key:
            value = settings.get(setting_key)
        return value

    return {
        "scheduler": pick("scheduler_name", "SCHEDULER_NAME"),
        "warmup_epochs": pick("warmup_epochs", "WARMUP_EPOCHS"),
        "epochs_planned": pick("epochs", "EPOCHS"),
        "deterministic": pick("deterministic_effective", "DETERMINISTIC_ALGORITHMS"),
        "lr": pick("lr", "LEARNING_RATE"),
        "batch_size": pick("batch_size", "BATCH_SIZE"),
    }


def count_history_epochs(run_dir: Path) -> int:
    """``history.jsonl`` 的行数 = 实际训练轮数（含训练中途就停下的 run）。"""
    history = Path(run_dir) / "history.jsonl"
    if not history.is_file():
        return 0
    with history.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def read_best_val_miou(run_dir: Path, best_epoch: int | None):
    """从 ``history.jsonl`` 取 best_epoch 那一轮的验证 mIoU。

    U-Net 的结果文件里没有保存最佳验证指标，只能回查训练历史；找不到指定轮
    就退回全程最大值（两者在正常训练里相等）。
    """
    history = Path(run_dir) / "history.jsonl"
    if not history.is_file():
        return None
    fallback = None
    with history.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            value = (record.get("val") or {}).get("mIoU")
            if value is None:
                continue
            if best_epoch is not None and record.get("epoch") == best_epoch:
                return value
            fallback = value if fallback is None else max(fallback, value)
    return fallback


def read_scconv_scale(run_dir: Path, best_epoch: int | None) -> str:
    """取 best_epoch 那轮的 ScConv 残差 scale（多个位置用逗号连起来）。"""
    history = Path(run_dir) / "history.jsonl"
    if not history.is_file() or best_epoch is None:
        return ""
    with history.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("epoch") == best_epoch and "scconv_scale" in record:
                return ",".join(f"{value:.3f}" for value in record["scconv_scale"].values())
    return ""


def unet_parameter_count(num_classes: int = 4):
    """U-Net 参数量。

    旧结果没有记录这个字段，于是按同构模型现算一次——只依赖结构、不依赖权重，
    因此对任意历史 run 都成立。torch 不可用时返回 None 而不是让汇总失败。
    """
    if num_classes in _UNET_PARAMS:
        return _UNET_PARAMS[num_classes]
    try:
        project_dir = Path(__file__).resolve().parents[1]
        for directory in (project_dir / "config", project_dir / "models", project_dir / "scripts"):
            if str(directory) not in sys.path:
                sys.path.insert(0, str(directory))
        from UNet import build_unet  # noqa: WPS433

        model = build_unet(in_channels=3, base_channels=32, num_classes=num_classes)
        count = sum(parameter.numel() for parameter in model.parameters())
    except Exception as error:  # pragma: no cover - 只在环境缺 torch 时触发
        print(f"无法计算 U-Net 参数量：{error}", file=sys.stderr)
        return None
    _UNET_PARAMS[num_classes] = count
    return count


def _load_bisenet(run_dir: Path) -> dict:
    document = _load_json(run_dir / BISENET_METRICS)
    test = document["test_metrics"]
    best_epoch = document.get("best_epoch")
    per_class_iou = list(test["per_class_iou"])
    return {
        "format": BISENET_FORMAT,
        "params": document.get("parameter_count"),
        "best_epoch": best_epoch,
        "epochs_planned": document.get("epochs_planned"),
        "epochs_trained": document.get("epochs_trained") or count_history_epochs(run_dir),
        "val_miou": (document.get("best_val_metrics") or {}).get("mIoU"),
        "test_miou": test["mIoU"],
        "test_mdice": test["mDice"],
        "accuracy": test["accuracy"],
        "per_class_iou": per_class_iou,
        "confusion_matrix": test["confusion_matrix"],
        "num_classes": len(per_class_iou),
        "training_seconds": document.get("training_seconds"),
        "scconv_scale": read_scconv_scale(run_dir, best_epoch),
        "model_variant": document.get("model_variant", ""),
    }


def _load_unet(run_dir: Path) -> dict:
    document = _load_json(run_dir / UNET_METRICS)
    best_epoch = document.get("best_epoch")
    per_class_iou = list(document["per_class_iou"])
    num_classes = len(per_class_iou)
    params = document.get("parameter_count") or unet_parameter_count(num_classes)
    return {
        "format": UNET_FORMAT,
        "params": params,
        "best_epoch": best_epoch,
        "epochs_planned": document.get("epochs_planned"),
        "epochs_trained": document.get("epochs_trained") or count_history_epochs(run_dir),
        "val_miou": document.get("best_val_miou", read_best_val_miou(run_dir, best_epoch)),
        "test_miou": document["mIoU"],
        "test_mdice": document["mDice"],
        "accuracy": document["Acc"],
        "per_class_iou": per_class_iou,
        "confusion_matrix": document.get("confusion_matrix"),
        "num_classes": num_classes,
        "training_seconds": document.get("training_seconds"),
        "scconv_scale": "",
        "model_variant": f"U-Net ({document.get('model', 'U-Net')})",
    }


def load_run(run_dir: Path) -> dict | None:
    """读取单个 run 目录，返回统一字段；认不出格式时返回 None。"""
    run_dir = Path(run_dir)
    fmt = detect_format(run_dir)
    if fmt is None:
        return None
    record = _load_bisenet(run_dir) if fmt == BISENET_FORMAT else _load_unet(run_dir)

    protocol = read_protocol(run_dir)
    record["seed"] = read_seed(run_dir)
    record["run"] = run_dir.name
    if record.get("epochs_planned") is None:
        record["epochs_planned"] = protocol.get("epochs_planned")
    record["scheduler"] = protocol.get("scheduler", "")
    record["warmup_epochs"] = protocol.get("warmup_epochs")
    record["deterministic"] = protocol.get("deterministic")
    record["lr"] = protocol.get("lr")
    record["batch_size"] = protocol.get("batch_size")
    return record


def iter_runs(root: Path):
    """递归遍历 ``root`` 下的 run 目录。

    实际布局是 ``results2中断/<arm>/<run>/``，所以必须往下走两层；一旦某个目录本身
    就是 run（有结果文件）就不再往下钻，避免把 run 内部的子目录当成 run。
    """
    root = Path(root)
    if detect_format(root) is not None:
        yield root
        return
    if not root.is_dir():
        return
    for child in sorted(root.iterdir()):
        if not child.is_dir() or child.name.startswith("."):
            continue
        if detect_format(child) is not None:
            yield child
            continue
        yield from iter_runs(child)


def main() -> int:
    parser = argparse.ArgumentParser(description="列出结果目录里能被统一读取的 run")
    parser.add_argument("--root", default="results2中断", help="结果根目录或单个 run 目录")
    args = parser.parse_args()

    project_dir = Path(__file__).resolve().parents[1]
    root = Path(args.root)
    if not root.is_absolute():
        root = project_dir / root

    rows = []
    for run_dir in iter_runs(root):
        record = load_run(run_dir)
        if record:
            rows.append(record)
    if not rows:
        print(f"{root} 下没有可识别的 run（需要 metrics.json 或 test_metrics.json）", file=sys.stderr)
        return 1

    print(f"{'run':44s} {'fmt':10s} {'seed':>5s} {'ep':>4s} {'mIoU':>7s} {'params':>10s}")
    print("-" * 88)
    for record in rows:
        params = f"{record['params']:,}" if record["params"] else "—"
        seed = record["seed"] if record["seed"] is not None else "—"
        print(
            f"{record['run']:44s} {record['format']:10s} {str(seed):>5s} "
            f"{str(record['epochs_trained']):>4s} {record['test_miou']:.4f} {params:>10s}"
        )
    print(f"\n共 {len(rows)} 个 run。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
