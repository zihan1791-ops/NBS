"""使用 ScConv 增强 BiSeNetV2，并在 NEU-Seg 上训练与评估。"""

import argparse
import json
import math
import os
import random
import time
from pathlib import Path, PosixPath, WindowsPath

# cuBLAS 的确定性工作区必须在首次调用 cuBLAS 之前设置，否则确定性算法不会生效。
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import LambdaLR, ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm

from bisenetv2_scconv import build_bisenetv2_scconv
import configScConv as config
from configScConv import (
    BEST_CHECKPOINT_NAME,
    CLASS_NAMES,
    COLOR_MAP_BGR,
    CONFIG_SNAPSHOT_NAME,
    DATA_ROOT,
    DICE_LOSS_WEIGHT,
    EARLY_STOPPING_PATIENCE,
    EPOCHS,
    IMAGE_DIR_NAME,
    IMAGE_SIZE,
    IMAGE_VALUE_SCALE,
    LEARNING_RATE,
    MAIN_HEAD_CHANNELS,
    MASK_DIR_NAME,
    METRICS_NAME,
    MIN_LEARNING_RATE,
    NUM_CLASSES,
    NUM_WORKERS,
    OUTPUT_DIR,
    PIN_MEMORY,
    PERSISTENT_WORKERS,
    RANDOM_SEED,
    SAVE_VISUALIZATIONS,
    SCHEDULER_FACTOR,
    SCHEDULER_MODE,
    SCHEDULER_PATIENCE,
    TEST_SPLIT_NAME,
    TRAIN_HISTORY_NAME,
    TRAIN_SPLIT_NAME,
    VALIDATION_RATIO,
    VISUALIZATION_DIR_NAME,
    VISUALIZATION_FONT_SCALE,
    VISUALIZATION_FONT_THICKNESS,
    VISUALIZATION_MAX_SAMPLES,
    VISUALIZATION_TEXT_COLOR_BGR,
    VISUALIZATION_TEXT_ORIGIN,
    WEIGHT_DECAY,
)
from dataset import build_loaders
from metrics import DiceLoss, metrics_from_confusion, update_confusion_matrix
from plots import (
    save_evaluation_plots,
    save_training_history_jsonl,
    save_training_history_text,
    save_training_results_plot,
    update_probability_histogram,
)


def set_seed(seed: int, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = config.CUDNN_DETERMINISTIC
    torch.backends.cudnn.benchmark = config.CUDNN_BENCHMARK
    if deterministic:
        # warn_only=True：仍有算子不支持确定性实现时只告警不中断，告警即发现问题点。
        torch.use_deterministic_algorithms(True, warn_only=True)
    else:
        torch.use_deterministic_algorithms(False)


def make_generator(seed: int) -> torch.Generator:
    """训练集 shuffle 与 worker 种子专用生成器，与全局 RNG 解耦。"""
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def parse_args():
    parser = argparse.ArgumentParser(description="选择数据集训练 ScConv-BiSeNetV2")
    parser.add_argument("--dataset", choices=("neu", "sd-saliency", "magnetic-tile"), default="neu",
                        help="数据集：neu、sd-saliency 或 magnetic-tile")
    parser.add_argument("--data_root", type=Path, default=None,
                        help="覆盖所选数据集的默认目录；通常只需设置 --dataset")
    parser.add_argument("--output_dir", "--output_root", dest="output_dir", type=Path, default=OUTPUT_DIR,
                        help="结果根目录；训练时自动创建带编号的运行目录，评估时填写具体实验目录")
    parser.add_argument("--image_size", type=int, default=IMAGE_SIZE)
    parser.add_argument("--num_classes", type=int, default=None,
                        help="可选校验类别数；默认根据数据集自动设置")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch_size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--weight_decay", type=float, default=WEIGHT_DECAY)
    parser.add_argument("--val_ratio", type=float, default=VALIDATION_RATIO,
                        help="NEU 关闭官方 test 验证模式时，从训练集划出的验证集比例")
    parser.add_argument("--aux_weight", type=float, default=config.AUXILIARY_LOSS_WEIGHT,
                        help="辅助分割头平均损失的权重")
    parser.add_argument("--ce_weight", type=float, default=config.CROSS_ENTROPY_LOSS_WEIGHT)
    parser.add_argument("--dice_weight", type=float, default=DICE_LOSS_WEIGHT)
    parser.add_argument("--main_head_channels", type=int, default=MAIN_HEAD_CHANNELS,
                        help="BiSeNetV2 主分割头通道数；显存不足时可调低")
    parser.add_argument("--num_workers", type=int, default=NUM_WORKERS)
    parser.add_argument("--patience", type=int, default=EARLY_STOPPING_PATIENCE)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--device", type=str, default=config.DEVICE,
                        help="填写 auto、cuda 或 cpu")
    parser.add_argument("--scheduler_factor", type=float, default=SCHEDULER_FACTOR)
    parser.add_argument("--scheduler_patience", type=int, default=SCHEDULER_PATIENCE)
    parser.add_argument("--min_lr", type=float, default=MIN_LEARNING_RATE)
    parser.add_argument("--scheduler_name", type=str,
                        default=getattr(config, "SCHEDULER_NAME", "plateau"),
                        help="plateau 或 cosine；cosine 使用 warmup + 余弦退火")
    parser.add_argument("--warmup_epochs", type=int,
                        default=getattr(config, "WARMUP_EPOCHS", 0),
                        help="cosine 调度器的线性 warmup 轮数")
    parser.add_argument("--deterministic", choices=("auto", "on", "off"), default="auto",
                        help="是否启用 torch 确定性算法；auto 跟随配置 DETERMINISTIC_ALGORITHMS")
    parser.add_argument("--no_augment", action="store_true")
    parser.add_argument("--no_visualization", action="store_true")
    parser.add_argument("--vis_samples", type=int, default=VISUALIZATION_MAX_SAMPLES)
    parser.add_argument("--eval_only", action="store_true",
                        help="读取 --output_dir 中已有 best.pth，只评估并生成图表")
    return parser.parse_args()


def create_run_directory(output_root: Path, dataset_tag: str, epochs: int) -> Path:
    """以原子建目录方式分配从 0 开始的唯一实验编号。"""
    output_root.mkdir(parents=True, exist_ok=True)
    index = 0
    while True:
        run_dir = output_root / f"{dataset_tag}_{epochs}_{index}"
        try:
            run_dir.mkdir()
            return run_dir
        except FileExistsError:
            index += 1


def make_loaders(args, device):
    return build_loaders(args, device, config, generator=make_generator(args.seed))


def build_scheduler(optimizer, args):
    """按配置创建学习率调度器，返回 (调度器, 名称)。"""
    name = str(args.scheduler_name).lower()
    if name == "plateau":
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode=SCHEDULER_MODE,
            factor=args.scheduler_factor,
            patience=args.scheduler_patience,
            min_lr=args.min_lr,
        )
        return scheduler, "plateau"
    if name == "cosine":
        warmup = max(args.warmup_epochs, 0)
        decay_epochs = max(args.epochs - warmup, 1)
        floor = args.min_lr / args.lr if args.lr > 0 else 0.0

        def lr_lambda(step: int) -> float:
            if step < warmup:
                return (step + 1) / warmup
            progress = min((step - warmup) / decay_epochs, 1.0)
            return floor + (1.0 - floor) * 0.5 * (1.0 + math.cos(math.pi * progress))

        return LambdaLR(optimizer, lr_lambda), "cosine"
    raise ValueError(f"不支持的学习率调度器：{args.scheduler_name}")


def segmentation_loss(
    outputs, masks, ce_loss, dice_loss, num_classes, aux_weight, ce_weight, dice_weight
):
    if isinstance(outputs, dict):
        logits = outputs["out"]
        aux_logits = outputs.get("aux", [])
    else:
        logits = outputs
        aux_logits = []

    main_loss = (
        ce_weight * ce_loss(logits, masks)
        + dice_weight * dice_loss(logits, masks, num_classes)
    )
    if aux_logits:
        auxiliary_loss = torch.stack([
            ce_weight * ce_loss(aux, masks)
            + dice_weight * dice_loss(aux, masks, num_classes)
            for aux in aux_logits
        ]).mean()
        total_loss = main_loss + aux_weight * auxiliary_loss
    else:
        total_loss = main_loss
    return total_loss, main_loss, logits


def run_epoch(
    model,
    loader,
    optimizer,
    ce_loss,
    dice_loss,
    args,
    device,
    description,
    probability_histogram=None,
):
    is_training = optimizer is not None
    model.train(is_training)
    confusion = torch.zeros(
        (args.num_classes, args.num_classes), dtype=torch.int64, device=device
    )
    total_loss = 0.0
    total_main_loss = 0.0
    sample_count = 0

    progress = tqdm(loader, desc=description, leave=False)
    for batch in progress:
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)
        batch_size = images.shape[0]

        if is_training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_training):
            outputs = model(images)
            loss, main_loss, logits = segmentation_loss(
                outputs, masks, ce_loss, dice_loss,
                args.num_classes, args.aux_weight, args.ce_weight, args.dice_weight,
            )
            if is_training:
                loss.backward()
                optimizer.step()

        total_loss += loss.item() * batch_size
        total_main_loss += main_loss.item() * batch_size
        sample_count += batch_size
        update_confusion_matrix(confusion, logits.detach(), masks, args.num_classes)
        if probability_histogram is not None:
            probabilities = torch.softmax(logits.detach(), dim=1).cpu().numpy()
            update_probability_histogram(
                probability_histogram,
                probabilities,
                masks.detach().cpu().numpy(),
            )
        progress.set_postfix(loss=f"{loss.item():.4f}")

    metrics = metrics_from_confusion(confusion.cpu())
    metrics["loss"] = total_loss / max(sample_count, 1)
    metrics["main_loss"] = total_main_loss / max(sample_count, 1)
    return metrics


def load_checkpoint_safely(path: Path, device):
    """安全加载项目检查点，并兼容旧检查点中的 pathlib 路径元数据。"""
    with torch.serialization.safe_globals([WindowsPath, PosixPath]):
        return torch.load(path, map_location=device, weights_only=True)


def load_training_history(path: Path):
    """读取机器可读历史，并兼容旧版 JSONL 格式的 result.txt。"""
    if not path.is_file():
        path = path.with_name(config.TRAIN_HISTORY_NAME)
    if not path.is_file():
        return []
    records = []
    with path.open("r", encoding="utf-8") as history_file:
        for line in history_file:
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def _model_variant(cfg) -> str:
    """给人读的模型名，写进 metrics.json 的 model_variant 字段。"""
    if not cfg.USE_SC_CONV:
        return "BiSeNetV2 (ScConv disabled)"
    form = " (residual)" if cfg.SC_CONV_RESIDUAL else " (direct)"
    fused_block = getattr(cfg, "SC_CONV_FUSED_BLOCK", "scconv")
    if fused_block != "scconv":
        # C2a/C2b/C2c：fused 位置放的是普通块，模型里并没有 ScConv。
        return f"BiSeNetV2 + {fused_block} control block{form}"
    return "BiSeNetV2 + ScConv %s%s" % ("+".join(cfg.SC_CONV_PLACEMENT), form)


def write_config_snapshot(args):
    """保存当前代码配置和本次实际命令行参数。"""
    resolved_config = {
        "settings": config.to_dict(),
        "effective_arguments": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
    }
    config_path = args.output_dir / CONFIG_SNAPSHOT_NAME
    with config_path.open("w", encoding="utf-8") as config_file:
        json.dump(resolved_config, config_file, ensure_ascii=False, indent=2, default=str)
    return resolved_config


def evaluate_and_save(
    model,
    test_loader,
    ce_loss,
    dice_loss,
    args,
    device,
    checkpoint,
    history,
    parameter_count,
    training_seconds=None,
):
    """评估最佳检查点，保存指标、图表和预测预览。"""
    bins = config.PROBABILITY_HISTOGRAM_BINS
    probability_histogram = (
        np.zeros((args.num_classes, bins), dtype=np.int64),
        np.zeros((args.num_classes, bins), dtype=np.int64),
    )
    test_description = (
        "Final test (used for validation/selection)"
        if args.validation_uses_official_test else "Final test"
    )
    test_metrics = run_epoch(
        model,
        test_loader,
        None,
        ce_loss,
        dice_loss,
        args,
        device,
        test_description,
        probability_histogram=probability_histogram,
    )
    curve_metrics = save_evaluation_plots(
        probability_histogram,
        test_metrics["confusion_matrix"],
        tuple(args.class_names),
        args.output_dir,
        plot_class_names=tuple(args.plot_class_names),
    )
    save_training_history_jsonl(
        history,
        args.output_dir / config.TRAIN_HISTORY_JSONL_NAME,
    )
    summary = {
        "model_variant": _model_variant(config),
        "scheduler": f"{args.scheduler_name} (warmup={args.warmup_epochs})",
        "epochs_planned": args.epochs,
        "epochs_trained": len(history),
        "scconv_enabled": config.USE_SC_CONV,
        "scconv_placement": list(config.SC_CONV_PLACEMENT),
        "scconv_residual": config.SC_CONV_RESIDUAL,
        "scconv_residual_initial_scale": config.SC_CONV_RESIDUAL_SCALE,
        "scconv_gamma_normalise": getattr(config, "SC_CONV_GAMMA_NORMALISE", True),
        "scconv_fused_block": getattr(config, "SC_CONV_FUSED_BLOCK", "scconv"),
        "scconv_replace": list(getattr(config, "SC_CONV_REPLACE", ())),
        "scconv_replaced_layers": list(getattr(model, "scconv_replaced_layers", [])),
        "dataset": args.dataset,
        "run_name": args.output_dir.name,
        "data_root": str(args.data_root),
        "best_epoch": checkpoint.get("epoch"),
        "best_val_metrics": checkpoint.get("val_metrics", {}),
        "test_metrics": test_metrics,
        "validation_source": args.validation_source,
        "official_test_used_for_checkpoint_selection": args.validation_uses_official_test,
        "test_metrics_are_independent": not args.validation_uses_official_test,
        "curve_metrics": curve_metrics,
        "class_names": args.class_names,
        "parameter_count": parameter_count,
        "training_seconds": training_seconds,
    }
    save_training_results_plot(
        history,
        args.output_dir / config.RESULTS_PLOT_NAME,
        best_epoch=checkpoint.get("epoch"),
        final_metrics=test_metrics,
        validation_source=summary["validation_source"],
    )
    save_training_history_text(
        history,
        args.output_dir / config.TRAIN_HISTORY_NAME,
        model_name=f"BiSeNetV2 + ScConv | {args.dataset_tag}",
        validation_source=summary["validation_source"],
        best_epoch=checkpoint.get("epoch"),
        final_metrics=test_metrics,
        training_seconds=training_seconds,
        class_names=tuple(args.class_names),
    )
    metrics_path = args.output_dir / METRICS_NAME
    temporary_metrics_path = metrics_path.with_name(metrics_path.name + ".tmp")
    with temporary_metrics_path.open("w", encoding="utf-8") as metrics_file:
        json.dump(summary, metrics_file, ensure_ascii=False, indent=2)
    temporary_metrics_path.replace(metrics_path)

    print(
        f"Test metrics (best checkpoint, epoch {summary['best_epoch']}; "
        f"used for selection={args.validation_uses_official_test}): "
        f"mIoU {test_metrics['mIoU']:.4f}, mDice {test_metrics['mDice']:.4f}, "
        f"Acc {test_metrics['accuracy']:.4f}"
    )
    print("Per-class test IoU:")
    for name, score in zip(args.class_names, test_metrics["per_class_iou"]):
        print(f"  {name}: {score:.4f}")
    for class_name, values in curve_metrics.items():
        print(
            f"{class_name}: AP {values['average_precision']:.4f}, "
            f"ROC-AUC {values['roc_auc']:.4f}"
        )

    if SAVE_VISUALIZATIONS and not args.no_visualization:
        vis_dir = args.output_dir / VISUALIZATION_DIR_NAME
        count = save_visualizations(model, test_loader, device, vis_dir, args.vis_samples)
        print(f"Saved {count} test previews to {vis_dir}")
    print(f"Charts: {args.output_dir}")
    print(f"Metrics: {args.output_dir / METRICS_NAME}")
    return summary


@torch.no_grad()
def save_visualizations(model, loader, device, output_dir: Path, max_samples: int):
    output_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    saved = 0
    for batch in tqdm(loader, desc="Saving test previews", leave=False):
        images = batch["image"].to(device)
        logits = model(images)
        predictions = logits.argmax(dim=1).cpu().numpy()
        images = images.cpu().numpy()
        masks = batch["mask"].numpy()
        ids = batch["id"]

        for image, mask, prediction, sample_id in zip(images, masks, predictions, ids):
            image_rgb = np.clip(
                image.transpose(1, 2, 0) * IMAGE_VALUE_SCALE,
                0,
                IMAGE_VALUE_SCALE,
            ).astype(np.uint8)
            image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
            gt_color = colorize(mask)
            pred_color = colorize(prediction)
            panels = [image_bgr, gt_color, pred_color]
            for panel, label in zip(panels, config.VISUALIZATION_PANEL_LABELS):
                cv2.putText(
                    panel,
                    label,
                    VISUALIZATION_TEXT_ORIGIN,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    VISUALIZATION_FONT_SCALE,
                    VISUALIZATION_TEXT_COLOR_BGR,
                    VISUALIZATION_FONT_THICKNESS,
                    cv2.LINE_AA,
                )
            row = np.hstack(panels)
            cv2.imwrite(str(output_dir / f"{sample_id}.png"), row)
            saved += 1
            if saved >= max_samples:
                return saved
    return saved


def colorize(mask: np.ndarray) -> np.ndarray:
    image = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for class_id, color in getattr(colorize, "color_map", COLOR_MAP_BGR).items():
        image[mask == class_id] = color
    return image


def main():
    args = parse_args()
    if not 0.0 < args.val_ratio < 1.0:
        raise ValueError("--val_ratio must be between 0 and 1")
    if args.num_workers < 0:
        raise ValueError("--num_workers must be non-negative")

    dataset_roots = {
        "neu": DATA_ROOT,
        "sd-saliency": DATA_ROOT.parent / "SD-saliency",
        "magnetic-tile": DATA_ROOT.parent / "Magnetic-tile",
    }
    dataset_tags = {"neu": "NEU", "sd-saliency": "SD-saliency", "magnetic-tile": "Magnetic-tile"}
    if args.data_root is None:
        args.data_root = dataset_roots[args.dataset]
    args.dataset_tag = dataset_tags[args.dataset]
    output_root = args.output_dir
    if args.deterministic == "on":
        deterministic = True
    elif args.deterministic == "off":
        deterministic = False
    else:
        deterministic = bool(getattr(config, "DETERMINISTIC_ALGORITHMS", False))
    args.deterministic_effective = deterministic
    set_seed(args.seed, deterministic)
    device_name = args.device
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    args.device = str(device)
    train_loader, val_loader, test_loader = make_loaders(args, device)
    colorize.color_map = args.color_map
    if not args.eval_only:
        args.output_dir = create_run_directory(output_root, args.dataset_tag, args.epochs)
    elif not args.output_dir.is_dir():
        raise FileNotFoundError(f"评估模式要求 --output_dir 指向已有实验目录：{args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    resolved_config = write_config_snapshot(args)
    print(f"Result directory: {args.output_dir}")

    # 模型初始化使用独立的 RNG 域：fork_rng 结束后全局 RNG 复原，保证不同变体之间
    # 的初始权重分布与训练集增强序列完全对齐（否则基线/ScConv 比的是两份随机序列）。
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(args.seed + 1000)
        model = build_bisenetv2_scconv(
            num_classes=args.num_classes,
            auxiliary=config.USE_AUXILIARY_HEADS,
            main_head_channels=args.main_head_channels,
            auxiliary_head_channels=config.AUXILIARY_HEAD_CHANNELS,
            dropout=config.HEAD_DROPOUT,
            in_channels=config.INPUT_CHANNELS,
            detail_channels=config.DETAIL_CHANNELS,
            detail_stage_repeats=config.DETAIL_STAGE_REPEATS,
            semantic_channels=config.SEMANTIC_CHANNELS,
            expansion=config.GE_EXPANSION,
            stage_repeats=config.STAGE_REPEATS,
            scconv_enabled=config.USE_SC_CONV,
            scconv_group_num=config.SC_CONV_GROUP_NUM,
            scconv_gate_threshold=config.SC_CONV_GATE_THRESHOLD,
            scconv_alpha=config.SC_CONV_ALPHA,
            scconv_squeeze_ratio=config.SC_CONV_SQUEEZE_RATIO,
            scconv_group_size=config.SC_CONV_GROUP_SIZE,
            scconv_group_kernel_size=config.SC_CONV_GROUP_KERNEL_SIZE,
            scconv_gamma_normalise=getattr(config, "SC_CONV_GAMMA_NORMALISE", True),
            scconv_placement=config.SC_CONV_PLACEMENT,
            scconv_residual=config.SC_CONV_RESIDUAL,
            scconv_residual_scale=config.SC_CONV_RESIDUAL_SCALE,
            scconv_replace=getattr(config, "SC_CONV_REPLACE", ()),
            scconv_replace_keep_relu=getattr(config, "SC_CONV_REPLACE_KEEP_RELU", True),
            scconv_fused_block=getattr(config, "SC_CONV_FUSED_BLOCK", "scconv"),
        )
    model = model.to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())

    print(f"BiSeNetV2 + ScConv / {args.dataset_tag}")
    print(f"Device: {device}; parameters: {parameter_count / 1e6:.2f}M")
    print(f"Data: {args.data_root}")
    print(f"Samples: train={len(train_loader.dataset)}, val={len(val_loader.dataset)}, "
          f"test={len(test_loader.dataset)}")
    print(
        f"ScConv: enabled={config.USE_SC_CONV}; placement={config.SC_CONV_PLACEMENT}; "
        f"fused_block={getattr(config, 'SC_CONV_FUSED_BLOCK', 'scconv')}; "
        f"residual={config.SC_CONV_RESIDUAL} (scale={config.SC_CONV_RESIDUAL_SCALE}); "
        f"groups={config.SC_CONV_GROUP_NUM}; gate={config.SC_CONV_GATE_THRESHOLD}; "
        f"gamma_normalise={getattr(config, 'SC_CONV_GAMMA_NORMALISE', True)}"
    )
    replaced = getattr(model, "scconv_replaced_layers", [])
    if replaced:
        print(f"ScConv 替换了 {len(replaced)} 个卷积层：")
        for item in replaced:
            print(f"  - {item}")
    if args.validation_uses_official_test:
        print("WARNING: 官方 test 集将逐轮用于验证和最佳检查点选择；最终 test 指标不是独立测试结果。")
    print(f"Input: {args.image_size}x{args.image_size}; classes={args.num_classes}; "
          f"auxiliary loss weight={args.aux_weight}")
    print(
        f"Schedule: {args.scheduler_name}"
        + (f" (warmup={args.warmup_epochs})" if str(args.scheduler_name).lower() == "cosine" else "")
        + f"; epochs={args.epochs}; lr={args.lr}; wd={args.weight_decay}; "
        f"deterministic_algorithms={args.deterministic_effective}"
    )

    ce_loss = nn.CrossEntropyLoss()
    dice_loss = DiceLoss()
    best_path = args.output_dir / BEST_CHECKPOINT_NAME
    log_path = args.output_dir / TRAIN_HISTORY_NAME
    history_jsonl_path = args.output_dir / config.TRAIN_HISTORY_JSONL_NAME

    if args.eval_only:
        if not best_path.is_file():
            raise FileNotFoundError(f"找不到已有最佳检查点：{best_path}")
        checkpoint = load_checkpoint_safely(best_path, device)
        model.load_state_dict(checkpoint["model_state"])
        evaluate_and_save(
            model,
            test_loader,
            ce_loss,
            dice_loss,
            args,
            device,
            checkpoint,
            load_training_history(history_jsonl_path),
            parameter_count,
        )
        return

    optimizer_name = config.OPTIMIZER_NAME.lower()
    if optimizer_name == "adamw":
        optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif optimizer_name == "adam":
        optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        raise ValueError(f"Unsupported optimizer in config.py: {config.OPTIMIZER_NAME}")
    scheduler, scheduler_name = build_scheduler(optimizer, args)

    history = []
    best_miou = -1.0
    patience_counter = 0
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(
            model, train_loader, optimizer, ce_loss, dice_loss, args, device,
            f"Train {epoch}/{args.epochs}",
        )
        validation_description = (
            f"Official test/Val {epoch}/{args.epochs}"
            if args.validation_uses_official_test
            else f"Val {epoch}/{args.epochs}"
        )
        val_metrics = run_epoch(
            model, val_loader, None, ce_loss, dice_loss, args, device,
            validation_description,
        )
        if scheduler_name == "plateau":
            scheduler.step(val_metrics["mIoU"])
        else:
            scheduler.step()
        learning_rate = optimizer.param_groups[0]["lr"]
        record = {
            "epoch": epoch,
            "lr": learning_rate,
            "validation_source": args.validation_source,
            "train": train_metrics,
            "val": val_metrics,
        }
        # 记录训练后的 ScConv 残差尺度，用于判断模块是否真的参与（Gate 1 判据）。
        scales = {
            name: float(parameter.detach())
            for name, parameter in model.named_parameters()
            if name.endswith(".scale")
        }
        if scales:
            record["scconv_scale"] = scales
        history.append(record)

        validation_label = (
            "official test (validation)"
            if args.validation_uses_official_test else "validation"
        )
        print(
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"train loss {train_metrics['loss']:.4f}, mIoU {train_metrics['mIoU']:.4f} | "
            f"{validation_label} loss {val_metrics['loss']:.4f}, "
            f"mIoU {val_metrics['mIoU']:.4f}, mDice {val_metrics['mDice']:.4f}, "
            f"Acc {val_metrics['accuracy']:.4f} | "
            f"lr {learning_rate:.2e}"
        )

        if val_metrics["mIoU"] > best_miou:
            best_miou = val_metrics["mIoU"]
            patience_counter = 0
            temporary_checkpoint_path = best_path.with_name(best_path.name + ".tmp")
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "epoch": epoch,
                    "val_metrics": val_metrics,
                    "num_classes": args.num_classes,
                    "config": resolved_config,
                    "args": {key: str(value) if isinstance(value, Path) else value
                             for key, value in vars(args).items()},
                },
                temporary_checkpoint_path,
            )
            temporary_checkpoint_path.replace(best_path)
        else:
            patience_counter += 1

        save_training_history_jsonl(history, history_jsonl_path)
        validation_source = args.validation_source
        save_training_history_text(
            history,
            log_path,
            model_name=f"BiSeNetV2 + ScConv | {args.dataset_tag}",
            validation_source=validation_source,
            class_names=tuple(args.class_names),
        )

        if patience_counter >= args.patience:
            print(f"Early stopping after {epoch} epochs")
            break

    elapsed = time.time() - start_time
    selection_label = "official test mIoU" if args.validation_uses_official_test else "validation mIoU"
    print(f"Training time: {elapsed / 60.0:.1f} min; best {selection_label}: {best_miou:.4f}")

    # 读取最佳检查点，并在官方测试集上生成最终指标和图表；test-as-validation 模式下该集合已参与选模。
    checkpoint = load_checkpoint_safely(best_path, device)
    model.load_state_dict(checkpoint["model_state"])
    evaluate_and_save(
        model,
        test_loader,
        ce_loss,
        dice_loss,
        args,
        device,
        checkpoint,
        history,
        parameter_count,
        training_seconds=elapsed,
    )
    print(f"Checkpoint: {best_path}")


if __name__ == "__main__":
    main()
