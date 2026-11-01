"""UNet 统一训练脚本，支持 NEU / SD-Saliency / Magnetic-Tile。

使用 --dataset 参数指定数据集：
  python train.py --model unet --config config_unet --dataset neu
  python train.py --model unet --config config_unet --dataset sd-saliency
  python train.py --model unet --config config_unet --dataset magnetic-tile

注意（2026-10 变更）：本脚本的数据管线已改为调用 ``scripts/dataset.py`` 的
``build_loaders``，与 BiSeNetV2 / ScConv 那条路径**完全共用**同一份固定划分清单、
同一套增强算子和同一个 shuffle 生成器。这样论文主表里 U-Net 与 BiSeNetV2 才真的
只差架构，而不是"差架构 + 差数据管线"。

代价是数据集名字统一成 ``dataset.py`` 的写法（``sd`` → ``sd-saliency``、
``magnetic`` → ``magnetic-tile``），并且不再支持 Severstal（``build_loaders``
没有这个数据集；本项目也没下载它）。原来的 ``datasetload`` 包仍然保留在仓库里，
只是这条路径不再使用。
"""

import os
import argparse
import json
import math
import random
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR, ReduceLROnPlateau
from tqdm import tqdm

# 模块级引用：build_loaders 需要整个配置模块，而不是几个散装常量。
# train.py 走 --config 时会把 sys.modules["config_unet"] 替换成派生配置，
# 因此这里的 `config` 也会跟着变成那一份（见 train.py 的 unet 分支）。
import config_unet as config
from config_unet import (
    DATASET_DEFAULTS,
    DATASET_PATHS,
    SAVE_DIR,
    LOG_FILE,
    EPOCHS,
    INITIAL_LR,
    WEIGHT_DECAY,
    LR_FACTOR,
    LR_PATIENCE,
    MIN_LR,
    MAX_GRAD_NORM,
    EARLY_STOPPING_PATIENCE,
    SEED,
    NUM_WORKERS,
    NO_AUGMENT,
    VIS_ENABLED,
    VIS_MAX_SAMPLES,
    PROBABILITY_HISTOGRAM_BINS,
    SCHEDULER_NAME,
    WARMUP_EPOCHS,
    DETERMINISTIC_ALGORITHMS,
)
# 数据管线统一走 scripts/dataset.py，与 BiSeNetV2 / ScConv 一侧共用同一份固定
# 划分清单、同一套增强算子。原来这里用 datasetload 包里自带的一套 Dataset，
# 它的增强实现（random 模块 + random_flip_rotate）与 dataset.py（np.random +
# 各自的翻转/旋转概率）并不等价，两条路径的"随机性来源"都不一样；主表要并列
# U-Net 与 BiSeNetV2，就必须先把数据管线对齐。
from dataset import build_loaders
from UNet import build_unet
from utils_losses_metrics import (
    DiceLoss,
    compute_confusion_matrix,
    metrics_from_confmat,
    evaluate_metrics,
)
from plots import (
    save_evaluation_plots,
    save_training_history_jsonl,
    save_training_history_text,
    save_training_results_plot,
    update_probability_histogram,
)


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------

def set_seed(seed: int, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if deterministic:
        # warn_only=True：仍有算子不支持确定性实现时只告警不中断，告警即问题点。
        torch.use_deterministic_algorithms(True, warn_only=True)
    else:
        torch.use_deterministic_algorithms(False)


def make_generator(seed: int) -> torch.Generator:
    """训练集 shuffle 与 worker 种子专用生成器，与全局 RNG 解耦。"""
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def jsonable(value):
    """递归把 pathlib.Path 转成 str，让配置能安全地写 JSON / 存进检查点。

    ``args.data_root`` 现在统一是 ``Path``（``dataset.build_loaders`` 要用 ``/``
    拼接子目录），但 ``json.dump`` 不认 Path，而检查点又要求能用
    ``torch.load(weights_only=True)`` 反序列化——一旦把 Path 存进去，加载时会报
    "Unsupported global"。所以两处都先过一遍这个函数。
    """
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


def build_scheduler(optimizer, args):
    """按配置创建学习率调度器，返回 (调度器, 名称)。"""
    name = str(args.scheduler_name).lower()
    if name == "plateau":
        scheduler = ReduceLROnPlateau(
            optimizer, mode="max", factor=args.lr_factor,
            patience=args.lr_patience, min_lr=args.min_lr,
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


def build_dataloaders(args, device, generator=None):
    """构建 train / val / test DataLoader。

    委托给 ``dataset.build_loaders``：与 BiSeNetV2 / ScConv 一侧完全同一份数据管线
    （同一份固定划分清单、同一套增强算子、同一个 shuffle 生成器）。原实现用
    ``datasetload`` 包里自带的一套 Dataset，增强的随机性来源和算子都与主线不同，
    直接并列进主表会混淆"架构差异"和"数据管线差异"。
    """
    return build_loaders(args, device, config, generator=generator)


# ---------------------------------------------------------------------------
# 训练 / 验证
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, optimizer, ce_loss, dice_loss, device, num_classes, max_norm=5.0):
    model.train()
    total_loss, total_ce, total_dice, num_batches = 0.0, 0.0, 0.0, 0
    confusion = torch.zeros((num_classes, num_classes), dtype=torch.int64, device=device)

    for batch in tqdm(loader, desc="Training", leave=False):
        x_ori = batch["image"].to(device)
        masks = batch["mask"].to(device)

        logits = model(x_ori)
        if logits.shape[-2:] != masks.shape[-2:]:
            logits = nn.functional.interpolate(logits, size=masks.shape[-2:],
                                               mode="bilinear", align_corners=False)

        loss_ce = ce_loss(logits, masks)
        loss_dice = dice_loss(logits, masks, num_classes=num_classes)
        loss = loss_ce + loss_dice

        optimizer.zero_grad()
        loss.backward()
        if max_norm > 0:
            nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()

        confusion += compute_confusion_matrix(torch.argmax(logits.detach(), dim=1), masks, num_classes)

        total_loss += loss.item()
        total_ce += loss_ce.item()
        total_dice += loss_dice.item()
        num_batches += 1

    train_metrics = metrics_from_confmat(confusion)
    return total_loss / num_batches, total_ce / num_batches, total_dice / num_batches, train_metrics


@torch.no_grad()
def validate(model, loader, ce_loss, dice_loss, device, num_classes, probability_histogram=None):
    model.eval()
    total_loss, total_ce, total_dice, num_batches = 0.0, 0.0, 0.0, 0
    all_preds, all_masks = [], []

    for batch in tqdm(loader, desc="Validating", leave=False):
        x_ori = batch["image"].to(device)
        masks = batch["mask"].to(device)

        logits = model(x_ori)
        if logits.shape[-2:] != masks.shape[-2:]:
            logits = nn.functional.interpolate(logits, size=masks.shape[-2:],
                                               mode="bilinear", align_corners=False)

        loss_ce = ce_loss(logits, masks)
        loss_dice = dice_loss(logits, masks, num_classes=num_classes)
        loss = loss_ce + loss_dice

        total_loss += loss.item()
        total_ce += loss_ce.item()
        total_dice += loss_dice.item()
        num_batches += 1

        preds = torch.argmax(logits, dim=1)
        all_preds.append(preds.cpu().numpy())
        all_masks.append(masks.cpu().numpy())
        if probability_histogram is not None:
            probabilities = torch.softmax(logits, dim=1).cpu().numpy()
            update_probability_histogram(
                probability_histogram, probabilities, masks.cpu().numpy()
            )

    avg_loss = total_loss / num_batches
    avg_ce = total_ce / num_batches
    avg_dice = total_dice / num_batches

    all_preds = np.concatenate(all_preds)
    all_masks = np.concatenate(all_masks)
    metrics = evaluate_metrics(all_masks, all_preds, num_classes)
    metrics["confusion_matrix"] = compute_confusion_matrix(
        torch.from_numpy(all_preds.astype(np.int64)),
        torch.from_numpy(all_masks.astype(np.int64)),
        num_classes,
    ).tolist()

    return avg_loss, avg_ce, avg_dice, metrics


# ---------------------------------------------------------------------------
# 测试集可视化
# ---------------------------------------------------------------------------

def _mask_to_colorimg(mask: np.ndarray, color_map: dict) -> np.ndarray:
    """将单通道掩膜转换为 BGR 彩色图像。"""
    h, w = mask.shape
    color_img = np.zeros((h, w, 3), dtype=np.uint8)
    for class_id, color in color_map.items():
        color_img[mask == class_id] = color
    return color_img


def _denormalize_image(img: np.ndarray) -> np.ndarray:
    """将归一化的 [0,1] 图像转换为 [0,255] uint8。"""
    # img: [C, H, W] or [H, W, C]
    if img.ndim == 3 and img.shape[0] in (1, 3):
        img = np.transpose(img, (1, 2, 0))
    img = np.clip(img * 255, 0, 255).astype(np.uint8)
    if img.shape[-1] == 1:
        img = np.repeat(img, 3, axis=-1)
    return img


def _add_legend(img_bgr: np.ndarray, class_names: list, color_map: dict,
                font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=0.55) -> np.ndarray:
    """在图像左上角叠加颜色图例。"""
    img_bgr = img_bgr.copy()
    x0, y0 = 5, 5
    line_h = 18
    for i, name in enumerate(class_names):
        color = tuple(color_map.get(i, [128, 128, 128]))
        cv2.rectangle(img_bgr, (x0, y0), (x0 + 20, y0 + line_h - 2), color, -1)
        cv2.putText(img_bgr, name, (x0 + 24, y0 + line_h - 5),
                    font, font_scale, (255, 255, 255), 1, cv2.LINE_AA)
        y0 += line_h
    return img_bgr


@torch.no_grad()
def visualize_predictions(model, loader, device, num_classes, class_names, color_map,
                         save_dir: str, max_samples: int = 20):
    """在测试集上推理，保存三列对比图 (原图 / GT / 预测)。

    ``class_names`` / ``color_map`` 由 ``dataset.build_loaders`` 写回 ``args``，
    与 BiSeNetV2 一侧的可视化用同一套配色。
    """
    model.eval()
    os.makedirs(save_dir, exist_ok=True)

    class_names = list(class_names) or [f"C{i}" for i in range(num_classes)]

    saved = 0
    for batch in tqdm(loader, desc="Visualizing", leave=False):
        x_ori = batch["image"].to(device)
        masks = batch["mask"].numpy()
        ids = batch["id"] if "id" in batch else [str(i) for i in range(len(masks))]

        logits = model(x_ori)
        if logits.shape[-2:] != masks.shape[-2:]:
            logits = nn.functional.interpolate(logits, size=masks.shape[-2:],
                                               mode="bilinear", align_corners=False)
        preds = torch.argmax(logits, dim=1).cpu().numpy()

        x_ori_np = x_ori.cpu().numpy()

        for i in range(len(ids)):
            sid = ids[i] if isinstance(ids[i], str) else str(ids[i].item())
            orig_img = _denormalize_image(x_ori_np[i])
            gt_color = _mask_to_colorimg(masks[i], color_map)
            pred_color = _mask_to_colorimg(preds[i], color_map)

            # 缩小宽高（Severstal 原始 1600x256 太大，缩放到 800 宽）
            scale = 800 / orig_img.shape[1] if orig_img.shape[1] > 800 else 1.0
            if scale < 1.0:
                new_w = int(orig_img.shape[1] * scale)
                new_h = int(orig_img.shape[0] * scale)
                orig_img = cv2.resize(orig_img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
                gt_color = cv2.resize(gt_color, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
                pred_color = cv2.resize(pred_color, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

            # 添加图例
            orig_img = _add_legend(orig_img, class_names, color_map)
            gt_color = _add_legend(gt_color, class_names, color_map)
            pred_color = _add_legend(pred_color, class_names, color_map)

            # 横向拼接三图
            row = np.hstack([orig_img, gt_color, pred_color])
            cv2.imwrite(os.path.join(save_dir, f"{sid}.png"), row)
            saved += 1
            if saved >= max_samples:
                break
        if saved >= max_samples:
            break

    print(f"  可视化完成: {saved} 张图保存至 {save_dir}")
    return saved


# ---------------------------------------------------------------------------
# 参数解析
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Train UNet on multiple datasets")
    p.add_argument("--dataset", type=str, default="neu",
                   choices=["neu", "sd-saliency", "magnetic-tile"],
                   help="数据集: neu(NEU_Seg), sd-saliency(SD-Saliency), magnetic-tile(Magnetic-tile)")
    p.add_argument("--data_root", type=str, default=None,
                   help="数据集根目录（默认从 config.py 读取）")
    p.add_argument("--epochs", type=int, default=EPOCHS)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--lr", type=float, default=INITIAL_LR)
    p.add_argument("--weight_decay", type=float, default=WEIGHT_DECAY)
    p.add_argument("--num_classes", type=int, default=None)
    p.add_argument("--image_size", type=int, default=None)
    p.add_argument("--num_workers", type=int, default=NUM_WORKERS)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--save_dir", type=str, default=SAVE_DIR)
    p.add_argument("--log_file", type=str, default=LOG_FILE)
    p.add_argument("--no_augment", action="store_true", default=NO_AUGMENT,
                   help="禁用数据增强")
    p.add_argument("--lr_factor", type=float, default=LR_FACTOR)
    p.add_argument("--lr_patience", type=int, default=LR_PATIENCE)
    p.add_argument("--patience", type=int, default=EARLY_STOPPING_PATIENCE,
                   help="Early stopping 耐心值")
    p.add_argument("--min_lr", type=float, default=MIN_LR)
    p.add_argument("--max_norm", type=float, default=MAX_GRAD_NORM,
                   help="梯度裁剪阈值；0 表示关闭（BiSeNetV2 那条路径不裁剪）")
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--no_vis", action="store_true", default=not VIS_ENABLED,
                   help="训练结束后跳过测试集可视化")
    p.add_argument("--vis_max_samples", type=int, default=VIS_MAX_SAMPLES,
                   help="测试集可视化最多保存多少张图")
    p.add_argument("--scheduler_name", type=str, default=SCHEDULER_NAME,
                   help="plateau 或 cosine；cosine 使用 warmup + 余弦退火")
    p.add_argument("--warmup_epochs", type=int, default=WARMUP_EPOCHS,
                   help="cosine 调度器的线性 warmup 轮数")
    p.add_argument("--deterministic", choices=("auto", "on", "off"), default="auto",
                   help="是否启用 torch 确定性算法；auto 跟随配置 DETERMINISTIC_ALGORITHMS")

    return p.parse_args()


# ---------------------------------------------------------------------------
# 主函数
# ---------------------------------------------------------------------------

def create_run_directory(output_root: str, dataset_name: str, epochs: int) -> str:
    """为每次实验创建从 0 开始且不覆盖旧结果的目录。"""
    os.makedirs(output_root, exist_ok=True)
    index = 0
    while True:
        run_dir = os.path.join(output_root, f"{dataset_name}_{epochs}_{index}")
        try:
            os.mkdir(run_dir)
            return run_dir
        except FileExistsError:
            index += 1


# 旧简称 -> dataset.build_loaders 认的名字（历史脚本用的是 sd / magnetic）
DATASET_ALIASES = {"sd": "sd-saliency", "magnetic": "magnetic-tile"}


def main():
    args = parse_args()

    # 应用数据集默认值：DATASET_DEFAULTS 仍以旧简称作键，先取值再映射名字
    defaults = DATASET_DEFAULTS[args.dataset]
    args.data_root = Path(args.data_root or DATASET_PATHS[args.dataset])
    args.num_classes = args.num_classes or defaults["num_classes"]
    args.batch_size = args.batch_size or defaults["batch_size"]
    if args.image_size is None and "image_size" in defaults:
        args.image_size = defaults["image_size"][0]
    args.dataset_original = args.dataset
    args.dataset = DATASET_ALIASES.get(args.dataset, args.dataset)

    if args.deterministic == "on":
        deterministic = True
    elif args.deterministic == "off":
        deterministic = False
    else:
        deterministic = bool(DETERMINISTIC_ALGORITHMS)
    args.deterministic_effective = deterministic
    set_seed(args.seed, deterministic)

    device = torch.device(args.device)
    print(f"\n{'='*70}")
    print(f"UNet Training — Dataset: {args.dataset.upper()}")
    print(f"{'='*70}")
    print(f"  Device: {device}, Seed: {args.seed}")
    print(f"  Data root: {args.data_root}")
    print(f"  Num classes: {args.num_classes}, Image size: {args.image_size}")
    print(f"  Epochs: {args.epochs}, Batch size: {args.batch_size}")
    print(f"  LR: {args.lr}, Weight decay: {args.weight_decay}")
    print(f"  LR scheduler: {args.scheduler_name}"
          + (f" (warmup={args.warmup_epochs})" if str(args.scheduler_name).lower() == "cosine" else ""))
    print(f"  Early stopping patience: {args.patience}")
    print(f"  Max grad norm: {args.max_norm} (0 = 关闭裁剪)")
    print(f"  Deterministic algorithms: {deterministic}")
    print(f"  Augment: {not args.no_augment}")
    print()

    train_loader, val_loader, test_loader = build_dataloaders(
        args, device, generator=make_generator(args.seed)
    )
    print(f"  Train: {len(train_loader.dataset)}, Val: {len(val_loader.dataset)}, Test: {len(test_loader.dataset)}")

    # 模型初始化使用独立的 RNG 域：fork_rng 结束后全局 RNG 复原，使不同架构/不同
    # 臂之间的数据增强序列只由 seed 决定（与 train_scconv.py 同一套做法）。
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(args.seed + 1000)
        model = build_unet(in_channels=3, base_channels=32, num_classes=args.num_classes)
    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  UNet params: {n_params / 1e6:.2f}M")

    # 损失 & 优化器
    ce_loss = nn.CrossEntropyLoss()
    dice_loss = DiceLoss()
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler, scheduler_name = build_scheduler(optimizer, args)

    # 保存路径
    save_dir = create_run_directory(args.save_dir, args.dataset.upper(), args.epochs)
    save_path = os.path.join(save_dir, "best.pth")
    if args.log_file is None:
        args.log_file = os.path.join(save_dir, "result.txt")
    log_lines = []
    run_config = jsonable(vars(args))
    run_config.update({
        "model": "U-Net",
        "split_counts": {
            "train": len(train_loader.dataset),
            "val": len(val_loader.dataset),
            "test": len(test_loader.dataset),
        },
    })
    with open(os.path.join(save_dir, "config.json"), "w", encoding="utf-8") as config_file:
        json.dump(run_config, config_file, ensure_ascii=False, indent=2, default=str)

    def log(msg):
        print(msg)
        log_lines.append(msg)

    log("=" * 70)
    log(f"UNet Training — Dataset: {args.dataset.upper()}")
    log(f"Data root: {args.data_root}")
    log(f"Num classes: {args.num_classes}, Image size: {args.image_size}")
    log(f"Epochs: {args.epochs}, Batch size: {args.batch_size}, LR: {args.lr}")
    log(f"Model params: {n_params / 1e6:.2f}M")
    log("=" * 70)

    best_miou = -1.0
    best_epoch = 0
    patience_counter = 0
    start_time = time.time()
    history = []
    history_jsonl_path = os.path.join(save_dir, "history.jsonl")

    for epoch in range(1, args.epochs + 1):
        should_stop = False
        t_loss, t_ce, t_dice, train_metrics = train_one_epoch(
            model, train_loader, optimizer, ce_loss, dice_loss,
            device, args.num_classes, args.max_norm,
        )
        v_loss, v_ce, v_dice, metrics = validate(
            model, val_loader, ce_loss, dice_loss, device, args.num_classes,
        )

        if scheduler_name == "plateau":
            scheduler.step(metrics['mIoU'])
        else:
            scheduler.step()
        lr_now = optimizer.param_groups[0]["lr"]
        train_summary = {
            "loss": t_loss, "main_loss": t_loss,
            "mIoU": train_metrics["mIoU"], "mDice": train_metrics["mDice"],
            "accuracy": train_metrics["Acc"],
        }
        val_summary = {
            "loss": v_loss, "main_loss": v_loss,
            "mIoU": metrics["mIoU"], "mDice": metrics["mDice"],
            "accuracy": metrics["Acc"],
        }
        history.append({
            "epoch": epoch, "lr": lr_now,
            "train": train_summary, "val": val_summary,
        })

        log(f"Epoch [{epoch}/{args.epochs}]")
        log(f"  Train: loss={t_loss:.4f} (ce={t_ce:.4f}, dice={t_dice:.4f})")
        log(f"  Train mIoU: {train_metrics['mIoU']:.4f}, mDice: {train_metrics['mDice']:.4f}, Acc: {train_metrics['Acc']:.4f}")
        log(f"  Val:   loss={v_loss:.4f} (ce={v_ce:.4f}, dice={v_dice:.4f})")
        log(f"  Val mIoU: {metrics['mIoU']:.4f}, mDice: {metrics['mDice']:.4f}, Acc: {metrics['Acc']:.4f}")
        log(f"  LR: {lr_now:.2e}")

        if metrics['mIoU'] > best_miou:
            best_miou = metrics['mIoU']
            best_epoch = epoch
            patience_counter = 0
            torch.save({
                "model_state": model.state_dict(),
                "epoch": epoch,
                "mIoU": metrics['mIoU'],
                "mDice": metrics['mDice'],
                "Acc": metrics['Acc'],
                "per_class_iou": metrics['per_class_iou'],
                "num_classes": args.num_classes,
                "args": jsonable(vars(args)),
            }, save_path)
            log(f"  >> Best model saved (mIoU={best_miou:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                log(f"Early stopping at epoch {epoch}")
                should_stop = True

        with open(args.log_file, 'w') as f:
            f.write('\n'.join(log_lines))
        save_training_history_jsonl(history, Path(history_jsonl_path))
        save_training_history_text(
            history, Path(args.log_file), model_name=f"U-Net | {args.dataset.upper()}",
            validation_source="固定验证集", best_epoch=best_epoch,
            class_names=tuple(args.class_names),
        )
        if should_stop:
            break

    elapsed = time.time() - start_time
    log(f"\nTraining completed in {elapsed:.1f}s")
    log(f"Best val mIoU: {best_miou:.4f}")

    # 使用最佳验证权重仅在训练结束后评估一次独立 test 集。
    best_checkpoint = torch.load(save_path, map_location=device, weights_only=True)
    model.load_state_dict(best_checkpoint["model_state"])
    bins = PROBABILITY_HISTOGRAM_BINS
    probability_histogram = (
        np.zeros((args.num_classes, bins), dtype=np.int64),
        np.zeros((args.num_classes, bins), dtype=np.int64),
    )
    test_loss, test_ce, test_dice, test_metrics = validate(
        model, test_loader, ce_loss, dice_loss, device, args.num_classes,
        probability_histogram=probability_histogram,
    )
    test_metrics_for_plots = dict(test_metrics)
    test_metrics_for_plots.update({
        "loss": test_loss, "main_loss": test_loss, "accuracy": test_metrics["Acc"],
    })
    test_summary = {
        "model": "U-Net", "dataset": args.dataset,
        "checkpoint": save_path, "best_epoch": best_checkpoint["epoch"],
        "split_counts": {"train": len(train_loader.dataset), "val": len(val_loader.dataset), "test": len(test_loader.dataset)},
        "test_metrics_are_independent": args.dataset == "neu",
        "test_loss": test_loss, "test_ce_loss": test_ce, "test_dice_loss": test_dice,
        # 以下字段供 scripts/run_metrics.py 把 U-Net 并入统一汇总表。老版本的
        # U-Net 结果没有它们，读取端会退回 history.jsonl 与现场计算参数量。
        "parameter_count": n_params,
        "epochs_planned": args.epochs,
        "epochs_trained": len(history),
        "training_seconds": elapsed,
        "best_val_miou": best_miou,
        "scheduler": f"{args.scheduler_name} (warmup={args.warmup_epochs})",
        "deterministic_algorithms": deterministic,
        "seed": args.seed,
        **test_metrics,
    }
    metrics_path = os.path.join(save_dir, "test_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as metrics_file:
        json.dump(test_summary, metrics_file, ensure_ascii=False, indent=2, default=str)
    curve_metrics = save_evaluation_plots(
        probability_histogram, test_metrics["confusion_matrix"],
        tuple(args.class_names), Path(save_dir),
        plot_class_names=tuple(args.plot_class_names),
    )
    test_summary["curve_metrics"] = curve_metrics
    with open(metrics_path, "w", encoding="utf-8") as metrics_file:
        json.dump(test_summary, metrics_file, ensure_ascii=False, indent=2, default=str)
    save_training_results_plot(
        history, Path(save_dir) / "results2中断.png", best_epoch=best_epoch,
        final_metrics=test_metrics_for_plots, validation_source="固定验证集",
    )
    log(f"Independent test: loss={test_loss:.4f}, mIoU={test_metrics['mIoU']:.4f}, mDice={test_metrics['mDice']:.4f}, Acc={test_metrics['Acc']:.4f}")
    log(f"Test metrics saved: {metrics_path}")
    with open(args.log_file, "w", encoding="utf-8") as log_file:
        log_file.write("\n".join(log_lines))
    save_training_history_text(
        history, Path(args.log_file), model_name=f"U-Net | {args.dataset.upper()}",
        validation_source="固定验证集", best_epoch=best_epoch,
        final_metrics=test_metrics_for_plots, training_seconds=elapsed,
        class_names=tuple(args.class_names),
    )

    # --- 测试集可视化 ---
    if not args.no_vis:
        print(f"\n生成测试集可视化...")
        vis_save_dir = os.path.join(save_dir, "visualization")
        # 加载最佳模型（device 已转回 cpu/gpu，需要与保存时一致）
        visualize_predictions(
            model=model, loader=test_loader,
            device=device, num_classes=args.num_classes,
            class_names=args.class_names, color_map=args.color_map,
            save_dir=vis_save_dir, max_samples=args.vis_max_samples,
        )

    print(f"\nDone! Best model: {save_path} (mIoU={best_miou:.4f})")


if __name__ == "__main__":
    main()


