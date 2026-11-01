"""SD900-only adapter: freeze one split and delegate to the existing trainer.

This file is intentionally separate from train.py and the NEU training path.
"""
from __future__ import annotations

import argparse
import importlib
import runpy
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
CONFIG_DIR = PROJECT_DIR / "config"
SCRIPTS_DIR = PROJECT_DIR / "scripts"
SPLIT_DIR = PROJECT_DIR / "data" / "SD-saliency" / "splits"
DATA_ROOT = PROJECT_DIR / "data" / "SD-saliency"
EXPECTED_COUNTS = {"train": 630, "val": 135, "test": 135}
EXPECTED_PER_PREFIX = {"train": 210, "val": 45, "test": 45}
PREFIXES = ("In", "Pa", "Sc")


def read_manifest() -> dict[str, list[str]]:
    manifests: dict[str, list[str]] = {}
    for split in ("train", "val", "test"):
        path = SPLIT_DIR / f"{split}.txt"
        if not path.is_file():
            raise FileNotFoundError(f"SD900 固定划分清单不存在：{path}")
        names = [line.strip() for line in path.read_text(encoding="utf-8-sig").splitlines()
                 if line.strip() and not line.lstrip().startswith("#")]
        if len(names) != EXPECTED_COUNTS[split] or len(set(names)) != len(names):
            raise ValueError(f"{path} 应含 {EXPECTED_COUNTS[split]} 个唯一样本，实际为 {len(names)}")
        for prefix in PREFIXES:
            count = sum(name.startswith(prefix + "_") for name in names)
            if count != EXPECTED_PER_PREFIX[split]:
                raise ValueError(f"{path} 中 {prefix} 样本应为 {EXPECTED_PER_PREFIX[split]}，实际为 {count}")
        manifests[split] = names
    sets = [set(manifests[key]) for key in ("train", "val", "test")]
    if any(sets[i] & sets[j] for i in range(3) for j in range(i + 1, 3)):
        raise ValueError("SD900 固定划分存在交叉样本")
    expected_all = {f"{prefix}_{index}" for prefix in PREFIXES for index in range(1, 301)}
    if set.union(*sets) != expected_all:
        missing = sorted(expected_all - set.union(*sets))[:5]
        extra = sorted(set.union(*sets) - expected_all)[:5]
        raise ValueError(f"SD900 固定划分与数据集不一致；缺失示例={missing}，多余示例={extra}")
    return manifests


def install_fixed_split() -> None:
    if not DATA_ROOT.is_dir():
        raise FileNotFoundError(f"SD900 数据目录不存在：{DATA_ROOT}")
    manifests = read_manifest()
    manifest_sets = {key: set(value) for key, value in manifests.items()}
    sys.path.insert(0, str(SCRIPTS_DIR))
    sys.path.insert(0, str(CONFIG_DIR))
    dataset_module = importlib.import_module("dataset")

    def split_from_manifest(samples, _seed: int, _test_ratio: float = 0.15):
        by_name = {sample[2]: sample for sample in samples}
        if len(by_name) != len(samples):
            raise ValueError("SD900 图像文件名存在重复，无法应用固定划分")
        if set(by_name) != set.union(*manifest_sets.values()):
            raise ValueError("当前 SD900 图像集合与固定划分清单不一致")
        return tuple([by_name[name] for name in manifests[split]] for split in ("train", "val", "test"))

    dataset_module._stratified_external_split = split_from_manifest


def make_c2c_autograd_safe(builder):
    """Keep C2c's ReLU math while disabling its unsafe in-place storage mutation."""
    import torch.nn as nn

    def build(*args, **kwargs):
        model = builder(*args, **kwargs)
        fused = getattr(model, "fused_scconv", None)
        if fused is not None:
            for module in fused.modules():
                if isinstance(module, nn.ReLU):
                    module.inplace = False
        return model
    return build


def smoke(model_name: str, config_name: str, batch_size: int, device_name: str) -> int:
    """Take one real batch through one optimizer step; never saves a checkpoint."""
    import torch
    import torch.nn.functional as F
    from types import SimpleNamespace

    if model_name not in ("unet", "bisenet", "scconv"):
        raise ValueError(f"不支持的模型：{model_name}")
    for path in (CONFIG_DIR, PROJECT_DIR / "models", SCRIPTS_DIR):
        sys.path.insert(0, str(path))
    if model_name == "unet":
        cfg = importlib.import_module(config_name)
        sys.modules["config_unet"] = cfg
    elif model_name == "bisenet":
        cfg = importlib.import_module(config_name)
        sys.modules["config"] = cfg
    else:
        importlib.import_module("config")
        importlib.import_module("configScConv")
        cfg = importlib.import_module(config_name)
        sys.modules["configScConv"] = cfg

    install_fixed_split()
    from dataset import build_loaders

    device = torch.device(device_name)
    args = SimpleNamespace(
        dataset="sd-saliency", data_root=DATA_ROOT, num_classes=None,
        image_size=200, batch_size=batch_size, num_workers=0,
        no_augment=True, seed=42,
    )
    train_loader, val_loader, test_loader = build_loaders(args, device, cfg)
    batch = next(iter(train_loader))
    images, masks = batch["image"].to(device), batch["mask"].to(device)

    if model_name == "unet":
        from UNet import build_unet
        model = build_unet(in_channels=3, base_channels=32, num_classes=2)
    elif model_name == "bisenet":
        if getattr(cfg, "USE_BRANCH_ORTHOGONALITY", False):
            from bisenetv2_orth import build_bisenetv2_orth
            builder = build_bisenetv2_orth
        else:
            from bisenetv2 import build_bisenetv2
            builder = build_bisenetv2
        model = builder(
            num_classes=2, auxiliary=cfg.USE_AUXILIARY_HEADS,
            main_head_channels=cfg.MAIN_HEAD_CHANNELS,
            auxiliary_head_channels=cfg.AUXILIARY_HEAD_CHANNELS,
            dropout=cfg.HEAD_DROPOUT,
        )
    else:
        from bisenetv2_scconv import build_bisenetv2_scconv
        model = build_bisenetv2_scconv(
            num_classes=2, auxiliary=cfg.USE_AUXILIARY_HEADS,
            main_head_channels=cfg.MAIN_HEAD_CHANNELS,
            auxiliary_head_channels=cfg.AUXILIARY_HEAD_CHANNELS,
            dropout=cfg.HEAD_DROPOUT,
            in_channels=cfg.INPUT_CHANNELS,
            detail_channels=cfg.DETAIL_CHANNELS,
            detail_stage_repeats=cfg.DETAIL_STAGE_REPEATS,
            semantic_channels=cfg.SEMANTIC_CHANNELS,
            expansion=cfg.GE_EXPANSION,
            stage_repeats=cfg.STAGE_REPEATS,
            scconv_enabled=cfg.USE_SC_CONV,
            scconv_group_num=cfg.SC_CONV_GROUP_NUM,
            scconv_gate_threshold=cfg.SC_CONV_GATE_THRESHOLD,
            scconv_alpha=cfg.SC_CONV_ALPHA,
            scconv_squeeze_ratio=cfg.SC_CONV_SQUEEZE_RATIO,
            scconv_group_size=cfg.SC_CONV_GROUP_SIZE,
            scconv_group_kernel_size=cfg.SC_CONV_GROUP_KERNEL_SIZE,
            scconv_gamma_normalise=getattr(cfg, "SC_CONV_GAMMA_NORMALISE", True),
            scconv_placement=cfg.SC_CONV_PLACEMENT,
            scconv_residual=cfg.SC_CONV_RESIDUAL,
            scconv_residual_scale=cfg.SC_CONV_RESIDUAL_SCALE,
            scconv_replace=getattr(cfg, "SC_CONV_REPLACE", ()),
            scconv_replace_keep_relu=getattr(cfg, "SC_CONV_REPLACE_KEEP_RELU", True),
            scconv_fused_block=getattr(cfg, "SC_CONV_FUSED_BLOCK", "scconv"),
        )
        if getattr(cfg, "SC_CONV_FUSED_BLOCK", "scconv") == "relu":
            import torch.nn as nn
            for module in model.fused_scconv.modules():
                if isinstance(module, nn.ReLU):
                    module.inplace = False
    model = model.to(device).train()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)
    optimizer.zero_grad(set_to_none=True)
    outputs = model(images)
    logits = outputs["out"] if isinstance(outputs, dict) else outputs
    if logits.shape != (images.shape[0], 2, 200, 200):
        raise RuntimeError(f"模型输出形状异常：{tuple(logits.shape)}")
    loss = F.cross_entropy(logits, masks)
    if model_name == "bisenet" and getattr(cfg, "USE_BRANCH_ORTHOGONALITY", False):
        from bisenetv2_orth import branch_orthogonality_loss
        warmup_epochs = max(1, cfg.BRANCH_ORTHOGONALITY_WARMUP_EPOCHS)
        loss = loss + (cfg.BRANCH_ORTHOGONALITY_WEIGHT / warmup_epochs) * (
            branch_orthogonality_loss(outputs["detail_feature"], outputs["semantic_feature"])
        )
    loss.backward()
    optimizer.step()
    print(
        f"SD900 one-step smoke OK | model={model_name} config={config_name} device={device} "
        f"batch={batch_size} train/val/test={len(train_loader.dataset)}/"
        f"{len(val_loader.dataset)}/{len(test_loader.dataset)} loss={loss.item():.4f}"
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="SD900 fixed-split training adapter")
    parser.add_argument("--smoke", action="store_true", help="只取一个 batch 做前向与损失检查，不训练、不保存")
    parser.add_argument("--model", choices=("unet", "bisenet", "scconv"), default="scconv")
    parser.add_argument("--config", default="configScConvAblation")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--device", default="cuda" if __import__("torch").cuda.is_available() else "cpu")
    parser.add_argument("--dataset", choices=("sd-saliency",), default="sd-saliency")
    parser.add_argument("--data_root", default=str(DATA_ROOT))
    args, remaining = parser.parse_known_args()
    if args.smoke:
        return smoke(args.model, args.config, args.batch_size, args.device)
    if Path(args.data_root).resolve() != DATA_ROOT.resolve():
        raise ValueError(f"SD900 固定清单仅对应数据目录：{DATA_ROOT}")
    install_fixed_split()
    if args.model == "scconv" and args.config == "configScConvC2c":
        sys.path.insert(0, str(PROJECT_DIR / "models"))
        model_module = importlib.import_module("bisenetv2_scconv")
        model_module.build_bisenetv2_scconv = make_c2c_autograd_safe(
            model_module.build_bisenetv2_scconv
        )
    train_entry = PROJECT_DIR / "train.py"
    sys.argv = [
        str(train_entry), "--model", args.model, "--config", args.config,
        "--dataset", args.dataset, "--data_root", args.data_root,
        "--batch_size", str(args.batch_size), "--device", args.device,
        *remaining,
    ]
    runpy.run_path(str(train_entry), run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
