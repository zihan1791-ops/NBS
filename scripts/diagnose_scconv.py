"""ScConv 训练结果的后置诊断：读取已训练检查点，回答"模块到底有没有参与"。

用法::

    python scripts/diagnose_scconv.py --run_dir results2中断/bisenetv2_scconv/NEU_300_0
    python scripts/diagnose_scconv.py --run_dir <dir> --with_data     # 需 GPU 空闲

输出：
1. 每个 ResidualScConv 的可学习残差尺度 scale（初始值 vs 训练后值）；
2. SRU 中 GroupNorm 的 γ 统计与归一化份额（γ/Σγ）；
3. ``--with_data`` 时额外在验证集上前向，统计 SRU 门控把多少元素判为 informative。

第 1 项最关键：若 scale 停在初始值附近，说明 ``x + scale * ScConv(x)`` 这条旁路
从未被真正学习，ScConv 对前向几乎无贡献。
"""
from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
for directory in (PROJECT_DIR / "config", PROJECT_DIR / "models", SCRIPT_DIR):
    sys.path.insert(0, str(directory))

import torch  # noqa: E402

import config as base_config  # noqa: E402
import configScConv as base_scconv  # noqa: E402

SRU_CLASS = "SpatialReconstructionUnit"


def settings_of(run_dir: Path) -> dict:
    config_path = run_dir / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"缺少配置快照：{config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)["settings"]


def pick(settings: dict, name: str, fallback):
    value = settings.get(name, fallback)
    return fallback if value is None else value


def build_model(settings: dict):
    scconv = importlib.import_module("bisenetv2_scconv")
    return scconv.build_bisenetv2_scconv(
        num_classes=pick(settings, "NUM_CLASSES", base_config.NUM_CLASSES),
        auxiliary=pick(settings, "USE_AUXILIARY_HEADS", base_config.USE_AUXILIARY_HEADS),
        main_head_channels=pick(settings, "MAIN_HEAD_CHANNELS", base_config.MAIN_HEAD_CHANNELS),
        auxiliary_head_channels=pick(
            settings, "AUXILIARY_HEAD_CHANNELS", base_config.AUXILIARY_HEAD_CHANNELS
        ),
        dropout=pick(settings, "HEAD_DROPOUT", base_config.HEAD_DROPOUT),
        in_channels=pick(settings, "INPUT_CHANNELS", base_config.INPUT_CHANNELS),
        detail_channels=pick(settings, "DETAIL_CHANNELS", base_config.DETAIL_CHANNELS),
        detail_stage_repeats=pick(
            settings, "DETAIL_STAGE_REPEATS", base_config.DETAIL_STAGE_REPEATS
        ),
        semantic_channels=pick(settings, "SEMANTIC_CHANNELS", base_config.SEMANTIC_CHANNELS),
        expansion=pick(settings, "GE_EXPANSION", base_config.GE_EXPANSION),
        stage_repeats=pick(settings, "STAGE_REPEATS", base_config.STAGE_REPEATS),
        scconv_enabled=pick(settings, "USE_SC_CONV", True),
        scconv_group_num=pick(settings, "SC_CONV_GROUP_NUM", base_scconv.SC_CONV_GROUP_NUM),
        scconv_gate_threshold=pick(
            settings, "SC_CONV_GATE_THRESHOLD", base_scconv.SC_CONV_GATE_THRESHOLD
        ),
        scconv_alpha=pick(settings, "SC_CONV_ALPHA", base_scconv.SC_CONV_ALPHA),
        scconv_squeeze_ratio=pick(
            settings, "SC_CONV_SQUEEZE_RATIO", base_scconv.SC_CONV_SQUEEZE_RATIO
        ),
        scconv_group_size=pick(settings, "SC_CONV_GROUP_SIZE", base_scconv.SC_CONV_GROUP_SIZE),
        scconv_group_kernel_size=pick(
            settings, "SC_CONV_GROUP_KERNEL_SIZE", base_scconv.SC_CONV_GROUP_KERNEL_SIZE
        ),
        scconv_gamma_normalise=pick(
            settings, "SC_CONV_GAMMA_NORMALISE",
            getattr(base_scconv, "SC_CONV_GAMMA_NORMALISE", True),
        ),
        scconv_placement=pick(settings, "SC_CONV_PLACEMENT", base_scconv.SC_CONV_PLACEMENT),
        scconv_residual=pick(settings, "SC_CONV_RESIDUAL", base_scconv.SC_CONV_RESIDUAL),
        scconv_residual_scale=pick(
            settings, "SC_CONV_RESIDUAL_SCALE", base_scconv.SC_CONV_RESIDUAL_SCALE
        ),
        scconv_replace=pick(settings, "SC_CONV_REPLACE", ()),
        scconv_replace_keep_relu=pick(settings, "SC_CONV_REPLACE_KEEP_RELU", True),
        # C2a/C2b/C2c 这些归因对照臂的 fused 位置放的是普通块，必须一起还原，
        # 否则会按默认值建出一个 ScConv 模型，权重对不上。
        scconv_fused_block=pick(settings, "SC_CONV_FUSED_BLOCK", "scconv"),
    )


def static_report(model, settings: dict) -> dict:
    """只用权重即可得到的信息：残差尺度与 SRU 的 γ 统计。"""
    report = {
        "scale_init": float(pick(settings, "SC_CONV_RESIDUAL_SCALE", 0.0)),
        "scales": {},
        "gamma": {},
    }
    for name, parameter in model.named_parameters():
        if name.endswith(".scale"):
            report["scales"][name] = float(parameter.detach())
    for name, module in model.named_modules():
        if module.__class__.__name__ != SRU_CLASS:
            continue
        gamma = module.group_norm.weight.detach().double()
        share = gamma / gamma.sum()
        report["gamma"][name] = {
            "gamma_normalise": bool(module.gamma_normalise),
            "mean": float(gamma.mean()),
            "std": float(gamma.std()),
            "min": float(gamma.min()),
            "max": float(gamma.max()),
            # 归一化份额的最大值：越接近 1/C 越均匀，越大说明 γ 越集中。
            "share_max": float(share.max()),
            "share_uniform": float(1.0 / share.numel()),
            "above_uniform": int((share > 1.0 / share.numel()).sum()),
            "channels": int(share.numel()),
        }
    return report


@torch.no_grad()
def gate_report(model, settings: dict, split: str, max_batches: int | None = None) -> dict:
    """在真实数据上统计 SRU 门控判为 informative 的元素比例。"""
    from dataset import NEUSegDataset
    from split_utils import read_neu_split

    data_root = Path(pick(settings, "DATA_ROOT", base_config.DATA_ROOT))
    dataset = NEUSegDataset(
        str(data_root / base_config.IMAGE_DIR_NAME / base_config.TRAIN_SPLIT_NAME),
        str(data_root / base_config.MASK_DIR_NAME / base_config.TRAIN_SPLIT_NAME),
        image_size=pick(settings, "IMAGE_SIZE", base_config.IMAGE_SIZE),
        augment=False,
        file_list=read_neu_split(data_root, split),
        num_classes=pick(settings, "NUM_CLASSES", base_config.NUM_CLASSES),
        salt_pepper_threshold=pick(
            settings, "SALT_PEPPER_THRESHOLD", base_config.SALT_PEPPER_THRESHOLD
        ),
    )
    loader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=False, num_workers=0)

    counters = {}
    hooks = []

    def register(name: str, module) -> None:
        counters[name] = {
            "hit": 0, "total": 0, "threshold": module.gate_threshold,
            "sum": 0.0, "sumsq": 0.0, "min": float("inf"), "max": float("-inf"),
            "near_threshold": 0,
        }

        def pre_hook(_module, inputs):
            x = inputs[0]
            normalised = _module.group_norm(x)
            weight = _module.group_norm.weight
            if _module.gamma_normalise:
                weight = weight / (weight.sum() + _module.epsilon)
            reweight = torch.sigmoid(normalised * weight.view(1, -1, 1, 1))
            entry = counters[name]
            values = reweight.detach().double()
            entry["hit"] += int((reweight > _module.gate_threshold).sum())
            entry["total"] += reweight.numel()
            entry["sum"] += float(values.sum())
            entry["sumsq"] += float((values * values).sum())
            entry["min"] = min(entry["min"], float(values.min()))
            entry["max"] = max(entry["max"], float(values.max()))
            # 落在阈值 ±0.01 内的元素比例：比例越高说明门控越"没有判断力"。
            entry["near_threshold"] += int(((values - _module.gate_threshold).abs() < 0.01).sum())
            return None

        hooks.append(module.register_forward_pre_hook(pre_hook))

    for name, module in model.named_modules():
        if module.__class__.__name__ == SRU_CLASS:
            register(name, module)
    if not counters:
        return {}

    device = next(model.parameters()).device
    for index, batch in enumerate(loader):
        model(batch["image"].to(device))
        if max_batches is not None and index + 1 >= max_batches:
            break
    for hook in hooks:
        hook.remove()

    result = {}
    for name, value in counters.items():
        total = max(value["total"], 1)
        mean = value["sum"] / total
        variance = max(value["sumsq"] / total - mean * mean, 0.0)
        result[name] = {
            "informative_ratio": value["hit"] / total,
            "threshold": value["threshold"],
            "reweight_mean": mean,
            "reweight_std": variance ** 0.5,
            "reweight_min": value["min"],
            "reweight_max": value["max"],
            "near_threshold_ratio": value["near_threshold"] / total,
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="诊断已训练的 ScConv 检查点")
    parser.add_argument("--run_dir", type=Path, action="append", required=True,
                        help="实验目录（含 best.pth 与 config.json），可重复指定")
    parser.add_argument("--checkpoint", default="best.pth")
    parser.add_argument("--with_data", action="store_true",
                        help="额外在验证集上前向统计门控行为（需要 GPU 空闲）")
    parser.add_argument("--split", default="val", choices=("train", "val", "test"))
    parser.add_argument("--max_batches", type=int, default=None,
                        help="--with_data 时最多前向多少个 batch（便于快速抽查）")
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"),
                        help="auto：--with_data 时优先 CUDA；cpu 可在 GPU 被占用时抽查")
    args = parser.parse_args()

    if args.device == "cuda":
        device = torch.device("cuda")
    elif args.device == "cpu":
        device = torch.device("cpu")
    else:
        device = torch.device(
            "cuda" if (args.with_data and torch.cuda.is_available()) else "cpu"
        )

    for run_dir in args.run_dir:
        run_dir = run_dir.resolve()
        settings = settings_of(run_dir)
        model = build_model(settings)
        checkpoint = torch.load(run_dir / args.checkpoint, map_location="cpu", weights_only=True)
        model.load_state_dict(checkpoint["model_state"])
        model.to(device)

        report = static_report(model, settings)
        best_epoch = checkpoint.get("epoch")
        best_miou = checkpoint.get("val_metrics", {}).get("mIoU")
        print(f"\n=== {run_dir} ===")
        print(f"placement={pick(settings, 'SC_CONV_PLACEMENT', None)} "
              f"residual={pick(settings, 'SC_CONV_RESIDUAL', None)} "
              f"scale_init={report['scale_init']} best_epoch={best_epoch} "
              f"best_val_mIoU={best_miou}")

        use_scconv = pick(settings, "USE_SC_CONV", True)
        fused_block = pick(settings, "SC_CONV_FUSED_BLOCK", "scconv")
        if fused_block != "scconv":
            print(f"  fused 位置是普通块（{fused_block}），本 run 不含任何 ScConv，"
                  f"没有 γ 与门控可诊断")
            continue
        if not report["scales"]:
            if not use_scconv:
                print("  ScConv 未启用（基线臂，无 ScConv 模块）")
            else:
                print("  无残差尺度参数（直插或替换式应用，ScConv 直接承担输出）")
        for name, value in report["scales"].items():
            delta = value - report["scale_init"]
            verdict = "几乎未学习（旁路近似关闭）" if abs(delta) < 0.05 else "已被学习"
            print(f"  scale {name:26s} {report['scale_init']:.4f} -> {value:.6f} "
                  f"(Δ={delta:+.6f})  {verdict}")
        for name, value in report["gamma"].items():
            print(f"  SRU γ {name:24s} gamma_normalise={value['gamma_normalise']!s:5s} "
                  f"mean={value['mean']:.4f} std={value['std']:.4f} "
                  f"min={value['min']:.4f} max={value['max']:.4f} "
                  f"份额max={value['share_max']:.4e} (均匀={value['share_uniform']:.4e}, "
                  f"高于均匀的通道 {value['above_uniform']}/{value['channels']})")

        if args.with_data:
            stats = gate_report(model, settings, args.split, args.max_batches)
            for name, value in stats.items():
                print(f"  门控 {name:24s} informative={value['informative_ratio']:.4f} "
                      f"(阈值 {value['threshold']}, split={args.split})")
                print(f"       reweight mean={value['reweight_mean']:.6f} "
                      f"std={value['reweight_std']:.6f} "
                      f"range=[{value['reweight_min']:.6f}, {value['reweight_max']:.6f}] "
                      f"阈值±0.01 内占比={value['near_threshold_ratio']:.4f}")


if __name__ == "__main__":
    main()
