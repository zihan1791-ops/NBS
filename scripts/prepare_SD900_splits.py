"""Create the immutable, per-defect 70/15/15 split used by all SD900 experiments."""
from __future__ import annotations

import argparse
import random
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = PROJECT_DIR / "data" / "SD-saliency"
DEFAULT_OUTPUT = DEFAULT_DATA_ROOT / "splits"
PREFIXES = ("In", "Pa", "Sc")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true", help="replace existing split manifests")
    args = parser.parse_args()
    if args.output_dir.exists() and not args.force:
        raise SystemExit(f"已存在 {args.output_dir}；如需重建请显式添加 --force")

    image_dir = args.data_root / "Source Images"
    mask_dir = args.data_root / "Ground truth"
    if not image_dir.is_dir() or not mask_dir.is_dir():
        raise SystemExit(f"缺少 SD900 图像或掩码目录：{args.data_root}")

    splits: dict[str, list[str]] = {"train": [], "val": [], "test": []}
    for prefix in PREFIXES:
        names = [f"{prefix}_{index}" for index in range(1, 301)]
        for name in names:
            if not (image_dir / f"{name}.png").is_file() or not (mask_dir / f"{name}.png").is_file():
                raise SystemExit(f"图像/掩码配对不完整：{name}.png")
        random.Random(args.seed + PREFIXES.index(prefix)).shuffle(names)
        splits["train"].extend(names[:210])
        splits["val"].extend(names[210:255])
        splits["test"].extend(names[255:])

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for split, names in splits.items():
        (args.output_dir / f"{split}.txt").write_text("\n".join(names) + "\n", encoding="utf-8")
    print(f"SD900 split ready: train={len(splits['train'])}, val={len(splits['val'])}, "
          f"test={len(splits['test'])}; seed={args.seed}; per-prefix=210/45/45")
    print(f"Wrote manifests to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
