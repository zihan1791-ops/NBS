"""NBS 全模型共享的 NEU 固定划分清单读取与校验。"""
from pathlib import Path

EXPECTED_COUNTS = {"train": 2904, "val": 726, "test": 840}
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def read_neu_split(data_root, split):
    """读取固定清单并验证数量以及图像/掩码配对。"""
    if split not in EXPECTED_COUNTS:
        raise ValueError(f"未知 NEU 划分：{split}")
    root = Path(data_root)
    manifest = root / "splits" / f"{split}.txt"
    if not manifest.is_file():
        raise FileNotFoundError(f"缺少固定划分文件：{manifest}")
    ids = [line.strip() for line in manifest.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    if len(ids) != EXPECTED_COUNTS[split]:
        raise ValueError(f"{manifest} 应有 {EXPECTED_COUNTS[split]} 项，实际 {len(ids)} 项")
    if len(set(ids)) != len(ids):
        raise ValueError(f"{manifest} 中存在重复样本编号")
    source_split = "test" if split == "test" else "train"
    image_dir = root / "images" / source_split
    mask_dir = root / "masks" / source_split
    image_ids = {p.stem for p in image_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS}
    mask_ids = {p.stem for p in mask_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS}
    missing_images = set(ids) - image_ids
    missing_masks = set(ids) - mask_ids
    if missing_images or missing_masks:
        raise FileNotFoundError(
            f"{split} 划分清单无法与图像/掩码配对；缺失图像={len(missing_images)}，缺失掩码={len(missing_masks)}"
        )
    if split in ("train", "val"):
        other = "val" if split == "train" else "train"
        other_path = root / "splits" / f"{other}.txt"
        if other_path.is_file():
            other_ids = {line.strip() for line in other_path.read_text(encoding="utf-8-sig").splitlines() if line.strip()}
            overlap = set(ids) & other_ids
            if overlap:
                raise ValueError(f"train 与 val 划分有 {len(overlap)} 个重叠样本")
    return ids


def validate_neu_splits(data_root):
    """验证全部三份清单及其合计数量。"""
    splits = {name: read_neu_split(data_root, name) for name in EXPECTED_COUNTS}
    total = sum(len(values) for values in splits.values())
    if total != 4470:
        raise ValueError(f"NEU 划分总量应为 4470，实际 {total}")
    return {name: len(values) for name, values in splits.items()}
