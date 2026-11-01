"""读取 NEU-Seg 数据，并沿用 SteelSegmentation 的标签编码方式。"""

import glob
import os
import random
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from split_utils import read_neu_split

from config import (
    AUGMENT_TRAINING,
    HORIZONTAL_FLIP_PROBABILITY,
    IMAGE_SIZE,
    IMAGE_VALUE_SCALE,
    MASK_CHANNEL_TO_CLASS,
    MEDIAN_BLUR_KERNEL,
    NUM_CLASSES,
    ROTATION_ANGLES,
    ROTATION_PROBABILITY,
    SALT_PEPPER_THRESHOLD,
    VERTICAL_FLIP_PROBABILITY,
    GRAYSCALE_EXTREME_HIGH,
    GRAYSCALE_EXTREME_LOW,
)


IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def collect_flat_pairs(images_dir: str, masks_dir: str) -> List[Tuple[str, str, str, int]]:
    """收集平铺目录中同名图像与掩码，默认把前景映射为类别 1。"""
    if not os.path.isdir(images_dir) or not os.path.isdir(masks_dir):
        raise FileNotFoundError(f"图像或掩码目录不存在：{images_dir} / {masks_dir}")
    samples = []
    for image_path in sorted(
        path for path in glob.glob(os.path.join(images_dir, "*"))
        if os.path.splitext(path)[1].lower() in IMAGE_EXTENSIONS
    ):
        stem = os.path.splitext(os.path.basename(image_path))[0]
        mask_path = next(
            (os.path.join(masks_dir, stem + ext) for ext in IMAGE_EXTENSIONS
             if os.path.isfile(os.path.join(masks_dir, stem + ext))),
            None,
        )
        if mask_path is None:
            raise FileNotFoundError(f"图像 {image_path} 没有同名掩码")
        samples.append((image_path, mask_path, stem, 1))
    return samples


def collect_magnetic_samples(data_root: str) -> List[Tuple[str, str, str, int]]:
    """收集 Magnetic-tile 六类目录；MT_Free 作为纯背景样本。"""
    category_ids = {
        "MT_Free": 0,
        "MT_Blowhole": 1,
        "MT_Break": 2,
        "MT_Crack": 3,
        "MT_Fray": 4,
        "MT_Uneven": 5,
    }
    samples = []
    for category, foreground_class in category_ids.items():
        image_dir = os.path.join(data_root, category, "Imgs")
        if not os.path.isdir(image_dir):
            raise FileNotFoundError(f"缺少数据目录：{image_dir}")
        for image_path in sorted(glob.glob(os.path.join(image_dir, "*.jpg"))):
            stem = os.path.splitext(os.path.basename(image_path))[0]
            mask_path = os.path.join(image_dir, stem + ".png")
            if not os.path.isfile(mask_path):
                raise FileNotFoundError(f"图像 {image_path} 没有配套 PNG 掩码")
            samples.append((image_path, mask_path, f"{category}_{stem}", foreground_class))
    return samples


def _stratified_external_split(samples, seed: int, test_ratio: float = 0.15):
    """按类别确定性拆分非 NEU 数据集，避免小类别只落入一个子集。"""
    grouped = {}
    for sample in samples:
        grouped.setdefault(sample[3], []).append(sample)
    rng = random.Random(seed)
    train, val, test = [], [], []
    for group in grouped.values():
        group = list(group)
        rng.shuffle(group)
        if len(group) < 3:
            raise RuntimeError("每个类别至少需要 3 个样本，才能拆分训练、验证和测试集")
        test_count = max(1, int(round(len(group) * test_ratio)))
        val_count = max(1, int(round(len(group) * test_ratio)))
        if test_count + val_count >= len(group):
            test_count = val_count = 1
        test.extend(group[:test_count])
        val.extend(group[test_count:test_count + val_count])
        train.extend(group[test_count + val_count:])
    return train, val, test


def build_loaders(args, device, cfg, generator=None):
    """依据 --dataset 选择数据格式、划分方式并创建三个数据加载器。

    ``generator`` 用于固定训练集的 shuffle 顺序与 worker 随机种子；为 None 时沿用
    PyTorch 默认行为（其它训练脚本不传该参数，行为保持不变）。
    """
    dataset_name = args.dataset
    if dataset_name == "neu":
        class_names = tuple(cfg.CLASS_NAMES)
        plot_class_names = tuple(cfg.PLOT_CLASS_NAMES[:len(class_names)])
        color_map = dict(cfg.COLOR_MAP_BGR)
        num_classes = len(class_names)
        if args.num_classes is not None and args.num_classes != num_classes:
            raise ValueError(f"NEU-Seg 的类别数固定为 {num_classes}")
        train_images = args.data_root / cfg.IMAGE_DIR_NAME / cfg.TRAIN_SPLIT_NAME
        train_masks = args.data_root / cfg.MASK_DIR_NAME / cfg.TRAIN_SPLIT_NAME
        test_images = args.data_root / cfg.IMAGE_DIR_NAME / cfg.TEST_SPLIT_NAME
        test_masks = args.data_root / cfg.MASK_DIR_NAME / cfg.TEST_SPLIT_NAME
        for required_dir in (train_images, train_masks, test_images, test_masks):
            if not required_dir.is_dir():
                raise FileNotFoundError(f"缺少 NEU 数据目录：{required_dir}")
        train_ids = read_neu_split(args.data_root, "train")
        val_ids = read_neu_split(args.data_root, "val")
        test_ids = read_neu_split(args.data_root, "test")
        if cfg.USE_OFFICIAL_TEST_AS_VALIDATION:
            raise ValueError("NBS 固定实验禁止把官方 test 用作逐轮验证；请保持 USE_OFFICIAL_TEST_AS_VALIDATION=False")
        common = {
            "image_size": args.image_size,
            "num_classes": num_classes,
            "salt_pepper_threshold": cfg.SALT_PEPPER_THRESHOLD,
        }
        train_set = NEUSegDataset(
            str(train_images), str(train_masks), file_list=train_ids,
            augment=cfg.AUGMENT_TRAINING and not args.no_augment, **common,
        )
        val_set = NEUSegDataset(
            str(train_images), str(train_masks), file_list=val_ids, augment=False, **common,
        )
        test_set = NEUSegDataset(
            str(test_images), str(test_masks), file_list=test_ids, augment=False, **common,
        )
        validation_source = "Fixed NEU holdout: train=2904, val=726; official test=840 reserved until training ends"
        args.validation_uses_official_test = False
    else:
        if args.num_classes is not None and args.num_classes != (2 if dataset_name == "sd-saliency" else 6):
            raise ValueError(f"{dataset_name} 固定类别数为 {2 if dataset_name == 'sd-saliency' else 6}")
        if dataset_name == "sd-saliency":
            samples = collect_flat_pairs(
                str(args.data_root / "Source Images"), str(args.data_root / "Ground truth")
            )
            class_names = ("背景", "缺陷")
            plot_class_names = ("Background", "Defect")
            foreground_colors = ((0, 220, 255),)
        elif dataset_name == "magnetic-tile":
            samples = collect_magnetic_samples(str(args.data_root))
            class_names = ("背景", "气孔", "断裂", "裂纹", "毛刺", "不平")
            plot_class_names = ("Background", "Blowhole", "Break", "Crack", "Fray", "Uneven")
            foreground_colors = (
                (255, 90, 40), (0, 200, 0), (40, 40, 255), (220, 0, 220), (0, 180, 255),
            )
        else:
            raise ValueError(f"不支持的数据集：{dataset_name}")
        num_classes = len(class_names)
        color_map = {0: (0, 0, 0)}
        color_map.update({i + 1: color for i, color in enumerate(foreground_colors)})
        train_samples, val_samples, test_samples = _stratified_external_split(
            samples, args.seed, cfg.EXTERNAL_TEST_RATIO
        )
        train_set = PairedSegmentationDataset(
            train_samples, args.image_size, num_classes,
            augment=cfg.AUGMENT_TRAINING and not args.no_augment,
        )
        val_set = PairedSegmentationDataset(val_samples, args.image_size, num_classes)
        test_set = PairedSegmentationDataset(test_samples, args.image_size, num_classes)
        validation_source = "Stratified validation split (fixed random seed)"
        args.validation_uses_official_test = False

    args.num_classes = num_classes
    args.class_names = class_names
    args.plot_class_names = plot_class_names
    args.color_map = color_map
    args.validation_source = validation_source
    pin_memory = cfg.PIN_MEMORY and device.type == "cuda"
    options = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": pin_memory,
        "persistent_workers": cfg.PERSISTENT_WORKERS and args.num_workers > 0,
    }
    train_loader = DataLoader(train_set, shuffle=True, generator=generator, **options)
    val_loader = DataLoader(val_set, shuffle=False, **options)
    test_loader = val_loader if dataset_name == "neu" and cfg.USE_OFFICIAL_TEST_AS_VALIDATION else DataLoader(
        test_set, shuffle=False, **options
    )
    return train_loader, val_loader, test_loader


def list_image_ids(images_dir: str) -> List[str]:
    paths = []
    for extension in IMAGE_EXTENSIONS:
        paths.extend(glob.glob(os.path.join(images_dir, f"*{extension}")))
        paths.extend(glob.glob(os.path.join(images_dir, f"*{extension.upper()}")))
    return sorted({os.path.splitext(os.path.basename(path))[0] for path in paths})


class NEUSegDataset(Dataset):
    """返回 ``image``（3×高×宽浮点张量）和 ``mask``（高×宽长整型张量）。"""

    def __init__(
        self,
        images_dir: str,
        masks_dir: str,
        image_size: int = IMAGE_SIZE,
        augment: bool = AUGMENT_TRAINING,
        file_list: Optional[List[str]] = None,
        num_classes: int = NUM_CLASSES,
        salt_pepper_threshold: float = SALT_PEPPER_THRESHOLD,
    ) -> None:
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.image_size = image_size
        self.augment = augment
        self.num_classes = num_classes
        self.salt_pepper_threshold = salt_pepper_threshold
        self.ids = list(file_list) if file_list is not None else list_image_ids(images_dir)
        if not self.ids:
            raise RuntimeError(f"No images found in {images_dir}")

    def __len__(self) -> int:
        return len(self.ids)

    @staticmethod
    def _find_file(directory: str, sample_id: str) -> str:
        for extension in IMAGE_EXTENSIONS:
            for ext in (extension, extension.upper()):
                candidate = os.path.join(directory, sample_id + ext)
                if os.path.isfile(candidate):
                    return candidate
        raise FileNotFoundError(f"Could not find {sample_id} in {directory}")

    @staticmethod
    def _read_mask(path: str) -> np.ndarray:
        raw = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if raw is None:
            raise RuntimeError(f"Could not read mask: {path}")

        if raw.ndim == 2:
            return raw.astype(np.uint8)

        if raw.shape[2] == 4:
            raw = cv2.cvtColor(raw, cv2.COLOR_BGRA2BGR)
        elif raw.shape[2] > 3:
            raw = raw[:, :, :3]

        is_background = np.all(raw[:, :, :3] == 0, axis=2)
        channel_index = np.argmax(raw[:, :, :3], axis=2)
        # SteelSegmentation 按 BGR 读取彩色掩码，并使用相同的类别映射。
        mask = np.choose(channel_index, MASK_CHANNEL_TO_CLASS).astype(np.uint8)
        mask[is_background] = 0
        return mask

    def __getitem__(self, index: int):
        sample_id = self.ids[index]
        image_path = self._find_file(self.images_dir, sample_id)
        mask_path = self._find_file(self.masks_dir, sample_id)

        image_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise RuntimeError(f"Could not read image: {image_path}")
        image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        mask = self._read_mask(mask_path)

        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        extreme_ratio = (
            np.count_nonzero(gray == GRAYSCALE_EXTREME_LOW)
            + np.count_nonzero(gray == GRAYSCALE_EXTREME_HIGH)
        ) / gray.size
        if extreme_ratio >= self.salt_pepper_threshold:
            image = cv2.medianBlur(image, MEDIAN_BLUR_KERNEL)

        if self.augment:
            if np.random.rand() < HORIZONTAL_FLIP_PROBABILITY:
                image = np.flip(image, axis=1).copy()
                mask = np.flip(mask, axis=1).copy()
            if np.random.rand() < VERTICAL_FLIP_PROBABILITY:
                image = np.flip(image, axis=0).copy()
                mask = np.flip(mask, axis=0).copy()
            if np.random.rand() < ROTATION_PROBABILITY and ROTATION_ANGLES:
                angle = int(np.random.choice(ROTATION_ANGLES))
                rotations = angle // 90
                image = np.rot90(image, rotations).copy()
                mask = np.rot90(mask, rotations).copy()

        size = (self.image_size, self.image_size)
        image = cv2.resize(image, size, interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, size, interpolation=cv2.INTER_NEAREST)
        image = image.astype(np.float32) / IMAGE_VALUE_SCALE
        mask = mask.astype(np.int64)
        mask[mask < 0] = 0
        mask[mask >= self.num_classes] = 0

        return {
            "id": sample_id,
            "image": torch.from_numpy(image).permute(2, 0, 1).contiguous(),
            "mask": torch.from_numpy(mask),
        }


class PairedSegmentationDataset(Dataset):
    """加载显式图像-掩码路径列表，支持二值掩码映射为语义类别。"""

    def __init__(
        self,
        samples: Sequence[Tuple[str, str, str, int]],
        image_size: int,
        num_classes: int,
        augment: bool = False,
        binary_threshold: int = 127,
    ) -> None:
        self.samples = list(samples)
        self.image_size = image_size
        self.num_classes = num_classes
        self.augment = augment
        self.binary_threshold = binary_threshold
        if not self.samples:
            raise RuntimeError("数据集中没有找到有效的图像-掩码配对")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        image_path, mask_path, sample_id, foreground_class = self.samples[index]
        image_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
        raw_mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
        if image_bgr is None:
            raise RuntimeError(f"无法读取图像：{image_path}")
        if raw_mask is None:
            raise RuntimeError(f"无法读取掩码：{mask_path}")

        image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        if raw_mask.ndim == 3:
            if raw_mask.shape[2] == 4:
                raw_mask = cv2.cvtColor(raw_mask, cv2.COLOR_BGRA2GRAY)
            else:
                raw_mask = cv2.cvtColor(raw_mask[:, :, :3], cv2.COLOR_BGR2GRAY)
        binary = raw_mask > self.binary_threshold
        mask = np.zeros(binary.shape, dtype=np.uint8)
        if foreground_class > 0:
            mask[binary] = foreground_class

        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        extreme_ratio = (
            np.count_nonzero(gray == GRAYSCALE_EXTREME_LOW)
            + np.count_nonzero(gray == GRAYSCALE_EXTREME_HIGH)
        ) / gray.size
        if extreme_ratio >= SALT_PEPPER_THRESHOLD:
            image = cv2.medianBlur(image, MEDIAN_BLUR_KERNEL)

        if self.augment:
            if np.random.rand() < HORIZONTAL_FLIP_PROBABILITY:
                image = np.flip(image, axis=1).copy()
                mask = np.flip(mask, axis=1).copy()
            if np.random.rand() < VERTICAL_FLIP_PROBABILITY:
                image = np.flip(image, axis=0).copy()
                mask = np.flip(mask, axis=0).copy()
            if np.random.rand() < ROTATION_PROBABILITY and ROTATION_ANGLES:
                angle = int(np.random.choice(ROTATION_ANGLES))
                rotations = angle // 90
                image = np.rot90(image, rotations).copy()
                mask = np.rot90(mask, rotations).copy()

        size = (self.image_size, self.image_size)
        image = cv2.resize(image, size, interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, size, interpolation=cv2.INTER_NEAREST)
        image = image.astype(np.float32) / IMAGE_VALUE_SCALE
        mask = mask.astype(np.int64)
        mask[mask >= self.num_classes] = 0
        return {
            "id": sample_id,
            "image": torch.from_numpy(image).permute(2, 0, 1).contiguous(),
            "mask": torch.from_numpy(mask),
        }
