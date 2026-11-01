import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import random
from typing import Optional, Tuple, Dict, Any


class SeverstalDataset(Dataset):
    """Severstal Steel 缺陷数据集

    数据集结构：
    - train/images/ + train/labels/ (YOLO 格式)
    - valid/images/ + valid/labels/
    - test/images/

    标签格式：YOLO (class_id x_center y_center width height)
    类别数：4 (0=无缺陷, 1-3=缺陷类型)
    """

    def __init__(
        self,
        data_root: str,
        split: str = "train",
        augment: bool = False,
        apply_clahe: bool = True,
        sp_threshold: float = 0.02,
        num_classes: int = 4,
        image_size: Tuple[int, int] = (1600, 256),
        random_seed: int = 42
    ):
        self.data_root = data_root
        self.split = split
        self.augment = augment
        self.apply_clahe = apply_clahe
        self.sp_threshold = sp_threshold
        self.num_classes = num_classes
        self.image_size = image_size

        random.seed(random_seed)
        np.random.seed(random_seed)

        self.samples = self._collect_samples()
        print(f"Severstal Dataset - {split}: {len(self.samples)} samples")

    def _collect_samples(self) -> list:
        samples = []

        if self.split == "test":
            images_dir = os.path.join(self.data_root, "test", "images")
            if os.path.exists(images_dir):
                for img_file in os.listdir(images_dir):
                    if not img_file.endswith('.jpg'):
                        continue
                    samples.append({
                        'image_path': os.path.join(images_dir, img_file),
                        'label_path': None,
                        'has_label': False,
                    })
        else:
            images_dir = os.path.join(self.data_root, self.split, "images")
            labels_dir = os.path.join(self.data_root, self.split, "labels")
            if os.path.exists(images_dir) and os.path.exists(labels_dir):
                for img_file in os.listdir(images_dir):
                    if not img_file.endswith('.jpg'):
                        continue
                    label_file = img_file.replace('.jpg', '.txt')
                    label_path = os.path.join(labels_dir, label_file)
                    has_label = os.path.exists(label_path)
                    samples.append({
                        'image_path': os.path.join(images_dir, img_file),
                        'label_path': label_path if has_label else None,
                        'has_label': has_label,
                    })
        return samples

    def _parse_yolo_label(self, label_path: str, img_shape: Tuple[int, int]) -> np.ndarray:
        if not os.path.exists(label_path):
            return np.zeros(img_shape, dtype=np.uint8)
        mask = np.zeros(img_shape, dtype=np.uint8)
        h, w = img_shape
        with open(label_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) < 5:
                    continue
                class_id = int(parts[0])
                x_c, y_c, bw, bh = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                x1 = int((x_c - bw / 2) * w)
                y1 = int((y_c - bh / 2) * h)
                x2 = int((x_c + bw / 2) * w)
                y2 = int((y_c + bh / 2) * h)
                x1, y1 = max(0, min(x1, w - 1)), max(0, min(y1, h - 1))
                x2, y2 = max(0, min(x2, w - 1)), max(0, min(y2, h - 1))
                if class_id < self.num_classes:
                    mask[y1:y2, x1:x2] = class_id
        return mask

    def _apply_clahe(self, image: np.ndarray) -> np.ndarray:
        if not self.apply_clahe:
            return image
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        lab = cv2.merge([l, a, b])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    def _augment(self, image: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if not self.augment:
            return image, mask
        if random.random() > 0.5:
            image = cv2.flip(image, 1)
            mask = cv2.flip(mask, 1)
        if random.random() > 0.5:
            image = cv2.flip(image, 0)
            mask = cv2.flip(mask, 0)
        if random.random() > 0.5:
            angle = random.uniform(-15, 15)
            h, w = image.shape[:2]
            M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
            image = cv2.warpAffine(image, M, (w, h))
            mask = cv2.warpAffine(mask, M, (w, h))
        return image, mask

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]

        image = cv2.imread(sample['image_path'])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if sample['has_label'] and sample['label_path']:
            mask = self._parse_yolo_label(sample['label_path'], image.shape[:2])
        else:
            mask = np.zeros(image.shape[:2], dtype=np.uint8)

        image = cv2.resize(image, self.image_size)
        mask = cv2.resize(mask, self.image_size, interpolation=cv2.INTER_NEAREST)

        image = self._apply_clahe(image)
        image, mask = self._augment(image, mask)

        image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        mask = torch.from_numpy(mask).long()

        return {
            'image': image,
            'mask': mask,
            'id': f"severstal_{self.split}_{idx}",
            'has_label': sample['has_label'],
        }
