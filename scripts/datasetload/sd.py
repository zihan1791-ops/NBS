import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import random
from typing import Optional, Tuple, Dict, Any


class SDSaliencyDataset(Dataset):
    """SD-Saliency-900 数据集

    数据集结构：
      - Source Images/In_{N}.png  (300张 Inclusion)
      - Source Images/Pa_{N}.png  (300张 Patch)
      - Source Images/Sc_{N}.png  (300张 Scratch)
      - Ground truth/In_{N}.png   (对应掩码)
      - Ground truth/Pa_{N}.png
      - Ground truth/Sc_{N}.png

    划分: train:val:test = 0.7:0.15:0.15
    掩码标签：0=背景, 1=显著区域（显著目标检测，二分类）
    """

    def __init__(
        self,
        data_root: str,
        split: str = "train",
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        augment: bool = False,
        apply_clahe: bool = True,
        num_classes: int = 4,
        image_size: Tuple[int, int] = (200, 200),
        random_seed: int = 42
    ):
        self.data_root = data_root
        self.split = split
        self.augment = augment
        self.apply_clahe = apply_clahe
        self.num_classes = num_classes
        self.image_size = image_size

        random.seed(random_seed)
        np.random.seed(random_seed)

        self.defect_types = {
            'In': 0,    # Inclusion → mask value 1
            'Pa': 1,    # Patch → mask value 2
            'Sc': 2,    # Scratch → mask value 3
        }

        self.src_dir = os.path.join(data_root, "Source Images")
        self.gt_dir = os.path.join(data_root, "Ground truth")

        self.samples = self._collect_samples()
        self.samples = self._split_dataset(train_ratio, val_ratio)

        print(f"SD-Saliency Dataset - {split}: {len(self.samples)} samples")

    def _collect_samples(self) -> list:
        samples = []
        for prefix, class_id in self.defect_types.items():
            for i in range(1, 301):
                img_name = f"{prefix}_{i}.png"
                img_path = os.path.join(self.src_dir, img_name)
                mask_path = os.path.join(self.gt_dir, img_name)
                if os.path.exists(img_path) and os.path.exists(mask_path):
                    samples.append({
                        'image_path': img_path,
                        'mask_path': mask_path,
                        'class_id': class_id,
                        'prefix': prefix,
                    })
        return samples

    def _split_dataset(self, train_ratio: float, val_ratio: float) -> list:
        samples_by_type = {}
        for s in self.samples:
            samples_by_type.setdefault(s['prefix'], []).append(s)

        train_list, val_list, test_list = [], [], []
        for prefix, items in samples_by_type.items():
            random.shuffle(items)
            n = len(items)
            n_train = int(n * train_ratio)
            n_val = int(n * val_ratio)
            train_list.extend(items[:n_train])
            val_list.extend(items[n_train:n_train + n_val])
            test_list.extend(items[n_train + n_val:])

        if self.split == "train":
            return train_list
        elif self.split == "val":
            return val_list
        else:
            return test_list

    def _apply_clahe(self, image: np.ndarray) -> np.ndarray:
        if not self.apply_clahe:
            return image
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(image)

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

        image = cv2.imread(sample['image_path'], cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(sample['mask_path'], cv2.IMREAD_GRAYSCALE)

        image = cv2.resize(image, self.image_size)
        mask = cv2.resize(mask, self.image_size, interpolation=cv2.INTER_NEAREST)

        mask = np.where(mask > 0, 1, 0).astype(np.uint8)

        image = self._apply_clahe(image)
        image, mask = self._augment(image, mask)

        # 灰度图扩展为 3 通道 (RGB)
        image = np.stack([image, image, image], axis=-1)
        image_t = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        mask_t = torch.from_numpy(mask).long()

        return {
            'image': image_t,
            'mask': mask_t,
            'id': f"{sample['prefix']}_{idx}",
            'class_id': sample['class_id'],
            'prefix': sample['prefix'],
        }


class DualSDSaliencyDataset(SDSaliencyDataset):
    """双输入版本：返回 image_ori 和 image_enh。"""

    def __getitem__(self, idx: int):
        sample = self.samples[idx]

        image = cv2.imread(sample['image_path'], cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(sample['mask_path'], cv2.IMREAD_GRAYSCALE)

        image = cv2.resize(image, self.image_size)
        mask = cv2.resize(mask, self.image_size, interpolation=cv2.INTER_NEAREST)
        mask = np.where(mask > 0, 1, 0).astype(np.uint8)

        image_enh = self._apply_clahe(image.copy())

        if self.augment:
            state = random.getstate()
            image, mask = self._augment(image, mask)
            random.setstate(state)
            image_enh, _ = self._augment(image_enh, mask.copy())

        image = np.stack([image, image, image], axis=-1)
        image_enh = np.stack([image_enh, image_enh, image_enh], axis=-1)

        image_ori_t = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        image_enh_t = torch.from_numpy(image_enh).permute(2, 0, 1).float() / 255.0
        mask_t = torch.from_numpy(mask).long()

        return {
            'image_ori': image_ori_t,
            'image_enh': image_enh_t,
            'mask': mask_t,
            'id': f"{sample['prefix']}_{idx}",
            'class_id': sample['class_id'],
            'prefix': sample['prefix'],
        }
