import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import random
from typing import Optional, Tuple, Dict, Any


class MagneticTileDataset(Dataset):
    """磁瓦缺陷数据集

    数据集结构：
    - MT_Blowhole/Imgs/ (气孔, class_id=0)
    - MT_Break/Imgs/ (破损, class_id=1)
    - MT_Crack/Imgs/ (裂纹, class_id=2)
    - MT_Fray/Imgs/ (磨损, class_id=3)
    - MT_Uneven/Imgs/ (不均匀, class_id=4)
    - MT_Free/Imgs/ (无缺陷, 不使用)

    掩码标签：class_id + 1，0 = 背景，共 6 类 (0-5)
    """

    def __init__(
        self,
        data_root: str,
        split: str = "train",
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        augment: bool = False,
        apply_clahe: bool = True,
        sp_threshold: float = 0.02,
        num_classes: int = 6,
        image_size: Tuple[int, int] = (256, 256),
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

        self.defect_types = {
            'MT_Blowhole': 0,
            'MT_Break': 1,
            'MT_Crack': 2,
            'MT_Fray': 3,
            'MT_Uneven': 4,
        }

        self.samples = self._collect_samples()
        self.samples = self._split_dataset(train_ratio, val_ratio)

        print(f"Magnetic Tile Dataset - {split}: {len(self.samples)} samples")

    def _collect_samples(self) -> list:
        samples = []
        for defect_type, class_id in self.defect_types.items():
            defect_dir = os.path.join(self.data_root, defect_type, "Imgs")
            if not os.path.exists(defect_dir):
                continue
            image_files = [f for f in os.listdir(defect_dir) if f.endswith('.jpg')]
            for img_file in image_files:
                img_path = os.path.join(defect_dir, img_file)
                mask_file = img_file.replace('.jpg', '.png')
                mask_path = os.path.join(defect_dir, mask_file)
                if os.path.exists(mask_path):
                    samples.append({
                        'image_path': img_path,
                        'mask_path': mask_path,
                        'class_id': class_id,
                        'defect_type': defect_type
                    })
        return samples

    def _split_dataset(self, train_ratio: float, val_ratio: float) -> list:
        samples_by_type = {}
        for sample in self.samples:
            defect_type = sample['defect_type']
            if defect_type not in samples_by_type:
                samples_by_type[defect_type] = []
            samples_by_type[defect_type].append(sample)

        train_samples, val_samples, test_samples = [], [], []
        for defect_type, type_samples in samples_by_type.items():
            random.shuffle(type_samples)
            n_total = len(type_samples)
            n_train = int(n_total * train_ratio)
            n_val = int(n_total * val_ratio)
            train_samples.extend(type_samples[:n_train])
            val_samples.extend(type_samples[n_train:n_train + n_val])
            test_samples.extend(type_samples[n_train + n_val:])

        if self.split == "train":
            return train_samples
        elif self.split == "val":
            return val_samples
        else:
            return test_samples

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
        mask = cv2.imread(sample['mask_path'], cv2.IMREAD_GRAYSCALE)

        image = cv2.resize(image, self.image_size)
        mask = cv2.resize(mask, self.image_size, interpolation=cv2.INTER_NEAREST)
        mask = np.where(mask > 0, sample['class_id'] + 1, 0).astype(np.uint8)

        image = self._apply_clahe(image)
        image, mask = self._augment(image, mask)

        image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        mask = torch.from_numpy(mask).long()

        return {
            'image': image,
            'mask': mask,
            'id': f"mt_{sample['defect_type']}_{idx}",
            'class_id': sample['class_id'],
            'defect_type': sample['defect_type'],
        }


class DualMagneticTileDataset(MagneticTileDataset):
    """双输入版本。"""

    def __getitem__(self, idx: int):
        sample = self.samples[idx]

        image = cv2.imread(sample['image_path'])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(sample['mask_path'], cv2.IMREAD_GRAYSCALE)

        image = cv2.resize(image, self.image_size)
        mask = cv2.resize(mask, self.image_size, interpolation=cv2.INTER_NEAREST)
        mask = np.where(mask > 0, sample['class_id'] + 1, 0).astype(np.uint8)

        image_enh = self._apply_clahe(image.copy())

        if self.augment:
            state = random.getstate()
            image, mask = self._augment(image, mask)
            random.setstate(state)
            image_enh, _ = self._augment(image_enh, mask.copy())

        image_ori_t = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        image_enh_t = torch.from_numpy(image_enh).permute(2, 0, 1).float() / 255.0
        mask_t = torch.from_numpy(mask).long()

        return {
            'image_ori': image_ori_t,
            'image_enh': image_enh_t,
            'mask': mask_t,
            'id': f"mt_{sample['defect_type']}_{idx}",
            'class_id': sample['class_id'],
            'defect_type': sample['defect_type'],
        }
