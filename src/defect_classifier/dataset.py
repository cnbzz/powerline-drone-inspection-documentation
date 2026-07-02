"""InsPLAD-fault PyTorch dataset for EfficientNetV2-S training."""
from pathlib import Path
from typing import Optional, Callable

import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T


# EfficientNetV2-S optimal image size (B0'ın 224'ünden büyük)
DEFAULT_IMAGE_SIZE = 384

# ImageNet normalization (timm pretrained ağırlıkları için)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class InsPLADFaultDataset(Dataset):
    """Equipment-specific defect classification dataset."""

    def __init__(
        self,
        root_dir: Path,
        equipment: str,
        split: str = 'train',
        transform: Optional[Callable] = None,
        image_size: int = DEFAULT_IMAGE_SIZE,
    ):
        self.root = Path(root_dir) / split / equipment
        self.equipment = equipment
        self.split = split

        if not self.root.exists():
            raise FileNotFoundError(f"Path not found: {self.root}")

        # Sınıfları sırala (reproducibility)
        self.classes = sorted([d.name for d in self.root.iterdir() if d.is_dir()])
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}

        # Tüm görüntüleri topla
        self.samples = []
        for cls in self.classes:
            cls_dir = self.root / cls
            for img_path in cls_dir.glob('*.jpg'):
                self.samples.append((img_path, self.class_to_idx[cls]))
            for img_path in cls_dir.glob('*.png'):
                self.samples.append((img_path, self.class_to_idx[cls]))

        # Augmentation transforms
        if transform is None:
            if split == 'train':
                # HEAVY augmentation (genelleme için kritik)
                self.transform = T.Compose([
                    T.Resize((image_size + 32, image_size + 32)),
                    T.RandomCrop((image_size, image_size)),
                    T.RandomHorizontalFlip(p=0.5),
                    T.RandomVerticalFlip(p=0.3),
                    T.RandomRotation(degrees=20),
                    T.ColorJitter(
                        brightness=0.3, contrast=0.3,
                        saturation=0.2, hue=0.05
                    ),
                    T.RandomAffine(
                        degrees=0,
                        translate=(0.1, 0.1),
                        scale=(0.85, 1.15)
                    ),
                    T.ToTensor(),
                    T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
                    T.RandomErasing(p=0.25, scale=(0.02, 0.20)),
                ])
            else:
                # Val/test: sadece resize + normalize
                self.transform = T.Compose([
                    T.Resize((image_size, image_size)),
                    T.ToTensor(),
                    T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
                ])
        else:
            self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]

        # Unicode-safe image loading
        try:
            with open(img_path, 'rb') as f:
                img = Image.open(f).convert('RGB')
        except Exception as e:
            print(f"Warning: cannot load {img_path}: {e}")
            return self.__getitem__((idx + 1) % len(self.samples))

        if self.transform:
            img = self.transform(img)

        return img, label

    def get_class_weights(self):
        """Inverse frequency class weights for imbalanced data."""
        from collections import Counter
        counter = Counter([label for _, label in self.samples])
        total = sum(counter.values())
        num_classes = len(self.classes)
        weights = [total / (num_classes * counter[i]) for i in range(num_classes)]
        return torch.FloatTensor(weights)
