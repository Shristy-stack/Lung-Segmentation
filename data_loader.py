import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class LungSegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None, augmentation=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.augmentation = augmentation
        self.images = [f for f in sorted(os.listdir(image_dir)) if os.path.isfile(os.path.join(image_dir, f))]
        self.masks = [f for f in sorted(os.listdir(mask_dir)) if os.path.isfile(os.path.join(mask_dir, f))]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.masks[idx])
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        if self.augmentation:
            image, mask = self.augmentation(image, mask)

        return image, mask
