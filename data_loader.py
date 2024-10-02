import os
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
import cv2
import numpy as np

class BrainHemorrhageDataset(Dataset):
    def __init__(self, images_dir, masks_dir, image_transform=None, mask_transform=None):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.image_transform = image_transform
        self.mask_transform = mask_transform
        self.image_filenames = self._collect_file_paths(images_dir)
        self.mask_filenames = self._collect_file_paths(masks_dir)

    def _collect_file_paths(self, dir_path):
        file_paths = []
        for root, _, files in os.walk(dir_path):
            for file in files:
                if file.endswith(('.png', '.jpg', '.jpeg')):
                    file_paths.append(os.path.join(root, file))
        return sorted(file_paths)

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        image_path = self.image_filenames[idx]
        mask_path = self.mask_filenames[idx]
        
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if self.image_transform:
            image = self.image_transform(image)
        if self.mask_transform:
            mask = self.mask_transform(mask)

        if isinstance(mask, torch.Tensor) and mask.ndim == 2:
            mask = mask.unsqueeze(0)
        elif isinstance(mask, np.ndarray) and mask.ndim == 2:
            mask = torch.tensor(mask).unsqueeze(0)

        return image, mask

def get_dataloaders(data_dir, batch_size=16, image_transform=None, mask_transform=None, val_split=0.2, test_split=0.1):
    full_dataset = BrainHemorrhageDataset(
        os.path.join(data_dir, 'png_volumes'),
        os.path.join(data_dir, 'png_masks'),
        image_transform=image_transform,
        mask_transform=mask_transform
    )

    test_size = int(len(full_dataset) * test_split)
    val_size = int(len(full_dataset) * val_split)
    train_size = len(full_dataset) - val_size - test_size

    train_dataset, val_dataset, test_dataset = random_split(full_dataset, [train_size, val_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader