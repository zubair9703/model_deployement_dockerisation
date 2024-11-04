import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from skimage import io, transform
import lightning.pytorch as pl
import numpy as np

# Custom Dataset for Images and Masks using scikit-image
class ImageMaskDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.image_names = os.listdir(image_dir)  # Assuming images and masks have the same naming

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_names[idx])
        mask_path = os.path.join(self.mask_dir, self.image_names[idx])  # Mask should have the same name

        # Load the image and mask using scikit-image
        image = io.imread(img_path) # Loaded as a NumPy array
        mask = io.imread(mask_path)
        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        return image, mask


# Lightning DataModule to Manage Dataloaders
class SegmentDataLoader(pl.LightningDataModule):
    def __init__(self, data_dir, batch_size=8):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        # self.image_size = image_size
        self.transform = transforms.Compose([
            transforms.Lambda(lambda x: torch.tensor(x, dtype=torch.float32).permute(2, 0, 1) / 255 if x.ndim == 3 else torch.tensor(x).long())
        ])

    def setup(self, stage=None):
        image_dir = os.path.join(self.data_dir, 'images')
        mask_dir = os.path.join(self.data_dir, 'masks')

        # Split data into train/validation
        dataset = ImageMaskDataset(image_dir, mask_dir, transform=self.transform)
        dataset_len = len(dataset)
        train_size = int(0.8 * dataset_len)
        val_size = dataset_len - train_size
        self.train_dataset, self.val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)  # Using val_dataset for testing as an example