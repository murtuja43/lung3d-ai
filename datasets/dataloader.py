import os
import torch
import numpy as np
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms

# ─────────────────────────────────────────
# Paths
# ─────────────────────────────────────────
MONTGOMERY_PATH = Path("datasets/montgomery/CXR_png")
SHENZHEN_PATH   = Path("datasets/shenzhen/CXR_png")

# ─────────────────────────────────────────
# Image size for CNN input
# ─────────────────────────────────────────
IMAGE_SIZE = 224  # Standard for transfer learning


# ─────────────────────────────────────────
# Collect all images + labels from both datasets
# ─────────────────────────────────────────
def collect_all_images():
    """
    Scan both dataset folders and return:
    - list of image paths
    - list of labels (0=Normal, 1=TB)
    """
    image_paths = []
    labels      = []

    for folder in [MONTGOMERY_PATH, SHENZHEN_PATH]:
        if not folder.exists():
            print(f"⚠️  Folder not found: {folder}")
            continue

        for img_path in sorted(folder.glob("*.png")):
            stem = img_path.stem  # filename without extension

            if stem.endswith('_0'):
                image_paths.append(img_path)
                labels.append(0)  # Normal
            elif stem.endswith('_1'):
                image_paths.append(img_path)
                labels.append(1)  # TB
            # skip unknown files

    return image_paths, labels


# ─────────────────────────────────────────
# Custom Dataset class
# ─────────────────────────────────────────
class TBDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels      = labels
        self.transform   = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load image
        img_path = self.image_paths[idx]
        image    = Image.open(img_path).convert('RGB')

        # Apply transforms
        if self.transform:
            image = self.transform(image)

        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return image, label


# ─────────────────────────────────────────
# Transforms (augmentation for training)
# ─────────────────────────────────────────
def get_transforms():
    """
    Training: augment images to prevent overfitting
    Validation: only resize and normalize
    """
    train_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2
        ),
        transforms.ToTensor(),
        # ImageNet mean/std (standard for transfer learning)
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])

    return train_transform, val_transform


# ─────────────────────────────────────────
# Split dataset into train / val / test
# ─────────────────────────────────────────
def split_dataset(image_paths, labels, train=0.7, val=0.15):
    """
    Split into 70% train, 15% val, 15% test
    Stratified — keeps TB/Normal ratio equal in each split
    """
    from sklearn.model_selection import train_test_split

    # First split: train vs (val + test)
    X_train, X_temp, y_train, y_temp = train_test_split(
        image_paths, labels,
        test_size=(1 - train),
        stratify=labels,
        random_state=42
    )

    # Second split: val vs test
    val_ratio = val / (1 - train)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=0.5,
        stratify=y_temp,
        random_state=42
    )

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


# ─────────────────────────────────────────
# Weighted sampler (handles class imbalance)
# ─────────────────────────────────────────
def get_sampler(labels):
    """
    Give higher weight to minority class
    so model sees equal TB and Normal samples
    """
    labels_tensor = torch.tensor(labels)
    class_counts  = torch.bincount(labels_tensor)
    class_weights = 1.0 / class_counts.float()
    sample_weights = class_weights[labels_tensor]
    sampler = WeightedRandomSampler(
        weights     = sample_weights,
        num_samples = len(sample_weights),
        replacement = True
    )
    return sampler


# ─────────────────────────────────────────
# Build all DataLoaders
# ─────────────────────────────────────────
def get_dataloaders(batch_size=16):
    """
    Returns train, val, test DataLoaders
    ready for PyTorch training
    """
    image_paths, labels = collect_all_images()

    print(f"✅ Total images loaded: {len(image_paths)}")
    print(f"   Normal : {labels.count(0)}")
    print(f"   TB     : {labels.count(1)}")

    # Split
    (X_train, y_train), \
    (X_val,   y_val),   \
    (X_test,  y_test)   = split_dataset(image_paths, labels)

    print(f"\n📊 Split Summary:")
    print(f"   Train : {len(X_train)} images")
    print(f"   Val   : {len(X_val)}   images")
    print(f"   Test  : {len(X_test)}  images")

    # Transforms
    train_transform, val_transform = get_transforms()

    # Datasets
    train_dataset = TBDataset(X_train, y_train, transform=train_transform)
    val_dataset   = TBDataset(X_val,   y_val,   transform=val_transform)
    test_dataset  = TBDataset(X_test,  y_test,  transform=val_transform)

    # Sampler for training (handle imbalance)
    sampler = get_sampler(y_train)

    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size  = batch_size,
        sampler     = sampler,
        num_workers = 0,
        pin_memory  = False
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size  = batch_size,
        shuffle     = False,
        num_workers = 0
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size  = batch_size,
        shuffle     = False,
        num_workers = 0
    )

    return train_loader, val_loader, test_loader


# ─────────────────────────────────────────
# Test the dataloader
# ─────────────────────────────────────────
if __name__ == "__main__":
    print("🔍 Testing DataLoader...\n")
    train_loader, val_loader, test_loader = get_dataloaders(batch_size=16)

    # Check one batch
    images, labels = next(iter(train_loader))
    print(f"\n✅ Batch loaded successfully!")
    print(f"   Image batch shape : {images.shape}")
    print(f"   Label batch shape : {labels.shape}")
    print(f"   Image dtype       : {images.dtype}")
    print(f"   Labels in batch   : {labels.tolist()}")
    print(f"   Min pixel value   : {images.min():.3f}")
    print(f"   Max pixel value   : {images.max():.3f}")
    print(f"\n✅ DataLoader is ready for training!")