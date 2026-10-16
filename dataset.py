from pathlib import Path
from collections import Counter

import numpy as np
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, WeightedRandomSampler

# Project Paths

PROJECT_ROOT = Path(__file__).resolve().parent

DEFAULT_DATA_DIR = PROJECT_ROOT / "Split_Data_PatientWise"


def get_transforms(img_size=(224, 224)):
    """
    Returns training and validation/test transforms.
    """

    train_transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=25),
        transforms.ColorJitter(
            brightness=0.3,
            contrast=0.3,
            saturation=0.2,
            hue=0.1
        ),
        transforms.RandomAffine(
            degrees=0,
            translate=(0.1, 0.1)
        ),
        transforms.ToTensor(), #transform image format (IPEG/PNG) to numerical tensors as neural network doesn't work with image formats
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    val_test_transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    return train_transform, val_test_transform


def get_dataloaders(
        data_dir=DEFAULT_DATA_DIR,
        batch_size=32,
        img_size=(224, 224)
):
   

    data_dir = Path(data_dir)

    train_dir = data_dir / "Train"
    val_dir = data_dir / "Validation"
    test_dir = data_dir / "Test"

    
    # Verify dataset folders exist
   

    for folder in [train_dir, val_dir, test_dir]:
        if not folder.exists():
            raise FileNotFoundError(
                f"Dataset folder not found:\n{folder}"
            )

    train_transform, val_test_transform = get_transforms(img_size)

    
    # ImageFolder datasets
    

    train_dataset = datasets.ImageFolder(
        train_dir,
        transform=train_transform
    )

    val_dataset = datasets.ImageFolder(
        val_dir,
        transform=val_test_transform
    )

    test_dataset = datasets.ImageFolder(
        test_dir,
        transform=val_test_transform
    )

    class_names = train_dataset.classes

    
    # Handle Class Imbalance
    

    train_labels = train_dataset.targets

    class_counts = np.bincount(train_labels)

    class_weights = np.zeros_like(
        class_counts,
        dtype=float
    )

    non_zero = class_counts > 0

    class_weights[non_zero] = (
        1.0 / class_counts[non_zero]
    )

    class_weights_tensor = torch.tensor(
        class_weights,
        dtype=torch.float32
    )

    sample_weights = [
        class_weights[label]
        for label in train_labels
    ]

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )

   
    # DataLoaders
    

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )

    return (
        train_loader,
        val_loader,
        test_loader,
        class_names,
        class_weights_tensor
    )



# Testing


if __name__ == "__main__":

    
    print("Testing dataset.py")
    

    print(f"\nDataset Path:\n{DEFAULT_DATA_DIR}\n")

    try:

        train_loader, val_loader, test_loader, classes, weights = get_dataloaders()

        print("Classes:")
        print(classes)

        print("\nLoss Weights:")
        print(weights)

        images, labels = next(iter(train_loader))

        print(f"\nBatch Shape : {images.shape}")
        print(f"Label Shape : {labels.shape}")

        train_dataset = train_loader.dataset

        original_counts = Counter(train_dataset.targets)

        print("\nOriginal Dataset Distribution")

        for idx, cls in enumerate(classes):
            print(f"{cls:<10}: {original_counts[idx]}")

        batch_counts = Counter(labels.tolist())

        print("\nSampled Batch Distribution")

        for idx, cls in enumerate(classes):
            print(f"{cls:<10}: {batch_counts.get(idx,0)}")

        print("\nDataset module working correctly.")

    except Exception as e:

        print("\nERROR")
        print(e)