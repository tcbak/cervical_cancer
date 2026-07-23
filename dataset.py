import os
from collections import Counter
import numpy as np
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, WeightedRandomSampler

# Default dataset path relative to the project workspace
DEFAULT_DATA_DIR = "D:/MSC AI/Split_Data"

def get_transforms(img_size=(224, 224)):
    """
    Returns train and val/test transforms.
    Train transforms include strong data augmentation to combat class imbalance.
    """
    train_transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=25),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    
    val_test_transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_test_transform

def get_dataloaders(data_dir=DEFAULT_DATA_DIR, batch_size=32, img_size=(224, 224)):
    """
    Loads ImageFolder datasets and configures balanced loaders using WeightedRandomSampler.
    """
    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "val")
    test_dir = os.path.join(data_dir, "test")
    
    # Verify directories
    for path in [train_dir, val_dir, test_dir]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Directory not found: {path}")
            
    train_transform, val_test_transform = get_transforms(img_size)
    
    # Load Image Folders
    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    val_dataset = datasets.ImageFolder(val_dir, transform=val_test_transform)
    test_dataset = datasets.ImageFolder(test_dir, transform=val_test_transform)
    
    class_names = train_dataset.classes
    num_classes = len(class_names)
    
    # Handle Class Imbalance
    train_labels = train_dataset.targets
    class_counts = np.bincount(train_labels)
    
    # Compute class weights (inverse frequency)
    # If a class count is 0, set weight to 0
    class_weights = np.zeros_like(class_counts, dtype=float)
    non_zero_mask = class_counts > 0
    class_weights[non_zero_mask] = 1.0 / class_counts[non_zero_mask]
    
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float)
    
    # Compute weight for each training sample
    sample_weights = [class_weights[label] for label in train_labels]
    
    # Weighted sampler to balance batches
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
        num_workers=0,  # Safer defaults for execution environments
        pin_memory=True if torch.cuda.is_available() else False
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
    
    return train_loader, val_loader, test_loader, class_names, class_weights_tensor

if __name__ == "__main__":
    print("Testing Dataset Module...")
    try:
        train_loader, val_loader, test_loader, classes, weights = get_dataloaders()
        print("Success! Classes loaded:", classes)
        print("Class Loss Weights:", weights)
        
        # Test one batch
        images, labels = next(iter(train_loader))
        print(f"Batch shape: {images.shape}, Labels shape: {labels.shape}")
        
        # Check distribution in train set (original)
        train_dataset = train_loader.dataset
        original_counts = Counter(train_dataset.targets)
        print("Original Train Counts:", {classes[k]: original_counts[k] for k in original_counts})
        
        # Check distribution in one sampled batch
        batch_counts = Counter(labels.tolist())
        print("Sampled Batch Class Distribution:", {classes[k]: batch_counts.get(k, 0) for k in range(len(classes))})
        
    except Exception as e:
        print("Error during test run:", e)
