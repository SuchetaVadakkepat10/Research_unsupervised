"""
Data loading and preprocessing utilities for Brain MRI Classification
"""
import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import train_test_split
import numpy as np
import config


class BrainMRIDataset(Dataset):
    """Custom Dataset for Brain MRI images"""
    
    def __init__(self, image_paths, labels, transform=None):
        """
        Args:
            image_paths: List of paths to images
            labels: List of corresponding labels
            transform: Optional transform to be applied on images
        """
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load image
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('L')  # Convert to grayscale
        
        if self.transform:
            image = self.transform(image)
        
        label = self.labels[idx]
        return image, label


def get_data_transforms(augment=True):
    """
    Get data transformation pipelines
    
    Args:
        augment: Whether to include data augmentation (for training)
    
    Returns:
        transform: torchvision transforms composition
    """
    if augment:
        transform = transforms.Compose([
            transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.ToTensor(),
            transforms.Normalize(mean=config.MEAN, std=config.STD)
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=config.MEAN, std=config.STD)
        ])
    
    return transform


def load_dataset():
    """
    Load all images and labels from the data directory
    
    Returns:
        image_paths: List of image file paths
        labels: List of corresponding label indices
        class_names: List of class names
    """
    image_paths = []
    labels = []
    
    for label_idx, class_name in enumerate(config.CLASSES):
        class_dir = os.path.join(config.DATA_DIR, class_name)
        
        if not os.path.exists(class_dir):
            print(f"Warning: Directory {class_dir} does not exist. Skipping...")
            continue
        
        # Get all PNG files in the class directory
        for img_name in os.listdir(class_dir):
            if img_name.lower().endswith('.png'):
                img_path = os.path.join(class_dir, img_name)
                image_paths.append(img_path)
                labels.append(label_idx)
    
    print(f"Loaded {len(image_paths)} images from {len(config.CLASSES)} classes")
    for idx, class_name in enumerate(config.CLASSES):
        count = sum(1 for label in labels if label == idx)
        print(f"  {class_name}: {count} images")
    
    return image_paths, labels, config.CLASSES


def create_data_loaders(random_seed=None):
    """
    Create train, validation, and test data loaders
    
    Args:
        random_seed: Random seed for reproducibility
    
    Returns:
        train_loader, val_loader, test_loader: DataLoader objects
    """
    if random_seed is None:
        random_seed = config.RANDOM_SEED
    
    # Load all data
    image_paths, labels, class_names = load_dataset()
    
    # Convert to numpy arrays for splitting
    image_paths = np.array(image_paths)
    labels = np.array(labels)
    
    # First split: separate test set
    train_val_paths, test_paths, train_val_labels, test_labels = train_test_split(
        image_paths, labels,
        test_size=config.TEST_RATIO,
        random_state=random_seed,
        stratify=labels
    )
    
    # Second split: separate train and validation
    val_size = config.VAL_RATIO / (config.TRAIN_RATIO + config.VAL_RATIO)
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        train_val_paths, train_val_labels,
        test_size=val_size,
        random_state=random_seed,
        stratify=train_val_labels
    )
    
    print(f"\nDataset split (seed={random_seed}):")
    print(f"  Train: {len(train_paths)} images")
    print(f"  Val:   {len(val_paths)} images")
    print(f"  Test:  {len(test_paths)} images")
    
    # Create datasets with appropriate transforms
    train_dataset = BrainMRIDataset(
        train_paths.tolist(), train_labels.tolist(),
        transform=get_data_transforms(augment=True)
    )
    
    val_dataset = BrainMRIDataset(
        val_paths.tolist(), val_labels.tolist(),
        transform=get_data_transforms(augment=False)
    )
    
    test_dataset = BrainMRIDataset(
        test_paths.tolist(), test_labels.tolist(),
        transform=get_data_transforms(augment=False)
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=0,  # Set to 0 for Windows compatibility
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # Test the data loader
    print("Testing data loader...")
    train_loader, val_loader, test_loader = create_data_loaders()
    
    # Get a batch
    images, labels = next(iter(train_loader))
    print(f"\nBatch shape: {images.shape}")
    print(f"Labels shape: {labels.shape}")
    print(f"Image value range: [{images.min():.2f}, {images.max():.2f}]")
