import torch
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from torchvision import transforms
import pandas as pd
import numpy as np
from PIL import Image
import os
from . import config
from . import utils

# Transforms

# Normalization parameters
MEAN = [0.9093]
STD = [0.1305]

# Transforms for the training set
train_transforms = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((config.IMAGE_HEIGHT, config.IMAGE_WIDTH)),
    transforms.RandomRotation(degrees=5),
    transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05), shear=3),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 1.0)),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD)
])

# Transforms for the validation/test set
val_transforms = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((config.IMAGE_HEIGHT, config.IMAGE_WIDTH)),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD)
])

class CaptchaDataset(Dataset):
    """Custom Dataset for loading Captcha images and labels."""
    def __init__(self, csv_file, img_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            img_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        try:
            self.labels_df = pd.read_csv(csv_file)
        except FileNotFoundError:
            raise FileNotFoundError(f"Error: Annotation file not found at {csv_file}")

        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        """Returns the total number of samples."""
        return len(self.labels_df)

    def __getitem__(self, idx):
        """
        Fetches the sample (image and label) at the given index, applies transforms,
        and encodes the label.
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        try:
            relative_img_path = self.labels_df.iloc[idx, 0]
            img_name = os.path.basename(relative_img_path)
            img_path = os.path.join(self.img_dir, img_name)

            # Load image using PIL
            # Convert to RGB first to handle potential palette issues, then grayscale if needed by transforms
            image = Image.open(img_path).convert('RGB')

            # Get label string
            label_string = self.labels_df.iloc[idx, 1] # Second column: captcha_text

        except FileNotFoundError:
            print(f"Warning: Image file not found at {img_path} for index {idx}. Skipping.")
            raise FileNotFoundError(f"Image not found: {img_path}")
        except Exception as e:
            print(f"Error loading item at index {idx}, path {img_path}: {e}")
            raise e

        # Apply transforms
        if self.transform:
            image = self.transform(image)

        # Encode the label string to a tensor of indices
        encoded_label = utils.encode_label(label_string)

        if encoded_label is None:
            # Handle cases where encoding failed (e.g., unexpected characters, wrong length)
            print(f"Warning: Failed to encode label '{label_string}' for image {img_path}. Skipping?")
            # Again, requires careful handling. Let's assume valid labels.
            raise ValueError(f"Label encoding failed for: {label_string}")

        return image, encoded_label

def get_dataloaders(csv_path=config.LABEL_FILE,
                    img_path=config.IMAGE_DIR,
                    batch_size=config.BATCH_SIZE,
                    val_split=config.VALIDATION_SPLIT,
                    random_seed=config.RANDOM_SEED,
                    num_workers=4,
                    pin_memory=True):
    """
    Creates and returns the training and validation DataLoaders.

    Args:
        csv_path (string): Path to the labels CSV file.
        img_path (string): Path to the image directory.
        batch_size (int): How many samples per batch to load.
        val_split (float): Fraction of the dataset to use for validation.
        random_seed (int): Seed for reproducible train/val split.
        num_workers (int): How many subprocesses to use for data loading.
        pin_memory (bool): If True, copies tensors into CUDA pinned memory before returning them.

    Returns:
        tuple: (train_loader, val_loader)
    """
    # Create two instances of the dataset: one for training, one for validation
    train_dataset = CaptchaDataset(csv_file=csv_path, img_dir=img_path, transform=train_transforms)
    val_dataset = CaptchaDataset(csv_file=csv_path, img_dir=img_path, transform=val_transforms)

    # Get dataset size and create indices
    dataset_size = len(train_dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(val_split * dataset_size))

    # Shuffle indices
    np.random.seed(random_seed)
    np.random.shuffle(indices)

    # Split indices
    train_indices, val_indices = indices[split:], indices[:split]

    # Create samplers
    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)

    # Create data loaders
    # The samplers will ensure each loader only gets indices from its respective split,
    # while drawing data from the appropriate Dataset instance
    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              sampler=train_sampler,
                              num_workers=num_workers,
                              pin_memory=pin_memory)

    val_loader = DataLoader(val_dataset,
                            batch_size=batch_size,
                            sampler=val_sampler,
                            num_workers=num_workers,
                            pin_memory=pin_memory)

    print(f"Dataset size: {dataset_size}")
    print(f"Train split: {len(train_indices)}, Validation split: {len(val_indices)}")
    print(f"Train batches: {len(train_loader)}, Validation batches: {len(val_loader)}")

    return train_loader, val_loader