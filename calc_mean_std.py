import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
import os
from tqdm import tqdm
from src import config
from src.data_loader import CaptchaDataset

def calculate_mean_std(dataloader):
    """
    Calculates the mean and standard deviation of a dataset yielded by a DataLoader.
    Assumes images are scaled to [0, 1] by ToTensor().
    """
    channels_sum, channels_squared_sum, num_batches = 0, 0, 0
    total_pixels = 0

    print("Calculating mean and std... this may take a while.")
    # Use tqdm for a progress bar
    for data, _ in tqdm(dataloader, total=len(dataloader)):
        # data shape is (batch_size, channels, height, width)
        # Ensure data is float for calculations
        data = data.float()

        # Sum over all dimensions except the channel dimension (dim=1 for C,H,W)
        # For grayscale, channel dim is 1, so sum everything
        # Summing over H, W, and Batch (dims 0, 2, 3)
        channels_sum += torch.sum(data)
        channels_squared_sum += torch.sum(data**2)
        num_batches += 1
        total_pixels += data.numel()

    if num_batches == 0 or total_pixels == 0:
        return None, None

    # Calculate mean
    mean = channels_sum / total_pixels

    # Calculate variance and then std deviation
    # Var = E[X^2] - (E[X])^2
    variance = (channels_squared_sum / total_pixels) - (mean**2)
    std = torch.sqrt(variance)

    return mean.item(), std.item()

if __name__ == "__main__":
    # Configuration
    # Mminimal transform
    calc_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((config.IMAGE_HEIGHT, config.IMAGE_WIDTH)),
        transforms.ToTensor() # Scales images to [0.0, 1.0]
    ])

    # Dataset
    # Create dataset instance using the minimal transform
    # Uses paths from config.py
    print(f"Loading dataset...")
    print(f"Label file: {config.LABEL_FILE}")
    print(f"Image directory: {config.IMAGE_DIR}")

    try:
        # Load the entire dataset for calculation.
        dataset = CaptchaDataset(
            csv_file=config.LABEL_FILE,
            img_dir=config.IMAGE_DIR,
            transform=calc_transform
        )
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please ensure data/labels.csv and data/images/ exist relative to the project root.")
        exit()
    except Exception as e:
        print(f"An unexpected error occurred loading the dataset: {e}")
        exit()

    # DataLoader
    # Use a relatively large batch size for efficiency
    BATCH_SIZE_CALC = 128
    NUM_WORKERS_CALC = 4

    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE_CALC,
        shuffle=False,
        num_workers=NUM_WORKERS_CALC,
        pin_memory=False
    )

    # Calculate
    if len(dataloader) > 0:
        mean, std = calculate_mean_std(dataloader)

        if mean is not None and std is not None:
            print(f"\nCalculation Complete:")
            print(f"Calculated Mean: {mean:.4f}")
            print(f"Calculated Std Dev: {std:.4f}")
            print("\nUpdate the MEAN and STD constants in 'src/data_loader.py':")
            print(f"MEAN = [{mean:.4f}]")
            print(f"STD = [{std:.4f}]")
        else:
            print("\nCalculation failed. DataLoader might be empty.")
    else:
        print("\nError: DataLoader has zero length. Check dataset path and contents.")