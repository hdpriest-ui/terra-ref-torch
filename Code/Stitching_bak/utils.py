import os
import torch
import numpy as np
import matplotlib.pyplot as plt


def save_checkpoint(model, optimizer, epoch, best_loss, checkpoint_dir="checkpoints"):
    """
    Save the model checkpoint.

    Args:
        model (torch.nn.Module): The model to save.
        optimizer (torch.optim.Optimizer): The optimizer to save.
        epoch (int): The current epoch.
        best_loss (float): The best loss achieved so far.
        checkpoint_dir (str): Directory to save the checkpoint.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, f"model_epoch_{epoch}.pth")
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_loss': best_loss,
    }, checkpoint_path)
    print(f"Checkpoint saved at {checkpoint_path}")


def load_checkpoint(checkpoint_path, model, optimizer=None):
    """
    Load a model checkpoint.

    Args:
        checkpoint_path (str): Path to the checkpoint file.
        model (torch.nn.Module): The model to load the checkpoint into.
        optimizer (torch.optim.Optimizer, optional): The optimizer to load the checkpoint into.
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    print(f"Checkpoint loaded from {checkpoint_path}")


def create_output_dirs(output_root):
    """
    Create output directories for saving results.

    Args:
        output_root (str): Root directory for outputs.
    """
    if os.path.exists(output_root):
        raise FileExistsError(f"Output directory already exists: {output_root}")

    os.makedirs(output_root, exist_ok=True)
    os.makedirs(os.path.join(output_root, "heatmaps"), exist_ok=True)
    os.makedirs(os.path.join(output_root, "refined_outputs"), exist_ok=True)
    os.makedirs(os.path.join(output_root, "coarse_outputs"), exist_ok=True)
    print(f"Output directories created at {output_root}")


def save_heatmap(output, target, idx, output_root):
    """
    Save a high-quality heatmap showing the difference between the output and target.

    Args:
        output (torch.Tensor): The model's output tensor.
        target (torch.Tensor): The ground truth tensor.
        idx (int): Index of the current sample.
        output_root (str): Root directory for outputs.
    """
    # Compute the difference between the output and target
    difference = torch.abs(output - target).mean(dim=0).cpu().numpy()

    # Normalize the difference to [0, 1]
    difference = (difference - difference.min()) / (difference.max() - difference.min() + 1e-8)

    # Save the heatmap as a high-quality figure
    heatmap_path = os.path.join(output_root, "heatmaps", f"heatmap_{idx}.png")
    plt.figure(figsize=(10, 10), dpi=300)  # High resolution for publication-quality
    plt.imshow(difference, cmap="hot", interpolation="nearest")
    plt.colorbar(label="Difference Intensity", fraction=0.046, pad=0.04)
    plt.title(f"Heatmap {idx}", fontsize=16, fontweight="bold")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(heatmap_path, bbox_inches="tight", pad_inches=0.1)
    plt.close()
    print(f"High-quality heatmap saved at {heatmap_path}")