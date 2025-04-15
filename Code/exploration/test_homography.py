import os, sys
import argparse
from pathlib import Path
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from Homography.models import HomographyEstimator
from Homography.utilities import HomographyInputLoader
from Stitching.models.SSL import StructureStitchingLayer
from Homography.tensorDLT import solve_DLT
import time

def get_device():
    """Get the device to use for computation"""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_model(model_path):
    """Load the homography model from checkpoint"""
    device = get_device()
    model = HomographyEstimator(batch_size=1)  # Using batch size 1 for testing
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    return model, device

def load_image(filename, resize_height=128, resize_width=128):
    """Load and preprocess an image frame.
    
    Args:
        filename (str): Path to the image file
        resize_height (int): Height to resize the image to
        resize_width (int): Width to resize the image to
        
    Returns:
        torch.Tensor: Preprocessed image tensor in CHW format
    """
    image = cv2.imread(str(filename))
    if image is None:
        raise ValueError(f"Could not load image: {filename}")
    
    # Resize if needed
    if resize_height is not None and resize_width is not None:
        image = cv2.resize(image, (resize_width, resize_height))
    
    # Convert to CHW format and create tensor
    image = torch.from_numpy(image).permute(2, 0, 1)  # HWC -> CHW
    image = image.float()
    
    # Normalize to [-1, 1] range
    image = (image / 127.5) - 1.0
    return image

def tensor_to_vis(image_tensor):
    """Convert a CHW tensor in [-1,1] range to HWC numpy array in [0,255] range for visualization"""
    # Convert from CHW to HWC
    image = image_tensor.permute(1, 2, 0).cpu().numpy()
    # Convert from [-1, 1] to [0, 255]
    image = ((image + 1.0) * 127.5).astype(np.uint8)
    return image

def create_heatmap(ref_image, target_image, alignment_score):
    """Create a heatmap visualization of the alignment"""
    # Convert tensors to visualization format
    ref_vis = tensor_to_vis(ref_image)
    target_vis = tensor_to_vis(target_image)
    
    # Create figure with subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot original images
    ax1.imshow(ref_vis)
    ax1.set_title('Reference Image')
    ax1.axis('off')
    
    ax2.imshow(target_vis)
    ax2.set_title('Target Image')
    ax2.axis('off')
    
    # Create and plot heatmap
    heatmap = np.zeros_like(ref_vis)
    heatmap[:, :, 0] = alignment_score  # Red channel for heatmap
    ax3.imshow(heatmap)
    ax3.set_title(f'Alignment Score: {alignment_score:.4f}')
    ax3.axis('off')
    
    plt.tight_layout()
    return fig

def evaluate_alignment(model, ref_image, target_image, device):
    """Evaluate alignment between two images using the model"""
    stitched_height = 304
    stitched_width = 304
    with torch.no_grad():
        # Prepare inputs and move to correct device
        ref_tensor = ref_image.unsqueeze(0).to(device)  # Add batch dimension
        target_tensor = target_image.unsqueeze(0).to(device)  # Add batch dimension
        
        # Get model outputs
        outputs, _ = model(ref_tensor, target_tensor, None, is_stitching=True)
        
        # Step 3: Compute the homography matrix
        H = solve_DLT(outputs, patch_size=max(ref_tensor.shape[2], ref_tensor.shape[3]))
        print(H)
        
        # # Step 4: Apply the homography transformation using SSL
        # coarse_stitched = StructureStitchingLayer.apply([ref_image, target_image], H, stitched_height, stitched_width)
        
        # # Calculate alignment score (using MSE between outputs and identity matrix)
        # identity_matrix = torch.eye(3).unsqueeze(0).to(device)
        # alignment_score = torch.mean((outputs - identity_matrix) ** 2).item()
        
        return H

def main():
    parser = argparse.ArgumentParser(description='Test homography model on image tiles')
    parser.add_argument('--ref_tile', required=True, help='Path to reference tile image')
    parser.add_argument('--tile_dir', required=True, help='Directory containing target tiles')
    parser.add_argument('--model_path', required=True, help='Path to homography model checkpoint')
    parser.add_argument('--output_dir', default='results', help='Directory to save results')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Load model and get device
    model, device = load_model(args.model_path)
    print(f"Using device: {device}")
    
    # Load reference tile
    ref_image = load_image(args.ref_tile)
    
    # Process each tile in the directory
    tile_dir = Path(args.tile_dir)
    # Define supported image formats
    image_formats = ['*.png', '*.jpg', '*.jpeg', '*.tif', '*.tiff']
    
    # Process all supported image formats
    for format_pattern in image_formats:
        for tile_path in tile_dir.glob(format_pattern):
            if tile_path.name not in ["000500.jpg", "000439.jpg"]:
                continue
            print(f"Processing {args.ref_tile} vs {tile_path.name} ")
            
            # Load target tile
            target_image = load_image(tile_path)
            
            # Evaluate alignment
            alignment_score = evaluate_alignment(model, ref_image, target_image, device)
            
            # # Create and save heatmap
            # fig = create_heatmap(ref_image, target_image, alignment_score)
            # fig.savefig(output_dir / f'heatmap_{tile_path.stem}.png')
            # plt.close(fig)
            
            # print(f"Alignment score for {tile_path.name}: {alignment_score:.4f}")

if __name__ == '__main__':
    main() 