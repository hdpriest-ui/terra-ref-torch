import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import seaborn as sns
from pathlib import Path
import logging
from collections import defaultdict

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Metric mapping for display names to data keys
METRIC_MAPPING = {
    'Width': 'widths',
    'Height': 'heights',
    'Pixels': 'pixels'
}

def get_image_dimensions(image_path):
    """Get dimensions of an image file."""
    try:
        with Image.open(image_path) as img:
            width, height = img.size
            return width, height, width * height
    except Exception as e:
        logger.error(f"Error processing {image_path}: {str(e)}")
        return None, None, None

def process_directory(input_dir):
    """Process all images in the directory and collect statistics for each subdirectory."""
    # Dictionary to store data for each subdirectory
    subdir_data = defaultdict(lambda: {'widths': [], 'heights': [], 'pixels': []})
    total_data = {'widths': [], 'heights': [], 'pixels': []}
    
    # Supported image extensions
    image_extensions = {'.jpg', '.jpeg', '.png', '.tiff', '.bmp'}
    
    # Walk through the directory
    for root, _, files in os.walk(input_dir):
        # Get relative path from input_dir
        rel_path = os.path.relpath(root, input_dir)
        if rel_path == '.':
            rel_path = 'root'
            
        for file in files:
            if Path(file).suffix.lower() in image_extensions:
                image_path = os.path.join(root, file)
                width, height, pixel_count = get_image_dimensions(image_path)
                if width is not None:
                    subdir_data[rel_path]['widths'].append(width)
                    subdir_data[rel_path]['heights'].append(height)
                    subdir_data[rel_path]['pixels'].append(pixel_count)
                    
                    total_data['widths'].append(width)
                    total_data['heights'].append(height)
                    total_data['pixels'].append(pixel_count)
    
    # Convert lists to numpy arrays
    for subdir in subdir_data:
        for key in subdir_data[subdir]:
            subdir_data[subdir][key] = np.array(subdir_data[subdir][key])
    
    for key in total_data:
        total_data[key] = np.array(total_data[key])
    
    return subdir_data, total_data

def print_statistics(subdir_data, total_data):
    """Print statistics about image dimensions for each subdirectory and overall."""
    def print_stats(values, prefix=""):
        logger.info(f"{prefix}Minimum: {np.min(values)}")
        logger.info(f"{prefix}Maximum: {np.max(values)}")
        logger.info(f"{prefix}Average: {np.mean(values):.2f}")
        logger.info(f"{prefix}Median: {np.median(values)}")
        logger.info(f"{prefix}Count: {len(values)}")
    
    # Print statistics for each subdirectory
    for subdir, data in sorted(subdir_data.items()):
        if len(data['widths']) > 0:
            logger.info(f"\nStatistics for subdirectory: {subdir}")
            logger.info("Width Statistics:")
            print_stats(data['widths'], "  ")
            logger.info("Height Statistics:")
            print_stats(data['heights'], "  ")
            logger.info("Pixel Statistics:")
            print_stats(data['pixels'], "  ")
    
    # Print overall statistics
    logger.info("\nOverall Statistics:")
    logger.info("Width Statistics:")
    print_stats(total_data['widths'], "  ")
    logger.info("Height Statistics:")
    print_stats(total_data['heights'], "  ")
    logger.info("Pixel Statistics:")
    print_stats(total_data['pixels'], "  ")

def create_distribution_plot(data, title, output_path):
    """Create a publication-quality figure showing the distributions for a single dataset."""
    try:
        # Set style for publication
        plt.style.use('seaborn-v0_8')  # Use the correct style name
        sns.set_context("paper", font_scale=1.5)
    except Exception as e:
        logger.warning(f"Could not set matplotlib style: {str(e)}")
        logger.warning("Using default style instead")
        # Set some basic style parameters manually
        plt.rcParams.update({
            'font.size': 12,
            'axes.labelsize': 14,
            'axes.titlesize': 16,
            'xtick.labelsize': 12,
            'ytick.labelsize': 12,
            'legend.fontsize': 12
        })
    
    # Create figure with subplots
    fig, axes = plt.subplots(3, 1, figsize=(12, 18))
    
    # Plot distributions
    for i, (display_metric, ax) in enumerate(zip(['Width', 'Height', 'Pixels'], axes)):
        try:
            data_key = METRIC_MAPPING[display_metric]
            values = data[data_key]
            
            if len(values) > 0:
                # Create histogram with KDE
                sns.histplot(data=values, ax=ax, kde=True, color='blue', stat='density')
                
                # Add vertical lines for statistics
                ax.axvline(np.mean(values), color='red', linestyle='--', label=f'Mean: {np.mean(values):.0f}')
                ax.axvline(np.median(values), color='green', linestyle='--', label=f'Median: {np.median(values):.0f}')
                
                ax.set_title(f'Distribution of Image {display_metric}s')
                ax.set_xlabel(f'{display_metric} (pixels)')
                ax.legend()
            else:
                logger.warning(f"No data available for {display_metric}")
                ax.set_title(f'No data available for {display_metric}')
        except Exception as e:
            logger.error(f"Error plotting {display_metric} distribution: {str(e)}")
            continue
    
    # Add overall title
    fig.suptitle(title, fontsize=16, y=0.95)
    
    # Adjust layout and save
    plt.tight_layout()
    try:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Successfully saved figure to {output_path}")
    except Exception as e:
        logger.error(f"Error saving figure: {str(e)}")
        raise
    finally:
        plt.close()

def create_all_plots(subdir_data, total_data, output_dir):
    """Create distribution plots for each subdirectory and the overall dataset."""
    # Create overall plot
    overall_path = os.path.join(output_dir, 'overall_distribution.png')
    create_distribution_plot(total_data, 'Overall Image Dimensions Distribution', overall_path)
    
    # Create individual plots for each subdirectory
    for subdir, data in subdir_data.items():
        if len(data['widths']) > 0:
            # Create a safe filename from the subdirectory name
            safe_subdir = subdir.replace('/', '_').replace('\\', '_')
            if safe_subdir == 'root':
                safe_subdir = 'root_directory'
            
            subdir_path = os.path.join(output_dir, f'{safe_subdir}_distribution.png')
            create_distribution_plot(data, f'Image Dimensions Distribution: {subdir}', subdir_path)

def main():
    parser = argparse.ArgumentParser(description='Profile image dimensions in a directory')
    parser.add_argument('--input_dir', type=str, required=True,
                      help='Input directory containing images')
    parser.add_argument('--output_dir', type=str, required=True,
                      help='Output directory for the figure')
    
    args = parser.parse_args()
    
    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Process the directory
    logger.info(f"Processing images in {args.input_dir}")
    subdir_data, total_data = process_directory(args.input_dir)
    
    if len(total_data['widths']) == 0:
        logger.error("No valid images found in the input directory")
        return
    
    # Print statistics
    print_statistics(subdir_data, total_data)
    
    # Create all plots
    create_all_plots(subdir_data, total_data, args.output_dir)

if __name__ == '__main__':
    main()
