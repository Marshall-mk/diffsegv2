#!/usr/bin/env python3
"""
Data preparation script for Diffusion Segmentation
Helps organize and validate your dataset structure
"""

import argparse
import shutil
from pathlib import Path
from typing import List, Tuple, Dict, Any
import json
from PIL import Image
import numpy as np
from tqdm import tqdm

from utils.data_utils import SegmentationDataset


def analyze_dataset(image_dir: str, mask_dir: str) -> Dict[str, Any]:
    """Analyze dataset and provide statistics"""
    
    image_path = Path(image_dir)
    mask_path = Path(mask_dir)
    
    # Find all image files
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    image_files = []
    for ext in image_extensions:
        image_files.extend(image_path.glob(f"*{ext}"))
        image_files.extend(image_path.glob(f"*{ext.upper()}"))
    
    # Find all mask files
    mask_files = []
    for ext in image_extensions:
        mask_files.extend(mask_path.glob(f"*{ext}"))
        mask_files.extend(mask_path.glob(f"*{ext.upper()}"))
    
    # Analyze image sizes
    image_sizes = []
    mask_sizes = []
    
    print("Analyzing dataset...")
    for img_file in tqdm(image_files[:50]):  # Sample first 50 images
        try:
            with Image.open(img_file) as img:
                image_sizes.append(img.size)
        except Exception as e:
            print(f"Error reading {img_file}: {e}")
    
    for mask_file in tqdm(mask_files[:50]):  # Sample first 50 masks
        try:
            with Image.open(mask_file) as mask:
                mask_sizes.append(mask.size)
        except Exception as e:
            print(f"Error reading {mask_file}: {e}")
    
    # Calculate statistics
    stats = {
        'total_images': len(image_files),
        'total_masks': len(mask_files),
        'image_sizes': {
            'unique_sizes': len(set(image_sizes)),
            'most_common_size': max(set(image_sizes), key=image_sizes.count) if image_sizes else None,
            'min_width': min(s[0] for s in image_sizes) if image_sizes else None,
            'max_width': max(s[0] for s in image_sizes) if image_sizes else None,
            'min_height': min(s[1] for s in image_sizes) if image_sizes else None,
            'max_height': max(s[1] for s in image_sizes) if image_sizes else None,
        },
        'mask_sizes': {
            'unique_sizes': len(set(mask_sizes)),
            'most_common_size': max(set(mask_sizes), key=mask_sizes.count) if mask_sizes else None,
        }
    }
    
    return stats


def validate_dataset(image_dir: str, mask_dir: str, fix_issues: bool = False) -> List[str]:
    """Validate dataset and optionally fix common issues"""
    
    issues = []
    
    try:
        dataset = SegmentationDataset(image_dir, mask_dir, image_size=(256, 256))
        print(f"✓ Dataset loaded successfully with {len(dataset)} pairs")
        
        # Check a few samples
        for i in range(min(5, len(dataset))):
            try:
                image, mask = dataset[i]
                info = dataset.get_item_info(i)
                
                # Check tensor shapes
                if image.shape[0] != 3:
                    issues.append(f"Image {info['image_name']} has {image.shape[0]} channels, expected 3")
                
                if mask.shape[0] != 1:
                    issues.append(f"Mask {info['mask_name']} has {mask.shape[0]} channels, expected 1")
                
                # Check value ranges
                if image.min() < -1.1 or image.max() > 1.1:
                    issues.append(f"Image {info['image_name']} values out of expected range [-1, 1]")
                
                if mask.min() < -0.1 or mask.max() > 1.1:
                    issues.append(f"Mask {info['mask_name']} values out of expected range [0, 1]")
                
            except Exception as e:
                issues.append(f"Error loading sample {i}: {e}")
        
    except Exception as e:
        issues.append(f"Failed to create dataset: {e}")
    
    return issues


def organize_dataset(source_dir: str, output_dir: str, train_split: float = 0.8):
    """Organize dataset into train/val splits"""
    
    source_path = Path(source_dir)
    output_path = Path(output_dir)
    
    # Create output directories
    train_images_dir = output_path / "train" / "images"
    train_masks_dir = output_path / "train" / "masks"
    val_images_dir = output_path / "val" / "images"
    val_masks_dir = output_path / "val" / "masks"
    
    for dir_path in [train_images_dir, train_masks_dir, val_images_dir, val_masks_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    # Find all image files
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    all_files = []
    
    for ext in image_extensions:
        all_files.extend(source_path.glob(f"**/*{ext}"))
        all_files.extend(source_path.glob(f"**/*{ext.upper()}"))
    
    # Split files
    import random
    random.shuffle(all_files)
    
    split_idx = int(len(all_files) * train_split)
    train_files = all_files[:split_idx]
    val_files = all_files[split_idx:]
    
    print(f"Organizing {len(train_files)} training files and {len(val_files)} validation files...")
    
    # Copy files
    for file_list, img_dir, mask_dir in [(train_files, train_images_dir, train_masks_dir), 
                                        (val_files, val_images_dir, val_masks_dir)]:
        for file_path in tqdm(file_list):
            # Determine if it's an image or mask based on filename or directory
            if any(keyword in str(file_path).lower() for keyword in ['mask', 'gt', 'label', 'annotation']):
                dest_dir = mask_dir
            else:
                dest_dir = img_dir
            
            dest_path = dest_dir / file_path.name
            shutil.copy2(file_path, dest_path)


def create_sample_dataset(output_dir: str, num_samples: int = 100):
    """Create a sample synthetic dataset for testing"""
    
    output_path = Path(output_dir)
    images_dir = output_path / "images"
    masks_dir = output_path / "masks"
    
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Creating {num_samples} synthetic samples...")
    
    for i in tqdm(range(num_samples)):
        # Create random image
        image = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
        
        # Create simple geometric mask
        mask = np.zeros((256, 256), dtype=np.uint8)
        center_x = np.random.randint(64, 192)
        center_y = np.random.randint(64, 192)
        radius = np.random.randint(20, 60)
        
        y, x = np.ogrid[:256, :256]
        mask_circle = (x - center_x)**2 + (y - center_y)**2 <= radius**2
        mask[mask_circle] = 255
        
        # Save files
        Image.fromarray(image).save(images_dir / f"sample_{i:04d}.png")
        Image.fromarray(mask).save(masks_dir / f"sample_{i:04d}.png")
    
    print(f"Sample dataset created at {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Prepare data for Diffusion Segmentation')
    parser.add_argument('--action', type=str, required=True, 
                       choices=['analyze', 'validate', 'organize', 'create-sample'],
                       help='Action to perform')
    parser.add_argument('--image-dir', type=str, help='Directory containing images')
    parser.add_argument('--mask-dir', type=str, help='Directory containing masks')
    parser.add_argument('--source-dir', type=str, help='Source directory for organization')
    parser.add_argument('--output-dir', type=str, help='Output directory')
    parser.add_argument('--train-split', type=float, default=0.8, help='Training split ratio')
    parser.add_argument('--num-samples', type=int, default=100, help='Number of samples to create')
    parser.add_argument('--fix-issues', action='store_true', help='Attempt to fix validation issues')
    
    args = parser.parse_args()
    
    if args.action == 'analyze':
        if not args.image_dir or not args.mask_dir:
            print("Error: --image-dir and --mask-dir required for analyze")
            return
        
        stats = analyze_dataset(args.image_dir, args.mask_dir)
        
        print("\n" + "="*50)
        print("DATASET ANALYSIS")
        print("="*50)
        print(f"Total images: {stats['total_images']}")
        print(f"Total masks: {stats['total_masks']}")
        print(f"Image sizes: {stats['image_sizes']['unique_sizes']} unique sizes")
        print(f"Most common image size: {stats['image_sizes']['most_common_size']}")
        print(f"Image width range: {stats['image_sizes']['min_width']} - {stats['image_sizes']['max_width']}")
        print(f"Image height range: {stats['image_sizes']['min_height']} - {stats['image_sizes']['max_height']}")
        print(f"Most common mask size: {stats['mask_sizes']['most_common_size']}")
        
        # Save stats to file
        if args.output_dir:
            output_path = Path(args.output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            with open(output_path / "dataset_stats.json", 'w') as f:
                json.dump(stats, f, indent=2)
            print(f"\nStats saved to {output_path / 'dataset_stats.json'}")
    
    elif args.action == 'validate':
        if not args.image_dir or not args.mask_dir:
            print("Error: --image-dir and --mask-dir required for validate")
            return
        
        print("Validating dataset...")
        issues = validate_dataset(args.image_dir, args.mask_dir, args.fix_issues)
        
        if issues:
            print(f"\n❌ Found {len(issues)} issues:")
            for issue in issues:
                print(f"  • {issue}")
        else:
            print("\n✅ Dataset validation passed!")
    
    elif args.action == 'organize':
        if not args.source_dir or not args.output_dir:
            print("Error: --source-dir and --output-dir required for organize")
            return
        
        organize_dataset(args.source_dir, args.output_dir, args.train_split)
        print(f"Dataset organized at {args.output_dir}")
    
    elif args.action == 'create-sample':
        if not args.output_dir:
            print("Error: --output-dir required for create-sample")
            return
        
        create_sample_dataset(args.output_dir, args.num_samples)


if __name__ == "__main__":
    main()