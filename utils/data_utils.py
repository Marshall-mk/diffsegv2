import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, List, Union, Dict, Any
import PIL.Image as Image
import random
import json
import cv2


class DualTransform:
    """Base class for transforms that need to be applied to both image and mask"""
    
    def __call__(self, image: Image.Image, mask: Image.Image) -> Tuple[Image.Image, Image.Image]:
        raise NotImplementedError


class DualRandomHorizontalFlip(DualTransform):
    def __init__(self, p: float = 0.5):
        self.p = p
    
    def __call__(self, image: Image.Image, mask: Image.Image) -> Tuple[Image.Image, Image.Image]:
        if random.random() < self.p:
            image = TF.hflip(image)
            mask = TF.hflip(mask)
        return image, mask


class DualRandomVerticalFlip(DualTransform):
    def __init__(self, p: float = 0.5):
        self.p = p
    
    def __call__(self, image: Image.Image, mask: Image.Image) -> Tuple[Image.Image, Image.Image]:
        if random.random() < self.p:
            image = TF.vflip(image)
            mask = TF.vflip(mask)
        return image, mask


class DualRandomRotation(DualTransform):
    def __init__(self, degrees: Union[float, Tuple[float, float]] = 15):
        if isinstance(degrees, (int, float)):
            self.degrees = (-degrees, degrees)
        else:
            self.degrees = degrees
    
    def __call__(self, image: Image.Image, mask: Image.Image) -> Tuple[Image.Image, Image.Image]:
        angle = random.uniform(self.degrees[0], self.degrees[1])
        image = TF.rotate(image, angle, interpolation=Image.BILINEAR, fill=0)
        mask = TF.rotate(mask, angle, interpolation=Image.NEAREST, fill=0)
        return image, mask


class DualRandomCrop(DualTransform):
    def __init__(self, size: Tuple[int, int]):
        self.size = size
    
    def __call__(self, image: Image.Image, mask: Image.Image) -> Tuple[Image.Image, Image.Image]:
        i, j, h, w = transforms.RandomCrop.get_params(image, self.size)
        image = TF.crop(image, i, j, h, w)
        mask = TF.crop(mask, i, j, h, w)
        return image, mask


class DualCenterCrop(DualTransform):
    def __init__(self, size: Tuple[int, int]):
        self.size = size
    
    def __call__(self, image: Image.Image, mask: Image.Image) -> Tuple[Image.Image, Image.Image]:
        image = TF.center_crop(image, self.size)
        mask = TF.center_crop(mask, self.size)
        return image, mask


class DualResize(DualTransform):
    def __init__(self, size: Tuple[int, int]):
        self.size = size
    
    def __call__(self, image: Image.Image, mask: Image.Image) -> Tuple[Image.Image, Image.Image]:
        image = image.resize(self.size, Image.BILINEAR)
        mask = mask.resize(self.size, Image.NEAREST)
        return image, mask


class ColorJitter:
    """Color jittering for images only (not masks)"""
    def __init__(self, brightness: float = 0.2, contrast: float = 0.2, 
                 saturation: float = 0.2, hue: float = 0.1):
        self.transform = transforms.ColorJitter(brightness, contrast, saturation, hue)
    
    def __call__(self, image: Image.Image) -> Image.Image:
        return self.transform(image)


class SegmentationDataset(Dataset):
    def __init__(self, 
                 image_dir: str, 
                 mask_dir: str, 
                 image_size: Tuple[int, int] = (256, 256),
                 augmentations: Optional[List[DualTransform]] = None,
                 color_jitter: bool = True,
                 normalize: bool = True,
                 mask_threshold: float = 0.5,
                 file_extension: str = "auto"):
        """
        Dataset for segmentation tasks with augmentation support
        
        Args:
            image_dir: Directory containing input images
            mask_dir: Directory containing segmentation masks
            image_size: Target size for images and masks
            augmentations: List of dual transforms to apply
            color_jitter: Whether to apply color jittering to images
            normalize: Whether to normalize images to [-1, 1]
            mask_threshold: Threshold for binarizing masks (if needed)
            file_extension: File extension to look for ("auto", "jpg", "png", etc.)
        """
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.image_size = image_size
        self.augmentations = augmentations or []
        self.normalize = normalize
        self.mask_threshold = mask_threshold
        
        # Setup resize transform
        self.resize_transform = DualResize(image_size)
        
        # Setup color jittering
        self.color_jitter = ColorJitter() if color_jitter else None
        
        # Find image and mask files
        self.image_files, self.mask_files = self._find_files(file_extension)
        
        # Validate
        if len(self.image_files) != len(self.mask_files):
            raise ValueError(f"Number of images ({len(self.image_files)}) != number of masks ({len(self.mask_files)})")
        
        print(f"Found {len(self.image_files)} image-mask pairs")
    
    def _find_files(self, extension: str) -> Tuple[List[Path], List[Path]]:
        """Find and match image and mask files"""
        if extension == "auto":
            extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tiff"]
        else:
            extensions = [f"*.{extension}"]
        
        # Find all image files
        image_files = []
        for ext in extensions:
            image_files.extend(self.image_dir.glob(ext))
            image_files.extend(self.image_dir.glob(ext.upper()))
        
        # Find corresponding mask files
        mask_files = []
        for img_file in image_files:
            # Try different mask naming conventions
            possible_mask_names = [
                img_file.stem + ".png",  # Default: same name with .png
                img_file.stem + "_mask.png",
                img_file.stem + "_gt.png",
                img_file.stem + ".jpg",
                img_file.name,  # Exact same name
            ]
            
            mask_found = False
            for mask_name in possible_mask_names:
                mask_path = self.mask_dir / mask_name
                if mask_path.exists():
                    mask_files.append(mask_path)
                    mask_found = True
                    break
            
            if not mask_found:
                print(f"Warning: No mask found for {img_file.name}")
        
        # Sort both lists to ensure correspondence
        image_files = sorted([f for f in image_files if any(m.stem == f.stem for m in mask_files)])
        mask_files = sorted(mask_files)
        
        return image_files, mask_files
    
    def __len__(self) -> int:
        return len(self.image_files)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Load image and mask
        image = Image.open(self.image_files[idx]).convert('RGB')
        mask = Image.open(self.mask_files[idx]).convert('L')
        
        # Apply resize
        image, mask = self.resize_transform(image, mask)
        
        # Apply augmentations
        for transform in self.augmentations:
            image, mask = transform(image, mask)
        
        # Apply color jittering to image only
        if self.color_jitter is not None:
            image = self.color_jitter(image)
        
        # Convert to tensors
        image = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0
        mask = torch.from_numpy(np.array(mask)).unsqueeze(0).float() / 255.0
        
        # Normalize image to [-1, 1] range (diffusers standard)
        if self.normalize:
            image = image * 2.0 - 1.0
        
        # Threshold mask if needed
        if self.mask_threshold is not None:
            mask = (mask > self.mask_threshold).float()
        
        return image, mask
    
    def get_item_info(self, idx: int) -> Dict[str, Any]:
        """Get information about a specific item"""
        return {
            'image_path': str(self.image_files[idx]),
            'mask_path': str(self.mask_files[idx]),
            'image_name': self.image_files[idx].name,
            'mask_name': self.mask_files[idx].name
        }


def create_augmentations(mode: str = "medium") -> List[DualTransform]:
    """Create predefined augmentation pipelines"""
    
    if mode == "light":
        return [
            DualRandomHorizontalFlip(p=0.5),
        ]
    
    elif mode == "medium":
        return [
            DualRandomHorizontalFlip(p=0.5),
            DualRandomVerticalFlip(p=0.3),
            DualRandomRotation(degrees=15),
        ]
    
    elif mode == "heavy":
        return [
            DualRandomHorizontalFlip(p=0.5),
            DualRandomVerticalFlip(p=0.3),
            DualRandomRotation(degrees=30),
            DualRandomCrop(size=(224, 224)),  # Note: adjust based on your image_size
        ]
    
    else:
        raise ValueError(f"Unknown augmentation mode: {mode}")


def load_dataset(image_dir: str, 
                mask_dir: str, 
                batch_size: int = 8,
                image_size: Tuple[int, int] = (256, 256), 
                shuffle: bool = True,
                augmentation_mode: str = "medium",
                num_workers: int = 4,
                pin_memory: bool = True,
                val_split: float = 0.0) -> Union[DataLoader, Tuple[DataLoader, DataLoader]]:
    """
    Load dataset with optional validation split
    
    Args:
        image_dir: Directory containing images
        mask_dir: Directory containing masks
        batch_size: Batch size for training
        image_size: Target image size
        shuffle: Whether to shuffle training data
        augmentation_mode: "light", "medium", "heavy", or "none"
        num_workers: Number of worker processes
        pin_memory: Whether to pin memory for faster GPU transfer
        val_split: Fraction of data to use for validation (0.0 = no validation split)
    
    Returns:
        DataLoader or tuple of (train_loader, val_loader)
    """
    
    # Create augmentations
    if augmentation_mode == "none":
        augmentations = []
    else:
        augmentations = create_augmentations(augmentation_mode)
    
    # Create full dataset
    full_dataset = SegmentationDataset(
        image_dir=image_dir,
        mask_dir=mask_dir,
        image_size=image_size,
        augmentations=augmentations,
        color_jitter=True
    )
    
    if val_split > 0:
        # Split dataset
        val_size = int(len(full_dataset) * val_split)
        train_size = len(full_dataset) - val_size
        
        train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
        
        # Create validation dataset without augmentations
        val_dataset_clean = SegmentationDataset(
            image_dir=image_dir,
            mask_dir=mask_dir,
            image_size=image_size,
            augmentations=[],  # No augmentations for validation
            color_jitter=False
        )
        
        # Use subset of clean dataset for validation
        val_indices = val_dataset.indices
        val_dataset_subset = torch.utils.data.Subset(val_dataset_clean, val_indices)
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=shuffle, 
            num_workers=num_workers, 
            pin_memory=pin_memory
        )
        
        val_loader = DataLoader(
            val_dataset_subset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory
        )
        
        return train_loader, val_loader
    
    else:
        # Single dataset
        return DataLoader(
            full_dataset, 
            batch_size=batch_size, 
            shuffle=shuffle, 
            num_workers=num_workers, 
            pin_memory=pin_memory
        )


def preprocess_image(image_path: str, image_size: Tuple[int, int] = (256, 256)) -> torch.Tensor:
    image = Image.open(image_path).convert('RGB')
    image = image.resize(image_size, Image.BILINEAR)
    image = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0
    image = image * 2.0 - 1.0  # Normalize to [-1, 1] (diffusers standard)
    return image.unsqueeze(0)


def create_synthetic_data(batch_size: int = 4, image_size: Tuple[int, int] = (256, 256)) -> Tuple[torch.Tensor, torch.Tensor]:
    """Create synthetic data for testing"""
    # Random RGB images in [-1, 1] range (diffusers standard)
    images = torch.rand(batch_size, 3, *image_size) * 2.0 - 1.0
    
    # Random binary masks with some structure
    masks = torch.zeros(batch_size, 1, *image_size)
    for i in range(batch_size):
        # Create circular masks with random centers and sizes
        center_x = torch.randint(image_size[1] // 4, 3 * image_size[1] // 4, (1,))
        center_y = torch.randint(image_size[0] // 4, 3 * image_size[0] // 4, (1,))
        radius = torch.randint(20, min(image_size) // 4, (1,))
        
        y, x = torch.meshgrid(torch.arange(image_size[0]), torch.arange(image_size[1]), indexing='ij')
        mask = ((x - center_x) ** 2 + (y - center_y) ** 2) <= radius ** 2
        masks[i, 0] = mask.float()
    
    return images, masks