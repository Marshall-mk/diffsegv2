# Diffusion Segmentation

A PyTorch implementation of a diffusion-based image segmentation model that supports both traditional Gaussian diffusion and novel morphological diffusion for generating segmentation masks.

## Features

- **Dual Diffusion Modes**: 
  - **Gaussian Diffusion**: Traditional DDPM with noise-based forward process
  - **Morphological Diffusion**: Novel approach using morphological operations instead of Gaussian noise
- **Multiple UNet Backends**: 
  - HuggingFace Diffusers UNet2DModel
  - HuggingFace Diffusers UNet2DConditionModel
- **Morphological Operations**: Differentiable dilation, erosion, opening, and closing operations
- **Flexible architecture**: Supports different image sizes and number of classes
- **Advanced Loss Functions**: MSE, Dice, and morphological boundary-aware losses
- **Training utilities**: Complete training pipeline with checkpointing and visualization
- **Inference tools**: Batch processing, interactive mode, and single image inference
- **Synthetic data support**: Built-in synthetic data generation for testing

## Repository Structure

```
├── src/
│   ├── models/
│   │   ├── __init__.py
│   │   ├── blocks.py          # Neural network building blocks
│   │   ├── unet.py           # Custom UNet architecture
│   │   ├── diffusion_model.py # Main diffusion model with dual modes
│   │   └── morphological_ops.py # Morphological operations and losses
│   ├── training/             # Training utilities (future)
│   └── inference/            # Inference utilities (future)
├── utils/
│   ├── __init__.py
│   ├── data_utils.py         # Data loading and preprocessing
│   └── visualization.py     # Visualization utilities
├── train.py                  # Training script
├── inference.py              # Inference script
├── demo.py                   # Demo and testing script
├── test_unet_switch.py       # Test script for UNet switching
├── test_morphological_diffusion.py # Test script for morphological diffusion
├── requirements.txt          # Python dependencies
└── README.md                # This file
```

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd diffusion-segmentation
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Quick Start

### Demo
Run the demo to see the model in action with synthetic data:
```bash
python demo.py
```

### Training

#### Gaussian Diffusion (Traditional)
Train with Gaussian noise-based diffusion:
```bash
# With diffusers UNet2DModel (default)
python train.py --synthetic --epochs 50 --batch-size 8 --diffusion-type gaussian

# With diffusers UNet2DConditionModel
python train.py --synthetic --epochs 50 --unet-type diffusers_2d_cond --diffusion-type gaussian
```

#### Morphological Diffusion (Novel)
Train with morphological operations instead of noise:
```bash
# Dilation-based (starts from black image)
python train.py --synthetic --epochs 50 --diffusion-type morphological --morph-type dilation --use-morph-loss

# Erosion-based (starts from white image)
python train.py --synthetic --epochs 50 --diffusion-type morphological --morph-type erosion --use-morph-loss

# Mixed morphological operations
python train.py --synthetic --epochs 50 --diffusion-type morphological --morph-type mixed
```

#### Train with Real Data
```bash
python train.py --image-dir /path/to/images --mask-dir /path/to/masks --epochs 100 --diffusion-type morphological --morph-type dilation
```

### Inference

#### Gaussian Diffusion Inference
```bash
python inference.py --checkpoint /path/to/checkpoint.pth --input /path/to/image.jpg --diffusion-type gaussian
```

#### Morphological Diffusion Inference
```bash
# Dilation-based inference
python inference.py --checkpoint /path/to/checkpoint.pth --input /path/to/image.jpg --diffusion-type morphological --morph-type dilation

# Erosion-based inference
python inference.py --checkpoint /path/to/checkpoint.pth --input /path/to/image.jpg --diffusion-type morphological --morph-type erosion
```

#### Batch Processing
```bash
python inference.py --checkpoint /path/to/checkpoint.pth --input /path/to/images/ --batch --output /path/to/results/ --diffusion-type morphological
```

#### Interactive Mode
```bash
python inference.py --checkpoint /path/to/checkpoint.pth --interactive --diffusion-type morphological --morph-type dilation
```

## Model Architecture

### Dual Diffusion Modes

#### Gaussian Diffusion (Traditional DDPM)
1. **Forward Process**: Gradually adds Gaussian noise to ground truth masks
2. **Reverse Process**: Denoises from pure noise to clean segmentation masks
3. **Training**: Model learns to predict added noise at each timestep

#### Morphological Diffusion (Novel Approach)
1. **Forward Process**: Applies morphological operations (dilation/erosion) instead of noise
2. **Reverse Process**: Reconstructs clean masks from morphologically degraded versions
3. **Training**: Model learns to predict original clean masks
4. **Starting Points**:
   - **Dilation**: Starts from black image (all zeros)
   - **Erosion**: Starts from white image (all ones)
   - **Mixed**: Randomly alternates between operations

### UNet Architecture Options

#### HuggingFace Diffusers UNet2DModel
- Pre-built 2D UNet from diffusers library
- Configurable architecture with attention blocks
- Option to load pretrained weights

#### HuggingFace Diffusers UNet2DConditionModel
- Conditional 2D UNet with cross-attention
- Support for additional conditioning (can be used without conditioning)
- More advanced architecture for complex tasks

### Morphological Operations
- **Soft Dilation**: Differentiable max-pooling with temperature scaling
- **Soft Erosion**: Differentiable min-pooling with temperature scaling
- **Soft Opening**: Erosion followed by dilation
- **Soft Closing**: Dilation followed by erosion
- **Learnable Kernels**: Convolutional morphology with trainable structural elements

## Training

### Command Line Arguments

#### Core Training Parameters
- `--data-dir`: Path to dataset directory
- `--image-dir`: Path to images directory  
- `--mask-dir`: Path to masks directory
- `--output-dir`: Output directory (default: ./outputs)
- `--batch-size`: Batch size (default: 8)
- `--epochs`: Number of epochs (default: 100)
- `--lr`: Learning rate (default: 1e-4)
- `--image-size`: Image size (default: 256)
- `--timesteps`: Number of diffusion timesteps (default: 1000)
- `--synthetic`: Use synthetic data for testing
- `--wandb`: Enable Weights & Biases logging
- `--resume`: Path to checkpoint to resume from

#### UNet Architecture Options
- `--unet-type`: Type of UNet to use (default: diffusers_2d)
  - `diffusers_2d`: HuggingFace UNet2DModel
  - `diffusers_2d_cond`: HuggingFace UNet2DConditionModel
- `--pretrained-model`: Path or name of pretrained diffusers model

#### Diffusion Type Options
- `--diffusion-type`: Type of diffusion process (default: gaussian)
  - `gaussian`: Traditional DDPM with Gaussian noise
  - `morphological`: Novel morphological operations approach

#### Morphological Diffusion Parameters
- `--morph-type`: Type of morphological operation (default: dilation)
  - `dilation`: Dilation operations (starts from black)
  - `erosion`: Erosion operations (starts from white)
  - `mixed`: Randomly mixed operations
- `--morph-kernel-size`: Size of morphological kernel (default: 3)
- `--morph-schedule`: Schedule type for morphological intensity (default: linear)
  - `linear`: Linear intensity increase
  - `cosine`: Cosine-based intensity schedule  
  - `quadratic`: Quadratic intensity increase
- `--use-morph-loss`: Use morphological loss instead of MSE (combines L1, Dice, and boundary losses)

### Example Training Commands

#### Basic Examples
```bash
# Quick test with synthetic data (Gaussian)
python train.py --synthetic --epochs 10 --batch-size 4 --diffusion-type gaussian

# Quick test with morphological diffusion
python train.py --synthetic --epochs 10 --batch-size 4 --diffusion-type morphological --morph-type dilation --use-morph-loss
```

#### Advanced Examples
```bash
# Full training with real data using morphological diffusion
python train.py --image-dir ./data/images --mask-dir ./data/masks --epochs 100 --diffusion-type morphological --morph-type dilation --use-morph-loss --wandb

# Training with HuggingFace Diffusers UNet and morphological diffusion
python train.py --synthetic --epochs 50 --unet-type diffusers_2d --diffusion-type morphological --morph-type erosion --morph-kernel-size 5

# Mixed morphological operations with cosine schedule
python train.py --image-dir ./data/images --mask-dir ./data/masks --epochs 100 --diffusion-type morphological --morph-type mixed --morph-schedule cosine --use-morph-loss

# Resume training from checkpoint
python train.py --resume ./outputs/checkpoint_epoch_50.pth --epochs 100 --diffusion-type morphological --morph-type dilation
```

#### Comparison Training
```bash
# Train Gaussian baseline
python train.py --image-dir ./data/images --mask-dir ./data/masks --epochs 100 --diffusion-type gaussian --output-dir ./outputs_gaussian

# Train morphological version for comparison
python train.py --image-dir ./data/images --mask-dir ./data/masks --epochs 100 --diffusion-type morphological --morph-type dilation --use-morph-loss --output-dir ./outputs_morphological
```

## Inference

### Command Line Arguments

#### Core Inference Parameters
- `--checkpoint`: Path to model checkpoint (required)
- `--input`: Input image path or directory
- `--output`: Output directory (default: ./inference_results)
- `--batch`: Batch inference mode for directories
- `--interactive`: Interactive inference mode
- `--steps`: Number of inference steps (default: 50)
- `--image-size`: Image size for processing (default: 256)
- `--timesteps`: Number of diffusion timesteps (default: 1000)
- `--visualize-process`: Visualize the inference process
- `--no-vis`: Skip saving visualizations in batch mode

#### UNet Architecture Parameters (must match training)
- `--unet-type`: Type of UNet used during training (default: diffusers_2d)
- `--pretrained-model`: Path or name of pretrained diffusers model (if used during training)

#### Diffusion Type Parameters (must match training)
- `--diffusion-type`: Type of diffusion process (default: gaussian)
- `--morph-type`: Type of morphological operation (default: dilation)
- `--morph-kernel-size`: Size of morphological kernel (default: 3)
- `--morph-schedule`: Schedule type for morphological intensity (default: linear)

### Example Inference Commands

#### Gaussian Diffusion Inference
```bash
# Single image with Gaussian diffusion
python inference.py --checkpoint gaussian_model.pth --input image.jpg --diffusion-type gaussian

# Batch processing with diffusers UNet2DModel
python inference.py --checkpoint gaussian_model.pth --input ./images/ --batch --output ./results/ --diffusion-type gaussian --unet-type diffusers_2d

# With diffusers UNet2DConditionModel
python inference.py --checkpoint diffusers_model.pth --input image.jpg --diffusion-type gaussian --unet-type diffusers_2d_cond
```

#### Morphological Diffusion Inference
```bash
# Single image with dilation-based morphological diffusion
python inference.py --checkpoint morph_model.pth --input image.jpg --diffusion-type morphological --morph-type dilation

# Erosion-based inference
python inference.py --checkpoint morph_erosion_model.pth --input image.jpg --diffusion-type morphological --morph-type erosion

# Batch processing with morphological diffusion
python inference.py --checkpoint morph_model.pth --input ./images/ --batch --output ./results/ --diffusion-type morphological --morph-type dilation --morph-kernel-size 5

# Interactive morphological inference
python inference.py --checkpoint morph_model.pth --interactive --diffusion-type morphological --morph-type dilation
```

#### Advanced Inference Options
```bash
# Visualize morphological inference process
python inference.py --checkpoint morph_model.pth --input image.jpg --visualize-process --diffusion-type morphological --morph-type dilation

# Fast inference with fewer steps
python inference.py --checkpoint morph_model.pth --input image.jpg --steps 20 --diffusion-type morphological --morph-type dilation

# High-resolution inference
python inference.py --checkpoint morph_model.pth --input image.jpg --image-size 512 --diffusion-type morphological --morph-type dilation
```

## Custom Dataset Setup

### Data Format Requirements

The model expects:
- **Images**: RGB images in common formats (JPG, JPEG, PNG, BMP, TIFF)
- **Masks**: Grayscale masks (0-255, where 255 represents foreground/object)
- **Image-Mask Correspondence**: Each image must have a corresponding mask with matching filename

### Directory Structure Options

#### Option 1: Separate Directories (Recommended)
```
your_dataset/
├── images/
│   ├── image001.jpg
│   ├── image002.png
│   └── ...
└── masks/
    ├── image001.png      # Same name as image
    ├── image002.png
    └── ...
```

#### Option 2: Alternative Mask Naming
```
your_dataset/
├── images/
│   ├── photo_001.jpg
│   ├── photo_002.jpg
│   └── ...
└── masks/
    ├── photo_001_mask.png    # With "_mask" suffix
    ├── photo_002_gt.png      # With "_gt" suffix  
    └── ...
```

#### Option 3: Train/Val Pre-split
```
your_dataset/
├── train/
│   ├── images/
│   │   ├── train_001.jpg
│   │   └── ...
│   └── masks/
│       ├── train_001.png
│       └── ...
└── val/
    ├── images/
    │   ├── val_001.jpg
    │   └── ...
    └── masks/
        ├── val_001.png
        └── ...
```

### Data Preparation Tools

#### Analyze Your Dataset
```bash
# Get detailed statistics about your dataset
python prepare_data.py --action analyze --image-dir ./data/images --mask-dir ./data/masks --output-dir ./analysis

# This will show:
# - Number of images and masks
# - Image size statistics
# - Common issues and recommendations
```

#### Validate Dataset
```bash
# Check for common issues
python prepare_data.py --action validate --image-dir ./data/images --mask-dir ./data/masks

# Automatically fix simple issues
python prepare_data.py --action validate --image-dir ./data/images --mask-dir ./data/masks --fix-issues
```

#### Organize Dataset
```bash
# Automatically split into train/val
python prepare_data.py --action organize --source-dir ./raw_data --output-dir ./organized_data --train-split 0.8
```

#### Create Sample Dataset
```bash
# Generate synthetic data for testing
python prepare_data.py --action create-sample --output-dir ./sample_data --num-samples 50
```

### Data Loading & Augmentation

#### Basic Loading
```python
from utils.data_utils import load_dataset

# Simple loading
train_loader = load_dataset(
    image_dir="./data/images",
    mask_dir="./data/masks",
    batch_size=8,
    image_size=(256, 256)
)
```

#### With Validation Split
```python
# Automatic train/val split
train_loader, val_loader = load_dataset(
    image_dir="./data/images",
    mask_dir="./data/masks",
    batch_size=8,
    val_split=0.2,  # 20% for validation
    augmentation_mode="medium"
)
```

#### Advanced Configuration
```python
# Full control over data loading
train_loader = load_dataset(
    image_dir="./data/train/images",
    mask_dir="./data/train/masks",
    batch_size=16,
    image_size=(512, 512),
    augmentation_mode="heavy",  # "light", "medium", "heavy", "none"
    num_workers=8,
    pin_memory=True,
    shuffle=True
)
```

### Data Augmentation Strategies

The dataset includes several augmentation modes:

#### Light Augmentation
- Horizontal flipping (50% probability)
- Best for: Small datasets, quick experimentation

#### Medium Augmentation (Default)
- Horizontal flipping (50% probability)
- Vertical flipping (30% probability)
- Random rotation (±15 degrees)
- Color jittering on images only
- Best for: Most use cases

#### Heavy Augmentation
- All medium augmentations
- Stronger rotation (±30 degrees)
- Random cropping
- Best for: Large datasets, challenging tasks

#### Custom Augmentations
```python
from utils.data_utils import (
    SegmentationDataset, 
    DualRandomHorizontalFlip, 
    DualRandomRotation,
    ColorJitter
)

# Create custom augmentation pipeline
custom_augmentations = [
    DualRandomHorizontalFlip(p=0.7),
    DualRandomRotation(degrees=20),
    # Add more as needed
]

dataset = SegmentationDataset(
    image_dir="./data/images",
    mask_dir="./data/masks",
    augmentations=custom_augmentations,
    color_jitter=True
)
```

### Training with Custom Data

#### Basic Training
```bash
# Train with your dataset
python train.py \
    --image-dir ./data/images \
    --mask-dir ./data/masks \
    --epochs 100 \
    --batch-size 8 \
    --augmentation-mode medium
```

#### With Validation
```bash
# Train with automatic validation split
python train.py \
    --image-dir ./data/images \
    --mask-dir ./data/masks \
    --epochs 100 \
    --batch-size 8 \
    --val-split 0.2 \
    --augmentation-mode medium \
    --wandb  # Optional: for experiment tracking
```

#### Multi-size Training
```bash
# Train with larger images (requires more GPU memory)
python train.py \
    --image-dir ./data/images \
    --mask-dir ./data/masks \
    --image-size 512 \
    --batch-size 4 \
    --epochs 100
```

### Common Data Issues & Solutions

#### Issue: Mismatched Image/Mask Count
```bash
# Diagnosis
python prepare_data.py --action validate --image-dir ./data/images --mask-dir ./data/masks

# Solution: Check naming conventions and file extensions
```

#### Issue: Different Image Sizes
The dataset automatically resizes images, but for better results:
- Analyze your data: `python prepare_data.py --action analyze`
- Choose image_size based on your data's common size
- Consider aspect ratio preservation

#### Issue: Mask Values Not Binary
```python
# The dataset automatically handles thresholding
dataset = SegmentationDataset(
    image_dir="./data/images", 
    mask_dir="./data/masks",
    mask_threshold=0.5  # Adjust as needed
)
```

#### Issue: Memory Errors
- Reduce batch_size
- Reduce image_size
- Reduce num_workers
- Use pin_memory=False

### Dataset Recommendations

#### For Small Datasets (< 100 samples)
- Use "light" augmentation
- Larger learning rate
- More epochs
- Consider synthetic data mixing

#### For Medium Datasets (100-1000 samples)
- Use "medium" augmentation
- Standard parameters
- Validation split recommended

#### For Large Datasets (> 1000 samples)
- Use "heavy" augmentation
- Consider larger image sizes
- Multi-GPU training

### File Naming Conventions

The dataset automatically handles these naming patterns:

**Supported mask naming:**
- `image.jpg` → `image.png`
- `image.jpg` → `image_mask.png`
- `image.jpg` → `image_gt.png`
- `image.jpg` → `image.jpg` (same extension)

**Example valid pairs:**
```
photo_001.jpg ↔ photo_001.png
IMG_123.png   ↔ IMG_123_mask.png
data_05.tiff  ↔ data_05_gt.png
```

## Customization

### Model Parameters
Modify the model architecture in `src/models/diffusion_model.py`:
- `in_channels`: Number of input image channels (default: 3 for RGB)
- `num_classes`: Number of segmentation classes (default: 1 for binary)
- `timesteps`: Number of diffusion timesteps (default: 1000)

### Training Parameters
- Learning rate, batch size, and other hyperparameters can be adjusted via command line
- Loss function can be modified in the training script
- Data augmentation can be added to the dataset class

## Visualization

The package includes comprehensive visualization tools:
- Training loss curves
- Segmentation results comparison
- Forward diffusion process visualization
- Reverse diffusion process step-by-step
- Interactive plotting with matplotlib

## Novel Morphological Diffusion Approach

### Key Innovation

This implementation introduces a novel alternative to traditional Gaussian diffusion for image segmentation. Instead of adding random noise, the forward process applies **morphological operations** - fundamental tools in computer vision and image processing.

### Why Morphological Diffusion?

1. **Task-Appropriate**: Morphological operations are naturally suited for segmentation tasks
2. **Interpretable**: Each step has clear geometric meaning (dilation expands regions, erosion shrinks them)
3. **Controllable**: Starting from deterministic states (black/white) rather than random noise
4. **Differentiable**: All operations are fully differentiable and end-to-end trainable

### How It Works

#### Forward Process (Training)
1. **Start**: Clean ground truth segmentation mask
2. **Degrade**: Apply morphological operations with increasing intensity
3. **Learn**: Model learns to predict original mask from degraded version

#### Reverse Process (Inference)
1. **Start**: Deterministic initial state
   - **Dilation mode**: All-black image (zeros)
   - **Erosion mode**: All-white image (ones)
2. **Reconstruct**: Model iteratively predicts cleaner masks
3. **Result**: Final segmentation mask

### Morphological Operations

#### Soft Dilation
- **Effect**: Expands object boundaries
- **Implementation**: Temperature-scaled soft maximum over neighborhoods
- **Use case**: Growing regions, filling gaps

#### Soft Erosion  
- **Effect**: Shrinks object boundaries
- **Implementation**: Temperature-scaled soft minimum over neighborhoods
- **Use case**: Removing noise, separating connected objects

#### Composite Operations
- **Opening**: Erosion followed by dilation (removes small objects)
- **Closing**: Dilation followed by erosion (fills small holes)
- **Mixed**: Random combination during training

### Advantages Over Gaussian Diffusion

| Aspect | Gaussian Diffusion | Morphological Diffusion |
|--------|-------------------|------------------------|
| **Starting Point** | Random noise | Deterministic (black/white) |
| **Operations** | Add/remove noise | Geometric transformations |
| **Interpretability** | Low (noise is random) | High (geometric meaning) |
| **Task Relevance** | Generic | Segmentation-specific |
| **Control** | Stochastic process | Deterministic degradation |
| **Training Target** | Predict noise | Predict clean mask |

### Performance Characteristics

- **Convergence**: Often faster convergence due to task-specific operations
- **Quality**: Better preservation of geometric structures
- **Robustness**: Less sensitive to hyperparameters
- **Efficiency**: Fewer inference steps needed for good results

### When to Use Each Mode

#### Choose Gaussian Diffusion When:
- Working with general image generation tasks
- Need stochastic diversity in outputs
- Following established DDPM research
- Comparing against standard baselines

#### Choose Morphological Diffusion When:
- Primary task is segmentation
- Need interpretable generation process
- Working with geometric/structured data
- Want faster, more controlled inference


## Dependencies

- PyTorch >= 2.0.0
- torchvision >= 0.15.0
- numpy >= 1.21.0
- matplotlib >= 3.5.0
- seaborn >= 0.11.0
- Pillow >= 9.0.0
- tqdm >= 4.64.0
- opencv-python >= 4.5.0
- diffusers >= 0.30.0 (for HuggingFace UNet models)
- wandb >= 0.13.0 (optional, for experiment tracking)

## License

[Add your license information here]

## Citation

[Add citation information if this is based on published research]

## Contributing

[Add contribution guidelines if applicable]