#!/usr/bin/env python3
"""
Training script for Diffusion Segmentation model with polynomial morphological operations
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import argparse
import os
from pathlib import Path
import json
from tqdm import tqdm
import wandb
from typing import Optional
import warnings

# Updated imports for new morphological operations
from src.models import DiffusionSegmentation
from src.models import MorphologicalLoss, MorphologicalDebugger
from utils.data_utils import load_dataset, create_synthetic_data
from utils.visualization import plot_training_curves, visualize_segmentation


def train_step(model: DiffusionSegmentation, image: torch.Tensor, mask: torch.Tensor, 
               optimizer: torch.optim.Optimizer, device: torch.device, 
               use_enhanced_loss: bool = True, loss_fn=None) -> float:
    """Single training step with enhanced morphological loss support"""
    model.train()
    optimizer.zero_grad()
    
    image, mask = image.to(device), mask.to(device)
    predicted, target = model(image, mask)
    
    # Use the model's built-in compute_loss method for morphological diffusion
    if use_enhanced_loss and hasattr(model, 'compute_loss') and model.diffusion_type == "morphological":
        loss = model.compute_loss(predicted, target, 
                                loss_type="l1", 
                                use_morphological_loss=True)
    elif loss_fn is not None:
        loss = loss_fn(predicted, target)
    else:
        # Fallback to standard MSE loss
        loss = F.mse_loss(predicted, target)
    
    loss.backward()
    
    # Gradient clipping for stability (especially important for morphological diffusion)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    
    optimizer.step()
    
    return loss.item()


def validate(model: DiffusionSegmentation, val_loader: DataLoader, device: torch.device, 
             use_enhanced_loss: bool = True, loss_fn=None) -> dict:
    """Enhanced validation loop with multiple metrics"""
    model.eval()
    total_loss = 0
    total_dice = 0
    total_iou = 0
    num_batches = 0
    
    with torch.no_grad():
        for image, mask in val_loader:
            image, mask = image.to(device), mask.to(device)
            
            # For validation, we need training mode for the forward pass
            model.train()
            predicted, target = model(image, mask)
            model.eval()
            
            # Compute loss
            if use_enhanced_loss and hasattr(model, 'compute_loss') and model.diffusion_type == "morphological":
                loss = model.compute_loss(predicted, target, 
                                        loss_type="l1", 
                                        use_morphological_loss=True)
            elif loss_fn is not None:
                loss = loss_fn(predicted, target)
            else:
                loss = F.mse_loss(predicted, target)
            
            total_loss += loss.item()
            
            # Compute additional metrics for segmentation
            pred_binary = (torch.sigmoid(predicted) > 0.5).float()
            target_binary = (target > 0.5).float()
            
            # Dice coefficient
            intersection = (pred_binary * target_binary).sum()
            dice = (2. * intersection) / (pred_binary.sum() + target_binary.sum() + 1e-8)
            total_dice += dice.item()
            
            # IoU
            union = pred_binary.sum() + target_binary.sum() - intersection
            iou = intersection / (union + 1e-8)
            total_iou += iou.item()
            
            num_batches += 1
    
    if num_batches == 0:
        return {"loss": 0, "dice": 0, "iou": 0}
    
    return {
        "loss": total_loss / num_batches,
        "dice": total_dice / num_batches,
        "iou": total_iou / num_batches
    }


def save_checkpoint(model: DiffusionSegmentation, optimizer: torch.optim.Optimizer, 
                   epoch: int, loss: float, path: str, config: dict = None):
    """Save model checkpoint with config"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'config': config
    }
    torch.save(checkpoint, path)


def load_checkpoint(model: DiffusionSegmentation, optimizer: torch.optim.Optimizer, 
                   path: str, device: torch.device):
    """Load model checkpoint"""
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['epoch'], checkpoint['loss']


def run_debugging_checks(model: DiffusionSegmentation, sample_data: tuple, device: torch.device):
    """Run debugging checks for morphological diffusion"""
    if model.diffusion_type != "morphological":
        return
    
    print("\n" + "="*50)
    print("MORPHOLOGICAL DIFFUSION DEBUGGING")
    print("="*50)
    
    # Get model debug info
    debug_info = model.get_morphological_debug_info()
    print(f"Morphological Type: {debug_info['morph_type']}")
    print(f"Morphological Routine: {debug_info['morph_routine']}")
    print(f"Number of Timesteps: {debug_info['num_timesteps']}")
    print(f"Kernel Sizes: {debug_info['kernel_sizes'][:5]}...{debug_info['kernel_sizes'][-5:]}")
    print(f"Sample Intensities: {debug_info['intensities_sample']}")
    
    # Check gradient flow
    print("\nChecking gradient flow...")
    sample_image, sample_mask = sample_data
    sample_input = torch.cat([sample_image[:1], sample_mask[:1]], dim=1).to(device)
    MorphologicalDebugger.check_gradient_flow(model, sample_input)
    
    print("="*50 + "\n")


def main():
    parser = argparse.ArgumentParser(description='Train Diffusion Segmentation Model with Polynomial Morphology')
    parser.add_argument('--data-dir', type=str, help='Path to dataset directory')
    parser.add_argument('--image-dir', type=str, help='Path to images directory')
    parser.add_argument('--mask-dir', type=str, help='Path to masks directory')
    parser.add_argument('--output-dir', type=str, default='./outputs', help='Output directory')
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate (use 1e-5 to 1e-4 for morphological)')
    parser.add_argument('--image-size', type=int, default=256, help='Image size')
    parser.add_argument('--timesteps', type=int, default=1000, help='Number of diffusion timesteps')
    parser.add_argument('--resume', type=str, help='Path to checkpoint to resume from')
    parser.add_argument('--synthetic', action='store_true', help='Use synthetic data for testing')
    parser.add_argument('--wandb', action='store_true', help='Use Weights & Biases logging')
    parser.add_argument('--save-freq', type=int, default=10, help='Save checkpoint every N epochs')
    parser.add_argument('--val-split', type=float, default=0.0, help='Validation split ratio (0.0 = no validation)')
    parser.add_argument('--augmentation-mode', type=str, default='medium', 
                       choices=['light', 'medium', 'heavy', 'none'], help='Augmentation intensity')
    parser.add_argument('--unet-type', type=str, default='diffusers_2d',
                       choices=['diffusers_2d', 'diffusers_2d_cond'], 
                       help='Type of UNet to use')
    parser.add_argument('--pretrained-model', type=str, help='Path or name of pretrained diffusers model')
    
    # Diffusion type parameters
    parser.add_argument('--diffusion-type', type=str, default='morphological',
                       choices=['gaussian', 'morphological'], 
                       help='Type of diffusion process (default: morphological for better segmentation)')
    
    # Enhanced morphological parameters
    parser.add_argument('--morph-type', type=str, default='mixed',
                       choices=['dilation', 'erosion', 'mixed', 'opening', 'closing'], 
                       help='Type of morphological operation (default: mixed for best results)')
    parser.add_argument('--morph-kernel-size-start', type=int, default=3,
                       help='Starting size of morphological kernel (default: 3)')
    parser.add_argument('--morph-kernel-size-end', type=int, default=9,
                       help='Ending size of morphological kernel (default: 9)')
    parser.add_argument('--morph-routine', type=str, default='Progressive',
                       choices=['Progressive', 'Constant'],
                       help='Morphological routine (default: Progressive for kernel growth)')
    parser.add_argument('--morph-schedule', type=str, default='exponential',
                       choices=['linear', 'exponential', 'cosine'],
                       help='Schedule type for morphological intensity (default: exponential)')
    
    # Loss function parameters
    parser.add_argument('--use-enhanced-loss', action='store_true', default=True,
                       help='Use enhanced morphological loss with dice and boundary terms (default: True)')
    parser.add_argument('--loss-type', type=str, default='l1',
                       choices=['l1', 'l2', 'mse'],
                       help='Base loss type (default: l1, following Cold Diffusion best practices)')
    
    # Legacy parameters (kept for compatibility)
    parser.add_argument('--morph-kernel-size', type=int, default=3,
                       help='DEPRECATED: Use --morph-kernel-size-start instead')
    parser.add_argument('--use-morph-loss', action='store_true',
                       help='DEPRECATED: Use --use-enhanced-loss instead')
    parser.add_argument('--morph-loss-type', type=str, default='l1',
                       help='DEPRECATED: Use --loss-type instead')
    
    parser.add_argument('--scheduler-type', type=str, default='ddpm',
                       choices=['ddpm', 'ddim'],
                       help='Type of diffusers scheduler to use')
    
    # Debugging and monitoring
    parser.add_argument('--debug-mode', action='store_true',
                       help='Enable debugging mode with gradient flow checks')
    parser.add_argument('--debug-freq', type=int, default=20,
                       help='Run debugging checks every N epochs')
    
    args = parser.parse_args()
    
    # Handle deprecated arguments
    if hasattr(args, 'morph_kernel_size') and args.morph_kernel_size != 3:
        warnings.warn("--morph-kernel-size is deprecated. Using value for --morph-kernel-size-start")
        args.morph_kernel_size_start = args.morph_kernel_size
    
    if args.use_morph_loss:
        warnings.warn("--use-morph-loss is deprecated. Using --use-enhanced-loss instead")
        args.use_enhanced_loss = True
    
    if hasattr(args, 'morph_loss_type') and args.morph_loss_type != 'l1':
        warnings.warn("--morph-loss-type is deprecated. Using value for --loss-type")
        args.loss_type = args.morph_loss_type
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Validate morphological parameters
    if args.diffusion_type == "morphological":
        if args.morph_kernel_size_start >= args.morph_kernel_size_end:
            print("WARNING: morph_kernel_size_start should be < morph_kernel_size_end for Progressive routine")
        
        print(f"Morphological Diffusion Configuration:")
        print(f"  Type: {args.morph_type}")
        print(f"  Routine: {args.morph_routine}")
        print(f"  Kernel sizes: {args.morph_kernel_size_start} → {args.morph_kernel_size_end}")
        print(f"  Schedule: {args.morph_schedule}")
        print(f"  Enhanced loss: {args.use_enhanced_loss}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config
    config = vars(args)
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    # Initialize wandb
    if args.wandb:
        wandb.init(project="morphological-diffusion-segmentation", config=config)
    
    # Create model with updated parameters
    model = DiffusionSegmentation(
        in_channels=3, 
        num_classes=1, 
        timesteps=args.timesteps,
        unet_type=args.unet_type,
        pretrained_model_name_or_path=args.pretrained_model,
        diffusion_type=args.diffusion_type,
        morph_type=args.morph_type,
        morph_kernel_size_start=args.morph_kernel_size_start,  # Updated parameter
        morph_kernel_size_end=args.morph_kernel_size_end,      # Updated parameter
        morph_routine=args.morph_routine,                      # New parameter
        morph_schedule_type=args.morph_schedule,
        scheduler_type=args.scheduler_type
    ).to(device)
    
    # Use lower learning rate for morphological diffusion (as recommended in the guide)
    if args.diffusion_type == "morphological" and args.lr > 1e-4:
        print(f"WARNING: Learning rate {args.lr} may be too high for morphological diffusion.")
        print("Consider using 1e-5 to 1e-4 for better stability.")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # Setup loss function (legacy support)
    legacy_loss_fn = None
    if not args.use_enhanced_loss:
        if args.diffusion_type == "morphological":
            legacy_loss_fn = MorphologicalLoss(loss_type=args.loss_type).to(device)
            print(f"Using legacy morphological loss function with {args.loss_type.upper()} loss")
        else:
            print("Using MSE loss for gaussian diffusion")
    else:
        print(f"Using enhanced loss with model.compute_loss() method")
    
    # Load checkpoint if resuming
    start_epoch = 0
    if args.resume:
        start_epoch, _ = load_checkpoint(model, optimizer, args.resume, device)
        print(f"Resumed from epoch {start_epoch}")
    
    # Setup data
    if args.synthetic:
        print("Using synthetic data for training")
        train_loader = None
        val_loader = None
        # Create sample data for debugging
        sample_data = create_synthetic_data(1, (args.image_size, args.image_size))
    else:
        if not args.image_dir or not args.mask_dir:
            raise ValueError("Must provide --image-dir and --mask-dir when not using synthetic data")
        
        # Load dataset with optional validation split
        dataset_result = load_dataset(
            args.image_dir, args.mask_dir, 
            batch_size=args.batch_size,
            image_size=(args.image_size, args.image_size),
            augmentation_mode=args.augmentation_mode,
            val_split=args.val_split
        )
        
        if args.val_split > 0:
            train_loader, val_loader = dataset_result
            print(f"Dataset split: {len(train_loader.dataset)} train, {len(val_loader.dataset)} validation")
            # Get sample data for debugging
            sample_data = next(iter(train_loader))
            sample_data = (sample_data[0][:1], sample_data[1][:1])
        else:
            train_loader = dataset_result
            val_loader = None
            print(f"Training dataset: {len(train_loader.dataset)} samples")
            # Get sample data for debugging
            sample_data = next(iter(train_loader))
            sample_data = (sample_data[0][:1], sample_data[1][:1])
    
    # Run initial debugging checks
    if args.debug_mode:
        run_debugging_checks(model, sample_data, device)
    
    # Training loop
    train_losses = []
    val_losses = []
    val_dice_scores = []
    val_iou_scores = []
    
    print(f"Starting training for {args.epochs} epochs...")
    
    for epoch in range(start_epoch, args.epochs):
        epoch_losses = []
        
        # Training
        if args.synthetic:
            # Use synthetic data
            for batch_idx in tqdm(range(100), desc=f"Epoch {epoch+1}/{args.epochs}"):
                image, mask = create_synthetic_data(args.batch_size, (args.image_size, args.image_size))
                loss = train_step(model, image, mask, optimizer, device, 
                                args.use_enhanced_loss, legacy_loss_fn)
                epoch_losses.append(loss)
        else:
            # Use real data
            for batch_idx, (image, mask) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")):
                loss = train_step(model, image, mask, optimizer, device, 
                                args.use_enhanced_loss, legacy_loss_fn)
                epoch_losses.append(loss)
        
        avg_train_loss = sum(epoch_losses) / len(epoch_losses)
        train_losses.append(avg_train_loss)
        
        # Update learning rate
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        # Validation
        if val_loader:
            val_metrics = validate(model, val_loader, device, args.use_enhanced_loss, legacy_loss_fn)
            val_losses.append(val_metrics["loss"])
            val_dice_scores.append(val_metrics["dice"])
            val_iou_scores.append(val_metrics["iou"])
            
            print(f"Epoch {epoch+1}/{args.epochs} - Train Loss: {avg_train_loss:.6f}, "
                  f"Val Loss: {val_metrics['loss']:.6f}, Val Dice: {val_metrics['dice']:.4f}, "
                  f"Val IoU: {val_metrics['iou']:.4f}, LR: {current_lr:.2e}")
        else:
            print(f"Epoch {epoch+1}/{args.epochs} - Train Loss: {avg_train_loss:.6f}, LR: {current_lr:.2e}")
        
        # Logging
        if args.wandb:
            log_dict = {
                "epoch": epoch + 1,
                "train_loss": avg_train_loss,
                "learning_rate": current_lr
            }
            if val_loader:
                log_dict.update({
                    "val_loss": val_metrics["loss"],
                    "val_dice": val_metrics["dice"],
                    "val_iou": val_metrics["iou"]
                })
            wandb.log(log_dict)
        
        # Run debugging checks periodically
        if args.debug_mode and (epoch + 1) % args.debug_freq == 0:
            run_debugging_checks(model, sample_data, device)
        
        # Save checkpoint
        if (epoch + 1) % args.save_freq == 0:
            checkpoint_path = output_dir / f"checkpoint_epoch_{epoch+1}.pth"
            save_checkpoint(model, optimizer, epoch + 1, avg_train_loss, checkpoint_path, config)
            print(f"Saved checkpoint: {checkpoint_path}")
        
        # Generate sample predictions
        if (epoch + 1) % (args.save_freq * 2) == 0:
            model.eval()
            with torch.no_grad():
                if args.synthetic:
                    sample_image, sample_mask = create_synthetic_data(1, (args.image_size, args.image_size))
                else:
                    sample_image, sample_mask = sample_data
                
                sample_image = sample_image.to(device)
                predicted_mask = model(sample_image)
                
                # Save visualization
                save_path = output_dir / f"sample_epoch_{epoch+1}.png"
                visualize_segmentation(
                    sample_image.cpu(), 
                    predicted_mask=predicted_mask.cpu(),
                    ground_truth_mask=sample_mask,
                    title=f"Epoch {epoch+1} Results ({model.diffusion_type.title()} Diffusion)", 
                    save_path=str(save_path)
                )
    
    # Save final model
    final_model_path = output_dir / "final_model.pth"
    save_checkpoint(model, optimizer, args.epochs, train_losses[-1], final_model_path, config)
    print(f"Saved final model: {final_model_path}")
    
    # Plot training curves
    if val_losses:
        # Plot loss curves
        plot_training_curves(train_losses, val_losses, "Training vs Validation Loss", 
                           str(output_dir / "loss_curves.png"))
        
        # Plot dice and IoU curves
        if val_dice_scores:
            plot_training_curves(val_dice_scores, title="Validation Dice Score", 
                               save_path=str(output_dir / "dice_curves.png"))
            plot_training_curves(val_iou_scores, title="Validation IoU Score", 
                               save_path=str(output_dir / "iou_curves.png"))
    else:
        plot_training_curves(train_losses, title="Training Loss", 
                           save_path=str(output_dir / "training_curves.png"))
    
    # Final model evaluation
    if val_loader and model.diffusion_type == "morphological":
        print("\n" + "="*50)
        print("FINAL MODEL EVALUATION")
        print("="*50)
        final_metrics = validate(model, val_loader, device, args.use_enhanced_loss, legacy_loss_fn)
        print(f"Final Validation Metrics:")
        print(f"  Loss: {final_metrics['loss']:.6f}")
        print(f"  Dice Score: {final_metrics['dice']:.4f}")
        print(f"  IoU Score: {final_metrics['iou']:.4f}")
        print("="*50)
    
    if args.wandb:
        wandb.finish()
    
    print("Training completed!")


if __name__ == "__main__":
    main()