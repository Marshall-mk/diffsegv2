import matplotlib.pyplot as plt
import torch
import numpy as np
from typing import List, Optional
import seaborn as sns


def visualize_segmentation(image: torch.Tensor, mask: torch.Tensor, predicted_mask: Optional[torch.Tensor] = None, 
                          title: str = "Segmentation Results", save_path: Optional[str] = None):
    """Visualize image, ground truth mask, and optionally predicted mask"""
    
    # Convert tensors to numpy and handle different formats
    if image.dim() == 4:
        image = image[0]  # Take first batch item
    if mask.dim() == 4:
        mask = mask[0]
    if predicted_mask is not None and predicted_mask.dim() == 4:
        predicted_mask = predicted_mask[0]
    
    # Normalize image from [-1, 1] to [0, 1]
    image = (image + 1) / 2
    image = torch.clamp(image, 0, 1)
    
    # Convert to numpy
    image_np = image.permute(1, 2, 0).cpu().numpy()
    mask_np = mask.squeeze().cpu().numpy()
    
    # Setup subplot
    n_plots = 3 if predicted_mask is not None else 2
    fig, axes = plt.subplots(1, n_plots, figsize=(15, 5))
    
    # Original image
    axes[0].imshow(image_np)
    axes[0].set_title("Original Image")
    axes[0].axis('off')
    
    # Ground truth mask
    axes[1].imshow(mask_np, cmap='gray')
    axes[1].set_title("Ground Truth Mask")
    axes[1].axis('off')
    
    # Predicted mask
    if predicted_mask is not None:
        pred_mask_np = predicted_mask.squeeze().cpu().numpy()
        axes[2].imshow(pred_mask_np, cmap='gray')
        axes[2].set_title("Predicted Mask")
        axes[2].axis('off')
    
    plt.suptitle(title)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()


def plot_training_curves(train_losses: List[float], val_losses: Optional[List[float]] = None, 
                         title: str = "Training Loss", save_path: Optional[str] = None):
    """Plot training loss curves with optional validation losses"""
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    
    if val_losses is not None:
        plt.plot(val_losses, label='Validation Loss')
    
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()


def visualize_diffusion_process(model, image: torch.Tensor, timesteps: List[int] = [0, 100, 500, 900, 999],
                               save_path: Optional[str] = None):
    """Visualize the forward diffusion process at different timesteps"""
    
    if image.dim() == 4:
        image = image[0:1]  # Take first batch item
    
    device = next(model.parameters()).device
    image = image.to(device)
    
    # Create a dummy mask for forward diffusion
    mask = torch.ones(1, 1, image.shape[2], image.shape[3], device=device)
    
    fig, axes = plt.subplots(1, len(timesteps), figsize=(20, 4))
    
    model.eval()
    with torch.no_grad():
        for i, t in enumerate(timesteps):
            t_tensor = torch.tensor([t], device=device)
            noisy_mask, _ = model.forward_diffusion(mask, t_tensor)
            
            noisy_mask_np = noisy_mask.squeeze().cpu().numpy()
            axes[i].imshow(noisy_mask_np, cmap='gray')
            axes[i].set_title(f"t = {t}")
            axes[i].axis('off')
    
    plt.suptitle("Forward Diffusion Process")
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()


def plot_inference_steps(model, image: torch.Tensor, num_steps: int = 10, save_path: Optional[str] = None):
    """Visualize the reverse diffusion process during inference"""
    
    if image.dim() == 4:
        image = image[0:1]  # Take first batch item
    
    device = next(model.parameters()).device
    image = image.to(device)
    
    # Only works with Gaussian diffusion
    if model.diffusion_type != "gaussian":
        print("plot_inference_steps only works with Gaussian diffusion")
        return
    
    # Start with random noise
    mask = torch.randn(1, 1, image.shape[2], image.shape[3], device=device)
    
    fig, axes = plt.subplots(2, num_steps // 2, figsize=(20, 8))
    axes = axes.flatten()
    
    model.eval()
    
    # Use the model's scheduler for proper timestep handling
    model.scheduler.set_timesteps(num_steps)
    timesteps = model.scheduler.timesteps
    
    with torch.no_grad():
        for step_idx, t in enumerate(timesteps):
            if step_idx < len(axes):
                mask_np = torch.sigmoid(mask).squeeze().cpu().numpy()
                axes[step_idx].imshow(mask_np, cmap='gray')
                axes[step_idx].set_title(f"Step {step_idx}, t = {t}")
                axes[step_idx].axis('off')
            
            # Denoising step using the UNet
            t_tensor = torch.full((image.shape[0],), t, device=device, dtype=torch.long)
            x = torch.cat([image, mask], dim=1)
            predicted_noise = model._call_unet(x, t_tensor)
            
            # Use scheduler to compute previous sample
            mask = model.scheduler.step(predicted_noise, t, mask).prev_sample
    
    plt.suptitle("Reverse Diffusion Process (Inference)")
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()