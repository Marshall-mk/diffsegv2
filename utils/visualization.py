import matplotlib.pyplot as plt
import torch
import numpy as np
from typing import List, Optional, Dict, Tuple
import seaborn as sns
import os
from pathlib import Path


def visualize_segmentation(image: torch.Tensor, 
                          predicted_mask: Optional[torch.Tensor] = None,
                          ground_truth_mask: Optional[torch.Tensor] = None, 
                          title: str = "Segmentation Results", 
                          save_path: Optional[str] = None):
    """
    Visualize image with optional ground truth and/or predicted masks
    
    Args:
        image: Input image tensor
        predicted_mask: Optional predicted segmentation mask
        ground_truth_mask: Optional ground truth segmentation mask  
        title: Title for the plot
        save_path: Optional path to save the figure
    
    Note: At least one of predicted_mask or ground_truth_mask should be provided
    """
    
    # Convert tensors to numpy and handle different formats
    if image.dim() == 4:
        image = image[0]  # Take first batch item
    if predicted_mask is not None and predicted_mask.dim() == 4:
        predicted_mask = predicted_mask[0]
    if ground_truth_mask is not None and ground_truth_mask.dim() == 4:
        ground_truth_mask = ground_truth_mask[0]
    
    # Normalize image from [-1, 1] to [0, 1] or keep [0, 1] if already normalized
    if image.min() < 0:
        image = (image + 1) / 2
    image = torch.clamp(image, 0, 1)
    
    # Convert to numpy
    image_np = image.permute(1, 2, 0).cpu().numpy()
    
    # Determine number of subplots needed
    n_plots = 1  # Always show original image
    has_predicted = predicted_mask is not None
    has_ground_truth = ground_truth_mask is not None
    
    if has_predicted:
        n_plots += 1
    if has_ground_truth:
        n_plots += 1
    
    # Handle edge case where no masks are provided
    if not has_predicted and not has_ground_truth:
        print("Warning: No masks provided, showing only original image")
    
    # Setup subplot
    fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 5))
    
    # Handle single subplot case
    if n_plots == 1:
        axes = [axes]
    
    plot_idx = 0
    
    # Original image
    axes[plot_idx].imshow(image_np)
    axes[plot_idx].set_title("Original Image")
    axes[plot_idx].axis('off')
    plot_idx += 1
    
    # Ground truth mask (if provided)
    if has_ground_truth:
        gt_mask_np = ground_truth_mask.squeeze().cpu().numpy()
        axes[plot_idx].imshow(gt_mask_np, cmap='gray')
        axes[plot_idx].set_title("Ground Truth Mask")
        axes[plot_idx].axis('off')
        plot_idx += 1
    
    # Predicted mask (if provided)
    if has_predicted:
        pred_mask_np = predicted_mask.squeeze().cpu().numpy()
        axes[plot_idx].imshow(pred_mask_np, cmap='gray')
        axes[plot_idx].set_title("Predicted Mask")
        axes[plot_idx].axis('off')
        plot_idx += 1
    
    plt.suptitle(title)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()


# For backward compatibility, create an alias with old parameter names
def visualize_segmentation_legacy(image: torch.Tensor, mask: torch.Tensor, 
                                predicted_mask: Optional[torch.Tensor] = None, 
                                title: str = "Segmentation Results", 
                                save_path: Optional[str] = None):
    """Legacy version with old parameter names for backward compatibility"""
    return visualize_segmentation(
        image=image,
        predicted_mask=predicted_mask,
        ground_truth_mask=mask,
        title=title,
        save_path=save_path
    )


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


def visualize_training_flow(training_data: Dict, save_path: str = None, 
                           batch_idx: int = 0, max_timesteps: int = 8):
    """
    Visualize the complete training data flow through morphological diffusion.
    
    Args:
        training_data: Dictionary from model.visualize_training_flow()
        save_path: Path to save the visualization
        batch_idx: Which batch element to visualize
        max_timesteps: Maximum number of timesteps to show
    """
    timesteps = training_data['timesteps'][:max_timesteps]
    num_timesteps = len(timesteps)
    
    # Create a large figure with subplots
    fig = plt.figure(figsize=(20, 12))
    
    # Original image and mask
    original_image = training_data['original_image'][batch_idx]
    original_mask = training_data['original_mask'][batch_idx, 0]
    
    # Convert image to displayable format
    if original_image.shape[0] == 3:  # CHW format
        image_display = original_image.permute(1, 2, 0).numpy()
    else:
        image_display = original_image.numpy()
    
    if image_display.max() > 1.0:
        image_display = image_display / 255.0
    
    # Top row: Original image and mask
    ax1 = plt.subplot(4, max(num_timesteps, 4), 1)
    plt.imshow(image_display)
    plt.title('Original Image', fontsize=12, fontweight='bold')
    plt.axis('off')
    
    ax2 = plt.subplot(4, max(num_timesteps, 4), 2)
    plt.imshow(original_mask.numpy(), cmap='gray', vmin=0, vmax=1)
    plt.title('Original Mask', fontsize=12, fontweight='bold')
    plt.axis('off')
    
    # Second row: Degraded masks at different timesteps
    for i, (timestep, degraded_masks) in enumerate(zip(timesteps, training_data['degraded_masks'])):
        if i >= max_timesteps:
            break
            
        ax = plt.subplot(4, max(num_timesteps, 4), max(num_timesteps, 4) + 1 + i)
        degraded_mask = degraded_masks[batch_idx, 0].numpy()
        
        plt.imshow(degraded_mask, cmap='gray', vmin=0, vmax=1)
        
        # Add intensity info if available
        if i < len(training_data['intensities']):
            intensity = training_data['intensities'][i]
            if isinstance(intensity, (int, float)):
                plt.title(f't={timestep}\nIntensity={intensity:.3f}', fontsize=10)
            else:
                plt.title(f't={timestep}\n{intensity}', fontsize=10)
        else:
            plt.title(f't={timestep}', fontsize=10)
        plt.axis('off')
    
    # Third row: Model inputs (concatenated image + degraded mask)
    for i, concat_data in enumerate(training_data['concatenated_inputs'][:max_timesteps]):
        ax = plt.subplot(4, max(num_timesteps, 4), 2 * max(num_timesteps, 4) + 1 + i)
        
        # Show the concatenated input as RGB where:
        # R = Image red channel, G = Image green channel, B = Degraded mask
        model_input = concat_data['concatenated'][batch_idx]
        
        if model_input.shape[0] >= 4:  # At least image (3 ch) + mask (1 ch)
            # Create RGB visualization
            rgb_viz = torch.zeros(3, model_input.shape[1], model_input.shape[2])
            rgb_viz[0] = model_input[0]  # R from image
            rgb_viz[1] = model_input[1]  # G from image  
            rgb_viz[2] = model_input[3]  # Degraded mask as B channel
            
            # Normalize for display
            rgb_viz = torch.clamp(rgb_viz, 0, 1)
            rgb_display = rgb_viz.permute(1, 2, 0).numpy()
            
            plt.imshow(rgb_display)
            plt.title(f'Model Input\n(R:img_r, G:img_g, B:mask)', fontsize=9)
        else:
            # Fallback: show just the mask channel
            plt.imshow(model_input[-1].numpy(), cmap='gray')
            plt.title(f'Model Input\n(mask only)', fontsize=9)
        
        plt.axis('off')
    
    # Fourth row: Channel separation visualization
    for i, concat_data in enumerate(training_data['concatenated_inputs'][:max_timesteps]):
        ax = plt.subplot(4, max(num_timesteps, 4), 3 * max(num_timesteps, 4) + 1 + i)
        
        # Show channels side by side in a mini subplot
        model_input = concat_data['concatenated'][batch_idx]
        
        # Create a small visualization showing channel structure
        if model_input.shape[0] >= 4:
            # Show image channels and mask channel
            fig_small = plt.figure(figsize=(2, 1))
            
            # Mini grid: [IMG_R | IMG_G | IMG_B | MASK]
            for ch in range(min(4, model_input.shape[0])):
                plt.subplot(1, 4, ch + 1)
                plt.imshow(model_input[ch].numpy(), cmap='gray')
                if ch < 3:
                    plt.title(f'Ch{ch}', fontsize=6)
                else:
                    plt.title('Mask', fontsize=6)
                plt.axis('off')
            
            plt.tight_layout()
            plt.savefig(f'/tmp/channels_{i}.png', dpi=50, bbox_inches='tight')
            plt.close(fig_small)
            
            # Load and show the mini figure
            import matplotlib.image as mpimg
            if os.path.exists(f'/tmp/channels_{i}.png'):
                mini_img = mpimg.imread(f'/tmp/channels_{i}.png')
                plt.imshow(mini_img)
                plt.title(f'Channels t={concat_data["timestep"]}', fontsize=9)
        else:
            plt.imshow(model_input[0].numpy(), cmap='gray')
            plt.title(f'Single Channel', fontsize=9)
        
        plt.axis('off')
    
    plt.suptitle('Training Data Flow Visualization', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def visualize_batch_training_info(batch_info: Dict, save_path: str = None, 
                                 max_samples: int = 4):
    """
    Visualize batch training information including actual forward process results.
    
    Args:
        batch_info: Dictionary from model.get_batch_training_info()
        save_path: Path to save the visualization  
        max_samples: Maximum number of batch samples to show
    """
    batch_size = min(batch_info['batch_size'], max_samples)
    
    fig, axes = plt.subplots(batch_size, 6, figsize=(24, 4 * batch_size))
    if batch_size == 1:
        axes = axes.reshape(1, -1)
    
    for b in range(batch_size):
        # Original image
        orig_img = batch_info['original_images'][b]
        if orig_img.shape[0] == 3:
            img_display = orig_img.permute(1, 2, 0).numpy()
        else:
            img_display = orig_img.numpy()
        if img_display.max() > 1.0:
            img_display = img_display / 255.0
            
        axes[b, 0].imshow(img_display)
        axes[b, 0].set_title(f'Sample {b}\nOriginal Image')
        axes[b, 0].axis('off')
        
        # Original mask
        orig_mask = batch_info['original_masks'][b, 0].numpy()
        axes[b, 1].imshow(orig_mask, cmap='gray', vmin=0, vmax=1)
        axes[b, 1].set_title(f'Original Mask')
        axes[b, 1].axis('off')
        
        # Degraded mask
        degraded_mask = batch_info['degraded_masks'][b, 0].numpy()
        axes[b, 2].imshow(degraded_mask, cmap='gray', vmin=0, vmax=1)
        
        timestep = batch_info['timesteps'][b]
        if batch_info['diffusion_type'] == 'morphological':
            intensity = batch_info['intensities'][b]
            axes[b, 2].set_title(f'Degraded Mask\nt={timestep}, I={intensity:.3f}')
        else:
            axes[b, 2].set_title(f'Noisy Mask\nt={timestep}')
        axes[b, 2].axis('off')
        
        # Model input visualization (as RGB)
        model_input = batch_info['model_input'][b]
        if model_input.shape[0] >= 4:
            rgb_viz = torch.zeros(3, model_input.shape[1], model_input.shape[2])
            rgb_viz[0] = torch.clamp(model_input[0], 0, 1)  # R 
            rgb_viz[1] = torch.clamp(model_input[1], 0, 1)  # G
            rgb_viz[2] = torch.clamp(model_input[3], 0, 1)  # Mask as B
            
            rgb_display = rgb_viz.permute(1, 2, 0).numpy()
            axes[b, 3].imshow(rgb_display)
            axes[b, 3].set_title(f'Model Input\n(IMG+MASK)')
        else:
            axes[b, 3].imshow(model_input[0].numpy(), cmap='gray')
            axes[b, 3].set_title(f'Model Input')
        axes[b, 3].axis('off')
        
        # Target
        target = batch_info['target'][b, 0].numpy() 
        axes[b, 4].imshow(target, cmap='gray', vmin=0, vmax=1)
        if batch_info['diffusion_type'] == 'morphological':
            axes[b, 4].set_title(f'Target\n(Original Mask)')
        else:
            axes[b, 4].set_title(f'Target\n(Noise)')
        axes[b, 4].axis('off')
        
        # Difference visualization
        diff = np.abs(orig_mask - degraded_mask)
        im = axes[b, 5].imshow(diff, cmap='hot', vmin=0, vmax=1)
        axes[b, 5].set_title(f'|Original - Degraded|')
        axes[b, 5].axis('off')
        
        # Add colorbar for difference
        plt.colorbar(im, ax=axes[b, 5], fraction=0.046, pad=0.04)
    
    plt.suptitle(f'Batch Training Info - {batch_info["diffusion_type"].title()} Diffusion', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def create_training_flow_gif(model, sample_data: Tuple[torch.Tensor, torch.Tensor],
                           save_dir: str, num_frames: int = 20):
    """
    Create an animated GIF showing the morphological degradation process.
    
    Args:
        model: DiffusionSegmentation model
        sample_data: Tuple of (image, mask) tensors
        save_dir: Directory to save the GIF
        num_frames: Number of frames in the animation
    """
    try:
        from PIL import Image
        import imageio
    except ImportError:
        print("PIL and imageio required for GIF creation. Install with: pip install Pillow imageio")
        return
    
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    image, mask = sample_data
    device = next(model.parameters()).device
    image, mask = image[:1].to(device), mask[:1].to(device)  # Use first sample only
    
    # Create frames
    frames = []
    timesteps = torch.linspace(0, model.timesteps - 1, num_frames, dtype=torch.long)
    
    for i, timestep in enumerate(timesteps):
        # Create visualization data for this timestep
        t_batch = torch.full((1,), timestep, device=device, dtype=torch.long)
        
        if model.diffusion_type == "morphological":
            degraded_mask = model.forward_morphology_batch(mask, t_batch)
            intensity = model.morph_schedule[timestep.item()].item()
        else:
            degraded_mask, _ = model.forward_diffusion(mask, t_batch)
            intensity = "Gaussian"
        
        # Create frame
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        
        # Original image
        orig_img = image[0].cpu()
        if orig_img.shape[0] == 3:
            img_display = orig_img.permute(1, 2, 0).numpy()
        else:
            img_display = orig_img.numpy()
        if img_display.max() > 1.0:
            img_display = img_display / 255.0
            
        axes[0].imshow(img_display)
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        # Original mask
        axes[1].imshow(mask[0, 0].cpu().numpy(), cmap='gray', vmin=0, vmax=1)
        axes[1].set_title('Original Mask')
        axes[1].axis('off')
        
        # Degraded mask
        axes[2].imshow(degraded_mask[0, 0].cpu().numpy(), cmap='gray', vmin=0, vmax=1)
        if isinstance(intensity, (int, float)):
            axes[2].set_title(f'Degraded Mask\nt={timestep.item()}, I={intensity:.3f}')
        else:
            axes[2].set_title(f'Degraded Mask\nt={timestep.item()}')
        axes[2].axis('off')
        
        plt.tight_layout()
        
        # Save frame
        frame_path = save_dir / f'frame_{i:03d}.png'
        plt.savefig(frame_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        frames.append(str(frame_path))
    
    # Create GIF
    gif_path = save_dir / 'morphological_degradation.gif'
    with imageio.get_writer(gif_path, mode='I', duration=0.3) as writer:
        for frame_path in frames:
            image_frame = imageio.imread(frame_path)
            writer.append_data(image_frame)
    
    # Clean up frame files
    for frame_path in frames:
        os.remove(frame_path)
    
    print(f"Created GIF: {gif_path}")