#!/usr/bin/env python3
"""
Inference script for Diffusion Segmentation model with polynomial morphological operations
"""

import torch
import torch.nn.functional as F
import argparse
import os
from pathlib import Path
import json
import numpy as np
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import warnings

from src.models import DiffusionSegmentation
from utils.data_utils import preprocess_image
from utils.visualization import visualize_segmentation, plot_inference_steps


def load_model_config(checkpoint_path: str) -> dict:
    """Load model configuration from checkpoint if available"""
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        if 'config' in checkpoint:
            return checkpoint['config']
        else:
            print("No config found in checkpoint, using provided parameters")
            return {}
    except Exception as e:
        print(f"Warning: Could not load config from checkpoint: {e}")
        return {}


def load_model(checkpoint_path: str, device: torch.device, 
               # Updated parameters to match new model
               timesteps: int = 1000, 
               unet_type: str = "diffusers_2d", 
               pretrained_model_name_or_path: str = None,
               diffusion_type: str = "morphological",  # Changed default
               morph_type: str = "mixed",              # Changed default
               morph_kernel_size_start: int = 3,       # New parameter
               morph_kernel_size_end: int = 9,         # New parameter
               morph_routine: str = "Progressive",     # New parameter
               morph_schedule_type: str = "exponential", # Changed default
               scheduler_type: str = "ddpm",
               # Legacy parameters for backward compatibility
               morph_kernel_size: int = None) -> DiffusionSegmentation:
    """Load trained model from checkpoint with enhanced morphological support"""
    
    # Load config from checkpoint if available
    saved_config = load_model_config(checkpoint_path)
    
    # Use saved config if available, otherwise use provided parameters
    model_params = {
        'in_channels': 3,
        'num_classes': 1,
        'timesteps': saved_config.get('timesteps', timesteps),
        'unet_type': saved_config.get('unet_type', unet_type),
        'pretrained_model_name_or_path': saved_config.get('pretrained_model', pretrained_model_name_or_path),
        'diffusion_type': saved_config.get('diffusion_type', diffusion_type),
        'morph_type': saved_config.get('morph_type', morph_type),
        'scheduler_type': saved_config.get('scheduler_type', scheduler_type)
    }
    
    # Handle morphological parameters with backward compatibility
    if saved_config.get('diffusion_type', diffusion_type) == 'morphological':
        # Try to use new parameters from saved config
        model_params.update({
            'morph_kernel_size_start': saved_config.get('morph_kernel_size_start', morph_kernel_size_start),
            'morph_kernel_size_end': saved_config.get('morph_kernel_size_end', morph_kernel_size_end),
            'morph_routine': saved_config.get('morph_routine', morph_routine),
            'morph_schedule_type': saved_config.get('morph_schedule', morph_schedule_type)
        })
        
        # Handle legacy parameter
        if morph_kernel_size is not None:
            warnings.warn("morph_kernel_size is deprecated. Using as morph_kernel_size_start.")
            model_params['morph_kernel_size_start'] = morph_kernel_size
        
        # Legacy compatibility for old checkpoints
        if 'morph_kernel_size' in saved_config and 'morph_kernel_size_start' not in saved_config:
            warnings.warn("Loading legacy checkpoint. Converting morph_kernel_size to new parameters.")
            model_params['morph_kernel_size_start'] = saved_config['morph_kernel_size']
            model_params['morph_kernel_size_end'] = saved_config['morph_kernel_size'] + 4
            model_params['morph_routine'] = 'Constant'  # Safe default for legacy models
    
    # Create model with parameters
    try:
        model = DiffusionSegmentation(**model_params).to(device)
    except TypeError as e:
        # Fallback for very old model format
        print(f"Warning: {e}")
        print("Attempting to load with legacy parameters...")
        
        legacy_params = {
            'in_channels': 3,
            'num_classes': 1,
            'timesteps': model_params['timesteps'],
            'unet_type': model_params['unet_type'],
            'diffusion_type': model_params['diffusion_type'],
            'morph_type': model_params['morph_type'],
            'morph_kernel_size': morph_kernel_size or 3,
            'morph_schedule_type': 'linear',  # Legacy default
            'scheduler_type': model_params['scheduler_type']
        }
        model = DiffusionSegmentation(**legacy_params).to(device)
    
    # Load state dict
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Print model information
    epoch = checkpoint.get('epoch', 'unknown')
    print(f"Loaded model from {checkpoint_path}, trained for {epoch} epochs")
    
    # Print morphological configuration if using morphological diffusion
    if model.diffusion_type == 'morphological':
        debug_info = model.get_morphological_debug_info()
        print(f"Morphological Configuration:")
        print(f"  Type: {debug_info['morph_type']}")
        print(f"  Routine: {debug_info['morph_routine']}")
        print(f"  Timesteps: {debug_info['num_timesteps']}")
        if debug_info['kernel_sizes']:
            print(f"  Kernel sizes: {debug_info['kernel_sizes'][0]} → {debug_info['kernel_sizes'][-1]}")
    
    return model


def predict_single_image(model: DiffusionSegmentation, image_path: str, 
                        device: torch.device, num_inference_steps: int = 50,
                        image_size: tuple = (256, 256), 
                        show_progress: bool = False) -> torch.Tensor:
    """Predict segmentation mask for a single image"""
    
    # Load and preprocess image
    image = preprocess_image(image_path, image_size).to(device)
    
    # Add batch dimension if needed
    if len(image.shape) == 3:
        image = image.unsqueeze(0)
    
    # Generate prediction
    with torch.no_grad():
        if show_progress:
            print(f"Generating segmentation with {num_inference_steps} steps...")
        
        # Call the sample method directly to control num_inference_steps
        predicted_mask = model.sample(image, mask=None, num_inference_steps=num_inference_steps)
    
    return predicted_mask


def predict_with_intermediate_steps(model: DiffusionSegmentation, image_path: str,
                                  device: torch.device, num_inference_steps: int = 50,
                                  image_size: tuple = (256, 256), 
                                  save_steps: list = None) -> tuple:
    """Predict with intermediate step visualization for morphological diffusion"""
    
    if model.diffusion_type != 'morphological':
        # For non-morphological models, just return regular prediction
        mask = predict_single_image(model, image_path, device, num_inference_steps, image_size)
        return mask, []
    
    # Load and preprocess image
    image = preprocess_image(image_path, image_size).to(device)
    if len(image.shape) == 3:
        image = image.unsqueeze(0)
    
    if save_steps is None:
        save_steps = [0, num_inference_steps//4, num_inference_steps//2, 3*num_inference_steps//4, num_inference_steps-1]
    
    intermediate_results = []
    
    # Get initial state
    mask = model._get_initial_morphological_state(image)
    
    # Create timestep schedule
    step_size = max(1, model.timesteps // num_inference_steps)
    timesteps = list(range(model.timesteps - 1, -1, -step_size))
    if timesteps[-1] != 0:
        timesteps.append(0)
    
    print(f"Running inference with {len(timesteps)} steps...")
    
    with torch.no_grad():
        for i, current_t in enumerate(tqdm(timesteps, desc="Inference steps")):
            t_tensor = torch.full((image.shape[0],), current_t, device=device, dtype=torch.long)
            
            # Predict the clean mask
            x = torch.cat([image, mask], dim=1)
            predicted_clean_mask = model._call_unet(x, t_tensor)
            
            if current_t > 0:
                # Progressive blending
                alpha = 1.0 - (current_t / model.timesteps)
                mask = alpha * predicted_clean_mask + (1 - alpha) * mask
            else:
                mask = predicted_clean_mask
            
            mask = torch.clamp(mask, 0, 1)
            
            # Save intermediate results
            if i in save_steps:
                intermediate_results.append({
                    'step': i,
                    'timestep': current_t,
                    'mask': mask.clone().cpu(),
                    'predicted_clean': predicted_clean_mask.clone().cpu()
                })
    
    # Final sigmoid
    final_mask = torch.sigmoid(mask)
    
    return final_mask, intermediate_results


def batch_inference(model: DiffusionSegmentation, input_dir: str, output_dir: str,
                   device: torch.device, num_inference_steps: int = 50,
                   image_size: tuple = (256, 256), save_visualizations: bool = True,
                   save_intermediate: bool = False):
    """Run inference on all images in a directory"""
    
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Find all image files
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    image_files = []
    for ext in image_extensions:
        image_files.extend(input_path.glob(f"*{ext}"))
        image_files.extend(input_path.glob(f"*{ext.upper()}"))
    
    if not image_files:
        print(f"No image files found in {input_dir}")
        return
    
    print(f"Found {len(image_files)} images to process")
    
    # Create subdirectories
    if save_visualizations:
        vis_dir = output_path / "visualizations"
        vis_dir.mkdir(exist_ok=True)
    
    if save_intermediate:
        inter_dir = output_path / "intermediate_steps"
        inter_dir.mkdir(exist_ok=True)
    
    results_summary = []
    
    for image_file in tqdm(image_files, desc="Processing images"):
        try:
            # Predict mask
            if save_intermediate and model.diffusion_type == 'morphological':
                predicted_mask, intermediate_steps = predict_with_intermediate_steps(
                    model, str(image_file), device, num_inference_steps, image_size
                )
                
                # Save intermediate steps
                if intermediate_steps:
                    step_dir = inter_dir / image_file.stem
                    step_dir.mkdir(exist_ok=True)
                    
                    for step_data in intermediate_steps:
                        step_mask = step_data['mask'].squeeze().numpy()
                        step_mask = (step_mask * 255).astype(np.uint8)
                        step_image = Image.fromarray(step_mask, mode='L')
                        step_file = step_dir / f"step_{step_data['step']:03d}_t{step_data['timestep']:04d}.png"
                        step_image.save(step_file)
            else:
                predicted_mask = predict_single_image(
                    model, str(image_file), device, num_inference_steps, image_size
                )
            
            # Save mask as image
            mask_np = predicted_mask.squeeze().cpu().numpy()
            mask_np = (mask_np * 255).astype(np.uint8)
            mask_image = Image.fromarray(mask_np, mode='L')
            
            output_file = output_path / f"{image_file.stem}_mask.png"
            mask_image.save(output_file)
            
            # Save visualization if requested
            if save_visualizations:
                image = preprocess_image(str(image_file), image_size)
                vis_file = vis_dir / f"{image_file.stem}_visualization.png"
                visualize_segmentation(
                    image, predicted_mask.cpu(), None,
                    title=f"Segmentation: {image_file.name}",
                    save_path=str(vis_file)
                )
            
            # Calculate basic metrics
            mask_mean = float(mask_np.mean() / 255.0)
            mask_std = float(mask_np.std() / 255.0)
            
            results_summary.append({
                'image': image_file.name,
                'mask_file': output_file.name,
                'mask_mean': mask_mean,
                'mask_std': mask_std
            })

        except Exception as e:
            print(f"Error processing {image_file}: {e}")
            results_summary.append({
                'image': image_file.name,
                'error': str(e)
            })
    
    # Save results summary
    summary_file = output_path / "inference_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    print(f"Inference completed. Results saved to {output_dir}")
    print(f"Summary saved to {summary_file}")


def interactive_inference(model: DiffusionSegmentation, device: torch.device,
                         num_inference_steps: int = 50, image_size: tuple = (256, 256)):
    """Interactive inference mode with enhanced features"""
    
    print("Interactive Inference Mode")
    print("Commands:")
    print("  <image_path> - Process an image")
    print("  'info' - Show model information") 
    print("  'steps <N>' - Set number of inference steps")
    print("  'debug <image_path>' - Show intermediate steps (morphological only)")
    print("  'quit' - Exit")
    
    current_steps = num_inference_steps
    
    while True:
        command = input(f"\n[{current_steps} steps] > ").strip()
        
        if command.lower() in ['quit', 'exit', 'q']:
            break
        
        if command.lower() == 'info':
            print(f"Model Type: {model.diffusion_type}")
            print(f"UNet Type: {model.unet_type}")
            print(f"Timesteps: {model.timesteps}")
            if hasattr(model, 'get_morphological_debug_info'):
                debug_info = model.get_morphological_debug_info()
                if debug_info.get('morph_type'):
                    print(f"Morphological Type: {debug_info['morph_type']}")
                    print(f"Morphological Routine: {debug_info['morph_routine']}")
            continue
        
        if command.startswith('steps '):
            try:
                current_steps = int(command.split()[1])
                print(f"Set inference steps to {current_steps}")
            except (ValueError, IndexError):
                print("Usage: steps <number>")
            continue
        
        if command.startswith('debug '):
            image_path = command[6:].strip()
            if not os.path.exists(image_path):
                print(f"File not found: {image_path}")
                continue
            
            if model.diffusion_type != 'morphological':
                print("Debug mode only available for morphological diffusion models")
                continue
            
            try:
                print("Generating segmentation with intermediate steps...")
                predicted_mask, intermediate_steps = predict_with_intermediate_steps(
                    model, image_path, device, current_steps, image_size, 
                    save_steps=[0, current_steps//4, current_steps//2, 3*current_steps//4, current_steps-1]
                )
                
                print(f"Generated {len(intermediate_steps)} intermediate steps")
                for step_data in intermediate_steps:
                    print(f"  Step {step_data['step']}: timestep {step_data['timestep']}")
                
                # Show final result
                image = preprocess_image(image_path, image_size)
                visualize_segmentation(
                    image, predicted_mask.cpu(), None,
                    title=f"Final Result: {os.path.basename(image_path)}"
                )
                
            except Exception as e:
                print(f"Error: {e}")
            continue
        
        # Regular image processing
        image_path = command
        if not os.path.exists(image_path):
            print(f"File not found: {image_path}")
            continue
        
        try:
            # Predict mask
            print(f"Generating segmentation with {current_steps} steps...")
            predicted_mask = predict_single_image(
                model, image_path, device, current_steps, image_size, show_progress=True
            )
            
            # Load original image for visualization
            image = preprocess_image(image_path, image_size)
            
            # Show results
            visualize_segmentation(
                image, predicted_mask.cpu(), None,
                title=f"Segmentation: {os.path.basename(image_path)}"
            )
            
            # Ask if user wants to save
            save = input("Save result? (y/n): ").strip().lower()
            if save in ['y', 'yes']:
                output_path = input("Output path (default: mask.png): ").strip()
                if not output_path:
                    output_path = "mask.png"
                
                mask_np = predicted_mask.squeeze().cpu().numpy()
                mask_np = (mask_np * 255).astype(np.uint8)
                mask_image = Image.fromarray(mask_np, mode='L')
                mask_image.save(output_path)
                print(f"Saved to {output_path}")
        
        except Exception as e:
            print(f"Error: {e}")


def main():
    parser = argparse.ArgumentParser(description='Run inference with Enhanced Diffusion Segmentation Model')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--input', type=str, help='Input image path or directory')
    parser.add_argument('--output', type=str, default='./inference_results', help='Output directory')
    parser.add_argument('--batch', action='store_true', help='Batch inference mode (process directory)')
    parser.add_argument('--interactive', action='store_true', help='Interactive inference mode')
    parser.add_argument('--steps', type=int, default=50, help='Number of inference steps')
    parser.add_argument('--image-size', type=int, default=256, help='Image size for processing')
    
    # Model parameters (will be overridden by checkpoint config if available)
    parser.add_argument('--timesteps', type=int, default=1000, help='Number of diffusion timesteps')
    parser.add_argument('--unet-type', type=str, default='diffusers_2d',
                       choices=['diffusers_2d', 'diffusers_2d_cond'], 
                       help='Type of UNet to use')
    parser.add_argument('--pretrained-model', type=str, help='Path or name of pretrained diffusers model')
    parser.add_argument('--diffusion-type', type=str, default='morphological',
                       choices=['gaussian', 'morphological'], 
                       help='Type of diffusion process')
    parser.add_argument('--morph-type', type=str, default='mixed',
                       choices=['dilation', 'erosion', 'mixed', 'opening', 'closing'], 
                       help='Type of morphological operation')
    
    # New morphological parameters
    parser.add_argument('--morph-kernel-size-start', type=int, default=3,
                       help='Starting size of morphological kernel')
    parser.add_argument('--morph-kernel-size-end', type=int, default=9,
                       help='Ending size of morphological kernel')
    parser.add_argument('--morph-routine', type=str, default='Progressive',
                       choices=['Progressive', 'Constant'],
                       help='Morphological routine')
    parser.add_argument('--morph-schedule', type=str, default='exponential',
                       choices=['linear', 'exponential', 'cosine'],
                       help='Schedule type for morphological intensity')
    
    # Legacy parameters
    parser.add_argument('--morph-kernel-size', type=int, help='DEPRECATED: Use morph-kernel-size-start')
    
    parser.add_argument('--scheduler-type', type=str, default='ddpm',
                       choices=['ddpm', 'ddim'],
                       help='Type of diffusers scheduler to use')
    
    # Visualization options
    parser.add_argument('--visualize-process', action='store_true', 
                       help='Visualize inference process (morphological only)')
    parser.add_argument('--save-intermediate', action='store_true',
                       help='Save intermediate steps (batch mode, morphological only)')
    parser.add_argument('--no-vis', action='store_true', help='Skip saving visualizations in batch mode')
    
    args = parser.parse_args()
    
    # Handle deprecated parameters
    if args.morph_kernel_size is not None:
        warnings.warn("--morph-kernel-size is deprecated. Use --morph-kernel-size-start")
        args.morph_kernel_size_start = args.morph_kernel_size
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    print(f"Loading model from {args.checkpoint}...")
    model = load_model(
        args.checkpoint, device, 
        timesteps=args.timesteps,
        unet_type=args.unet_type, 
        pretrained_model_name_or_path=args.pretrained_model,
        diffusion_type=args.diffusion_type, 
        morph_type=args.morph_type, 
        morph_kernel_size_start=args.morph_kernel_size_start,
        morph_kernel_size_end=args.morph_kernel_size_end,
        morph_routine=args.morph_routine,
        morph_schedule_type=args.morph_schedule, 
        scheduler_type=args.scheduler_type,
        morph_kernel_size=args.morph_kernel_size  # Legacy support
    )
    
    image_size = (args.image_size, args.image_size)
    
    if args.interactive:
        # Interactive mode
        interactive_inference(model, device, args.steps, image_size)
    
    elif args.batch:
        # Batch inference mode
        if not args.input:
            print("Error: --input directory required for batch mode")
            return
        
        batch_inference(
            model, args.input, args.output, device, 
            args.steps, image_size, not args.no_vis, args.save_intermediate
        )
    
    else:
        # Single image inference
        if not args.input:
            print("Error: --input image path required")
            return
        
        if not os.path.exists(args.input):
            print(f"Error: Input file not found: {args.input}")
            return
        
        print(f"Processing {args.input}...")
        
        # Predict mask with optional intermediate steps
        if args.visualize_process and model.diffusion_type == 'morphological':
            predicted_mask, intermediate_steps = predict_with_intermediate_steps(
                model, args.input, device, args.steps, image_size
            )
            print(f"Generated {len(intermediate_steps)} intermediate visualizations")
        else:
            predicted_mask = predict_single_image(model, args.input, device, args.steps, image_size, show_progress=True)
        
        # Load original image
        image = preprocess_image(args.input, image_size)
        
        # Create output directory
        output_path = Path(args.output)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save mask
        mask_np = predicted_mask.squeeze().cpu().numpy()
        mask_np = (mask_np * 255).astype(np.uint8)
        mask_image = Image.fromarray(mask_np, mode='L')
        
        input_name = Path(args.input).stem
        mask_file = output_path / f"{input_name}_mask.png"
        mask_image.save(mask_file)
        print(f"Saved mask: {mask_file}")
        
        # Save visualization
        vis_file = output_path / f"{input_name}_visualization.png"
        visualize_segmentation(
            image, predicted_mask.cpu(), None,
            title=f"Segmentation: {Path(args.input).name}",
            save_path=str(vis_file)
        )
        plt.close()
        
        print(f"Saved visualization: {vis_file}")
        
        # Visualize inference process if requested
        if args.visualize_process and hasattr(model, 'diffusion_type') and model.diffusion_type == 'morphological':
            print("Generating process visualization...")
            process_file = output_path / f"{input_name}_process.png"
            # This would need to be implemented in utils/visualization.py
            try:
                plot_inference_steps(
                    model, image.to(device), num_steps=10, 
                    save_path=str(process_file)
                )
                print(f"Saved process visualization: {process_file}")
            except Exception as e:
                print(f"Could not create process visualization: {e}")


if __name__ == "__main__":
    main()