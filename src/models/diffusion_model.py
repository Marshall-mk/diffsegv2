import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Optional, Union

try:
    from diffusers import UNet2DModel, UNet2DConditionModel, DDPMScheduler, DDIMScheduler
    DIFFUSERS_AVAILABLE = True
except ImportError:
    DIFFUSERS_AVAILABLE = False

# Updated imports to use polynomial morphology
from .morphological_ops import PolynomialMorphology, MorphologicalDegradation, create_morphological_schedule


class DiffusionSegmentation(nn.Module):
    def __init__(self, 
                 in_channels: int = 3, 
                 num_classes: int = 1, 
                 timesteps: int = 1000,
                 unet_type: str = "diffusers_2d",
                 pretrained_model_name_or_path: Optional[str] = None,
                 diffusion_type: str = "gaussian",
                 morph_type: str = "mixed",  # Changed default to "mixed" for better results
                 morph_kernel_size_start: int = 3,  # Start kernel size
                 morph_kernel_size_end: int = 9,    # End kernel size  
                 morph_schedule_type: str = "exponential",  # Changed to exponential
                 morph_routine: str = "Progressive",  # Added morphological routine
                 scheduler_type: str = "ddpm"):
        """
        Initialize DiffusionSegmentation model with polynomial morphological operations.
        
        Args:
            in_channels: Number of input image channels (default: 3)
            num_classes: Number of segmentation classes (default: 1) 
            timesteps: Number of diffusion timesteps (default: 1000)
            unet_type: Type of UNet to use. Options: "diffusers_2d", "diffusers_2d_cond"
            pretrained_model_name_or_path: Path or name of pretrained diffusers model
            diffusion_type: Type of diffusion process. Options: "gaussian", "morphological"
            morph_type: Type of morphological operation. Options: "dilation", "erosion", "mixed", "opening", "closing"
            morph_kernel_size_start: Starting size of morphological kernel (default: 3)
            morph_kernel_size_end: Ending size of morphological kernel (default: 9)
            morph_schedule_type: Schedule type for morphological intensity. Options: "linear", "exponential", "cosine"
            morph_routine: Morphological routine. Options: "Progressive", "Constant"
            scheduler_type: Type of diffusers scheduler. Options: "ddpm", "ddim"
        """
        super().__init__()
        self.num_classes = num_classes
        self.timesteps = timesteps
        self.unet_type = unet_type
        self.diffusion_type = diffusion_type
        self.morph_type = morph_type
        self.morph_routine = morph_routine
        
        if not DIFFUSERS_AVAILABLE:
            raise ImportError("diffusers library not available. Install with: pip install diffusers")
        
        # Initialize UNet based on type
        if unet_type == "diffusers_2d":
            if pretrained_model_name_or_path:
                self.unet = UNet2DModel.from_pretrained(pretrained_model_name_or_path)
            else:
                self.unet = UNet2DModel(
                    sample_size=None,  # Will be inferred from input
                    in_channels=in_channels + num_classes,
                    out_channels=num_classes,
                    layers_per_block=2,
                    block_out_channels=(64, 128, 256, 512),
                    down_block_types=("DownBlock2D", "DownBlock2D", "DownBlock2D", "AttnDownBlock2D"),
                    up_block_types=("AttnUpBlock2D", "UpBlock2D", "UpBlock2D", "UpBlock2D"),
                )
        elif unet_type == "diffusers_2d_cond":
            if pretrained_model_name_or_path:
                self.unet = UNet2DConditionModel.from_pretrained(pretrained_model_name_or_path)
            else:
                self.unet = UNet2DConditionModel(
                    sample_size=None,  # Will be inferred from input
                    in_channels=in_channels + num_classes,
                    out_channels=num_classes,
                    layers_per_block=2,
                    block_out_channels=(64, 128, 256, 512),
                    down_block_types=("CrossAttnDownBlock2D", "CrossAttnDownBlock2D", "CrossAttnDownBlock2D", "DownBlock2D"),
                    up_block_types=("UpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D"),
                    cross_attention_dim=768,  # Standard dimension for conditioning
                )
        else:
            raise ValueError(f"Unsupported unet_type: {unet_type}. Choose from: 'diffusers_2d', 'diffusers_2d_cond'")
        
        # Initialize diffusion schedules based on type
        if diffusion_type == "gaussian":
            # Use diffusers scheduler instead of custom implementation
            if scheduler_type == "ddpm":
                self.scheduler = DDPMScheduler(
                    num_train_timesteps=timesteps,
                    beta_schedule="scaled_linear",
                    beta_start=0.00085,
                    beta_end=0.012,
                    clip_sample=False,
                )
            elif scheduler_type == "ddim":
                self.scheduler = DDIMScheduler(
                    num_train_timesteps=timesteps,
                    beta_schedule="scaled_linear",
                    beta_start=0.00085,
                    beta_end=0.012,
                    clip_sample=False,
                )
            else:
                raise ValueError(f"Unsupported scheduler_type: {scheduler_type}. Choose from: 'ddpm', 'ddim'")
                
        elif diffusion_type == "morphological":
            # NEW: Use polynomial-based morphological operations
            self.morph_degradation = MorphologicalDegradation(
                morph_routine=morph_routine,
                kernel_size_start=morph_kernel_size_start,
                kernel_size_end=morph_kernel_size_end,
                num_timesteps=timesteps,
                channels=num_classes,
                morph_type=morph_type,
                device_of_kernel='cuda' if torch.cuda.is_available() else 'cpu'
            )
            
            # Create intensity schedule using the new function
            self.register_buffer('morph_schedule', create_morphological_schedule(timesteps, morph_schedule_type))
            
            # Move to GPU if available
            if torch.cuda.is_available():
                self.morph_degradation = self.morph_degradation.cuda()
        else:
            raise ValueError(f"Unsupported diffusion_type: {diffusion_type}. Choose from: 'gaussian', 'morphological'")

    def forward_diffusion(self, x0: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward diffusion process using diffusers scheduler"""
        noise = torch.randn_like(x0)
        noisy_x = self.scheduler.add_noise(x0, noise, t)
        return noisy_x, noise

    # def forward_morphology(self, x0: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    #     """Apply forward morphological process using polynomial operations."""
    #     batch_size = x0.shape[0]
    #     device = x0.device
        
    #     result = torch.zeros_like(x0)
        
    #     for i in range(batch_size):
    #         # Get timestep for this batch element
    #         timestep = t[i].item()
            
    #         # Get intensity from schedule
    #         intensity = self.morph_schedule[timestep].item()
            
    #         # Extract single mask and ensure it's in [0,1] range
    #         current_mask = x0[i:i+1]
    #         current_mask = torch.clamp(current_mask, 0, 1)
            
    #         # Apply morphological degradation using polynomial operations
    #         degraded_mask = self.morph_degradation.forward(current_mask, timestep, intensity)
            
    #         result[i:i+1] = degraded_mask
        
    #     return result

    # def forward_morphology_batch(self, x0: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    #     """
    #     Optimized batch version of morphological forward process.
    #     """
    #     # Ensure x0 is in [0,1] range
    #     x0 = torch.clamp(x0, 0, 1)
        
    #     # For batch processing, we need to handle different timesteps
    #     # Group by timestep for efficiency
    #     unique_timesteps = torch.unique(t)
    #     result = torch.zeros_like(x0)
        
    #     for timestep in unique_timesteps:
    #         # Find all batch elements with this timestep
    #         mask_indices = (t == timestep)
    #         if mask_indices.sum() == 0:
    #             continue
                
    #         # Get intensity for this timestep
    #         intensity = self.morph_schedule[timestep.item()].item()
            
    #         # Extract masks for this timestep
    #         batch_masks = x0[mask_indices]
            
    #         # Apply morphological degradation
    #         degraded_batch = self.morph_degradation.forward(batch_masks, timestep.item(), intensity)
            
    #         # Store results
    #         result[mask_indices] = degraded_batch
        
    #     return result

    def forward_morphology(self, x0: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Apply forward morphological process using polynomial operations - CUMULATIVE VERSION."""
        batch_size = x0.shape[0]
        device = x0.device
        
        result = torch.zeros_like(x0)
        
        for i in range(batch_size):
            # Get timestep for this batch element
            timestep = t[i].item()
            
            # Extract single mask and ensure it's in [0,1] range
            current_mask = x0[i:i+1]
            current_mask = torch.clamp(current_mask, 0, 1)
            
            # CUMULATIVE DEGRADATION: Apply step by step from 0 to timestep
            for step in range(timestep + 1):
                # Get intensity from schedule for this step
                intensity = self.morph_schedule[step].item()
                
                # Apply morphological degradation for this single step
                current_mask = self.morph_degradation.forward(current_mask, step, intensity)
            
            result[i:i+1] = current_mask
        
        return result


    def forward_morphology_batch(self, x0: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Optimized batch version of morphological forward process - CUMULATIVE VERSION.
        """
        # Ensure x0 is in [0,1] range
        x0 = torch.clamp(x0, 0, 1)
        
        # For cumulative processing, we need to process each unique timestep sequence
        # Group by timestep for efficiency
        unique_timesteps = torch.unique(t)
        result = torch.zeros_like(x0)
        
        for timestep in unique_timesteps:
            # Find all batch elements with this timestep
            mask_indices = (t == timestep)
            if mask_indices.sum() == 0:
                continue
            
            # Extract masks for this timestep
            batch_masks = x0[mask_indices]
            
            # CUMULATIVE DEGRADATION: Apply step by step from 0 to timestep
            current_batch = batch_masks.clone()
            for step in range(timestep.item() + 1):
                # Get intensity for this step
                intensity = self.morph_schedule[step].item()
                
                # Apply morphological degradation for this single step to entire batch
                current_batch = self.morph_degradation.forward(current_batch, step, intensity)
            
            # Store results
            result[mask_indices] = current_batch
        
        return result

    def _call_unet(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Call UNet with appropriate parameters based on type."""
        if self.unet_type == "diffusers_2d":
            # UNet2DModel expects (sample, timestep)
            return self.unet(x, t).sample
        elif self.unet_type == "diffusers_2d_cond":
            # UNet2DConditionModel expects (sample, timestep, encoder_hidden_states)
            # For segmentation, we don't use conditioning, so pass None
            return self.unet(x, t, encoder_hidden_states=None).sample
        else:
            raise ValueError(f"Unsupported unet_type: {self.unet_type}")

    def forward(self, image: torch.Tensor, mask: torch.Tensor = None, t: torch.Tensor = None) -> torch.Tensor:
        if self.training:
            if mask is None:
                raise ValueError("Mask is required during training")
            if t is None:
                t = torch.randint(0, self.timesteps, (image.shape[0],), device=image.device)
            
            if self.diffusion_type == "gaussian":
                # Gaussian diffusion: predict noise
                noisy_mask, noise = self.forward_diffusion(mask, t)
                
                # Concatenate image and noisy mask
                x = torch.cat([image, noisy_mask], dim=1)
                
                # Predict noise
                predicted_noise = self._call_unet(x, t)
                return predicted_noise, noise
                
            elif self.diffusion_type == "morphological":
                # Morphological diffusion: predict original mask
                # Use batch version for efficiency
                morphed_mask = self.forward_morphology_batch(mask, t)
                
                # Concatenate image and morphed mask
                x = torch.cat([image, morphed_mask], dim=1)
                
                # Predict original mask
                predicted_mask = self._call_unet(x, t)
                return predicted_mask, mask
                
        else:
            # Inference mode
            if self.diffusion_type == "gaussian":
                # Start from pure noise
                if mask is None:
                    mask = torch.randn(image.shape[0], self.num_classes, image.shape[2], image.shape[3], device=image.device)
            elif self.diffusion_type == "morphological":
                # Start from appropriate initial state based on morphological type
                if mask is None:
                    mask = self._get_initial_morphological_state(image)
            
            return self.sample(image, mask)

    def _get_initial_morphological_state(self, image: torch.Tensor) -> torch.Tensor:
        """Get appropriate initial state for morphological diffusion based on operation type."""
        shape = (image.shape[0], self.num_classes, image.shape[2], image.shape[3])
        device = image.device
        
        if self.morph_type in ["erosion", "opening"]:
            # Start from all ones for operations that reduce the mask
            return torch.ones(shape, device=device)
        elif self.morph_type in ["dilation", "closing"]:
            # Start from all zeros for operations that expand the mask
            return torch.zeros(shape, device=device)
        elif self.morph_type == "mixed":
            # For mixed operations, start from a middle state
            return torch.full(shape, 0.5, device=device)
        else:
            # Default to zeros
            return torch.zeros(shape, device=device)

    # def sample(self, image: torch.Tensor, mask: torch.Tensor = None, num_inference_steps: int = 50) -> torch.Tensor:
    #     """
    #     Enhanced sampling with proper morphological reverse process.
    #     """
    #     # Initialize mask if not provided
    #     if mask is None:
    #         if self.diffusion_type == "gaussian":
    #             # Start from pure noise
    #             mask = torch.randn(image.shape[0], self.num_classes, image.shape[2], image.shape[3], device=image.device)
    #         elif self.diffusion_type == "morphological":
    #             # Start from appropriate morphological state
    #             mask = self._get_initial_morphological_state(image)
        
    #     if self.diffusion_type == "gaussian":
    #         # Use diffusers scheduler for sampling
    #         self.scheduler.set_timesteps(num_inference_steps)
            
    #         for t in self.scheduler.timesteps:
    #             t_tensor = torch.full((image.shape[0],), t, device=image.device, dtype=torch.long)
                
    #             # Predict noise
    #             with torch.no_grad():
    #                 x = torch.cat([image, mask], dim=1)
    #                 predicted_noise = self._call_unet(x, t_tensor)
                
    #             # Use scheduler to compute previous sample
    #             mask = self.scheduler.step(predicted_noise, t, mask).prev_sample
            
    #         return torch.sigmoid(mask)
            
    #     elif self.diffusion_type == "morphological":
    #         # Enhanced reverse morphological process
    #         step_size = max(1, self.timesteps // num_inference_steps)
            
    #         # Create timestep schedule
    #         timesteps = list(range(self.timesteps - 1, -1, -step_size))
    #         if timesteps[-1] != 0:
    #             timesteps.append(0)  # Ensure we end at t=0
            
    #         for i, current_t in enumerate(timesteps):
    #             t_tensor = torch.full((image.shape[0],), current_t, device=image.device, dtype=torch.long)
                
    #             # Predict the clean mask
    #             with torch.no_grad():
    #                 x = torch.cat([image, mask], dim=1)
    #                 predicted_clean_mask = self._call_unet(x, t_tensor)
                
    #             if current_t > 0:
    #                 # We're still in the reverse process
    #                 # Progressive blending: trust prediction more as we approach t=0
    #                 alpha = 1.0 - (current_t / self.timesteps)
                    
    #                 # Enhanced blending with morphological consistency
    #                 if i > 0:  # Not the first step
    #                     # Apply light morphological smoothing to maintain structure
    #                     prev_t = timesteps[i-1] if i > 0 else self.timesteps
    #                     morphological_weight = min(0.3, (prev_t - current_t) / self.timesteps)
                        
    #                     # Apply inverse morphological operation for smoothing
    #                     if self.morph_type in ["dilation", "mixed"]:
    #                         # Light erosion to counter excessive dilation
    #                         smoothed = self.morph_degradation.morph_operators[0].polynomial_erosion(predicted_clean_mask)
    #                         predicted_clean_mask = (1 - morphological_weight) * predicted_clean_mask + morphological_weight * smoothed
    #                     elif self.morph_type in ["erosion"]:
    #                         # Light dilation to counter excessive erosion
    #                         smoothed = self.morph_degradation.morph_operators[0].polynomial_dilation(predicted_clean_mask)
    #                         predicted_clean_mask = (1 - morphological_weight) * predicted_clean_mask + morphological_weight * smoothed
                    
    #                 # Blend current mask with prediction
    #                 mask = alpha * predicted_clean_mask + (1 - alpha) * mask
    #             else:
    #                 # Final step: use the predicted clean mask
    #                 mask = predicted_clean_mask
                
    #             # Apply constraints and ensure valid range
    #             mask = torch.clamp(mask, 0, 1)
                
    #             # Optional: Apply slight sharpening in later steps
    #             if current_t < self.timesteps * 0.3:  # Last 30% of steps
    #                 # Gentle sharpening to improve boundary definition
    #                 sharpening_strength = (self.timesteps * 0.3 - current_t) / (self.timesteps * 0.3)
    #                 sharpened = torch.sigmoid((mask - 0.5) * (1 + sharpening_strength * 2)) 
    #                 mask = 0.8 * mask + 0.2 * sharpened
            
    #         # Final sigmoid for clean output
    #         return torch.sigmoid(mask)
    def sample(self, image: torch.Tensor, mask: torch.Tensor = None, num_inference_steps: int = 50) -> torch.Tensor:
        """
        Enhanced sampling with proper morphological reverse process - FIXED FOR CUMULATIVE DEGRADATION.
        """
        # Initialize mask if not provided
        if mask is None:
            if self.diffusion_type == "gaussian":
                # Start from pure noise
                mask = torch.randn(image.shape[0], self.num_classes, image.shape[2], image.shape[3], device=image.device)
            elif self.diffusion_type == "morphological":
                # Start from appropriate morphological state
                mask = self._get_initial_morphological_state(image)
        
        if self.diffusion_type == "gaussian":
            # Use diffusers scheduler for sampling (unchanged)
            self.scheduler.set_timesteps(num_inference_steps)
            
            for t in self.scheduler.timesteps:
                t_tensor = torch.full((image.shape[0],), t, device=image.device, dtype=torch.long)
                
                # Predict noise
                with torch.no_grad():
                    x = torch.cat([image, mask], dim=1)
                    predicted_noise = self._call_unet(x, t_tensor)
                
                # Use scheduler to compute previous sample
                mask = self.scheduler.step(predicted_noise, t, mask).prev_sample
            
            return torch.sigmoid(mask)
            
        elif self.diffusion_type == "morphological":
            # SIMPLIFIED reverse morphological process for cumulative degradation
            step_size = max(1, self.timesteps // num_inference_steps)
            
            # Create timestep schedule
            timesteps = list(range(self.timesteps - 1, -1, -step_size))
            if timesteps[-1] != 0:
                timesteps.append(0)  # Ensure we end at t=0
            
            for i, current_t in enumerate(timesteps):
                t_tensor = torch.full((image.shape[0],), current_t, device=image.device, dtype=torch.long)
                
                # Predict the clean mask
                with torch.no_grad():
                    x = torch.cat([image, mask], dim=1)
                    predicted_clean_mask = self._call_unet(x, t_tensor)
                
                if current_t > 0:
                    # SIMPLIFIED blending for cumulative degradation
                    # The network is trained to predict clean masks directly, so trust it more
                    alpha = 1.0 - (current_t / self.timesteps)
                    
                    # Simple progressive blending - no complex morphological operations needed
                    mask = alpha * predicted_clean_mask + (1 - alpha) * mask
                else:
                    # Final step: use the predicted clean mask directly
                    mask = predicted_clean_mask
                
                # Apply constraints and ensure valid range
                mask = torch.clamp(mask, 0, 1)
            
            # Final sigmoid for clean output
            return torch.sigmoid(mask)

    def compute_loss(self, predicted: torch.Tensor, target: torch.Tensor, 
                    loss_type: str = "mse", use_morphological_loss: bool = True) -> torch.Tensor:
        """
        Compute loss with optional morphological consistency terms.
        """
        # Primary reconstruction loss
        if loss_type == "mse":
            main_loss = F.mse_loss(predicted, target)
        elif loss_type == "l1":
            main_loss = F.l1_loss(predicted, target)
        else:
            raise ValueError(f"Unsupported loss_type: {loss_type}")
        
        if not use_morphological_loss or self.diffusion_type != "morphological":
            return main_loss
        
        # Add morphological consistency loss
        total_loss = main_loss
        
        # Dice loss for better segmentation
        smooth = 1e-5
        pred_flat = predicted.view(-1)
        target_flat = target.view(-1)
        intersection = (pred_flat * target_flat).sum()
        dice = (2. * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)
        dice_loss = 1 - dice
        
        # Boundary loss
        if predicted.shape[-1] > 1 and predicted.shape[-2] > 1:
            pred_grad_x = torch.abs(predicted[:, :, :, 1:] - predicted[:, :, :, :-1])
            pred_grad_y = torch.abs(predicted[:, :, 1:, :] - predicted[:, :, :-1, :])
            target_grad_x = torch.abs(target[:, :, :, 1:] - target[:, :, :, :-1])
            target_grad_y = torch.abs(target[:, :, 1:, :] - target[:, :, :-1, :])
            
            boundary_loss = F.mse_loss(pred_grad_x, target_grad_x) + F.mse_loss(pred_grad_y, target_grad_y)
        else:
            boundary_loss = torch.tensor(0.0, device=predicted.device)
        
        # Combine losses with appropriate weights
        total_loss = main_loss + 0.3 * dice_loss + 0.2 * boundary_loss
        
        return total_loss

    def get_morphological_debug_info(self) -> dict:
        """Get debugging information about morphological operations."""
        if self.diffusion_type != "morphological":
            return {"message": "Not using morphological diffusion"}
        
        info = {
            "morph_type": self.morph_type,
            "morph_routine": self.morph_routine,
            "num_timesteps": self.timesteps,
            "schedule_type": "exponential",  # We're using exponential
            "kernel_sizes": [],
            "intensities_sample": []
        }
        
        # Get kernel sizes for each timestep
        for i, op in enumerate(self.morph_degradation.morph_operators):
            if hasattr(op, 'kernel_size'):
                info["kernel_sizes"].append(op.kernel_size)
            if i < 10:  # Sample first 10 intensities
                info["intensities_sample"].append(self.morph_schedule[i].item())
        
        return info

    def get_batch_training_info(self, image: torch.Tensor, mask: torch.Tensor, 
                               t: torch.Tensor = None) -> dict:
        """
        Get detailed information about a training batch including the actual forward process.
        
        Args:
            image: Input image tensor [B, C, H, W]
            mask: Ground truth mask tensor [B, 1, H, W] 
            t: Timestep tensor [B] (if None, will be randomly sampled)
            
        Returns:
            Dictionary with batch training information
        """
        batch_size = image.shape[0]
        device = image.device
        
        # Sample timesteps if not provided
        if t is None:
            t = torch.randint(0, self.timesteps, (batch_size,), device=device)
        
        batch_info = {
            'batch_size': batch_size,
            'image_shape': list(image.shape),
            'mask_shape': list(mask.shape),
            'timesteps': t.cpu().tolist(),
            'original_images': image.cpu(),
            'original_masks': mask.cpu(),
            'diffusion_type': self.diffusion_type,
        }
        
        if self.diffusion_type == "gaussian":
            # Apply gaussian noise
            noisy_masks, noise = self.forward_diffusion(mask, t)
            model_input = torch.cat([image, noisy_masks], dim=1)
            
            batch_info.update({
                'noisy_masks': noisy_masks.cpu(),
                'noise': noise.cpu(),
                'model_input': model_input.cpu(),
                'target': noise.cpu()  # For gaussian, target is the noise
            })
            
        elif self.diffusion_type == "morphological":
            # Apply morphological degradation
            degraded_masks = self.forward_morphology_batch(mask, t)
            model_input = torch.cat([image, degraded_masks], dim=1)
            
            # Get intensities for each timestep
            intensities = [self.morph_schedule[timestep.item()].item() for timestep in t]
            
            batch_info.update({
                'intensities': intensities,
                'degraded_masks': degraded_masks.cpu(),
                'model_input': model_input.cpu(),
                'target': mask.cpu()  # For morphological, target is the original mask
            })
        
        return batch_info

    def visualize_training_flow(self, image: torch.Tensor, mask: torch.Tensor, 
                               num_timesteps_to_show: int = 6) -> dict:
        """
        Create training flow visualization data showing degradation process.
        
        Args:
            image: Input image tensor [B, C, H, W]
            mask: Ground truth mask tensor [B, 1, H, W]
            num_timesteps_to_show: Number of timesteps to visualize
            
        Returns:
            Dictionary with visualization data
        """
        device = image.device
        batch_size = image.shape[0]
        
        # Select timesteps to visualize
        timestep_indices = torch.linspace(0, self.timesteps - 1, num_timesteps_to_show).long()
        
        flow_data = {
            'original_images': image.cpu(),
            'original_masks': mask.cpu(),
            'timesteps': timestep_indices.tolist(),
            'degraded_masks': [],
            'intensities': [],
            'diffusion_type': self.diffusion_type
        }
        
        if self.diffusion_type == "gaussian":
            # Show gaussian noise progression
            for timestep in timestep_indices:
                t_tensor = torch.full((batch_size,), timestep, device=device)
                noisy_masks, _ = self.forward_diffusion(mask, t_tensor)
                flow_data['degraded_masks'].append(noisy_masks.cpu())
                flow_data['intensities'].append(timestep.item() / self.timesteps)
                
        elif self.diffusion_type == "morphological":
            # Show morphological degradation progression
            for timestep in timestep_indices:
                t_tensor = torch.full((batch_size,), timestep, device=device)
                degraded_masks = self.forward_morphology_batch(mask, t_tensor)
                intensity = self.morph_schedule[timestep.item()].item()
                
                flow_data['degraded_masks'].append(degraded_masks.cpu())
                flow_data['intensities'].append(intensity)
        
        return flow_data