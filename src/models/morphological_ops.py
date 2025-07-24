"""
Enhanced morphological operations for diffusion segmentation.
Implements polynomial-based morphological operations for numerical stability and better performance.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional


class PolynomialMorphology(nn.Module):
    """Polynomial-based morphological operations - numerically stable alternative to temperature scaling."""
    
    def __init__(self, kernel_size: int = 3, channels: int = 1):
        super().__init__()
        self.kernel_size = kernel_size
        self.channels = channels
        
        # Create structural element (circular kernel)
        self.register_buffer('kernel', self._create_circular_kernel(kernel_size))
        
        # Padding for maintaining spatial dimensions
        self.padding = kernel_size // 2
    
    def _create_circular_kernel(self, size: int) -> torch.Tensor:
        """Create a circular structural element."""
        center = size // 2
        y, x = torch.meshgrid(torch.arange(size), torch.arange(size), indexing='ij')
        kernel = ((x - center)**2 + (y - center)**2) <= (center**2)
        return kernel.float()
    
    def _extract_neighborhoods(self, x: torch.Tensor) -> torch.Tensor:
        """Extract neighborhoods using unfold operation - memory efficient."""
        # Use unfold to get sliding windows
        unfolded = F.unfold(x, kernel_size=self.kernel_size, padding=self.padding)
        B, _, num_patches = unfolded.shape
        H = W = int(np.sqrt(num_patches))
        
        # Reshape to [B, C, kernel_size^2, H, W]
        neighborhoods = unfolded.view(B, self.channels, self.kernel_size**2, H, W)
        
        # Apply structural element mask
        kernel_mask = self.kernel.view(1, 1, -1, 1, 1).to(neighborhoods.device)
        masked_neighborhoods = neighborhoods * kernel_mask
        
        return masked_neighborhoods
    
    def polynomial_dilation(self, x: torch.Tensor) -> torch.Tensor:
        """Polynomial-based dilation: 1 - product(1-xi) for each neighborhood."""
        neighborhoods = self._extract_neighborhoods(x)
        
        # Convert max operation to polynomial: max(x1,x2,...,xn) ≈ 1 - ∏(1-xi)
        complement = 1.0 - neighborhoods
        
        # Only consider pixels within the structural element
        kernel_mask = self.kernel.view(1, 1, -1, 1, 1).to(neighborhoods.device)
        complement = complement * kernel_mask + (1 - kernel_mask)  # Set masked areas to 1
        
        # Compute product along neighborhood dimension
        product = torch.prod(complement, dim=2)
        result = 1.0 - product
        
        # Ensure output is in valid range
        return torch.clamp(result, 0, 1)
    
    def polynomial_erosion(self, x: torch.Tensor) -> torch.Tensor:
        """Polynomial-based erosion: product(xi) for each neighborhood."""
        neighborhoods = self._extract_neighborhoods(x)
        
        # Erosion: min(x1,x2,...,xn) ≈ ∏(xi)
        kernel_mask = self.kernel.view(1, 1, -1, 1, 1).to(neighborhoods.device)
        masked_neighborhoods = neighborhoods * kernel_mask + (1 - kernel_mask)  # Set masked areas to 1
        
        # Compute product along neighborhood dimension
        result = torch.prod(masked_neighborhoods, dim=2)
        
        # Ensure output is in valid range
        return torch.clamp(result, 0, 1)
    
    def polynomial_opening(self, x: torch.Tensor) -> torch.Tensor:
        """Opening = erosion followed by dilation."""
        return self.polynomial_dilation(self.polynomial_erosion(x))
    
    def polynomial_closing(self, x: torch.Tensor) -> torch.Tensor:
        """Closing = dilation followed by erosion."""
        return self.polynomial_erosion(self.polynomial_dilation(x))


class MorphologicalDegradation(nn.Module):
    """
    Morphological degradation process for diffusion segmentation.
    """
    
    def __init__(self, 
                 morph_routine='Progressive', 
                 kernel_size_start=3,
                 kernel_size_end=9,
                 num_timesteps=50,
                 channels=1,
                 morph_type='mixed',
                 device_of_kernel='cuda',
                 morph_operators=None):
        super().__init__()
        
        self.morph_routine = morph_routine
        self.kernel_size_start = kernel_size_start
        self.kernel_size_end = kernel_size_end
        self.num_timesteps = num_timesteps
        self.channels = channels
        self.morph_type = morph_type
        self.device_of_kernel = device_of_kernel
        
        # Create morphological operators for each timestep
        self.morph_operators = self._create_morphological_schedule()
    
    def _create_morphological_schedule(self):
        """Create schedule of morphological operators with progressive kernel growth."""
        operators = []
        
        if self.morph_routine == 'Progressive':
            # Linearly increase kernel size over timesteps
            kernel_sizes = np.linspace(self.kernel_size_start, self.kernel_size_end, self.num_timesteps)
            kernel_sizes = [int(k) if int(k) % 2 == 1 else int(k) + 1 for k in kernel_sizes]  # Ensure odd
            
            for i, kernel_size in enumerate(kernel_sizes):
                morph_op = PolynomialMorphology(kernel_size=kernel_size, channels=self.channels)
                if self.device_of_kernel == 'cuda':
                    morph_op = morph_op.cuda()
                operators.append(morph_op)
        
        elif self.morph_routine == 'Constant':
            # Use constant kernel size
            for i in range(self.num_timesteps):
                morph_op = PolynomialMorphology(kernel_size=self.kernel_size_start, channels=self.channels)
                if self.device_of_kernel == 'cuda':
                    morph_op = morph_op.cuda()
                operators.append(morph_op)
        
        return nn.ModuleList(operators)
    
    def forward(self, x: torch.Tensor, i: int, intensity: float = None) -> torch.Tensor:
        """Apply morphological degradation at timestep i."""
        if intensity is None:
            # Default intensity based on timestep
            intensity = (i + 1) / self.num_timesteps
        
        # Clamp intensity to valid range
        intensity = max(0.0, min(1.0, intensity))
        
        if intensity == 0:
            return x.clone()
        
        result = x.clone()
        morph_op = self.morph_operators[min(i, len(self.morph_operators) - 1)]
        
        if self.morph_type == "dilation":
            morphed = morph_op.polynomial_dilation(result)
            result = (1 - intensity) * result + intensity * morphed
            
        elif self.morph_type == "erosion":
            morphed = morph_op.polynomial_erosion(result)
            result = (1 - intensity) * result + intensity * morphed
            
        elif self.morph_type == "opening":
            morphed = morph_op.polynomial_opening(result)
            result = (1 - intensity) * result + intensity * morphed
            
        elif self.morph_type == "closing":
            morphed = morph_op.polynomial_closing(result)
            result = (1 - intensity) * result + intensity * morphed
            
        elif self.morph_type == "mixed":
            # Adaptive mixing based on intensity
            if intensity < 0.5:
                # More erosion for early timesteps
                weight = intensity * 2
                morphed = morph_op.polynomial_erosion(result)
                result = (1 - weight) * result + weight * morphed
            else:
                # More dilation for later timesteps
                weight = (intensity - 0.5) * 2
                morphed = morph_op.polynomial_dilation(result)
                result = (1 - weight) * result + weight * morphed
        
        # Ensure output is in valid range and check for NaN
        result = torch.clamp(result, 0, 1)
        
        if torch.isnan(result).any():
            print(f"Warning: NaN detected at timestep {i}, returning original")
            return x.clone()
        
        return result
    
    def total_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply complete morphological degradation (all timesteps)."""
        result = x.clone()
        for i in range(self.num_timesteps):
            result = self.forward(result, i)
        return result
    
    @torch.no_grad()
    def reset_parameters(self, batch_size=32):
        """Reset parameters - required for compatibility."""
        pass
    
    # def get_debug_info(self) -> dict:
    #     """Get debugging information about morphological operations."""
    #     info = {
    #         "morph_type": self.morph_type,
    #         "morph_routine": self.morph_routine,
    #         "num_timesteps": self.num_timesteps,
    #         "kernel_sizes": [],
    #         "intensities_sample": []
    #     }
        
    #     # Get kernel sizes for each timestep
    #     for i, op in enumerate(self.morph_operators):
    #         if hasattr(op, 'kernel_size'):
    #             info["kernel_sizes"].append(op.kernel_size)
        
    #     return info
    
    # def get_batch_training_info(self, image: torch.Tensor, mask: torch.Tensor, 
    #                            t: torch.Tensor = None) -> dict:
    #     """
    #     Get detailed information about a training batch including the actual forward process.
        
    #     Args:
    #         image: Input image tensor [B, C, H, W]
    #         mask: Ground truth mask tensor [B, 1, H, W] 
    #         t: Timestep tensor [B] (if None, will be randomly sampled)
            
    #     Returns:
    #         Dictionary with batch training information
    #     """
    #     batch_size = image.shape[0]
    #     device = image.device
        
    #     # Sample timesteps if not provided
    #     if t is None:
    #         t = torch.randint(0, self.num_timesteps, (batch_size,), device=device)
        
    #     batch_info = {
    #         'batch_size': batch_size,
    #         'image_shape': list(image.shape),
    #         'mask_shape': list(mask.shape),
    #         'timesteps': t.cpu().tolist(),
    #         'original_images': image.cpu(),
    #         'original_masks': mask.cpu(),
    #         'diffusion_type': 'morphological',
    #     }
        
    #     # Apply morphological degradation
    #     degraded_masks = torch.zeros_like(mask)
    #     intensities = []
        
    #     for i in range(batch_size):
    #         timestep = t[i].item()
    #         # Use default intensity calculation
    #         intensity = (timestep + 1) / self.num_timesteps
    #         intensities.append(intensity)
            
    #         # Apply degradation
    #         degraded_mask = self.forward(mask[i:i+1], timestep, intensity)
    #         degraded_masks[i:i+1] = degraded_mask
        
    #     model_input = torch.cat([image, degraded_masks], dim=1)
        
    #     batch_info.update({
    #         'intensities': intensities,
    #         'degraded_masks': degraded_masks.cpu(),
    #         'model_input': model_input.cpu(),
    #         'target': mask.cpu()  # For morphological, target is the original mask
    #     })
        
    #     return batch_info

# # Keep the old SoftMorphology class for backward compatibility, but mark as deprecated
# class SoftMorphology(nn.Module):
#     """
#     DEPRECATED: Use PolynomialMorphology instead for better numerical stability.
    
#     Differentiable morphological operations using soft approximations.
#     This class is kept for backward compatibility but should not be used in new code.
#     """
    
#     def __init__(self, kernel_size: int = 3, temperature: float = 1.0):
#         super().__init__()
#         print("WARNING: SoftMorphology is deprecated. Use PolynomialMorphology for better results.")
#         self.kernel_size = kernel_size
#         self.temperature = max(temperature, 0.1)
        
#         # Create structural element (circular kernel)
#         self.register_buffer('kernel', self._create_circular_kernel(kernel_size))
    
#     def _create_circular_kernel(self, size: int) -> torch.Tensor:
#         """Create a circular structural element."""
#         center = size // 2
#         y, x = torch.meshgrid(torch.arange(size), torch.arange(size), indexing='ij')
#         kernel = ((x - center)**2 + (y - center)**2) <= (center**2)
#         return kernel.float()
    
#     def soft_dilation(self, x: torch.Tensor) -> torch.Tensor:
#         """Soft dilation using max-pooling with temperature scaling."""
#         # Unfold to get neighborhoods
#         unfolded = F.unfold(x, kernel_size=self.kernel_size, padding=self.kernel_size//2)
#         # Reshape: [B, C, kernel_size^2, H*W]
#         B, C = x.shape[:2]
#         H, W = x.shape[2:]
#         unfolded = unfolded.view(B, C, self.kernel_size**2, H, W)
        
#         # Apply structural element mask
#         kernel_mask = self.kernel.view(1, 1, -1, 1, 1)
#         # Use large negative value instead of -inf to avoid NaN
#         masked = unfolded * kernel_mask + (1 - kernel_mask) * (-1e6)
        
#         # Soft maximum using temperature-scaled softmax
#         weights = F.softmax(masked / self.temperature, dim=2)
#         result = (unfolded * weights).sum(dim=2)
        
#         # Ensure output is in valid range
#         return torch.clamp(result, 0, 1)
    
#     def soft_erosion(self, x: torch.Tensor) -> torch.Tensor:
#         """Soft erosion using min-pooling with temperature scaling."""
#         # Unfold to get neighborhoods
#         unfolded = F.unfold(x, kernel_size=self.kernel_size, padding=self.kernel_size//2)
#         B, C = x.shape[:2]
#         H, W = x.shape[2:]
#         unfolded = unfolded.view(B, C, self.kernel_size**2, H, W)
        
#         # Apply structural element mask
#         kernel_mask = self.kernel.view(1, 1, -1, 1, 1)
#         # Use large positive value instead of inf to avoid NaN
#         masked = unfolded * kernel_mask + (1 - kernel_mask) * 1e6
        
#         # Soft minimum using temperature-scaled softmin
#         weights = F.softmax(-masked / self.temperature, dim=2)
#         result = (unfolded * weights).sum(dim=2)
        
#         # Ensure output is in valid range
#         return torch.clamp(result, 0, 1)
    
#     def soft_opening(self, x: torch.Tensor) -> torch.Tensor:
#         """Opening = erosion followed by dilation."""
#         return self.soft_dilation(self.soft_erosion(x))
    
#     def soft_closing(self, x: torch.Tensor) -> torch.Tensor:
#         """Closing = dilation followed by erosion."""
#         return self.soft_erosion(self.soft_dilation(x))


# class ConvolutionalMorphology(nn.Module):
#     """Differentiable morphological operations using learned convolutions."""
    
#     def __init__(self, kernel_size: int = 3, operation: str = 'dilation'):
#         super().__init__()
#         self.operation = operation
        
#         # Learnable morphological kernel
#         self.morph_conv = nn.Conv2d(1, 1, kernel_size, padding=kernel_size//2, bias=False)
        
#         # Initialize with circular pattern
#         with torch.no_grad():
#             center = kernel_size // 2
#             y, x = torch.meshgrid(torch.arange(kernel_size), torch.arange(kernel_size), indexing='ij')
#             circular_mask = ((x - center)**2 + (y - center)**2) <= (center**2)
#             self.morph_conv.weight.data = circular_mask.float().unsqueeze(0).unsqueeze(0)
    
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         """Apply morphological operation."""
#         # Apply convolution
#         conv_result = self.morph_conv(x)
        
#         if self.operation == 'dilation':
#             # Dilation: any neighbor > 0 makes output > 0
#             return torch.sigmoid(conv_result * 10)  # Sharp transition
#         elif self.operation == 'erosion':
#             # Erosion: all neighbors must be > 0
#             kernel_size = self.morph_conv.weight.shape[-1]
#             num_ones = (self.morph_conv.weight > 0.5).sum()
#             return torch.sigmoid((conv_result - num_ones + 1) * 10)
#         else:
#             return conv_result


class MorphologicalLoss(nn.Module):
    """Enhanced loss function for morphological diffusion with proper component weighting."""
    
    def __init__(self, 
                 loss_type: str = "l1", 
                 dice_weight: float = 0.3, 
                 boundary_weight: float = 0.2,
                 morphological_weight: float = 0.2,
                 perceptual_weight: float = 0.1):
        super().__init__()
        self.loss_type = loss_type.lower()
        self.dice_weight = dice_weight
        self.boundary_weight = boundary_weight
        self.morphological_weight = morphological_weight
        self.perceptual_weight = perceptual_weight
        
        # Initialize morphological operator for consistency loss
        self.morph_op = PolynomialMorphology(kernel_size=3, channels=1)
        
        if self.loss_type not in ['l1', 'l2', 'mse']:
            raise ValueError(f"Unsupported loss_type: {loss_type}")
    
    def reconstruction_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Primary reconstruction loss."""
        if self.loss_type == 'l1':
            return F.l1_loss(pred, target)
        elif self.loss_type == 'l2' or self.loss_type == 'mse':
            return F.mse_loss(pred, target)
    
    def dice_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Numerically stable Dice loss."""
        smooth = 1e-5
        
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        
        intersection = (pred_flat * target_flat).sum()
        union = pred_flat.sum() + target_flat.sum()
        
        dice = (2. * intersection + smooth) / (union + smooth)
        dice = torch.clamp(dice, 0.0, 1.0)
        
        return -torch.log(dice + 1e-8)
    
    def boundary_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Boundary-preserving loss using gradients."""
        if pred.shape[-1] < 2 or pred.shape[-2] < 2:
            return torch.tensor(0.0, device=pred.device, dtype=pred.dtype)
        
        # Compute gradients
        pred_grad_x = torch.abs(pred[:, :, :, 1:] - pred[:, :, :, :-1])
        pred_grad_y = torch.abs(pred[:, :, 1:, :] - pred[:, :, :-1, :])
        
        target_grad_x = torch.abs(target[:, :, :, 1:] - target[:, :, :, :-1])
        target_grad_y = torch.abs(target[:, :, 1:, :] - target[:, :, :-1, :])
        
        if self.loss_type == 'l1':
            loss_x = F.l1_loss(pred_grad_x, target_grad_x)
            loss_y = F.l1_loss(pred_grad_y, target_grad_y)
        else:
            loss_x = F.mse_loss(pred_grad_x, target_grad_x)
            loss_y = F.mse_loss(pred_grad_y, target_grad_y)
        
        return (loss_x + loss_y) / 2
    
    def morphological_consistency_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Ensure morphological operations are consistent between pred and target."""
        # Apply same morphological operation to both
        pred_dilated = self.morph_op.polynomial_dilation(pred)
        target_dilated = self.morph_op.polynomial_dilation(target)
        
        pred_eroded = self.morph_op.polynomial_erosion(pred)
        target_eroded = self.morph_op.polynomial_erosion(target)
        
        if self.loss_type == 'l1':
            dilation_loss = F.l1_loss(pred_dilated, target_dilated)
            erosion_loss = F.l1_loss(pred_eroded, target_eroded)
        else:
            dilation_loss = F.mse_loss(pred_dilated, target_dilated)
            erosion_loss = F.mse_loss(pred_eroded, target_eroded)
        
        return (dilation_loss + erosion_loss) / 2
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Combined loss with proper weighting."""
        # Primary reconstruction loss
        main_loss = self.reconstruction_loss(pred, target)
        total_loss = main_loss
        
        # Add component losses with proper weighting
        if self.dice_weight > 0:
            dice = self.dice_loss(pred, target)
            total_loss = total_loss + self.dice_weight * dice
        
        if self.boundary_weight > 0:
            boundary = self.boundary_loss(pred, target)
            total_loss = total_loss + self.boundary_weight * boundary
        
        if self.morphological_weight > 0:
            morph_consistency = self.morphological_consistency_loss(pred, target)
            total_loss = total_loss + self.morphological_weight * morph_consistency
        
        # Ensure loss is non-negative
        total_loss = torch.clamp(total_loss, min=0.0)
        
        return total_loss


# def create_morph_schedule(timesteps: int, schedule_type: str = "linear") -> torch.Tensor:
#     """
#     DEPRECATED: Use create_morphological_schedule instead.
#     Create schedule for morphological operation intensity.
#     """
#     print("WARNING: create_morph_schedule is deprecated. Use create_morphological_schedule instead.")
#     return create_morphological_schedule(timesteps, schedule_type)


def create_morphological_schedule(timesteps: int, schedule_type: str = "exponential") -> torch.Tensor:
    """Create proper intensity schedule for morphological operations."""
    if schedule_type == "linear":
        return torch.linspace(0, 1, timesteps)
    elif schedule_type == "exponential":
        # Exponential schedule for better morphological learning
        x = torch.linspace(0, 1, timesteps)
        return 1 - torch.exp(-3 * x)  # Rapid initial change, slower later
    elif schedule_type == "cosine":
        x = torch.linspace(0, 1, timesteps)
        return (1 - torch.cos(x * np.pi)) / 2
    elif schedule_type == "quadratic":
        x = torch.linspace(0, 1, timesteps)
        return x ** 2
    else:
        raise ValueError(f"Unknown schedule type: {schedule_type}")


def apply_morphological_degradation(mask: torch.Tensor, 
                                  intensity: float, 
                                  morph_type: str = "dilation",
                                  soft_morph: PolynomialMorphology = None) -> torch.Tensor:
    """
    Apply morphological degradation with given intensity using polynomial operations.
    
    Args:
        mask: Input mask tensor
        intensity: Degradation intensity (0-1)
        morph_type: Type of morphological operation
        soft_morph: Morphological operator (if None, creates a default one)
    """
    if soft_morph is None:
        soft_morph = PolynomialMorphology(kernel_size=3, channels=mask.shape[1] if len(mask.shape) > 1 else 1)
        if mask.is_cuda:
            soft_morph = soft_morph.cuda()
    
    # Ensure intensity is in valid range
    intensity = torch.clamp(torch.tensor(intensity), 0, 1).item()
    
    if intensity == 0:
        return mask.clone()
    
    # Apply morphological operation based on intensity
    result = mask.clone()
    
    if morph_type == "dilation":
        # Gradually dilate based on intensity
        morphed = soft_morph.polynomial_dilation(result)
        result = (1 - intensity) * result + intensity * morphed
    elif morph_type == "erosion":
        # Gradually erode based on intensity
        morphed = soft_morph.polynomial_erosion(result)
        result = (1 - intensity) * result + intensity * morphed
    elif morph_type == "opening":
        # Apply opening with intensity
        morphed = soft_morph.polynomial_opening(result)
        result = (1 - intensity) * result + intensity * morphed
    elif morph_type == "closing":
        # Apply closing with intensity
        morphed = soft_morph.polynomial_closing(result)
        result = (1 - intensity) * result + intensity * morphed
    elif morph_type == "mixed":
        # Mix dilation and erosion based on intensity
        if intensity < 0.5:
            # More erosion
            weight = intensity * 2
            morphed = soft_morph.polynomial_erosion(result)
            result = (1 - weight) * result + weight * morphed
        else:
            # More dilation
            weight = (intensity - 0.5) * 2
            morphed = soft_morph.polynomial_dilation(result)
            result = (1 - weight) * result + weight * morphed
    
    # Ensure output is in valid range and check for NaN
    result = torch.clamp(result, 0, 1)
    
    # Check for NaN values and replace with original mask if found
    if torch.isnan(result).any():
        print("Warning: NaN values detected in morphological degradation, returning original mask")
        return mask.clone()
    
    return result


# Debugging and visualization utilities
class MorphologicalDebugger:
    """Debugging utilities for morphological diffusion models."""
    
    @staticmethod
    def check_gradient_flow(model, sample_input, timesteps_to_check=[0, 10, 25, 49]):
        """Check gradient magnitudes at different timesteps."""
        model.train()
        
        for t in timesteps_to_check:
            sample_input.requires_grad_(True)
            t_tensor = torch.full((sample_input.shape[0],), t, dtype=torch.long, device=sample_input.device)
            
            # Forward pass
            if hasattr(model, 'compute_loss'):
                # For DiffusionSegmentation model
                pred, target = model(sample_input[:, :3], sample_input[:, 3:], t_tensor)
                loss = model.compute_loss(pred, target)
            else:
                # For other models
                loss = model(sample_input, t_tensor)
            
            # Backward pass
            loss.backward()
            
            grad_norm = sample_input.grad.norm().item()
            print(f"Timestep {t}: Loss = {loss.item():.6f}, Gradient norm = {grad_norm:.6f}")
            
            sample_input.grad.zero_()
    
    @staticmethod
    def visualize_degradation_sequence(mask, morph_degradation, num_steps=10):
        """Visualize the morphological degradation sequence."""
        sequence = [mask.clone()]
        current = mask.clone()
        
        step_size = morph_degradation.num_timesteps // num_steps
        
        for i in range(0, morph_degradation.num_timesteps, step_size):
            current = morph_degradation.forward(current, i)
            sequence.append(current.clone())
        
        return sequence
    
    @staticmethod
    def compare_morphological_operations(mask, kernel_sizes=[3, 5, 7]):
        """Compare different morphological operations and kernel sizes."""
        results = {}
        
        for kernel_size in kernel_sizes:
            morph_op = PolynomialMorphology(kernel_size=kernel_size, channels=mask.shape[1])
            if mask.is_cuda:
                morph_op = morph_op.cuda()
            
            results[f'dilation_{kernel_size}'] = morph_op.polynomial_dilation(mask)
            results[f'erosion_{kernel_size}'] = morph_op.polynomial_erosion(mask)
            results[f'opening_{kernel_size}'] = morph_op.polynomial_opening(mask)
            results[f'closing_{kernel_size}'] = morph_op.polynomial_closing(mask)
        
        return results