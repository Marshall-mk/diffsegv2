#!/usr/bin/env python3
"""
Demo script for visualizing morphological forward diffusion processes

This script tests different morphological operations (dilation, erosion, mixed, etc.)
and creates visualizations showing how each operation degrades an input image over time.

Requirements:
- Your MorphologicalDegradation class with proper schedule support
- create_morphological_schedule function for creating intensity schedules
- Input image (preferably a binary mask or segmentation)

Usage Examples:
    # Basic demo with default parameters
    python morphological_demo.py --input sample_mask.png
    
    # Comprehensive comparison of all operations
    python morphological_demo.py --input sample_mask.png \
        --timesteps 1000 --morph-types "dilation,erosion,mixed,opening,closing" \
        --schedule-type exponential --create-individual-plots --debug-mode
    
    # Focus on specific timesteps with custom schedule
    python morphological_demo.py --input sample_mask.png \
        --timesteps-to-plot "0,50,150,300,500,1000" \
        --schedule-type cosine --kernel-size-start 3 --kernel-size-end 15

Note: The script uses create_morphological_schedule() to create intensity schedules
      and then passes them to MorphologicalDegradation via the 'schedule' parameter.
"""

import torch
import torch.nn.functional as F
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import warnings

# Import your morphological operations
try:
    from src.models import MorphologicalDegradation, MorphologicalDebugger
    from src.models.morphological_ops import create_morphological_schedule
except ImportError as e:
    print(f"Error importing morphological operations: {e}")
    print("Please ensure your morphological_ops.py file contains:")
    print("- MorphologicalDegradation class")
    print("- MorphologicalDebugger class")
    print("- create_morphological_schedule function")
    raise e


def load_and_preprocess_image(
    image_path: str, size: Tuple[int, int] = (256, 256)
) -> torch.Tensor:
    """Load and preprocess an image for morphological diffusion"""

    # Load image
    image = Image.open(image_path).convert("L")  # Convert to grayscale
    image = image.resize(size, Image.Resampling.LANCZOS)

    # Convert to tensor and normalize to [0, 1]
    image_tensor = torch.tensor(np.array(image), dtype=torch.float32) / 255.0

    # Add batch and channel dimensions: (H, W) -> (1, 1, H, W)
    image_tensor = image_tensor.unsqueeze(0).unsqueeze(0)

    return image_tensor


def create_morphological_ops(
    morph_type: str,
    timesteps: int,
    kernel_size_start: int,
    kernel_size_end: int,
    morph_routine: str,
    schedule_type: str,
) -> MorphologicalDegradation:
    """Create morphological operations with specified parameters"""

    # Create the morphological schedule first
    schedule = create_morphological_schedule(
        schedule_type=schedule_type, timesteps=timesteps
    )

    # Create morphological operations with the schedule
    morph_ops = MorphologicalDegradation(
        morph_type=morph_type,
        num_timesteps=timesteps,
        kernel_size_start=kernel_size_start,
        kernel_size_end=kernel_size_end,
        morph_routine=morph_routine,
        channels=1,  # Assuming single-channel images
        morph_operators=schedule,  # Pass the created schedule
    )

    return morph_ops


def run_forward_process(
    image: torch.Tensor,
    morph_ops: MorphologicalDegradation,
    timesteps_to_plot: List[int],
    device: torch.device,
    use_training_mode: bool = True,
) -> Dict[int, torch.Tensor]:
    """Run forward morphological diffusion process"""

    image = image.to(device)
    results = {}

    print(f"Running forward process for {len(timesteps_to_plot)} timesteps...")
    print(
        f"Mode: {'Training-style (direct timestep)' if use_training_mode else 'Demo-style (cumulative)'}"
    )

    if use_training_mode:
        # TRAINING MODE: Apply degradation directly for each timestep (matches training)
        # This is what actually happens during training
        for timestep in timesteps_to_plot:
            if timestep == 0:
                results[timestep] = image.clone()
            else:
                # Apply degradation directly for this timestep (like in training)
                # This matches what forward_morphology_batch does
                t_tensor = torch.full((1,), timestep, device=device, dtype=torch.long)

                # Get intensity from the schedule for this specific timestep
                # This should match the schedule used in training
                schedule = create_morphological_schedule(
                    morph_ops.num_timesteps, "exponential"
                )

                # FIX: Ensure timestep index is within bounds
                schedule_index = min(timestep, len(schedule) - 1)
                intensity = schedule[schedule_index].item()

                # Apply morphological degradation for this specific timestep
                # Use the schedule_index for the morph_ops.forward call as well
                degraded = morph_ops.forward(image.clone(), schedule_index, intensity)
                results[timestep] = degraded
    else:
        # DEMO MODE: Apply degradation step-by-step (original demo behavior)
        # This creates cumulative degradation which is why it's so severe
        current_image = image.clone()
        results[0] = current_image.clone()

        max_timestep = (
            max(timesteps_to_plot) if timesteps_to_plot else morph_ops.num_timesteps
        )

        for t in range(1, max_timestep + 1):
            # Get intensity from schedule
            schedule = create_morphological_schedule(
                morph_ops.num_timesteps, "exponential"
            )

            # FIX: Ensure schedule index is within bounds
            schedule_index = min(t - 1, len(schedule) - 1)
            intensity = schedule[schedule_index].item()

            # Apply one step of degradation to the already degraded image
            current_image = morph_ops.forward(current_image, schedule_index, intensity)

            if t in timesteps_to_plot:
                results[t] = current_image.clone()

    return results


def create_comparison_collage(
    results_dict: Dict[str, Dict[int, torch.Tensor]],
    timesteps_to_plot: List[int],
    original_image: torch.Tensor,
    title: str,
    save_path: Optional[str] = None,
) -> None:
    """Create a collage comparing different morphological operations"""

    morph_types = list(results_dict.keys())
    n_morphs = len(morph_types)
    n_timesteps = len(timesteps_to_plot)

    # Create figure with appropriate size
    fig_width = max(15, n_timesteps * 2.5)
    fig_height = max(10, n_morphs * 2.5)

    fig = plt.figure(figsize=(fig_width, fig_height))
    gs = gridspec.GridSpec(
        n_morphs + 1, n_timesteps, height_ratios=[1] + [1] * n_morphs
    )

    # Add original image row
    for i, timestep in enumerate(timesteps_to_plot):
        ax = fig.add_subplot(gs[0, i])
        if timestep == 0:
            img_np = original_image.squeeze().cpu().numpy()
            ax.imshow(img_np, cmap="gray", vmin=0, vmax=1)
            ax.set_title(f"Original\n(t={timestep})", fontsize=10, fontweight="bold")
        else:
            ax.axis("off")
            ax.text(
                0.5,
                0.5,
                f"t={timestep}",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=12,
                fontweight="bold",
            )
        ax.axis("off")

    # Add morphological operation rows
    for row, morph_type in enumerate(morph_types, 1):
        results = results_dict[morph_type]

        for col, timestep in enumerate(timesteps_to_plot):
            ax = fig.add_subplot(gs[row, col])

            if timestep in results:
                img_np = results[timestep].squeeze().cpu().numpy()
                ax.imshow(img_np, cmap="gray", vmin=0, vmax=1)

                # Add timestep label
                if col == 0:
                    ax.set_ylabel(
                        f"{morph_type.title()}", fontsize=12, fontweight="bold"
                    )

                # Add intensity statistics
                mean_intensity = img_np.mean()
                ax.text(
                    0.02,
                    0.02,
                    f"μ={mean_intensity:.3f}",
                    transform=ax.transAxes,
                    fontsize=8,
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8),
                )
            else:
                ax.text(
                    0.5,
                    0.5,
                    "N/A",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                    fontsize=12,
                )

            ax.axis("off")

    plt.suptitle(title, fontsize=16, fontweight="bold", y=0.98)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"Saved collage: {save_path}")

    plt.show()


def create_individual_progression(
    results: Dict[int, torch.Tensor],
    morph_type: str,
    timesteps_to_plot: List[int],
    config_info: str,
    save_path: Optional[str] = None,
) -> None:
    """Create detailed progression plot for a single morphological operation"""

    n_timesteps = len(timesteps_to_plot)
    cols = min(6, n_timesteps)
    rows = (n_timesteps + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))

    # Handle axes indexing properly
    if rows == 1 and cols == 1:
        axes = [axes]
    elif rows == 1 or cols == 1:
        axes = axes.flatten()
    else:
        axes = axes.flatten()

    for i, timestep in enumerate(timesteps_to_plot):
        ax = axes[i]

        if timestep in results:
            img_np = results[timestep].squeeze().cpu().numpy()
            im = ax.imshow(img_np, cmap="gray", vmin=0, vmax=1)

            # Add colorbar for the first image
            if i == 0:
                cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                cbar.set_label("Intensity", fontsize=10)

            # Statistics
            mean_intensity = img_np.mean()
            std_intensity = img_np.std()
            min_intensity = img_np.min()
            max_intensity = img_np.max()

            ax.set_title(
                f"t={timestep}\nμ={mean_intensity:.3f}, σ={std_intensity:.3f}\n"
                f"[{min_intensity:.3f}, {max_intensity:.3f}]",
                fontsize=9,
            )
        else:
            ax.text(
                0.5,
                0.5,
                f"t={timestep}\nNot computed",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=12,
            )

        ax.axis("off")

    # Hide unused subplots
    for i in range(len(timesteps_to_plot), len(axes)):
        axes[i].axis("off")

    title = f"{morph_type.title()} Morphological Forward Process\n{config_info}"
    plt.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"Saved progression: {save_path}")

    plt.show()


def print_debug_info(
    morph_ops: MorphologicalDegradation, morph_type: str, schedule_type: str
) -> str:
    """Print debug information about morphological operations"""

    print(f"\n{'=' * 50}")
    print(f"DEBUG INFO: {morph_type.upper()} MORPHOLOGICAL OPERATIONS")
    print(f"{'=' * 50}")

    # Get debug information
    debug_info = morph_ops.get_debug_info()

    print(f"Morphological Type: {debug_info['morph_type']}")
    print(f"Morphological Routine: {debug_info['morph_routine']}")
    print(f"Total Timesteps: {debug_info['num_timesteps']}")
    print(f"Schedule Type: {schedule_type}")

    if debug_info["kernel_sizes"]:
        print(
            f"Kernel Sizes: {debug_info['kernel_sizes'][0]} → {debug_info['kernel_sizes'][-1]}"
        )
        print(
            f"Sample Kernel Progression: {debug_info['kernel_sizes'][:5]}...{debug_info['kernel_sizes'][-5:]}"
        )

    if debug_info.get("intensities_sample"):
        print(f"Sample Intensities: {debug_info['intensities_sample']}")

    # Print schedule information if available
    if hasattr(morph_ops, "schedule"):
        schedule_values = morph_ops.schedule
        print(
            f"Schedule Values: {schedule_values[:5].tolist()}...{schedule_values[-5:].tolist()}"
        )
        print(
            f"Schedule Range: [{schedule_values.min().item():.6f}, {schedule_values.max().item():.6f}]"
        )

    # Create configuration string for plots
    config_str = (
        f"Type: {debug_info['morph_type']}, "
        f"Routine: {debug_info['morph_routine']}, "
        f"Schedule: {schedule_type}, "
        f"Timesteps: {debug_info['num_timesteps']}, "
        f"Kernels: {debug_info['kernel_sizes'][0]}→{debug_info['kernel_sizes'][-1]}"
    )

    print(f"{'=' * 50}\n")

    return config_str


def main():
    parser = argparse.ArgumentParser(
        description="Demo Morphological Forward Diffusion Processes"
    )

    # Input parameters
    parser.add_argument("--input", type=str, required=True, help="Input image path")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./morph_demo_results",
        help="Output directory for results",
    )

    # Diffusion parameters
    parser.add_argument(
        "--timesteps",
        type=int,
        default=1000,
        help="Total number of diffusion timesteps",
    )
    parser.add_argument(
        "--timesteps-to-plot",
        type=str,
        default="0,50,100,200,400,600,800,1000",
        help='Comma-separated timesteps to visualize (e.g., "0,100,500,1000")',
    )

    # Morphological parameters
    parser.add_argument(
        "--morph-types",
        type=str,
        default="dilation,erosion,mixed,opening,closing",
        help="Comma-separated morphological operation types to test",
    )
    parser.add_argument(
        "--kernel-size-start", type=int, default=3, help="Starting kernel size"
    )
    parser.add_argument(
        "--kernel-size-end", type=int, default=9, help="Ending kernel size"
    )
    parser.add_argument(
        "--morph-routine",
        type=str,
        default="Progressive",
        choices=["Progressive", "Constant"],
        help="Morphological routine",
    )
    parser.add_argument(
        "--schedule-type",
        type=str,
        default="exponential",
        choices=["linear", "exponential", "cosine"],
        help="Intensity schedule type",
    )

    # Visualization parameters
    parser.add_argument(
        "--image-size", type=int, default=256, help="Image size for processing"
    )
    parser.add_argument(
        "--create-individual-plots",
        action="store_true",
        help="Create individual progression plots for each morphological type",
    )
    parser.add_argument(
        "--no-collage", action="store_true", help="Skip creating comparison collage"
    )
    parser.add_argument(
        "--debug-mode", action="store_true", help="Print detailed debug information"
    )

    # Add argument for degradation mode
    parser.add_argument(
        "--use-sequence-mode",
        action="store_true",
        default=True,
        help="Use direct timestep degradation instead of cumulative (default: True)",
    )
    parser.add_argument(
        "--use-cummulative-mode",
        action="store_true",
        help="Use cumulative degradation (overrides --use-training-mode)",
    )

    args = parser.parse_args()

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Parse parameters
    morph_types = [t.strip() for t in args.morph_types.split(",")]
    timesteps_to_plot = [int(t.strip()) for t in args.timesteps_to_plot.split(",")]

    # Validate timesteps
    timesteps_to_plot = [t for t in timesteps_to_plot if 0 <= t <= args.timesteps]
    # Ensure timestep 1000 is mapped to index 999 for a 1000-timestep schedule
    timesteps_to_plot = [
        min(t, args.timesteps - 1) if t == args.timesteps else t
        for t in timesteps_to_plot
    ]
    timesteps_to_plot = sorted(
        list(set(timesteps_to_plot))
    )  # Remove duplicates and sort

    print(f"Morphological types to test: {morph_types}")
    print(f"Timesteps to visualize: {timesteps_to_plot}")

    # Load and preprocess image
    print(f"Loading image: {args.input}")
    image_size = (args.image_size, args.image_size)
    original_image = load_and_preprocess_image(args.input, image_size)

    print(f"Image shape: {original_image.shape}")
    print(
        f"Image range: [{original_image.min().item():.3f}, {original_image.max().item():.3f}]"
    )

    # Run forward process for each morphological type
    results_dict = {}
    config_strings = {}

    for morph_type in morph_types:
        print(f"\n{'=' * 60}")
        print(f"PROCESSING: {morph_type.upper()} MORPHOLOGICAL OPERATION")
        print(f"{'=' * 60}")

        try:
            # Create the morphological schedule first
            print(
                f"Creating {args.schedule_type} schedule for {args.timesteps} timesteps..."
            )

            try:
                schedule = create_morphological_schedule(
                    schedule_type=args.schedule_type, timesteps=args.timesteps
                )
            except TypeError as e:
                # Try alternative parameter names if the first attempt fails
                print(f"Trying alternative parameter names for schedule creation...")
                try:
                    schedule = create_morphological_schedule(
                        args.schedule_type, args.timesteps
                    )
                except Exception as e2:
                    print(f"Error creating schedule with alternative parameters: {e2}")
                    print(
                        f"Please check the create_morphological_schedule function signature"
                    )
                    raise e2
            except Exception as e:
                print(f"Error creating {args.schedule_type} schedule: {e}")
                print(f"Available schedule types might be: linear, exponential, cosine")
                raise e

            print(
                f"Schedule created successfully. Sample values: {schedule[:5].tolist()}...{schedule[-5:].tolist()}"
            )

            # Create morphological operations
            morph_ops = MorphologicalDegradation(
                morph_type=morph_type,
                num_timesteps=args.timesteps,
                kernel_size_start=args.kernel_size_start,
                kernel_size_end=args.kernel_size_end,
                morph_routine=args.morph_routine,
                channels=1,
                morph_operators=schedule,  # Pass the created schedule
            )

            # Print debug info if requested
            if args.debug_mode:
                config_str = print_debug_info(morph_ops, morph_type, args.schedule_type)
                config_strings[morph_type] = config_str
            else:
                config_strings[morph_type] = (
                    f"Type: {morph_type}, "
                    f"Routine: {args.morph_routine}, "
                    f"Schedule: {args.schedule_type}, "
                    f"Timesteps: {args.timesteps}"
                )

            # Determine which mode to use
            use_training_mode = args.use_sequence_mode and not args.use_cummulative_mode

            # Run forward process
            results = run_forward_process(
                original_image,
                morph_ops,
                timesteps_to_plot,
                device,
                use_training_mode=use_training_mode,
            )

            results_dict[morph_type] = results

            print(f"Successfully processed {morph_type} with {len(results)} timesteps")

            # Create individual progression plot if requested
            if args.create_individual_plots:
                individual_save_path = output_dir / f"{morph_type}_progression.png"
                create_individual_progression(
                    results,
                    morph_type,
                    timesteps_to_plot,
                    config_strings[morph_type],
                    str(individual_save_path),
                )

        except Exception as e:
            print(f"Error processing {morph_type}: {e}")
            warnings.warn(f"Skipping {morph_type} due to error: {e}")
            continue

    # Create comparison collage
    if not args.no_collage and results_dict:
        print(f"\nCreating comparison collage...")

        # Create title with configuration info
        config_summary = (
            f"Morphological Forward Diffusion Comparison\n"
            f"Routine: {args.morph_routine}, "
            f"Schedule: {args.schedule_type}, "
            f"Kernels: {args.kernel_size_start}→{args.kernel_size_end}, "
            f"Timesteps: {args.timesteps}"
        )

        collage_save_path = output_dir / "morphological_comparison_collage.png"
        create_comparison_collage(
            results_dict,
            timesteps_to_plot,
            original_image,
            config_summary,
            str(collage_save_path),
        )

    # Save configuration summary
    config_file = output_dir / "demo_config.txt"
    with open(config_file, "w") as f:
        f.write("Morphological Forward Diffusion Demo Configuration\n")
        f.write("=" * 50 + "\n")
        f.write(f"Input Image: {args.input}\n")
        f.write(f"Image Size: {image_size}\n")
        f.write(f"Total Timesteps: {args.timesteps}\n")
        f.write(f"Visualized Timesteps: {timesteps_to_plot}\n")
        f.write(f"Morphological Types: {morph_types}\n")
        f.write(
            f"Kernel Size Range: {args.kernel_size_start} → {args.kernel_size_end}\n"
        )
        f.write(f"Morphological Routine: {args.morph_routine}\n")
        f.write(f"Schedule Type: {args.schedule_type}\n")
        f.write(f"Device: {device}\n")
        f.write("\nResults:\n")
        for morph_type, results in results_dict.items():
            f.write(f"  {morph_type}: {len(results)} timesteps processed\n")

    print(f"\nDemo completed! Results saved to: {output_dir}")
    print(f"Configuration saved to: {config_file}")

    # Summary statistics
    print(f"\nSUMMARY:")
    print(f"- Processed {len(results_dict)} morphological operation types")
    print(f"- Visualized {len(timesteps_to_plot)} timesteps each")
    print(
        f"- Individual plots: {'Created' if args.create_individual_plots else 'Skipped'}"
    )
    print(
        f"- Comparison collage: {'Created' if not args.no_collage and results_dict else 'Skipped'}"
    )

    # Interpretation guide
    print(f"\nINTERPRETATION GUIDE:")
    print(f"- Dilation: Should expand white regions (objects grow)")
    print(f"- Erosion: Should shrink white regions (objects shrink)")
    print(f"- Opening: Erosion then dilation (removes small objects)")
    print(f"- Closing: Dilation then erosion (fills small holes)")
    print(f"- Mixed: Combination of operations (complex behavior)")
    print(f"- Higher timesteps = more degradation")
    print(f"- Progressive routine = kernel size grows with time")
    print(f"- Exponential schedule = faster degradation at higher timesteps")

    if results_dict:
        print(f"\nNEXT STEPS:")
        print(
            f"1. Examine the comparison collage to see how different operations behave"
        )
        print(f"2. Check individual progressions for detailed analysis")
        print(f"3. Compare different schedule types by running with --schedule-type")
        print(f"4. Test different kernel size ranges with --kernel-size-start/end")
        print(f"5. Use these insights to choose optimal parameters for training")


if __name__ == "__main__":
    main()
