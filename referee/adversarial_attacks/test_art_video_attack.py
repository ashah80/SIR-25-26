"""
Test ART (Adversarial Robustness Toolbox) video-only attacks on Referee model.

This script uses ART's PGD implementation to attack only the video modality
while keeping audio unchanged. This is useful for:
- Comparing ART's implementation with our custom PGD
- Testing different ART attack algorithms in the future
- Standardized benchmarking

Usage:
    python test_art_video_attack.py --num-samples 3 --output-dir ./art-testing
    python test_art_video_attack.py --num-samples 5 --eps 0.3 --max-iter 100

Requirements:
    pip install adversarial-robustness-toolbox
"""

import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, Dict, Any
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import shutil
import argparse
import time

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

# Check for ART
try:
    from art.estimators.classification import PyTorchClassifier
    from art.attacks.evasion import ProjectedGradientDescent
    HAVE_ART = True
except ImportError:
    HAVE_ART = False
    print("=" * 60)
    print("ERROR: ART not installed!")
    print("Install with: pip install adversarial-robustness-toolbox")
    print("=" * 60)

# Check for OpenCV
try:
    import cv2
    HAVE_CV2 = True
except ImportError:
    HAVE_CV2 = False
    print("Warning: OpenCV (cv2) not found. Video saving will be limited.")


# ==============================================================================
# ART Wrapper for Video-Only Attacks
# ==============================================================================

class ARTRefereeVideoWrapper(nn.Module):
    """
    Wrapper to make Referee model compatible with ART for video-only attacks.

    This wrapper:
    - Takes only video input (flattened by ART)
    - Holds target_audio, ref_video, ref_audio fixed
    - Returns real/fake logits for ART to optimize against
    """

    def __init__(self, referee_model: nn.Module,
                 target_audio: torch.Tensor,
                 ref_video: torch.Tensor,
                 ref_audio: torch.Tensor,
                 device: str = 'cuda'):
        super().__init__()
        self.referee = referee_model
        self.device = device

        # Store fixed inputs (these won't be attacked)
        self.register_buffer('target_audio', target_audio.to(device))
        self.register_buffer('ref_video', ref_video.to(device))
        self.register_buffer('ref_audio', ref_audio.to(device))

        # Video shape for reshaping
        self.video_shape = (8, 16, 3, 224, 224)  # (S, T, C, H, W)
        self.flattened_size = int(np.prod(self.video_shape))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for ART compatibility.

        Args:
            x: Flattened video tensor from ART (B, flattened_size)

        Returns:
            logits: Real/fake classification logits (B, 2)
        """
        batch_size = x.shape[0]

        # Reshape flattened input back to video tensor
        video = x.view(batch_size, *self.video_shape)

        # Expand fixed inputs to match batch size if needed
        if batch_size != 1:
            target_audio = self.target_audio.expand(batch_size, -1, -1, -1, -1)
            ref_video = self.ref_video.expand(batch_size, -1, -1, -1, -1, -1)
            ref_audio = self.ref_audio.expand(batch_size, -1, -1, -1, -1)
        else:
            target_audio = self.target_audio
            ref_video = self.ref_video
            ref_audio = self.ref_audio

        # Call Referee model
        logits_rf, _ = self.referee(
            target_vis=video,
            target_aud=target_audio,
            ref_vis=ref_video,
            ref_aud=ref_audio
        )

        return logits_rf


# ==============================================================================
# Model Loading
# ==============================================================================

def load_real_model(checkpoint_path: Optional[str] = None, device: str = 'cuda'):
    """Load the real Referee model."""
    from model.referee import Referee
    from omegaconf import OmegaConf

    config_path = PROJECT_ROOT / "configs" / "pair_sync.yaml"
    cfg = OmegaConf.load(config_path)
    model = Referee(cfg)

    if checkpoint_path is None:
        checkpoint_path = PROJECT_ROOT / "model" / "pretrained" / "pretrained.pth"

    checkpoint_path = Path(checkpoint_path)
    if checkpoint_path.exists():
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint

        new_state_dict = {}
        for k, v in state_dict.items():
            new_k = k[7:] if k.startswith('module.') else k
            new_state_dict[new_k] = v

        missing, unexpected = model.load_state_dict(new_state_dict, strict=False)
        if missing:
            print(f"  Missing keys: {len(missing)} keys")
        if unexpected:
            print(f"  Unexpected keys: {len(unexpected)} keys")
        print("Checkpoint loaded successfully!")
    else:
        print(f"Warning: Checkpoint not found at {checkpoint_path}")

    model = model.to(device)
    model.eval()
    return model


# ==============================================================================
# Data Loading
# ==============================================================================

def load_sample_by_index(dataset, idx: int, device: str = 'cuda'):
    """Load a specific sample from the dataset by index."""
    sample = dataset[idx]

    target_audio = sample['target_audio'].unsqueeze(0).to(device)
    target_video = sample['target_video'].unsqueeze(0).to(device)
    ref_audio = sample['reference_audio'].unsqueeze(0).to(device)
    ref_video = sample['reference_video'].unsqueeze(0).to(device)
    labels_rf = torch.tensor([sample['fake_label']], device=device)
    sample_info = sample['sample_info']

    return target_audio, target_video, ref_audio, ref_video, labels_rf, sample_info


# ==============================================================================
# Output Saving Functions
# ==============================================================================

def denormalize_video(video: torch.Tensor, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)) -> torch.Tensor:
    """Denormalize video tensor from [-1, 1] back to [0, 1] range."""
    mean = torch.tensor(mean, device=video.device).view(1, 1, 1, 3, 1, 1)
    std = torch.tensor(std, device=video.device).view(1, 1, 1, 3, 1, 1)

    if video.dim() == 5:
        mean = mean.squeeze(0)
        std = std.squeeze(0)

    video = video * std + mean
    return torch.clamp(video, 0, 1)


def save_video_as_mp4(video_tensor: torch.Tensor, save_path: Path, fps: int = 25):
    """Save video tensor as MP4 file."""
    if not HAVE_CV2:
        print("  Skipping video save (OpenCV not available)")
        return

    video_denorm = denormalize_video(video_tensor)

    if video_denorm.dim() == 6:
        video_denorm = video_denorm[0]  # Remove batch dim

    S, T, C, H, W = video_denorm.shape
    frames = video_denorm.reshape(S * T, C, H, W)
    frames = frames.permute(0, 2, 3, 1).cpu().numpy()
    frames = (frames * 255).astype(np.uint8)
    frames = np.clip(frames, 0, 255)
    frames = frames[:, :, :, ::-1]  # RGB to BGR

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(save_path), fourcc, fps, (W, H))

    for frame in frames:
        out.write(frame)

    out.release()
    print(f"  Saved video: {save_path}")


def save_video_comparison(original_video: torch.Tensor, adversarial_video: torch.Tensor,
                          save_path: Path, num_frames: int = 8):
    """Save a grid comparing original vs adversarial video frames."""
    orig_denorm = denormalize_video(original_video)
    adv_denorm = denormalize_video(adversarial_video)

    if orig_denorm.dim() == 6:
        orig_denorm = orig_denorm[0]
        adv_denorm = adv_denorm[0]

    S, T, C, H, W = orig_denorm.shape
    total_frames = S * T
    frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)

    orig_flat = orig_denorm.reshape(total_frames, C, H, W)
    adv_flat = adv_denorm.reshape(total_frames, C, H, W)

    fig, axes = plt.subplots(3, num_frames, figsize=(num_frames * 2, 6))

    for i, idx in enumerate(frame_indices):
        orig_frame = orig_flat[idx].permute(1, 2, 0).cpu().numpy()
        axes[0, i].imshow(orig_frame)
        axes[0, i].axis('off')
        if i == 0:
            axes[0, i].set_ylabel('Original', fontsize=12)

        adv_frame = adv_flat[idx].permute(1, 2, 0).cpu().numpy()
        axes[1, i].imshow(adv_frame)
        axes[1, i].axis('off')
        if i == 0:
            axes[1, i].set_ylabel('Adversarial', fontsize=12)

        diff = np.abs(adv_frame - orig_frame)
        diff_enhanced = np.clip(diff * 10, 0, 1)
        axes[2, i].imshow(diff_enhanced)
        axes[2, i].axis('off')
        if i == 0:
            axes[2, i].set_ylabel('Diff (10x)', fontsize=12)

    plt.suptitle('Video Frame Comparison: Original vs ART Adversarial', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved comparison: {save_path}")


def copy_original_video(source_path: str, dest_path: Path):
    """Copy original video file from dataset."""
    if source_path and Path(source_path).exists():
        shutil.copy2(source_path, dest_path)
        print(f"  Copied original video: {dest_path}")
    else:
        print(f"  Warning: Could not copy original video from {source_path}")


def save_attack_stats(stats: Dict[str, Any], save_path: Path):
    """Save attack statistics to a text file."""
    with open(save_path, 'w') as f:
        f.write("ART Video-Only Attack Results\n")
        f.write("=" * 50 + "\n\n")

        for key, value in stats.items():
            if isinstance(value, float):
                f.write(f"{key}: {value:.6f}\n")
            else:
                f.write(f"{key}: {value}\n")

    print(f"  Saved statistics: {save_path}")


# ==============================================================================
# ART Attack Function
# ==============================================================================

def run_art_video_attack(
    referee_model: nn.Module,
    target_audio: torch.Tensor,
    target_video: torch.Tensor,
    ref_audio: torch.Tensor,
    ref_video: torch.Tensor,
    labels_rf: torch.Tensor,
    device: str = 'cuda',
    eps: float = 0.3,           # ART default
    eps_step: float = 0.1,      # ART default
    max_iter: int = 100,        # ART default
    norm: int = np.inf,         # ART default (Linf)
    verbose: bool = True
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """
    Run ART's PGD attack on video only.

    Args:
        referee_model: Referee model
        target_audio: (1, S, 1, F, T) - kept fixed
        target_video: (1, S, T, C, H, W) - to be attacked
        ref_audio: (1, S, 1, F, T) - kept fixed
        ref_video: (1, S, T, C, H, W) - kept fixed
        labels_rf: (1,) label tensor
        device: Device
        eps: Perturbation budget (ART default: 0.3)
        eps_step: Step size (ART default: 0.1)
        max_iter: Max iterations (ART default: 100)
        norm: Norm type (ART default: np.inf)
        verbose: Print progress

    Returns:
        adversarial_video: (1, S, T, C, H, W)
        attack_info: Dictionary with attack statistics
    """
    if not HAVE_ART:
        raise ImportError("ART not installed. Run: pip install adversarial-robustness-toolbox")

    # Create ART wrapper
    wrapper = ARTRefereeVideoWrapper(
        referee_model=referee_model,
        target_audio=target_audio,
        ref_video=ref_video,
        ref_audio=ref_audio,
        device=device
    )
    wrapper.eval()

    # Get initial prediction
    with torch.no_grad():
        logits_orig = referee_model(target_video, target_audio, ref_video, ref_audio)[0]
        probs_orig = F.softmax(logits_orig, dim=1)
        orig_fake_prob = probs_orig[0, 1].item()
        orig_real_prob = probs_orig[0, 0].item()

    if verbose:
        print(f"Initial confidence - Real: {orig_real_prob:.4f}, Fake: {orig_fake_prob:.4f}")

    # Create ART classifier
    classifier = PyTorchClassifier(
        model=wrapper,
        loss=nn.CrossEntropyLoss(),
        input_shape=(wrapper.flattened_size,),
        nb_classes=2,
        clip_values=(-1.0, 1.0),  # Video is normalized to [-1, 1]
        device_type='gpu' if device == 'cuda' else 'cpu'
    )

    # Create ART PGD attack (using ART defaults)
    attack = ProjectedGradientDescent(
        estimator=classifier,
        norm=norm,
        eps=eps,
        eps_step=eps_step,
        max_iter=max_iter,
        targeted=False,
        num_random_init=1,
        batch_size=1,
        verbose=verbose
    )

    # Prepare input for ART (flatten video)
    video_flat = target_video.reshape(1, -1).cpu().numpy()

    # Labels for ART (we want to flip from fake(1) to real(0))
    labels_np = labels_rf.cpu().numpy()

    if verbose:
        print(f"Running ART PGD attack...")
        print(f"  eps={eps}, eps_step={eps_step}, max_iter={max_iter}, norm={norm}")

    # Run attack
    start_time = time.time()
    adv_video_flat = attack.generate(x=video_flat, y=labels_np)
    attack_time = time.time() - start_time

    # Convert back to tensor and reshape
    adv_video = torch.from_numpy(adv_video_flat).float().to(device)
    adv_video = adv_video.reshape(target_video.shape)

    # Get final prediction
    with torch.no_grad():
        logits_adv = referee_model(adv_video, target_audio, ref_video, ref_audio)[0]
        probs_adv = F.softmax(logits_adv, dim=1)
        adv_fake_prob = probs_adv[0, 1].item()
        adv_real_prob = probs_adv[0, 0].item()

    # Compute perturbation stats
    perturbation = adv_video - target_video
    l2_norm = torch.norm(perturbation).item()
    linf_norm = torch.max(torch.abs(perturbation)).item()

    confidence_change = adv_real_prob - orig_real_prob

    if verbose:
        print(f"Final confidence - Real: {adv_real_prob:.4f}, Fake: {adv_fake_prob:.4f}")
        print(f"Confidence change: {confidence_change:+.4f}")
        print(f"Attack time: {attack_time:.2f}s")

    attack_info = {
        'method': 'ART_PGD',
        'original_real_prob': orig_real_prob,
        'original_fake_prob': orig_fake_prob,
        'adversarial_real_prob': adv_real_prob,
        'adversarial_fake_prob': adv_fake_prob,
        'confidence_change': confidence_change,
        'attack_success': adv_real_prob > 0.5,
        'perturbation_l2_norm': l2_norm,
        'perturbation_linf_norm': linf_norm,
        'eps': eps,
        'eps_step': eps_step,
        'max_iter': max_iter,
        'norm': 'Linf' if norm == np.inf else f'L{norm}',
        'attack_time_seconds': attack_time
    }

    return adv_video, attack_info


# ==============================================================================
# Main Test Function
# ==============================================================================

def run_art_test(
    num_samples: int = 3,
    output_dir: str = './art-testing',
    eps: float = 0.3,
    eps_step: float = 0.1,
    max_iter: int = 100,
    device: str = 'cuda'
):
    """
    Run ART video-only attack test on multiple samples.

    Args:
        num_samples: Number of samples to test
        output_dir: Output directory for results
        eps: Epsilon (perturbation budget)
        eps_step: Step size
        max_iter: Maximum iterations
        device: Device to use
    """
    print("=" * 70)
    print("ART Video-Only Attack Test")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"Number of samples: {num_samples}")
    print(f"Output directory: {output_dir}")
    print(f"Attack params: eps={eps}, eps_step={eps_step}, max_iter={max_iter}")
    print()

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load model
    print("Loading Referee model...")
    model = load_real_model(device=device)

    # Load dataset
    print("Loading dataset...")
    from adversarial_attacks.real_data_loader import AdversarialTestDataset
    dataset = AdversarialTestDataset(device=device)

    # Get fake samples (indices where fake_label=1)
    fake_indices = [i for i, s in enumerate(dataset.samples) if s.get('fake_label', 0) == 1]
    print(f"Found {len(fake_indices)} fake samples in dataset")

    # Select samples to test
    test_indices = fake_indices[:num_samples]

    # Store results for summary
    all_results = []

    # Run attack on each sample
    for i, sample_idx in enumerate(test_indices):
        print()
        print("=" * 70)
        print(f"Sample {i+1}/{num_samples} (dataset index: {sample_idx})")
        print("=" * 70)

        # Create sample output directory
        sample_out = output_path / f"sample_{i+1}"
        sample_out.mkdir(exist_ok=True)

        # Load sample
        target_audio, target_video, ref_audio, ref_video, labels_rf, sample_info = \
            load_sample_by_index(dataset, sample_idx, device)

        print(f"Source: {sample_info.get('target_path', 'unknown')}")
        print(f"Audio shape: {target_audio.shape}")
        print(f"Video shape: {target_video.shape}")
        print()

        # Run ART attack
        try:
            adv_video, attack_info = run_art_video_attack(
                referee_model=model,
                target_audio=target_audio,
                target_video=target_video,
                ref_audio=ref_audio,
                ref_video=ref_video,
                labels_rf=labels_rf,
                device=device,
                eps=eps,
                eps_step=eps_step,
                max_iter=max_iter,
                verbose=True
            )

            # Add sample info to results
            attack_info['sample_index'] = sample_idx
            attack_info['source_path'] = sample_info.get('target_path', 'unknown')
            all_results.append(attack_info)

            # Save outputs
            print(f"\nSaving outputs to {sample_out}...")

            # Copy original video
            if 'target_path' in sample_info and sample_info['target_path']:
                copy_original_video(sample_info['target_path'], sample_out / "original_video_file.mp4")

            # Save adversarial video
            save_video_as_mp4(adv_video, sample_out / "adversarial_video.mp4")

            # Save original processed video (for comparison)
            save_video_as_mp4(target_video, sample_out / "original_video_processed.mp4")

            # Save video comparison
            save_video_comparison(target_video, adv_video, sample_out / "video_comparison.png")

            # Save statistics
            save_attack_stats(attack_info, sample_out / "attack_stats.txt")

        except Exception as e:
            print(f"Error attacking sample {sample_idx}: {e}")
            import traceback
            traceback.print_exc()
            all_results.append({
                'sample_index': sample_idx,
                'error': str(e)
            })

    # Print summary
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)

    successful_attacks = [r for r in all_results if r.get('attack_success', False)]
    errors = [r for r in all_results if 'error' in r]

    print(f"Total samples tested: {len(all_results)}")
    print(f"Successful attacks (real_prob > 0.5): {len(successful_attacks)}")
    print(f"Errors: {len(errors)}")
    print()

    # Compute average statistics
    valid_results = [r for r in all_results if 'confidence_change' in r]
    if valid_results:
        avg_conf_change = np.mean([r['confidence_change'] for r in valid_results])
        avg_l2_norm = np.mean([r['perturbation_l2_norm'] for r in valid_results])
        avg_linf_norm = np.mean([r['perturbation_linf_norm'] for r in valid_results])
        avg_time = np.mean([r['attack_time_seconds'] for r in valid_results])

        print(f"Average confidence change: {avg_conf_change:+.4f}")
        print(f"Average perturbation L2 norm: {avg_l2_norm:.4f}")
        print(f"Average perturbation Linf norm: {avg_linf_norm:.4f}")
        print(f"Average attack time: {avg_time:.2f}s")
        print()

        print("Per-sample results:")
        for r in valid_results:
            status = "SUCCESS" if r.get('attack_success') else "FAILED"
            print(f"  Sample {r['sample_index']}: {r['confidence_change']:+.4f} [{status}]")

    # Save summary to file
    summary_path = output_path / "summary.txt"
    with open(summary_path, 'w') as f:
        f.write("ART Video-Only Attack Summary\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Attack parameters:\n")
        f.write(f"  eps: {eps}\n")
        f.write(f"  eps_step: {eps_step}\n")
        f.write(f"  max_iter: {max_iter}\n\n")
        f.write(f"Total samples: {len(all_results)}\n")
        f.write(f"Successful: {len(successful_attacks)}\n")
        f.write(f"Errors: {len(errors)}\n\n")

        if valid_results:
            f.write(f"Average confidence change: {avg_conf_change:+.4f}\n")
            f.write(f"Average L2 norm: {avg_l2_norm:.4f}\n")
            f.write(f"Average Linf norm: {avg_linf_norm:.4f}\n")
            f.write(f"Average time: {avg_time:.2f}s\n\n")

            f.write("Per-sample results:\n")
            for r in valid_results:
                status = "SUCCESS" if r.get('attack_success') else "FAILED"
                f.write(f"  Sample {r['sample_index']}: {r['confidence_change']:+.4f} [{status}]\n")

    print(f"\nSummary saved to: {summary_path}")
    print(f"All outputs saved to: {output_path.absolute()}")
    print()
    print("=" * 70)
    print("Done!")
    print("=" * 70)


# ==============================================================================
# Main Entry Point
# ==============================================================================

if __name__ == "__main__":
    if not HAVE_ART:
        print("\nPlease install ART first:")
        print("  pip install adversarial-robustness-toolbox")
        sys.exit(1)

    parser = argparse.ArgumentParser(description="Test ART video-only attacks on Referee model")
    parser.add_argument("--num-samples", type=int, default=3,
                        help="Number of samples to test (default: 3)")
    parser.add_argument("--output-dir", type=str, default="./art-testing",
                        help="Output directory (default: ./art-testing)")
    parser.add_argument("--eps", type=float, default=0.3,
                        help="Perturbation budget epsilon (ART default: 0.3)")
    parser.add_argument("--eps-step", type=float, default=0.1,
                        help="Step size (ART default: 0.1)")
    parser.add_argument("--max-iter", type=int, default=100,
                        help="Maximum iterations (ART default: 100)")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use (default: cuda)")

    args = parser.parse_args()

    run_art_test(
        num_samples=args.num_samples,
        output_dir=args.output_dir,
        eps=args.eps,
        eps_step=args.eps_step,
        max_iter=args.max_iter,
        device=args.device
    )
