"""
Save Adversarial Outputs for Qualitative Analysis

This script runs adversarial attacks on REAL samples from the FakeAVCeleb dataset
and saves the original and adversarial outputs for qualitative analysis.

Saves:
- Original video file (copied from dataset)
- Adversarial video as MP4
- Audio spectrograms as images (PNG)
- Perturbation visualizations
- Side-by-side comparison frames
- Attack statistics

Usage:
    python save_adversarial_outputs.py --mode all
    python save_adversarial_outputs.py --mode audio --eps-audio 3.0
    python save_adversarial_outputs.py --mode video --eps-video 12.0
"""

import sys
from pathlib import Path
import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional, Dict, Any
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import shutil
import argparse

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

from adversarial_attacks.pgd_attack import RefereeMultiModalPGD
from adversarial_attacks.multimodal_wrapper import create_attack_wrapper

try:
    import cv2
    HAVE_CV2 = True
except ImportError:
    HAVE_CV2 = False
    print("Warning: OpenCV (cv2) not found. Video saving will be limited.")
    print("Install with: pip install opencv-python")


def load_real_model(checkpoint_path: Optional[str] = None, device: str = 'cuda'):
    """Load the real Referee model."""
    from model.referee import Referee
    from omegaconf import OmegaConf

    config_path = PROJECT_ROOT / "configs" / "pair_sync.yaml"
    cfg = OmegaConf.load(config_path)

    model = Referee(cfg.model.params)

    if checkpoint_path is None:
        checkpoint_path = PROJECT_ROOT / "model" / "pretrained" / "pretrained.pth"

    if Path(checkpoint_path).exists():
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)
        print("Checkpoint loaded successfully!")
    else:
        print(f"Warning: Checkpoint not found at {checkpoint_path}")

    model = model.to(device)
    model.eval()
    return model


def create_dummy_model(device: str = 'cuda'):
    """Create a dummy model for testing."""
    class DummyReferee(nn.Module):
        def __init__(self):
            super().__init__()
            self.classifier = nn.Sequential(
                nn.Linear(2, 256), nn.ReLU(),
                nn.Linear(256, 128), nn.ReLU(),
                nn.Linear(128, 64), nn.ReLU(),
                nn.Linear(64, 2)
            )

        def forward(self, target_vis, target_aud, ref_vis, ref_aud, **kwargs):
            audio_feat = torch.mean(target_aud, dim=list(range(1, target_aud.ndim)))
            video_feat = torch.mean(target_vis, dim=list(range(1, target_vis.ndim)))
            combined = torch.stack([audio_feat, video_feat], dim=1) * 100.0
            logits_rf = self.classifier(combined)
            logits_id = self.classifier(combined * 0.8)
            return logits_rf, logits_id

    return DummyReferee().to(device)


def load_real_data(device: str = 'cuda', sample_type: str = 'fake'):
    """Load real data from the FakeAVCeleb dataset."""
    from adversarial_attacks.real_data_loader import load_real_sample
    return load_real_sample(device=device, sample_type=sample_type)


def create_dummy_data(batch_size: int = 1, device: str = 'cuda'):
    """Create dummy data for testing."""
    target_audio = torch.randn(batch_size, 8, 1, 128, 66, device=device)
    ref_audio = torch.randn(batch_size, 8, 1, 128, 66, device=device)
    target_video = torch.rand(batch_size, 8, 16, 3, 224, 224, device=device)
    ref_video = torch.rand(batch_size, 8, 16, 3, 224, 224, device=device)
    labels_rf = torch.ones(batch_size, dtype=torch.long, device=device)
    sample_info = {"source": "dummy", "target_path": None, "reference_path": None}
    return target_audio, target_video, ref_audio, ref_video, labels_rf, sample_info


def denormalize_video(video: torch.Tensor, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)) -> torch.Tensor:
    """
    Denormalize video tensor back to [0, 1] range.

    Args:
        video: (B, S, T, C, H, W) or (S, T, C, H, W) normalized tensor

    Returns:
        Video tensor in [0, 1] range
    """
    mean = torch.tensor(mean, device=video.device).view(1, 1, 1, 3, 1, 1)
    std = torch.tensor(std, device=video.device).view(1, 1, 1, 3, 1, 1)

    if video.dim() == 5:  # (S, T, C, H, W)
        mean = mean.squeeze(0)
        std = std.squeeze(0)

    video = video * std + mean
    return torch.clamp(video, 0, 1)


def denormalize_audio(audio: torch.Tensor, mean=-4.2677393, std=4.5689974) -> torch.Tensor:
    """
    Denormalize audio mel-spectrogram.

    Args:
        audio: (B, S, 1, F, T) normalized tensor

    Returns:
        Audio tensor in original log-mel scale
    """
    return audio * (2 * std) + mean


def save_audio_spectrogram(
    audio_tensor: torch.Tensor,
    save_path: Path,
    title: str = "Audio Spectrogram",
    sample_idx: int = 0,
    segment_idx: int = 0
):
    """Save audio mel-spectrogram as image."""
    # Denormalize
    audio_denorm = denormalize_audio(audio_tensor)

    # Extract single segment: (F, Ta)
    if audio_denorm.dim() == 5:  # (B, S, 1, F, T)
        spec = audio_denorm[sample_idx, segment_idx, 0].cpu().numpy()
    else:  # (S, 1, F, T)
        spec = audio_denorm[segment_idx, 0].cpu().numpy()

    # Create figure
    plt.figure(figsize=(12, 6))
    plt.imshow(spec, aspect='auto', origin='lower', cmap='viridis')
    plt.colorbar(label='Log Magnitude')
    plt.xlabel('Time Frames')
    plt.ylabel('Mel Frequency Bins')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  Saved audio spectrogram: {save_path}")


def save_audio_comparison(
    original_audio: torch.Tensor,
    adversarial_audio: torch.Tensor,
    save_path: Path,
    sample_idx: int = 0,
    segment_idx: int = 0
):
    """Save side-by-side comparison of original vs adversarial audio."""
    # Denormalize
    orig_denorm = denormalize_audio(original_audio)
    adv_denorm = denormalize_audio(adversarial_audio)

    if orig_denorm.dim() == 5:
        orig_spec = orig_denorm[sample_idx, segment_idx, 0].cpu().numpy()
        adv_spec = adv_denorm[sample_idx, segment_idx, 0].cpu().numpy()
    else:
        orig_spec = orig_denorm[segment_idx, 0].cpu().numpy()
        adv_spec = adv_denorm[segment_idx, 0].cpu().numpy()

    diff = adv_spec - orig_spec

    # Create 3-panel figure
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    im0 = axes[0].imshow(orig_spec, aspect='auto', origin='lower', cmap='viridis')
    axes[0].set_title('Original Audio')
    axes[0].set_xlabel('Time Frames')
    axes[0].set_ylabel('Mel Frequency Bins')
    plt.colorbar(im0, ax=axes[0])

    im1 = axes[1].imshow(adv_spec, aspect='auto', origin='lower', cmap='viridis')
    axes[1].set_title('Adversarial Audio')
    axes[1].set_xlabel('Time Frames')
    axes[1].set_ylabel('Mel Frequency Bins')
    plt.colorbar(im1, ax=axes[1])

    vmax = np.abs(diff).max()
    im2 = axes[2].imshow(diff, aspect='auto', origin='lower', cmap='RdBu_r',
                         vmin=-vmax, vmax=vmax)
    axes[2].set_title('Perturbation (Adversarial - Original)')
    axes[2].set_xlabel('Time Frames')
    axes[2].set_ylabel('Mel Frequency Bins')
    plt.colorbar(im2, ax=axes[2])

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  Saved audio comparison: {save_path}")


def save_video_frames_as_mp4(
    video_tensor: torch.Tensor,
    save_path: Path,
    sample_idx: int = 0,
    fps: int = 25
):
    """
    Save video tensor as MP4 file.

    Args:
        video_tensor: (B, S, T, C, H, W) normalized video tensor
        save_path: Path to save MP4
        sample_idx: Batch index
        fps: Frames per second
    """
    if not HAVE_CV2:
        print("  Skipping video save (OpenCV not available)")
        return

    # Denormalize video
    video_denorm = denormalize_video(video_tensor)

    # Extract all segments and concatenate frames: (S, T, C, H, W)
    if video_denorm.dim() == 6:
        video_denorm = video_denorm[sample_idx]  # (S, T, C, H, W)

    S, T, C, H, W = video_denorm.shape

    # Reshape to (S*T, C, H, W) then to (S*T, H, W, C)
    frames = video_denorm.view(S * T, C, H, W)
    frames = frames.permute(0, 2, 3, 1).cpu().numpy()  # (N, H, W, C)

    # Scale to [0, 255] and convert to uint8
    frames = (frames * 255).astype(np.uint8)
    frames = np.clip(frames, 0, 255)

    # Convert RGB to BGR for OpenCV
    frames = frames[:, :, :, ::-1]

    # Write video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(save_path), fourcc, fps, (W, H))

    for frame in frames:
        out.write(frame)

    out.release()
    print(f"  Saved video: {save_path}")


def save_video_comparison_frames(
    original_video: torch.Tensor,
    adversarial_video: torch.Tensor,
    save_path: Path,
    sample_idx: int = 0,
    num_frames: int = 16
):
    """Save a grid comparing original vs adversarial video frames."""
    # Denormalize
    orig_denorm = denormalize_video(original_video)
    adv_denorm = denormalize_video(adversarial_video)

    if orig_denorm.dim() == 6:
        orig_denorm = orig_denorm[sample_idx]  # (S, T, C, H, W)
        adv_denorm = adv_denorm[sample_idx]

    S, T, C, H, W = orig_denorm.shape

    # Select frames evenly spaced across all segments
    total_frames = S * T
    frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)

    orig_flat = orig_denorm.view(total_frames, C, H, W)
    adv_flat = adv_denorm.view(total_frames, C, H, W)

    # Create comparison figure
    fig, axes = plt.subplots(3, num_frames, figsize=(num_frames * 2, 6))

    for i, idx in enumerate(frame_indices):
        # Original
        orig_frame = orig_flat[idx].permute(1, 2, 0).cpu().numpy()
        axes[0, i].imshow(orig_frame)
        axes[0, i].axis('off')
        if i == 0:
            axes[0, i].set_ylabel('Original', fontsize=12)

        # Adversarial
        adv_frame = adv_flat[idx].permute(1, 2, 0).cpu().numpy()
        axes[1, i].imshow(adv_frame)
        axes[1, i].axis('off')
        if i == 0:
            axes[1, i].set_ylabel('Adversarial', fontsize=12)

        # Difference (amplified)
        diff = np.abs(adv_frame - orig_frame) * 10  # Amplify for visibility
        diff = np.clip(diff, 0, 1)
        axes[2, i].imshow(diff)
        axes[2, i].axis('off')
        if i == 0:
            axes[2, i].set_ylabel('Diff (10x)', fontsize=12)

        axes[0, i].set_title(f'Frame {idx}', fontsize=8)

    plt.suptitle('Video Comparison: Original vs Adversarial', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  Saved video comparison frames: {save_path}")


def copy_original_video(source_path: str, dest_path: Path):
    """Copy the original video file to output directory."""
    if source_path is None or not Path(source_path).exists():
        print(f"  Warning: Original video not found at {source_path}")
        return False

    try:
        shutil.copy2(source_path, dest_path)
        print(f"  Copied original video: {dest_path}")
        return True
    except Exception as e:
        print(f"  Warning: Could not copy original video: {e}")
        return False


def run_attack_and_save_outputs(
    attack_mode: str = 'joint',
    eps_audio: float = 3.0,
    eps_video: float = 8.0,
    output_dir: str = './adversarial_outputs',
    use_real_data: bool = True,
    use_real_model: bool = True,
    checkpoint_path: Optional[str] = None,
):
    """
    Run an adversarial attack and save all outputs for qualitative analysis.

    Args:
        attack_mode: 'audio', 'video', or 'joint'
        eps_audio: Audio epsilon
        eps_video: Video epsilon
        output_dir: Directory to save outputs
        use_real_data: Whether to use real dataset samples
        use_real_model: Whether to use real Referee model
        checkpoint_path: Path to model checkpoint
    """
    print("=" * 70)
    print(f"Running {attack_mode.upper()} Attack and Saving Outputs")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    print(f"Using real data: {use_real_data}")
    print(f"Using real model: {use_real_model}")

    output_path = Path(output_dir) / attack_mode
    output_path.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_path}\n")

    # Load model
    print("Loading model...")
    if use_real_model:
        try:
            model = load_real_model(checkpoint_path, device)
        except Exception as e:
            print(f"Failed to load real model: {e}")
            print("Falling back to dummy model.")
            model = create_dummy_model(device)
    else:
        model = create_dummy_model(device)
    model.eval()

    # Load data
    print("Loading data...")
    if use_real_data:
        try:
            target_audio, target_video, ref_audio, ref_video, labels_rf, sample_info = load_real_data(device)
            print(f"Loaded sample from: {sample_info.get('target_path', 'unknown')}")
        except Exception as e:
            print(f"Failed to load real data: {e}")
            print("Falling back to dummy data.")
            target_audio, target_video, ref_audio, ref_video, labels_rf, sample_info = create_dummy_data(1, device)
    else:
        target_audio, target_video, ref_audio, ref_video, labels_rf, sample_info = create_dummy_data(1, device)

    print(f"Audio shape: {target_audio.shape}")
    print(f"Video shape: {target_video.shape}")

    # Get initial confidence
    wrapper = create_attack_wrapper(model, ref_audio, ref_video, labels_rf)
    initial_conf = wrapper.get_confidence(target_audio, target_video)
    print(f"\nInitial confidence: fake_prob={initial_conf['rf_fake_prob']:.4f}, "
          f"real_prob={initial_conf['rf_real_prob']:.4f}")

    # Run attack
    print(f"\nRunning {attack_mode} attack...")
    attacker = RefereeMultiModalPGD(
        model,
        attack_mode=attack_mode,
        eps_audio=eps_audio,
        eps_video=eps_video,
        eps_step_audio=eps_audio / 3,
        eps_step_video=eps_video / 4,
        max_iter=40,
        temporal_weight=0.0,
        early_stop=True,
        verbose=True
    )

    adv_audio, adv_video, attack_info = attacker.generate(
        target_audio, target_video, ref_audio, ref_video, labels_rf
    )

    # Get final confidence
    final_conf = wrapper.get_confidence(adv_audio, adv_video)

    print(f"\nAttack completed!")
    print(f"  Iterations: {len(attack_info['losses'])}")
    print(f"  Converged: {attack_info.get('converged', False)}")
    print(f"  Final confidence: fake_prob={final_conf['rf_fake_prob']:.4f}, "
          f"real_prob={final_conf['rf_real_prob']:.4f}")
    print(f"  Confidence change: {final_conf['rf_real_prob'] - initial_conf['rf_real_prob']:+.4f}")

    # Save outputs
    print(f"\nSaving outputs to {output_path}...")

    # Copy original video file
    if sample_info.get('target_path'):
        orig_video_path = output_path / 'original_video_file.mp4'
        copy_original_video(sample_info['target_path'], orig_video_path)

        # Also copy reference video
        if sample_info.get('reference_path'):
            ref_video_path = output_path / 'reference_video_file.mp4'
            copy_original_video(sample_info['reference_path'], ref_video_path)

    # Audio outputs
    if attack_mode in ['audio', 'joint']:
        print("\nSaving audio outputs...")
        save_audio_spectrogram(
            target_audio,
            output_path / 'original_audio_spectrogram.png',
            title='Original Audio Spectrogram'
        )
        save_audio_spectrogram(
            adv_audio,
            output_path / 'adversarial_audio_spectrogram.png',
            title='Adversarial Audio Spectrogram'
        )
        save_audio_comparison(
            target_audio,
            adv_audio,
            output_path / 'audio_comparison.png'
        )

    # Video outputs
    if attack_mode in ['video', 'joint']:
        print("\nSaving video outputs...")
        save_video_frames_as_mp4(
            target_video,
            output_path / 'original_video_frames.mp4'
        )
        save_video_frames_as_mp4(
            adv_video,
            output_path / 'adversarial_video_frames.mp4'
        )
        save_video_comparison_frames(
            target_video,
            adv_video,
            output_path / 'video_frame_comparison.png'
        )

    # Save attack statistics
    print("\nSaving attack statistics...")
    stats_path = output_path / 'attack_info.txt'
    with open(stats_path, 'w') as f:
        f.write("ADVERSARIAL ATTACK RESULTS\n")
        f.write("=" * 50 + "\n\n")

        f.write("Sample Information:\n")
        f.write("-" * 30 + "\n")
        for k, v in sample_info.items():
            f.write(f"  {k}: {v}\n")

        f.write("\nAttack Configuration:\n")
        f.write("-" * 30 + "\n")
        f.write(f"  Attack Mode: {attack_mode}\n")
        f.write(f"  Epsilon Audio: {eps_audio}\n")
        f.write(f"  Epsilon Video: {eps_video}\n")
        f.write(f"  Max Iterations: 40\n")

        f.write("\nResults:\n")
        f.write("-" * 30 + "\n")
        f.write(f"  Iterations Used: {len(attack_info['losses'])}\n")
        f.write(f"  Converged: {attack_info.get('converged', False)}\n")

        f.write("\nConfidence Changes:\n")
        f.write("-" * 30 + "\n")
        f.write(f"  Initial fake_prob: {initial_conf['rf_fake_prob']:.4f}\n")
        f.write(f"  Initial real_prob: {initial_conf['rf_real_prob']:.4f}\n")
        f.write(f"  Final fake_prob: {final_conf['rf_fake_prob']:.4f}\n")
        f.write(f"  Final real_prob: {final_conf['rf_real_prob']:.4f}\n")
        f.write(f"  Change in real_prob: {final_conf['rf_real_prob'] - initial_conf['rf_real_prob']:+.4f}\n")

        attack_success = final_conf['rf_real_prob'] > 0.5
        f.write(f"\n  ATTACK SUCCESS: {'YES' if attack_success else 'NO'}\n")

    print(f"  Saved statistics: {stats_path}")

    print("\n" + "=" * 70)
    print("All outputs saved successfully!")
    print("=" * 70)
    print(f"\nOutput directory: {output_path.absolute()}")
    print("\nFiles created:")
    for f in sorted(output_path.glob('*')):
        print(f"  - {f.name}")

    # Cleanup
    del model, target_audio, target_video, ref_audio, ref_video
    del adv_audio, adv_video, attacker, attack_info, wrapper
    if device == 'cuda':
        torch.cuda.empty_cache()


def main():
    """Run attacks and save outputs."""
    parser = argparse.ArgumentParser(description='Generate and save adversarial outputs')
    parser.add_argument('--mode', type=str, default='all',
                       choices=['audio', 'video', 'joint', 'all'],
                       help='Attack mode to run')
    parser.add_argument('--eps-audio', type=float, default=3.0,
                       help='Audio epsilon')
    parser.add_argument('--eps-video', type=float, default=8.0,
                       help='Video epsilon')
    parser.add_argument('--output-dir', type=str, default='./adversarial_outputs',
                       help='Output directory')
    parser.add_argument('--dummy-data', action='store_true',
                       help='Use dummy data instead of real dataset')
    parser.add_argument('--dummy-model', action='store_true',
                       help='Use dummy model instead of real Referee model')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Path to model checkpoint')

    args = parser.parse_args()

    if args.mode == 'all':
        modes = ['audio', 'video', 'joint']
    else:
        modes = [args.mode]

    for mode in modes:
        run_attack_and_save_outputs(
            attack_mode=mode,
            eps_audio=args.eps_audio,
            eps_video=args.eps_video,
            output_dir=args.output_dir,
            use_real_data=not args.dummy_data,
            use_real_model=not args.dummy_model,
            checkpoint_path=args.checkpoint,
        )
        print("\n\n")


if __name__ == "__main__":
    main()
