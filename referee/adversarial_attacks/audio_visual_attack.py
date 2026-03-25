"""
Combined Audio-Visual Adversarial Attack for Referee Deepfake Detection Model.

This script runs both FlickeringAttack (video) and ImprovedPsychoacousticAttack (audio)
on the same samples and evaluates the combined effect.

Usage:
    python adversarial_attacks/audio_visual_attack.py --num-samples 3
    python adversarial_attacks/audio_visual_attack.py --num-samples 5 --output-dir ./my-results
"""

import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import numpy as np
from typing import Tuple, Dict, Any
import argparse
import time
import shutil
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

try:
    import cv2
    HAVE_CV2 = True
except ImportError:
    HAVE_CV2 = False
    print("Warning: OpenCV not found. Video saving will be disabled.")

try:
    import soundfile as sf
    HAVE_SOUNDFILE = True
except ImportError:
    HAVE_SOUNDFILE = False
    print("Warning: soundfile not found. Install with: pip install soundfile")

# Import attack classes
from adversarial_attacks.improved_video_attack import FlickeringAttack, save_video_non_overlapping
from adversarial_attacks.improved_audio_attacks import (
    ImprovedPsychoacousticAttack,
    DifferentiableMelTransform,
    extract_audio_from_video
)
from adversarial_attacks.real_data_loader import AdversarialTestDataset


def load_model(device: str = 'cuda'):
    """Load the Referee model with pretrained weights."""
    from model.referee import Referee
    from omegaconf import OmegaConf

    config_path = PROJECT_ROOT / "configs" / "pair_sync.yaml"
    cfg = OmegaConf.load(config_path)
    model = Referee(cfg)

    checkpoint_path = PROJECT_ROOT / "model" / "pretrained" / "pretrained.pth"
    if checkpoint_path.exists():
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint

        new_state_dict = {(k[7:] if k.startswith('module.') else k): v
                         for k, v in state_dict.items()}
        model.load_state_dict(new_state_dict, strict=False)
        print("Checkpoint loaded!")

    model = model.to(device)
    model.eval()
    return model


def save_frame_comparison(
    orig_video: torch.Tensor,
    adv_video: torch.Tensor,
    save_path: Path,
    num_frames: int = 8
):
    """Save a visual comparison of original vs adversarial frames with perturbation."""
    mean = torch.tensor([0.5, 0.5, 0.5], device=orig_video.device).view(1, 1, 1, 3, 1, 1)
    std = torch.tensor([0.5, 0.5, 0.5], device=orig_video.device).view(1, 1, 1, 3, 1, 1)

    orig = torch.clamp(orig_video * std + mean, 0, 1)[0]
    adv = torch.clamp(adv_video * std + mean, 0, 1)[0]

    S, T, C, H, W = orig.shape
    total = S * T
    indices = np.linspace(0, total - 1, num_frames, dtype=int)

    orig_flat = orig.reshape(total, C, H, W)
    adv_flat = adv.reshape(total, C, H, W)

    fig, axes = plt.subplots(3, num_frames, figsize=(num_frames * 2, 6))

    for i, idx in enumerate(indices):
        orig_f = orig_flat[idx].permute(1, 2, 0).cpu().numpy()
        adv_f = adv_flat[idx].permute(1, 2, 0).cpu().numpy()
        diff = np.abs(adv_f - orig_f)

        axes[0, i].imshow(orig_f)
        axes[0, i].axis('off')
        if i == 0:
            axes[0, i].set_ylabel('Original', fontsize=10)

        axes[1, i].imshow(adv_f)
        axes[1, i].axis('off')
        if i == 0:
            axes[1, i].set_ylabel('Adversarial', fontsize=10)

        axes[2, i].imshow(np.clip(diff * 10, 0, 1))
        axes[2, i].axis('off')
        if i == 0:
            axes[2, i].set_ylabel('Diff (10x)', fontsize=10)

    plt.suptitle('Video Perturbation Visualization', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved frame comparison: {save_path}")


def save_mel_comparison(
    original_waveform: torch.Tensor,
    adversarial_waveform: torch.Tensor,
    save_path: Path,
    sample_rate: int = 16000
):
    """Save mel-spectrogram comparison: original, adversarial, and difference."""
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=1024,
        hop_length=160,
        win_length=400,
        n_mels=128
    )

    if original_waveform.dim() == 2:
        original_waveform = original_waveform[0]
    if adversarial_waveform.dim() == 2:
        adversarial_waveform = adversarial_waveform[0]

    orig_mel = mel_transform(original_waveform.cpu())
    adv_mel = mel_transform(adversarial_waveform.cpu())

    orig_mel_db = 10 * torch.log10(orig_mel + 1e-10)
    adv_mel_db = 10 * torch.log10(adv_mel + 1e-10)
    diff_mel_db = adv_mel_db - orig_mel_db

    fig, axes = plt.subplots(3, 1, figsize=(12, 10))

    im0 = axes[0].imshow(orig_mel_db.numpy(), aspect='auto', origin='lower', cmap='viridis')
    axes[0].set_title('Original Audio Mel-Spectrogram')
    axes[0].set_ylabel('Mel Bin')
    plt.colorbar(im0, ax=axes[0], label='dB')

    im1 = axes[1].imshow(adv_mel_db.numpy(), aspect='auto', origin='lower', cmap='viridis')
    axes[1].set_title('Adversarial Audio Mel-Spectrogram')
    axes[1].set_ylabel('Mel Bin')
    plt.colorbar(im1, ax=axes[1], label='dB')

    vmax = max(abs(diff_mel_db.min().item()), abs(diff_mel_db.max().item()))
    im2 = axes[2].imshow(diff_mel_db.numpy(), aspect='auto', origin='lower',
                         cmap='RdBu_r', vmin=-vmax, vmax=vmax)
    axes[2].set_title('Difference (Adversarial - Original)')
    axes[2].set_ylabel('Mel Bin')
    axes[2].set_xlabel('Time Frame')
    plt.colorbar(im2, ax=axes[2], label='dB Difference')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved mel comparison: {save_path}")


def combine_video_audio(video_path: Path, audio_path: Path, output_path: Path):
    """Combine video file with audio file using ffmpeg."""
    import subprocess

    cmd = [
        'ffmpeg', '-y',
        '-i', str(video_path),
        '-i', str(audio_path),
        '-c:v', 'copy',
        '-c:a', 'aac',
        '-strict', 'experimental',
        '-shortest',
        str(output_path)
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, timeout=60)
        if result.returncode == 0:
            print(f"  Combined video+audio: {output_path}")
        else:
            print(f"  Warning: ffmpeg failed - {result.stderr.decode()[:200]}")
    except Exception as e:
        print(f"  Warning: Could not combine video+audio: {e}")


def save_stats(info: Dict, save_path: Path):
    """Save attack statistics to a text file."""
    with open(save_path, 'w') as f:
        f.write("Combined Audio-Visual Attack Results\n")
        f.write("=" * 50 + "\n\n")
        for k, v in info.items():
            if isinstance(v, float):
                f.write(f"{k}: {v:.6f}\n")
            else:
                f.write(f"{k}: {v}\n")
    print(f"  Saved stats: {save_path}")


class CombinedAudioVisualAttack:
    """
    Combined audio-visual attack that runs both modality attacks and evaluates combined effect.
    """

    def __init__(
        self,
        model: nn.Module,
        # Video attack params
        video_eps: float = 0.05,
        video_iterations: int = 200,
        video_step_size: float = 0.05,
        flicker_freq: float = 2.5,
        spatial_freq: int = 8,
        num_basis: int = 4,
        smoothness_weight: float = 2.0,
        # Audio attack params
        audio_eps: float = 0.15,
        audio_iterations: int = 300,
        target_snr_db: float = 35.0,
        snr_weight: float = 0.1,
        masking_strength: float = 0.5,
        device: str = 'cuda'
    ):
        self.model = model
        self.device = device

        # Initialize video attack
        self.video_attack = FlickeringAttack(
            model=model,
            eps=video_eps,
            flicker_freq=flicker_freq,
            spatial_freq=spatial_freq,
            num_basis=num_basis,
            num_iterations=video_iterations,
            step_size=video_step_size,
            smoothness_weight=smoothness_weight,
            device=device
        )

        # Initialize audio attack
        self.audio_attack = ImprovedPsychoacousticAttack(
            model=model,
            eps=audio_eps,
            num_iterations=audio_iterations,
            target_snr_db=target_snr_db,
            snr_weight=snr_weight,
            masking_strength=masking_strength,
            device=device
        )

        # Mel transform for converting adversarial waveform to mel
        self.mel_transform = DifferentiableMelTransform(device=device).to(device)

    def attack(
        self,
        target_video: torch.Tensor,
        target_audio_mel: torch.Tensor,
        ref_video: torch.Tensor,
        ref_audio: torch.Tensor,
        labels: torch.Tensor,
        original_waveform: torch.Tensor,
        verbose: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """
        Run combined audio-visual attack.

        Args:
            target_video: (1, S, T, C, H, W) input video
            target_audio_mel: (1, S, 1, F, Ta) input audio mel-spectrogram
            ref_video: (1, S, T, C, H, W) reference video
            ref_audio: (1, S, 1, F, Ta) reference audio
            labels: (1,) ground truth label
            original_waveform: (1, T) raw audio waveform

        Returns:
            adv_video: Adversarial video tensor
            adv_waveform: Adversarial audio waveform
            info: Dictionary with attack statistics
        """
        self.model.eval()
        start_time = time.time()

        # Get initial prediction
        with torch.no_grad():
            logits = self.model(target_video, target_audio_mel, ref_video, ref_audio)[0]
            probs = F.softmax(logits, dim=1)
            orig_real_prob = probs[0, 0].item()
            orig_fake_prob = probs[0, 1].item()

        if verbose:
            print(f"\n{'='*60}")
            print("COMBINED AUDIO-VISUAL ATTACK")
            print(f"{'='*60}")
            print(f"Initial prediction: Real={orig_real_prob:.4f}, Fake={orig_fake_prob:.4f}")

        # Run video attack
        if verbose:
            print(f"\n--- Running Video Attack (FlickeringAttack) ---")
        adv_video, video_info = self.video_attack.attack(
            target_video, target_audio_mel, ref_video, ref_audio, labels, verbose=verbose
        )

        # Run audio attack
        if verbose:
            print(f"\n--- Running Audio Attack (ImprovedPsychoacousticAttack) ---")
        adv_waveform, audio_info = self.audio_attack.attack(
            original_waveform, target_video, ref_audio, ref_video, labels, verbose=verbose
        )

        # Convert adversarial waveform to mel for combined evaluation
        adv_audio_mel = self.mel_transform(adv_waveform)

        # Evaluate combined effect
        if verbose:
            print(f"\n--- Evaluating Combined Effect ---")

        with torch.no_grad():
            # Combined: both adversarial video and adversarial audio
            logits_combined = self.model(adv_video, adv_audio_mel, ref_video, ref_audio)[0]
            probs_combined = F.softmax(logits_combined, dim=1)
            combined_real_prob = probs_combined[0, 0].item()
            combined_fake_prob = probs_combined[0, 1].item()

        total_time = time.time() - start_time

        # Compile results
        info = {
            'method': 'CombinedAudioVisualAttack',
            # Original
            'original_real_prob': orig_real_prob,
            'original_fake_prob': orig_fake_prob,
            # Combined results
            'combined_real_prob': combined_real_prob,
            'combined_fake_prob': combined_fake_prob,
            'combined_confidence_change': combined_real_prob - orig_real_prob,
            'combined_attack_success': combined_real_prob > 0.5,
            # Video attack info
            'video_real_prob': video_info['adversarial_real_prob'],
            'video_confidence_change': video_info['confidence_change'],
            'video_attack_success': video_info['attack_success'],
            'video_perturbation_linf': video_info['perturbation_linf_norm'],
            'video_perturbation_l2': video_info['perturbation_l2_norm'],
            'video_eps': video_info['eps'],
            # Audio attack info
            'audio_real_prob': audio_info['adversarial_real_prob'],
            'audio_confidence_change': audio_info['confidence_change'],
            'audio_attack_success': audio_info['attack_success'],
            'audio_perturbation_linf': audio_info['perturbation_linf'],
            'audio_perturbation_snr_db': audio_info['perturbation_snr_db'],
            'audio_eps': audio_info['eps'],
            # Timing
            'total_attack_time_seconds': total_time,
            'video_attack_time_seconds': video_info['attack_time_seconds'],
            'audio_attack_time_seconds': audio_info['attack_time_seconds'],
        }

        if verbose:
            print(f"\n{'='*60}")
            print("RESULTS SUMMARY")
            print(f"{'='*60}")
            print(f"Original:  Real={orig_real_prob:.4f}")
            print(f"Combined:  Real={combined_real_prob:.4f} (change: {info['combined_confidence_change']:+.4f})")
            print(f"Success:   {'YES' if info['combined_attack_success'] else 'NO'}")
            print(f"Time:      {total_time:.1f}s")

        return adv_video, adv_waveform, info


def main():
    parser = argparse.ArgumentParser(description="Combined Audio-Visual Attack for Referee")
    parser.add_argument("--num-samples", type=int, default=3,
                        help="Number of samples to attack")
    parser.add_argument("--output-dir", type=str, default="./audiovisualattack",
                        help="Directory to save results")
    # Video params
    parser.add_argument("--video-eps", type=float, default=0.05,
                        help="Video perturbation budget")
    parser.add_argument("--video-iter", type=int, default=200,
                        help="Video attack iterations")
    parser.add_argument("--video-step-size", type=float, default=0.05,
                        help="Video optimizer learning rate")
    parser.add_argument("--flicker-freq", type=float, default=2.5,
                        help="Temporal flicker frequency in Hz")
    parser.add_argument("--spatial-freq", type=int, default=8,
                        help="Spatial pattern frequency")
    parser.add_argument("--num-basis", type=int, default=4,
                        help="Number of basis patterns")
    parser.add_argument("--smoothness-weight", type=float, default=2.0,
                        help="Temporal smoothness regularization")
    # Audio params
    parser.add_argument("--audio-eps", type=float, default=0.05,
                        help="Audio perturbation budget")
    parser.add_argument("--audio-iter", type=int, default=300,
                        help="Audio attack iterations")
    parser.add_argument("--target-snr", type=float, default=35.0,
                        help="Target SNR in dB")
    parser.add_argument("--snr-weight", type=float, default=0.02,
                        help="SNR regularization weight")
    parser.add_argument("--masking-strength", type=float, default=0.1,
                        help="Psychoacoustic masking strength")
    # General
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()

    print("=" * 70)
    print("COMBINED AUDIO-VISUAL ADVERSARIAL ATTACK")
    print("=" * 70)
    print(f"\nVideo params: eps={args.video_eps}, iter={args.video_iter}")
    print(f"Audio params: eps={args.audio_eps}, iter={args.audio_iter}, SNR={args.target_snr}dB")
    print()

    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("Loading model...")
    model = load_model(args.device)

    print("Loading dataset...")
    dataset = AdversarialTestDataset(device=args.device)

    # Get fake samples
    fake_indices = [i for i, s in enumerate(dataset.samples) if s.get('fake_label', 0) == 1]
    test_indices = fake_indices[:args.num_samples]

    print(f"Found {len(fake_indices)} fake samples, testing {len(test_indices)}")

    # Initialize combined attack
    attack = CombinedAudioVisualAttack(
        model=model,
        video_eps=args.video_eps,
        video_iterations=args.video_iter,
        video_step_size=args.video_step_size,
        flicker_freq=args.flicker_freq,
        spatial_freq=args.spatial_freq,
        num_basis=args.num_basis,
        smoothness_weight=args.smoothness_weight,
        audio_eps=args.audio_eps,
        audio_iterations=args.audio_iter,
        target_snr_db=args.target_snr,
        snr_weight=args.snr_weight,
        masking_strength=args.masking_strength,
        device=args.device
    )

    results = []

    for i, sample_idx in enumerate(test_indices):
        print()
        print("=" * 70)
        print(f"SAMPLE {i+1}/{args.num_samples}")
        print("=" * 70)

        sample_out = output_path / f"sample_{i+1}"
        sample_out.mkdir(exist_ok=True)

        # Load sample
        sample = dataset[sample_idx]
        target_video = sample['target_video'].unsqueeze(0)
        target_audio_mel = sample['target_audio'].unsqueeze(0)
        ref_audio = sample['reference_audio'].unsqueeze(0)
        ref_video = sample['reference_video'].unsqueeze(0)
        labels = sample['fake_label'].unsqueeze(0)
        info = sample['sample_info']

        video_path = info.get('target_path')
        if not video_path or not Path(video_path).exists():
            print(f"  Skipping - video file not found: {video_path}")
            continue

        print(f"Source: {video_path}")

        try:
            # Extract original audio waveform
            original_waveform = extract_audio_from_video(video_path)
            waveform_tensor = torch.from_numpy(original_waveform).float().to(args.device)
            if waveform_tensor.dim() == 1:
                waveform_tensor = waveform_tensor.unsqueeze(0)

            # Run combined attack
            adv_video, adv_waveform, attack_info = attack.attack(
                target_video, target_audio_mel, ref_video, ref_audio,
                labels, waveform_tensor, verbose=True
            )

            # Save outputs
            print(f"\nSaving outputs to {sample_out}...")

            # 1. Original video file (with audio)
            shutil.copy2(video_path, sample_out / "01_original_video_with_audio.mp4")
            print(f"  Saved: 01_original_video_with_audio.mp4")

            # 2. Processed video file (without audio) - from tensor
            save_video_non_overlapping(target_video, sample_out / "02_original_video_processed.mp4")

            # 3. Original audio file
            if HAVE_SOUNDFILE:
                orig_wav = original_waveform.copy()
                max_val = np.max(np.abs(orig_wav))
                if max_val > 0:
                    orig_wav = orig_wav / max_val * 0.95
                sf.write(str(sample_out / "03_original_audio.wav"), orig_wav, 16000)
                print(f"  Saved: 03_original_audio.wav")

            # 4. Adversarial video file (without audio)
            save_video_non_overlapping(adv_video, sample_out / "04_adversarial_video.mp4")

            # 5. Adversarial audio file
            if HAVE_SOUNDFILE:
                adv_wav = adv_waveform[0].cpu().numpy()
                max_val = np.max(np.abs(adv_wav))
                if max_val > 0:
                    adv_wav = adv_wav / max_val * 0.95
                sf.write(str(sample_out / "05_adversarial_audio.wav"), adv_wav, 16000)
                print(f"  Saved: 05_adversarial_audio.wav")

            # 6. Final adversarial video (with audio)
            if HAVE_SOUNDFILE and HAVE_CV2:
                combine_video_audio(
                    sample_out / "04_adversarial_video.mp4",
                    sample_out / "05_adversarial_audio.wav",
                    sample_out / "06_final_adversarial_video.mp4"
                )

            # 7. Frame-level visualization
            save_frame_comparison(target_video, adv_video, sample_out / "07_video_perturbation.png")

            # 8. Mel-spectrogram visualization
            save_mel_comparison(
                waveform_tensor, adv_waveform,
                sample_out / "08_audio_mel_comparison.png"
            )

            # Save stats
            attack_info['source_file'] = str(video_path)
            attack_info['sample_index'] = sample_idx
            save_stats(attack_info, sample_out / "stats.txt")

            results.append(attack_info)

        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
            results.append({'error': str(e), 'sample_index': sample_idx})

    # Print summary
    print()
    print("=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)

    valid = [r for r in results if 'combined_confidence_change' in r]
    if valid:
        avg_change = np.mean([r['combined_confidence_change'] for r in valid])
        successes = sum(1 for r in valid if r.get('combined_attack_success', False))

        print(f"Samples tested: {len(valid)}")
        print(f"Combined attack success: {successes}/{len(valid)} ({100*successes/len(valid):.0f}%)")
        print(f"Avg confidence change: {avg_change:+.4f}")
        print()
        print("Per-sample results:")
        print("-" * 50)
        for r in valid:
            status = "SUCCESS" if r.get('combined_attack_success') else "FAILED"
            change = r.get('combined_confidence_change', 0)
            orig = r.get('original_real_prob', 0)
            final = r.get('combined_real_prob', 0)
            print(f"  {orig:.4f} -> {final:.4f} ({change:+.4f}) [{status}]")

    print()
    print(f"Outputs saved to: {output_path.absolute()}")


if __name__ == "__main__":
    main()
