"""
Waveform-Space Audio Attack for Referee Model (FIXED)

This module implements adversarial audio attacks that operate in the WAVEFORM domain
rather than the mel-spectrogram domain. This solves the reconstruction problem because:

1. Perturbations are applied directly to the raw audio waveform
2. The mel-spectrogram is computed on-the-fly during the forward pass
3. No inverse transformation is needed - the adversarial audio IS the output

FIXES from v1:
- Only perturb samples that are actually used by the model (center segment region)
- Use smooth L2-normalized gradients instead of harsh sign() updates
- Add low-pass filtering to make perturbations more imperceptible
- Proper gradient masking to avoid static artifacts

Usage:
    python waveform_audio_attack.py --num-samples 3 --output-dir ./waveform-attack-results
"""

import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import numpy as np
from typing import Tuple, Optional, Dict, Any
import argparse
import time
import math

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

try:
    import soundfile as sf
    HAVE_SOUNDFILE = True
except ImportError:
    HAVE_SOUNDFILE = False
    print("Warning: soundfile not found. Install with: pip install soundfile")


class DifferentiableMelSpectrogram(nn.Module):
    """
    Differentiable mel-spectrogram transform that matches Referee's preprocessing.

    This allows gradients to flow from mel-spectrogram loss back to waveform.
    Also returns the active region mask so we know which samples affect the model.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        n_fft: int = 1024,
        hop_length: int = 160,
        win_length: int = 400,
        n_mels: int = 128,
        audio_mean: float = -4.2677393,
        audio_std: float = 4.5689974,
        max_spec_t: int = 66,
        n_segments: int = 8,
        segment_size_vframes: int = 16,
        step_size_seg: float = 0.5,
        v_fps: int = 25,
    ):
        super().__init__()

        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.audio_mean = audio_mean
        self.audio_std = audio_std
        self.max_spec_t = max_spec_t
        self.n_segments = n_segments
        self.step_size_seg = step_size_seg

        # Compute segment sizes
        self.segment_size_vframes = segment_size_vframes
        self.v_fps = v_fps
        self.seg_size_aframes = int(segment_size_vframes / v_fps * sample_rate)
        self.stride_aframes = int(step_size_seg * self.seg_size_aframes)

        # Mel spectrogram transform (differentiable!)
        self.mel_spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            n_mels=n_mels,
        )

        # Store last computed active region for masking
        self.last_active_start = 0
        self.last_active_end = 0

    def get_active_region(self, T: int) -> Tuple[int, int]:
        """
        Calculate the sample indices that are actually used by the model.

        Args:
            T: Total waveform length

        Returns:
            (start_idx, end_idx): The range of samples used
        """
        # Calculate required length
        required_len = (self.n_segments - 1) * self.stride_aframes + self.seg_size_aframes

        # Pad simulation
        padded_T = max(T, required_len)

        # Center crop calculation (same as forward)
        seg_seq_len = self.n_segments * self.step_size_seg + (1 - self.step_size_seg)
        aframes_seg_seq_len = int(seg_seq_len * self.seg_size_aframes)
        max_a_start = max(padded_T - aframes_seg_seq_len, 0)
        a_start = max_a_start // 2

        # End position: last segment start + segment size
        a_end = a_start + (self.n_segments - 1) * self.stride_aframes + self.seg_size_aframes

        return a_start, min(a_end, T)

    def create_importance_mask(self, T: int, device: torch.device) -> torch.Tensor:
        """
        Create a smooth importance mask based on how many segments use each sample.

        Samples used by more segments get higher weight (more important for attack).
        This creates smooth transitions at boundaries instead of harsh edges.

        Args:
            T: Total waveform length
            device: Device for the mask tensor

        Returns:
            mask: (T,) tensor with importance weights [0, 1]
        """
        mask = torch.zeros(T, device=device)

        # Calculate positions
        required_len = (self.n_segments - 1) * self.stride_aframes + self.seg_size_aframes
        padded_T = max(T, required_len)

        seg_seq_len = self.n_segments * self.step_size_seg + (1 - self.step_size_seg)
        aframes_seg_seq_len = int(seg_seq_len * self.seg_size_aframes)
        max_a_start = max(padded_T - aframes_seg_seq_len, 0)
        a_start = max_a_start // 2

        # Count how many segments use each sample
        for s in range(self.n_segments):
            start = a_start + s * self.stride_aframes
            end = min(start + self.seg_size_aframes, T)
            if start < T:
                mask[start:end] += 1.0

        # Normalize to [0, 1]
        if mask.max() > 0:
            mask = mask / mask.max()

        return mask

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Convert waveform to normalized mel-spectrogram in model format.

        Args:
            waveform: (B, T) raw audio waveform

        Returns:
            mel: (B, S, 1, F, Ta) normalized log mel-spectrogram
        """
        B, T = waveform.shape
        device = waveform.device

        # Calculate required length
        required_len = (self.n_segments - 1) * self.stride_aframes + self.seg_size_aframes

        # Pad if necessary
        if T < required_len:
            pad_len = required_len - T
            waveform = F.pad(waveform, (0, pad_len), mode='constant', value=0)
            T = required_len

        # Calculate start position (center crop temporally)
        seg_seq_len = self.n_segments * self.step_size_seg + (1 - self.step_size_seg)
        aframes_seg_seq_len = int(seg_seq_len * self.seg_size_aframes)
        max_a_start = max(T - aframes_seg_seq_len, 0)
        a_start = max_a_start // 2

        # Store active region
        self.last_active_start = a_start
        self.last_active_end = a_start + (self.n_segments - 1) * self.stride_aframes + self.seg_size_aframes

        # Extract segments
        segments = []
        for s in range(self.n_segments):
            start = a_start + s * self.stride_aframes
            end = start + self.seg_size_aframes
            segment = waveform[:, start:end]  # (B, seg_size_aframes)
            segments.append(segment)

        # Stack segments: (B, S, seg_size_aframes)
        audio_segments = torch.stack(segments, dim=1)

        # Compute mel spectrogram for each segment
        # Reshape for batch processing: (B*S, seg_size_aframes)
        B, S, seg_len = audio_segments.shape
        audio_flat = audio_segments.view(B * S, seg_len)

        # Apply mel spectrogram: (B*S, n_mels, Ta)
        mel = self.mel_spec(audio_flat)

        # Log transform
        mel = torch.log(mel + 1e-6)

        # Pad or truncate time dimension
        Ta = mel.shape[-1]
        if Ta < self.max_spec_t:
            pad_size = self.max_spec_t - Ta
            mel = F.pad(mel, (0, pad_size), mode='constant', value=0)
        elif Ta > self.max_spec_t:
            mel = mel[..., :self.max_spec_t]

        # Normalize (AST normalization)
        mel = (mel - self.audio_mean) / (2 * self.audio_std)

        # Reshape: (B*S, F, Ta) -> (B, S, F, Ta) -> (B, S, 1, F, Ta)
        mel = mel.view(B, S, self.n_mels, self.max_spec_t)
        mel = mel.unsqueeze(2)  # Add channel dimension

        return mel


def lowpass_filter(x: torch.Tensor, cutoff_ratio: float = 0.3) -> torch.Tensor:
    """
    Apply a simple low-pass filter to smooth perturbations.

    This makes perturbations less audible by removing high-frequency components.

    Args:
        x: (B, T) waveform or perturbation
        cutoff_ratio: Fraction of Nyquist frequency to keep (0.3 = keep low 30%)

    Returns:
        filtered: (B, T) low-pass filtered signal
    """
    B, T = x.shape

    # FFT
    X = torch.fft.rfft(x, dim=1)

    # Create frequency mask
    n_freqs = X.shape[1]
    cutoff = int(n_freqs * cutoff_ratio)

    # Smooth transition (raised cosine)
    mask = torch.ones(n_freqs, device=x.device)
    transition_width = max(int(n_freqs * 0.1), 1)
    for i in range(transition_width):
        idx = cutoff + i
        if idx < n_freqs:
            mask[idx] = 0.5 * (1 + math.cos(math.pi * i / transition_width))
    mask[cutoff + transition_width:] = 0

    # Apply mask
    X_filtered = X * mask.unsqueeze(0)

    # Inverse FFT
    x_filtered = torch.fft.irfft(X_filtered, n=T, dim=1)

    return x_filtered


class WaveformAudioAttack:
    """
    PGD attack on audio in waveform space (FIXED VERSION).

    Key fixes from v1:
    1. Only perturb samples in the active region (used by model)
    2. Use smooth L2-normalized gradients instead of harsh sign()
    3. Apply importance masking based on segment overlap
    4. Optional low-pass filtering for imperceptibility

    Key advantages over mel-spectrogram space attacks:
    1. No lossy reconstruction needed
    2. Adversarial audio is directly playable
    3. Perturbations are in a perceptually meaningful space
    """

    def __init__(
        self,
        model: nn.Module,
        eps: float = 0.01,  # L-inf perturbation budget (max amplitude change)
        step_size: float = 0.002,
        num_iterations: int = 100,
        use_lowpass: bool = True,
        lowpass_cutoff: float = 0.5,
        use_smooth_grad: bool = True,
        device: str = 'cuda',
    ):
        """
        Args:
            model: Referee model
            eps: L-inf perturbation bound (max per-sample amplitude change)
                 0.01 = 1% of max amplitude, usually inaudible
                 0.03 = 3% of max amplitude, subtle
            step_size: PGD step size
            num_iterations: Number of attack iterations
            use_lowpass: Apply low-pass filter to perturbation (more imperceptible)
            lowpass_cutoff: Cutoff ratio for low-pass filter (0.5 = keep low 50%)
            use_smooth_grad: Use L2-normalized grads instead of sign() (smoother)
            device: Device to use
        """
        self.model = model
        self.eps = eps
        self.step_size = step_size
        self.num_iterations = num_iterations
        self.use_lowpass = use_lowpass
        self.lowpass_cutoff = lowpass_cutoff
        self.use_smooth_grad = use_smooth_grad
        self.device = device

        # Create differentiable mel-spectrogram transform
        self.mel_transform = DifferentiableMelSpectrogram().to(device)

    def attack(
        self,
        original_waveform: torch.Tensor,
        target_video: torch.Tensor,
        ref_audio: torch.Tensor,
        ref_video: torch.Tensor,
        labels: torch.Tensor,
        verbose: bool = True
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Run waveform-space audio attack.

        Args:
            original_waveform: (B, T) raw audio waveform
            target_video: (B, S, Tv, C, H, W) preprocessed video
            ref_audio: (B, S, 1, F, Ta) reference audio (mel-spectrogram)
            ref_video: (B, S, Tv, C, H, W) reference video
            labels: (B,) ground truth labels

        Returns:
            adv_waveform: (B, T) adversarial waveform (directly playable!)
            info: Attack statistics
        """
        self.model.eval()

        B, T = original_waveform.shape
        orig_waveform = original_waveform.clone().detach()

        # Get importance mask - samples used by more segments are more important
        importance_mask = self.mel_transform.create_importance_mask(T, self.device)
        importance_mask = importance_mask.unsqueeze(0).expand(B, -1)  # (B, T)

        # Get active region (the center portion that's actually used)
        active_start, active_end = self.mel_transform.get_active_region(T)

        if verbose:
            print(f"  Audio length: {T} samples ({T/16000:.2f}s)")
            print(f"  Active region: [{active_start}, {active_end}] ({(active_end-active_start)/16000:.2f}s)")
            print(f"  Inactive: first {active_start/16000:.2f}s, last {(T-active_end)/16000:.2f}s")

        # Initialize perturbation (only for active region)
        delta = torch.zeros_like(original_waveform, requires_grad=True)

        # Get initial prediction
        with torch.no_grad():
            orig_mel = self.mel_transform(orig_waveform)
            logits = self.model(target_video, orig_mel, ref_video, ref_audio)[0]
            probs = F.softmax(logits, dim=1)
            orig_real_prob = probs[0, 0].item()
            orig_fake_prob = probs[0, 1].item()

        if verbose:
            print(f"Initial: Real={orig_real_prob:.4f}, Fake={orig_fake_prob:.4f}")

        best_delta = delta.clone()
        best_real_prob = orig_real_prob

        start_time = time.time()

        for i in range(self.num_iterations):
            delta.requires_grad_(True)

            # Apply perturbation
            adv_waveform = orig_waveform + delta

            # Convert to mel-spectrogram (differentiable)
            adv_mel = self.mel_transform(adv_waveform)

            # Forward pass
            logits = self.model(target_video, adv_mel, ref_video, ref_audio)[0]
            probs = F.softmax(logits, dim=1)

            # Loss: maximize real probability (fool the fake classifier)
            loss = F.cross_entropy(logits, labels)

            # Backward
            self.model.zero_grad()
            loss.backward()

            # Get gradient
            grad = delta.grad.detach()

            # Apply importance mask to gradient (focus on active region)
            grad = grad * importance_mask

            # Update perturbation
            with torch.no_grad():
                if self.use_smooth_grad:
                    # Smooth L2-normalized update (less harsh than sign)
                    grad_norm = torch.norm(grad, dim=1, keepdim=True) + 1e-10
                    grad_normalized = grad / grad_norm
                    delta = delta + self.step_size * grad_normalized
                else:
                    # Standard FGSM-style sign update
                    delta = delta + self.step_size * grad.sign()

                # Apply low-pass filter to make perturbation less audible
                if self.use_lowpass:
                    delta = lowpass_filter(delta, self.lowpass_cutoff)

                # Project to L-inf ball
                delta = torch.clamp(delta, -self.eps, self.eps)

                # Zero out perturbation outside active region (clean boundaries)
                delta[:, :active_start] = 0
                delta[:, active_end:] = 0

                # Ensure valid audio range [-1, 1]
                adv_waveform = torch.clamp(orig_waveform + delta, -1.0, 1.0)
                delta = adv_waveform - orig_waveform

                delta = delta.detach()

            # Track best
            current_real_prob = probs[0, 0].item()
            if current_real_prob > best_real_prob:
                best_real_prob = current_real_prob
                best_delta = delta.clone()

            if verbose and (i % 20 == 0 or i == self.num_iterations - 1):
                print(f"Iter {i:3d}: Loss={loss.item():.4f}, Real={current_real_prob:.4f}")

            # Early stopping
            if current_real_prob > 0.95:
                if verbose:
                    print(f"  Early stop at iter {i}!")
                break

        attack_time = time.time() - start_time

        # Final adversarial waveform
        adv_waveform = orig_waveform + best_delta

        # Final evaluation
        with torch.no_grad():
            adv_mel = self.mel_transform(adv_waveform)
            logits = self.model(target_video, adv_mel, ref_video, ref_audio)[0]
            probs = F.softmax(logits, dim=1)
            final_real_prob = probs[0, 0].item()
            final_fake_prob = probs[0, 1].item()

        # Compute perturbation stats
        delta_final = adv_waveform - orig_waveform
        linf_norm = torch.max(torch.abs(delta_final)).item()
        l2_norm = torch.norm(delta_final).item()

        # SNR only for active region
        active_orig = orig_waveform[:, active_start:active_end]
        active_delta = delta_final[:, active_start:active_end]
        snr = 10 * torch.log10(
            torch.sum(active_orig ** 2) / (torch.sum(active_delta ** 2) + 1e-10)
        ).item()

        info = {
            'method': 'WaveformPGD_v2_fixed',
            'original_real_prob': orig_real_prob,
            'original_fake_prob': orig_fake_prob,
            'adversarial_real_prob': final_real_prob,
            'adversarial_fake_prob': final_fake_prob,
            'confidence_change': final_real_prob - orig_real_prob,
            'attack_success': final_real_prob > 0.5,
            'perturbation_linf': linf_norm,
            'perturbation_l2': l2_norm,
            'perturbation_snr_db': snr,
            'eps': self.eps,
            'num_iterations': self.num_iterations,
            'active_region_start': active_start,
            'active_region_end': active_end,
            'use_lowpass': self.use_lowpass,
            'attack_time_seconds': attack_time,
        }

        if verbose:
            print(f"Final: Real={final_real_prob:.4f}, Fake={final_fake_prob:.4f}")
            print(f"Confidence change: {info['confidence_change']:+.4f}")
            print(f"Perturbation SNR: {snr:.1f} dB (higher = less audible change)")

        return adv_waveform.detach(), info


def extract_audio_from_video(video_path: str, sample_rate: int = 16000) -> np.ndarray:
    """Extract audio from video file using ffmpeg."""
    import subprocess
    import tempfile
    import os

    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
        temp_path = f.name

    try:
        cmd = [
            'ffmpeg', '-y', '-i', str(video_path),
            '-vn', '-acodec', 'pcm_s16le',
            '-ar', str(sample_rate), '-ac', '1',
            temp_path
        ]
        subprocess.run(cmd, capture_output=True, timeout=30, check=True)
        audio, sr = sf.read(temp_path)
        return audio.astype(np.float32)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


def load_model(device: str = 'cuda'):
    """Load Referee model."""
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


def main():
    parser = argparse.ArgumentParser(description="Waveform-space audio adversarial attack (FIXED)")
    parser.add_argument("--num-samples", type=int, default=3)
    parser.add_argument("--output-dir", type=str, default="./waveform-attack-results")
    parser.add_argument("--eps", type=float, default=0.03,
                        help="L-inf perturbation budget (0.01-0.05 typical, default 0.03)")
    parser.add_argument("--max-iter", type=int, default=100)
    parser.add_argument("--step-size", type=float, default=0.003)
    parser.add_argument("--no-lowpass", action="store_true",
                        help="Disable low-pass filtering (perturbation will be more audible)")
    parser.add_argument("--lowpass-cutoff", type=float, default=0.5,
                        help="Low-pass filter cutoff ratio (0.3-0.7, lower = smoother)")
    parser.add_argument("--use-sign-grad", action="store_true",
                        help="Use sign(grad) instead of normalized grad (more aggressive)")
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()

    print("=" * 70)
    print("Waveform-Space Audio Adversarial Attack (FIXED v2)")
    print("=" * 70)
    print()
    print("FIXES from v1:")
    print("  - Only perturbs samples in the active region (used by model)")
    print("  - Uses smooth L2-normalized gradients (not harsh sign())")
    print("  - Applies low-pass filtering for imperceptibility")
    print("  - Clean boundaries (no static at beginning/end)")
    print()
    print(f"Settings: eps={args.eps}, lowpass={'disabled' if args.no_lowpass else f'cutoff={args.lowpass_cutoff}'}")
    print()

    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("Loading model...")
    model = load_model(args.device)

    print("Loading dataset...")
    from adversarial_attacks.real_data_loader import AdversarialTestDataset
    import json

    json_path = PROJECT_ROOT / "data" / "test_pairs_fixed.json"
    with open(json_path, "r") as f:
        loaded = json.load(f)
        if isinstance(loaded, dict) and "data" in loaded:
            samples_meta = loaded["data"]
        else:
            samples_meta = loaded

    dataset = AdversarialTestDataset(device=args.device)

    # Get fake samples
    fake_indices = [i for i, s in enumerate(dataset.samples) if s.get('fake_label', 0) == 1]
    test_indices = fake_indices[:args.num_samples]

    results = []

    for i, sample_idx in enumerate(test_indices):
        print()
        print("=" * 70)
        print(f"Sample {i+1}/{args.num_samples}")
        print("=" * 70)

        sample_out = output_path / f"sample_{i+1}"
        sample_out.mkdir(exist_ok=True)

        # Load preprocessed data for video and reference
        sample = dataset[sample_idx]
        target_video = sample['target_video'].unsqueeze(0)
        ref_audio = sample['reference_audio'].unsqueeze(0)
        ref_video = sample['reference_video'].unsqueeze(0)
        labels = sample['fake_label'].unsqueeze(0)
        info = sample['sample_info']

        print(f"Source: {info.get('target_path', 'unknown')}")

        try:
            # Extract ORIGINAL WAVEFORM from video
            video_path = info.get('target_path')
            if not video_path or not Path(video_path).exists():
                print(f"  Skipping - video file not found: {video_path}")
                continue

            print("  Extracting audio waveform from video...")
            original_waveform = extract_audio_from_video(video_path)
            print(f"  Waveform: {len(original_waveform)} samples ({len(original_waveform)/16000:.2f}s)")

            # Convert to tensor
            waveform_tensor = torch.from_numpy(original_waveform).float().unsqueeze(0).to(args.device)

            # Run waveform attack (FIXED version)
            attack = WaveformAudioAttack(
                model=model,
                eps=args.eps,
                step_size=args.step_size,
                num_iterations=args.max_iter,
                use_lowpass=not args.no_lowpass,
                lowpass_cutoff=args.lowpass_cutoff,
                use_smooth_grad=not args.use_sign_grad,
                device=args.device
            )

            adv_waveform, attack_info = attack.attack(
                waveform_tensor, target_video, ref_audio, ref_video, labels
            )

            results.append(attack_info)

            # Save outputs
            print(f"\nSaving to {sample_out}...")

            # Save original waveform
            orig_wav = original_waveform.copy()
            max_val = np.max(np.abs(orig_wav))
            if max_val > 0:
                orig_wav = orig_wav / max_val * 0.95
            sf.write(str(sample_out / "original_audio.wav"), orig_wav, 16000)
            print(f"  Saved: original_audio.wav")

            # Save adversarial waveform (DIRECTLY PLAYABLE!)
            adv_wav = adv_waveform[0].cpu().numpy()
            max_val = np.max(np.abs(adv_wav))
            if max_val > 0:
                adv_wav = adv_wav / max_val * 0.95
            sf.write(str(sample_out / "adversarial_audio.wav"), adv_wav, 16000)
            print(f"  Saved: adversarial_audio.wav (DIRECTLY PLAYABLE!)")

            # Save perturbation (boosted for audibility)
            perturbation = adv_wav - (orig_wav[:len(adv_wav)] if len(orig_wav) >= len(adv_wav)
                                      else np.pad(orig_wav, (0, len(adv_wav) - len(orig_wav))))
            pert_boosted = perturbation * 10  # Boost 10x to hear it
            max_val = np.max(np.abs(pert_boosted))
            if max_val > 0:
                pert_boosted = pert_boosted / max_val * 0.95
            sf.write(str(sample_out / "perturbation_10x.wav"), pert_boosted, 16000)
            print(f"  Saved: perturbation_10x.wav (perturbation boosted 10x)")

            # Save stats
            with open(sample_out / "stats.txt", 'w') as f:
                f.write("Waveform Audio Attack Results\n")
                f.write("=" * 50 + "\n\n")
                for k, v in attack_info.items():
                    if isinstance(v, float):
                        f.write(f"{k}: {v:.6f}\n")
                    else:
                        f.write(f"{k}: {v}\n")
            print(f"  Saved: stats.txt")

        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
            results.append({'error': str(e), 'sample_index': sample_idx})

    # Summary
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)

    valid = [r for r in results if 'confidence_change' in r]
    if valid:
        avg_change = np.mean([r['confidence_change'] for r in valid])
        successes = sum(1 for r in valid if r.get('attack_success', False))
        avg_snr = np.mean([r.get('perturbation_snr_db', 0) for r in valid])

        print(f"Samples tested: {len(valid)}")
        print(f"Successful attacks: {successes}/{len(valid)}")
        print(f"Average confidence change: {avg_change:+.4f}")
        print(f"Average perturbation SNR: {avg_snr:.1f} dB")
        print()

        print("Per-sample results:")
        for r in valid:
            status = "SUCCESS" if r.get('attack_success') else "FAILED"
            change = r.get('confidence_change', 0)
            snr = r.get('perturbation_snr_db', 0)
            print(f"  {change:+.4f} [{status}] SNR={snr:.1f}dB")

    print()
    print("KEY ADVANTAGE: Adversarial audio files are DIRECTLY PLAYABLE!")
    print("No Griffin-Lim reconstruction needed. No robotic artifacts.")
    print("Beginning/end of audio stay clean (only active region is perturbed).")
    print()
    print(f"Outputs saved to: {output_path.absolute()}")


if __name__ == "__main__":
    main()
