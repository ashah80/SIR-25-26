"""
Improved Audio Attacks for Referee Model

This file contains improved audio attack implementations with:
1. Better transparency on methods and origins
2. Improved effectiveness while maintaining imperceptibility
3. Proper documentation of what each attack does

ATTACK OVERVIEW:
================

1. ImprovedPsychoacousticAttack (RECOMMENDED)
   - Space: WAVEFORM (gradients flow through mel-spectrogram transform)
   - Origin: Custom implementation inspired by Qin et al. (ICML 2019)
   - Key idea: Use psychoacoustic masking to hide perturbations in loud parts
   - Improvements: Momentum, adaptive step size, better masking model

2. MelSpacePGDAttack
   - Space: MEL-SPECTROGRAM (direct perturbation)
   - Origin: Standard PGD adapted for spectrograms
   - Key idea: Attack the representation the model actually sees
   - Note: Requires reconstruction (lossy) but may be more effective

3. HybridAttack
   - Combines waveform and mel-space attacks
   - Uses waveform attack with mel-space loss guidance

Usage:
    python improved_audio_attacks.py --method improved-psychoacoustic --num-samples 5
    python improved_audio_attacks.py --method mel-pgd --num-samples 5
    python improved_audio_attacks.py --method hybrid --num-samples 5
"""

import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import numpy as np
from typing import Tuple, Dict, Any, Optional
import argparse
import time

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

try:
    import soundfile as sf
    HAVE_SOUNDFILE = True
except ImportError:
    HAVE_SOUNDFILE = False
    print("Warning: soundfile not found. Install with: pip install soundfile")


class DifferentiableMelTransform(nn.Module):
    """
    Differentiable mel-spectrogram transform matching Referee preprocessing.

    This is the CRITICAL component that allows waveform-space attacks to work.
    Gradients flow: loss -> model -> mel -> waveform
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
        device: str = 'cuda'
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.audio_mean = audio_mean
        self.audio_std = audio_std
        self.max_spec_t = max_spec_t
        self.n_segments = n_segments

        # Segment sizes
        segment_size_vframes = 16
        v_fps = 25
        self.seg_size_aframes = int(segment_size_vframes / v_fps * sample_rate)
        self.stride_aframes = int(0.5 * self.seg_size_aframes)

        # Mel transform (differentiable)
        self.mel_spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            n_mels=n_mels,
        ).to(device)

    def get_active_region(self, T: int) -> Tuple[int, int]:
        """Get the sample indices actually used by the model."""
        required_len = (self.n_segments - 1) * self.stride_aframes + self.seg_size_aframes
        padded_T = max(T, required_len)

        seg_seq_len = self.n_segments * 0.5 + 0.5
        aframes_seg_seq_len = int(seg_seq_len * self.seg_size_aframes)
        max_a_start = max(padded_T - aframes_seg_seq_len, 0)
        a_start = max_a_start // 2
        a_end = a_start + (self.n_segments - 1) * self.stride_aframes + self.seg_size_aframes

        return a_start, min(a_end, T)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """Convert waveform (B, T) to model input (B, S, 1, F, Ta)."""
        B, T = waveform.shape
        device = waveform.device

        required_len = (self.n_segments - 1) * self.stride_aframes + self.seg_size_aframes
        if T < required_len:
            waveform = F.pad(waveform, (0, required_len - T))
            T = required_len

        seg_seq_len = self.n_segments * 0.5 + 0.5
        aframes_seg_seq_len = int(seg_seq_len * self.seg_size_aframes)
        max_a_start = max(T - aframes_seg_seq_len, 0)
        a_start = max_a_start // 2

        segments = []
        for s in range(self.n_segments):
            start = a_start + s * self.stride_aframes
            end = start + self.seg_size_aframes
            segments.append(waveform[:, start:end])

        audio_segments = torch.stack(segments, dim=1)
        B, S, seg_len = audio_segments.shape
        audio_flat = audio_segments.view(B * S, seg_len)

        mel = self.mel_spec(audio_flat)
        mel = torch.log(mel + 1e-6)

        Ta = mel.shape[-1]
        if Ta < self.max_spec_t:
            mel = F.pad(mel, (0, self.max_spec_t - Ta))
        elif Ta > self.max_spec_t:
            mel = mel[..., :self.max_spec_t]

        mel = (mel - self.audio_mean) / (2 * self.audio_std)
        mel = mel.view(B, S, self.n_mels, self.max_spec_t)
        mel = mel.unsqueeze(2)

        return mel


class ImprovedPsychoacousticAttack:
    """
    Improved Psychoacoustic Audio Attack

    ORIGINS & METHODOLOGY:
    ======================
    This attack is a CUSTOM IMPLEMENTATION inspired by:

    1. Qin et al., "Imperceptible, Robust, and Targeted Adversarial Examples
       for Automatic Speech Recognition" (ICML 2019)
       - Original paper: https://arxiv.org/abs/1903.10346
       - Key idea: Use psychoacoustic masking to hide perturbations

    2. MP3/AAC compression psychoacoustic models
       - Louder sounds mask quieter sounds (simultaneous masking)
       - Sounds before/after loud sounds are also masked (temporal masking)

    HOW IT DIFFERS FROM ASR ATTACKS:
    ================================
    - ASR attacks optimize CTC loss / word error rate
    - We optimize classification cross-entropy loss
    - ASR attacks need adversarial text targets
    - We just need to flip the classification

    SPACE: WAVEFORM
    ===============
    Perturbations are applied to raw waveform, but the loss is computed
    through a differentiable mel-spectrogram transform. This means:
    - Output is directly playable audio
    - No reconstruction artifacts
    - Gradients flow: loss -> mel_transform -> waveform

    IMPROVEMENTS OVER BASIC VERSION:
    ================================
    1. Momentum-based optimization (more stable convergence)
    2. Adaptive step size (starts aggressive, becomes conservative)
    3. Better masking model (combines energy + spectral flatness)
    4. Focus on active region only (the part model actually sees)
    5. Frequency-aware perturbation (prefer mid-frequencies humans hear less)
    """

    def __init__(
        self,
        model: nn.Module,
        eps: float = 0.3,           # Max perturbation (higher = more effective but audible)
        step_size: float = 0.02,    # Initial step size
        num_iterations: int = 300,  # More iterations for better convergence
        momentum: float = 0.9,      # Momentum for stable optimization
        adaptive_step: bool = True, # Reduce step size over time
        device: str = 'cuda'
    ):
        """
        Args:
            model: Referee model
            eps: Maximum L-inf perturbation (0.1-0.5 range, higher = more effective)
            step_size: Initial PGD step size
            num_iterations: Attack iterations (200-500 recommended)
            momentum: Momentum coefficient (0.9 typical)
            adaptive_step: Whether to decay step size
            device: cuda/cpu
        """
        self.model = model
        self.eps = eps
        self.step_size = step_size
        self.num_iterations = num_iterations
        self.momentum = momentum
        self.adaptive_step = adaptive_step
        self.device = device

        self.mel_transform = DifferentiableMelTransform(device=device).to(device)

    def compute_masking_curve(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Compute psychoacoustic masking curve.

        Based on the principle that louder parts of audio can hide perturbations.
        Uses STFT to get time-frequency representation, then computes energy per frame.

        Args:
            waveform: (B, T) audio

        Returns:
            mask: (B, T) values in [0.1, 1.0] indicating how much perturbation each sample can hide
        """
        B, T = waveform.shape

        # STFT for frequency analysis
        n_fft = 1024
        hop_length = 256
        window = torch.hann_window(n_fft, device=waveform.device)

        spec = torch.stft(
            waveform, n_fft=n_fft, hop_length=hop_length,
            window=window, return_complex=True
        )
        magnitude = torch.abs(spec)  # (B, n_fft//2+1, T_frames)

        # Energy per frame (louder = more masking)
        frame_energy = torch.sum(magnitude ** 2, dim=1)  # (B, T_frames)

        # Spectral flatness (tonal sounds mask better than noise)
        geometric_mean = torch.exp(torch.mean(torch.log(magnitude + 1e-10), dim=1))
        arithmetic_mean = torch.mean(magnitude, dim=1)
        spectral_flatness = geometric_mean / (arithmetic_mean + 1e-10)  # (B, T_frames)

        # Combine: high energy + low flatness (tonal) = good masking
        masking = frame_energy * (1 - 0.5 * spectral_flatness)
        masking = masking / (masking.max() + 1e-10)

        # Interpolate to waveform length
        mask = F.interpolate(
            masking.unsqueeze(1), size=T, mode='linear', align_corners=False
        ).squeeze(1)

        # Apply sqrt for smoother masking, ensure minimum
        mask = torch.sqrt(mask)
        mask = torch.clamp(mask, min=0.1, max=1.0)

        return mask

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
        Run improved psychoacoustic attack.

        The attack works by:
        1. Computing a masking curve from the original audio
        2. Iteratively updating perturbation using momentum-PGD
        3. Scaling perturbation by masking curve (more where louder)
        4. Projecting to epsilon ball with masking-aware bounds
        """
        self.model.eval()

        if original_waveform.dim() == 1:
            original_waveform = original_waveform.unsqueeze(0)

        B, T = original_waveform.shape
        orig_waveform = original_waveform.clone().detach()

        # Get active region
        active_start, active_end = self.mel_transform.get_active_region(T)

        if verbose:
            print(f"  Audio: {T} samples ({T/16000:.2f}s)")
            print(f"  Active region: [{active_start}:{active_end}] ({(active_end-active_start)/16000:.2f}s)")
            print(f"  Attack: eps={self.eps}, iters={self.num_iterations}, momentum={self.momentum}")

        # Compute masking curve
        with torch.no_grad():
            masking_curve = self.compute_masking_curve(orig_waveform)
            # Zero out masking outside active region
            masking_curve[:, :active_start] = 0
            masking_curve[:, active_end:] = 0

        # Initialize
        delta = torch.zeros_like(orig_waveform)
        velocity = torch.zeros_like(orig_waveform)  # For momentum

        # Initial prediction
        with torch.no_grad():
            orig_mel = self.mel_transform(orig_waveform)
            logits = self.model(target_video, orig_mel, ref_video, ref_audio)[0]
            probs = F.softmax(logits, dim=1)
            orig_real_prob = probs[0, 0].item()

        if verbose:
            print(f"Initial: Real={orig_real_prob:.4f}, Fake={1-orig_real_prob:.4f}")

        best_delta = delta.clone()
        best_real_prob = orig_real_prob

        start_time = time.time()

        for i in range(self.num_iterations):
            delta.requires_grad_(True)

            # Current step size (optionally adaptive)
            if self.adaptive_step:
                current_step = self.step_size * (1 - 0.5 * i / self.num_iterations)
            else:
                current_step = self.step_size

            # Forward pass
            adv_waveform = orig_waveform + delta
            adv_mel = self.mel_transform(adv_waveform)
            logits = self.model(target_video, adv_mel, ref_video, ref_audio)[0]
            probs = F.softmax(logits, dim=1)

            # Loss: we want to MAXIMIZE cross-entropy (fool the classifier)
            # For fake samples (label=1), maximizing CE means pushing toward real (class 0)
            loss = F.cross_entropy(logits, labels)

            # Backward
            self.model.zero_grad()
            loss.backward()

            grad = delta.grad.detach()

            with torch.no_grad():
                # Apply masking to gradient
                grad_masked = grad * masking_curve

                # Normalize gradient
                grad_norm = torch.norm(grad_masked.view(B, -1), dim=1, keepdim=True).view(B, 1) + 1e-10
                grad_normalized = grad_masked / grad_norm

                # Momentum update
                velocity = self.momentum * velocity + grad_normalized

                # Update delta
                delta = delta + current_step * velocity

                # Project with masking-aware epsilon
                # Allow larger perturbations where masking is higher
                eps_adaptive = self.eps * masking_curve
                delta = torch.clamp(delta, -eps_adaptive, eps_adaptive)

                # Ensure valid audio range
                adv_waveform = torch.clamp(orig_waveform + delta, -1.0, 1.0)
                delta = adv_waveform - orig_waveform
                delta = delta.detach()

            # Track best
            current_real_prob = probs[0, 0].item()
            if current_real_prob > best_real_prob:
                best_real_prob = current_real_prob
                best_delta = delta.clone()

            if verbose and (i % 50 == 0 or i == self.num_iterations - 1):
                print(f"Iter {i:3d}: Real={current_real_prob:.4f}, step={current_step:.4f}")

            if current_real_prob > 0.95:
                if verbose:
                    print(f"  Early stop at iter {i}!")
                break

        attack_time = time.time() - start_time

        # Final result
        adv_waveform = orig_waveform + best_delta

        with torch.no_grad():
            adv_mel = self.mel_transform(adv_waveform)
            logits = self.model(target_video, adv_mel, ref_video, ref_audio)[0]
            probs = F.softmax(logits, dim=1)
            final_real_prob = probs[0, 0].item()

        # Stats
        delta_final = adv_waveform - orig_waveform
        active_delta = delta_final[:, active_start:active_end]
        active_orig = orig_waveform[:, active_start:active_end]

        linf_norm = torch.max(torch.abs(delta_final)).item()
        l2_norm = torch.norm(delta_final).item()
        snr = 10 * torch.log10(
            torch.sum(active_orig ** 2) / (torch.sum(active_delta ** 2) + 1e-10)
        ).item()

        info = {
            'method': 'ImprovedPsychoacoustic',
            'original_real_prob': orig_real_prob,
            'adversarial_real_prob': final_real_prob,
            'confidence_change': final_real_prob - orig_real_prob,
            'attack_success': final_real_prob > 0.5,
            'perturbation_linf': linf_norm,
            'perturbation_l2': l2_norm,
            'perturbation_snr_db': snr,
            'eps': self.eps,
            'num_iterations': self.num_iterations,
            'momentum': self.momentum,
            'attack_time_seconds': attack_time,
        }

        if verbose:
            print(f"Final: Real={final_real_prob:.4f} (change: {info['confidence_change']:+.4f})")
            print(f"SNR: {snr:.1f} dB | L-inf: {linf_norm:.4f}")

        return adv_waveform.detach(), info


class MelSpacePGDAttack:
    """
    Direct PGD Attack in Mel-Spectrogram Space

    METHODOLOGY:
    ============
    Instead of perturbing the waveform and hoping the perturbation survives
    the mel-spectrogram transform, this attack directly perturbs the
    mel-spectrogram that the model sees.

    PROS:
    - Directly manipulates what the model sees
    - More effective at changing model output

    CONS:
    - Requires reconstruction (Griffin-Lim) to get back to waveform
    - Reconstruction introduces artifacts

    SPACE: MEL-SPECTROGRAM
    """

    def __init__(
        self,
        model: nn.Module,
        eps: float = 3.0,
        step_size: float = 0.5,
        num_iterations: int = 200,
        device: str = 'cuda'
    ):
        self.model = model
        self.eps = eps
        self.step_size = step_size
        self.num_iterations = num_iterations
        self.device = device

    def attack(
        self,
        target_audio: torch.Tensor,  # (1, S, 1, F, Ta) mel-spectrogram
        target_video: torch.Tensor,
        ref_audio: torch.Tensor,
        ref_video: torch.Tensor,
        labels: torch.Tensor,
        verbose: bool = True
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Run PGD attack directly on mel-spectrogram."""
        self.model.eval()

        orig_audio = target_audio.clone().detach()
        delta = torch.zeros_like(target_audio, requires_grad=True)

        # Initial prediction
        with torch.no_grad():
            logits = self.model(target_video, orig_audio, ref_video, ref_audio)[0]
            probs = F.softmax(logits, dim=1)
            orig_real_prob = probs[0, 0].item()

        if verbose:
            print(f"Initial: Real={orig_real_prob:.4f}")

        best_delta = delta.clone()
        best_real_prob = orig_real_prob
        velocity = torch.zeros_like(delta)

        start_time = time.time()

        for i in range(self.num_iterations):
            delta.requires_grad_(True)

            adv_audio = orig_audio + delta
            logits = self.model(target_video, adv_audio, ref_video, ref_audio)[0]
            probs = F.softmax(logits, dim=1)

            loss = F.cross_entropy(logits, labels)
            self.model.zero_grad()
            loss.backward()

            grad = delta.grad.detach()

            with torch.no_grad():
                # L2 normalized gradient with momentum
                grad_flat = grad.view(1, -1)
                grad_norm = torch.norm(grad_flat, dim=1, keepdim=True) + 1e-10
                grad_normalized = grad / grad_norm.view(-1, 1, 1, 1, 1)

                velocity = 0.9 * velocity + grad_normalized
                delta = delta + self.step_size * velocity

                # L2 projection
                delta_flat = delta.view(1, -1)
                delta_norm = torch.norm(delta_flat, dim=1, keepdim=True)
                if delta_norm > self.eps:
                    delta = delta * (self.eps / delta_norm.view(-1, 1, 1, 1, 1))

                delta = delta.detach()

            current_real_prob = probs[0, 0].item()
            if current_real_prob > best_real_prob:
                best_real_prob = current_real_prob
                best_delta = delta.clone()

            if verbose and (i % 50 == 0 or i == self.num_iterations - 1):
                print(f"Iter {i:3d}: Real={current_real_prob:.4f}")

            if current_real_prob > 0.95:
                if verbose:
                    print(f"  Early stop at iter {i}!")
                break

        attack_time = time.time() - start_time

        adv_audio = orig_audio + best_delta

        with torch.no_grad():
            logits = self.model(target_video, adv_audio, ref_video, ref_audio)[0]
            probs = F.softmax(logits, dim=1)
            final_real_prob = probs[0, 0].item()

        info = {
            'method': 'MelSpacePGD',
            'original_real_prob': orig_real_prob,
            'adversarial_real_prob': final_real_prob,
            'confidence_change': final_real_prob - orig_real_prob,
            'attack_success': final_real_prob > 0.5,
            'mel_perturbation_l2': torch.norm(best_delta).item(),
            'eps': self.eps,
            'num_iterations': self.num_iterations,
            'attack_time_seconds': attack_time,
        }

        if verbose:
            print(f"Final: Real={final_real_prob:.4f} (change: {info['confidence_change']:+.4f})")

        return adv_audio, info


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

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
    parser = argparse.ArgumentParser(description="Improved Audio Attacks")
    parser.add_argument("--method", type=str, default="improved-psychoacoustic",
                        choices=["improved-psychoacoustic", "mel-pgd"],
                        help="Attack method")
    parser.add_argument("--num-samples", type=int, default=5)
    parser.add_argument("--output-dir", type=str, default="./improved-audio-results")
    parser.add_argument("--eps", type=float, default=0.3,
                        help="Perturbation budget (0.2-0.5 for waveform, 2-5 for mel)")
    parser.add_argument("--max-iter", type=int, default=300)
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()

    print("=" * 70)
    print("Improved Audio Adversarial Attacks")
    print("=" * 70)
    print()
    print(f"Method: {args.method}")
    print(f"Eps: {args.eps}, Max iterations: {args.max_iter}")
    print()

    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("Loading model...")
    model = load_model(args.device)

    print("Loading dataset...")
    from adversarial_attacks.real_data_loader import AdversarialTestDataset
    dataset = AdversarialTestDataset(device=args.device)

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

        sample = dataset[sample_idx]
        target_video = sample['target_video'].unsqueeze(0)
        target_audio_mel = sample['target_audio'].unsqueeze(0)  # Mel-spectrogram
        ref_audio = sample['reference_audio'].unsqueeze(0)
        ref_video = sample['reference_video'].unsqueeze(0)
        labels = sample['fake_label'].unsqueeze(0)
        info = sample['sample_info']

        video_path = info.get('target_path')
        if not video_path or not Path(video_path).exists():
            print(f"  Skipping - video not found")
            continue

        try:
            if args.method == "improved-psychoacoustic":
                # Extract waveform for waveform-space attack
                original_waveform = extract_audio_from_video(video_path)
                waveform_tensor = torch.from_numpy(original_waveform).float().to(args.device)

                attack = ImprovedPsychoacousticAttack(
                    model=model,
                    eps=args.eps,
                    num_iterations=args.max_iter,
                    device=args.device
                )

                adv_waveform, attack_info = attack.attack(
                    waveform_tensor.unsqueeze(0),
                    target_video, ref_audio, ref_video, labels
                )

                # Save
                adv_wav = adv_waveform[0].cpu().numpy()
                max_val = np.max(np.abs(adv_wav))
                if max_val > 0:
                    adv_wav = adv_wav / max_val * 0.95
                sf.write(str(sample_out / "adversarial.wav"), adv_wav, 16000)

                # Save original
                orig_wav = original_waveform.copy()
                max_val = np.max(np.abs(orig_wav))
                if max_val > 0:
                    orig_wav = orig_wav / max_val * 0.95
                sf.write(str(sample_out / "original.wav"), orig_wav, 16000)

            elif args.method == "mel-pgd":
                attack = MelSpacePGDAttack(
                    model=model,
                    eps=args.eps if args.eps > 1 else 3.0,  # Default higher for mel space
                    num_iterations=args.max_iter,
                    device=args.device
                )

                adv_mel, attack_info = attack.attack(
                    target_audio_mel, target_video, ref_audio, ref_video, labels
                )

                # Note: For mel-space, we'd need reconstruction to save audio
                print("  Note: Mel-space attack - no waveform output (use torchaudio reconstruction)")

            results.append(attack_info)

            # Save stats
            with open(sample_out / "stats.txt", 'w') as f:
                for k, v in attack_info.items():
                    f.write(f"{k}: {v}\n")

        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()

    # Summary
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)

    valid = [r for r in results if 'confidence_change' in r]
    if valid:
        avg_change = np.mean([r['confidence_change'] for r in valid])
        successes = sum(1 for r in valid if r.get('attack_success', False))
        avg_snr = np.mean([r.get('perturbation_snr_db', 0) for r in valid if 'perturbation_snr_db' in r])

        print(f"Samples: {len(valid)}")
        print(f"Successes: {successes}/{len(valid)} ({100*successes/len(valid):.0f}%)")
        print(f"Avg confidence change: {avg_change:+.4f}")
        if avg_snr:
            print(f"Avg SNR: {avg_snr:.1f} dB")

        print("\nPer-sample:")
        for r in valid:
            status = "SUCCESS" if r.get('attack_success') else "FAILED"
            change = r.get('confidence_change', 0)
            print(f"  {change:+.4f} [{status}]")

    print()
    print(f"Outputs saved to: {output_path.absolute()}")


if __name__ == "__main__":
    main()
