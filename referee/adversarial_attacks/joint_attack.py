"""
Joint Adversarial Attack: Video Flickering + Audio PGD

This script combines:
1. Video flickering attack (temporally smooth, imperceptible patterns)
2. Audio PGD attack (mel-spectrogram perturbations)

Both modalities are attacked together for maximum effectiveness against
Referee's multimodal architecture.

Outputs:
- Adversarial video MP4
- Adversarial audio WAV (reconstructed from mel-spectrogram)
- Comparison visualizations
- Attack statistics

Usage:
    python joint_attack.py --num-samples 3 --output-dir ./joint-results
    python joint_attack.py --video-eps 0.15 --audio-eps 5.0 --num-samples 5
"""

import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, Dict, Any, List
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import shutil
import argparse
import time

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

try:
    import cv2
    HAVE_CV2 = True
except ImportError:
    HAVE_CV2 = False
    print("Warning: OpenCV not found. Video saving limited.")

try:
    import librosa
    import soundfile as sf
    HAVE_LIBROSA = True
except ImportError:
    HAVE_LIBROSA = False
    print("Warning: librosa/soundfile not found. Audio WAV output disabled.")
    print("Install with: pip install librosa soundfile")


# Video Flickering Attack (FIXED)

class VideoFlickeringAttack(nn.Module):
    """
    Temporally consistent video perturbation using learnable basis patterns.
    Uses nn.Parameter to ensure leaf tensors for optimization.
    """

    def __init__(
        self,
        video_shape: Tuple[int, ...],  # (S, T, C, H, W)
        num_basis: int = 8,           # Increased from 4
        spatial_freq: int = 4,        # Decreased from 8 (larger patterns)
        flicker_freq: float = 5.0,    # Add flicker frequency control
        device: str = 'cuda'
    ):
        super().__init__()

        S, T, C, H, W = video_shape
        self.video_shape = video_shape
        self.device = device
        self.num_basis = num_basis
        self.spatial_freq = spatial_freq
        self.flicker_freq = flicker_freq

        # Learnable spatial basis patterns - use nn.Parameter (always leaf!)
        basis_size = max(W // spatial_freq, 16)
        self.basis_patterns = nn.Parameter(
            torch.randn(num_basis, C, basis_size, basis_size, device=device) * 0.01
        )

        # Learnable temporal coefficients
        self.temporal_coeffs = nn.Parameter(
            torch.randn(num_basis, S * T, device=device) * 0.1
        )

    def forward(self, eps: float = 0.1) -> torch.Tensor:
        """
        Generate the perturbation tensor.

        Args:
            eps: Maximum perturbation magnitude

        Returns:
            perturbation: (1, S, T, C, H, W) tensor
        """
        S, T, C, H, W = self.video_shape

        # Tile basis patterns to full resolution
        tiled_basis = F.interpolate(
            self.basis_patterns,
            size=(H, W),
            mode='bilinear',
            align_corners=False
        )  # (num_basis, C, H, W)

        # Build perturbation frame by frame
        perturbation = torch.zeros(S, T, C, H, W, device=self.device)

        for s in range(S):
            for t in range(T):
                frame_idx = s * T + t
                frame_pert = torch.zeros(C, H, W, device=self.device)

                for b in range(self.num_basis):
                    coeff = torch.tanh(self.temporal_coeffs[b, frame_idx])
                    frame_pert = frame_pert + coeff * tiled_basis[b]

                perturbation[s, t] = frame_pert

        # Clamp to epsilon ball
        perturbation = torch.clamp(perturbation, -eps, eps)

        return perturbation.unsqueeze(0)  # (1, S, T, C, H, W)


# Audio PGD Attack

class AudioPGDAttack:
    """
    PGD attack on audio mel-spectrograms.

    Uses L2 norm projection for smooth perturbations.
    """

    def __init__(
        self,
        eps: float = 3.0,
        step_size: float = 0.5,
        num_iterations: int = 40,
        device: str = 'cuda'
    ):
        self.eps = eps
        self.step_size = step_size
        self.num_iterations = num_iterations
        self.device = device

    def attack(
        self,
        model: nn.Module,
        target_audio: torch.Tensor,
        target_video: torch.Tensor,
        ref_audio: torch.Tensor,
        ref_video: torch.Tensor,
        labels: torch.Tensor,
        verbose: bool = True
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Run PGD attack on audio.

        Args:
            model: Referee model
            target_audio: (1, S, 1, F, T) mel-spectrogram
            target_video: (1, S, T, C, H, W) - kept unchanged
            ref_audio, ref_video: Reference inputs
            labels: Ground truth labels

        Returns:
            adv_audio: Adversarial audio
            info: Attack statistics
        """
        model.eval()

        # Clone and enable gradients
        adv_audio = target_audio.clone().detach().requires_grad_(True)
        orig_audio = target_audio.clone().detach()

        # Get initial prediction
        with torch.no_grad():
            logits = model(target_video, target_audio, ref_video, ref_audio)[0]
            probs = F.softmax(logits, dim=1)
            orig_real_prob = probs[0, 0].item()

        best_adv = adv_audio.clone()
        best_real_prob = orig_real_prob

        for i in range(self.num_iterations):
            adv_audio.requires_grad_(True)

            # Forward pass
            logits = model(target_video, adv_audio, ref_video, ref_audio)[0]

            # Loss: maximize real probability (untargeted attack on fake samples)
            loss = F.cross_entropy(logits, labels)

            # Backward
            model.zero_grad()
            loss.backward()

            # Get gradient and normalize
            grad = adv_audio.grad.detach()
            grad_flat = grad.reshape(1, -1)
            grad_norm = torch.norm(grad_flat, dim=1, keepdim=True) + 1e-10
            grad_normalized = grad / grad_norm.reshape(-1, 1, 1, 1, 1)

            # Gradient ASCENT (maximize loss = fool the model)
            with torch.no_grad():
                adv_audio = adv_audio + self.step_size * grad_normalized

                # Project to epsilon ball (L2)
                delta = adv_audio - orig_audio
                delta_flat = delta.reshape(1, -1)
                delta_norm = torch.norm(delta_flat, dim=1, keepdim=True)

                if delta_norm > self.eps:
                    delta = delta * (self.eps / delta_norm.reshape(-1, 1, 1, 1, 1))

                adv_audio = orig_audio + delta
                adv_audio = adv_audio.detach()

            # Track best
            with torch.no_grad():
                logits = model(target_video, adv_audio, ref_video, ref_audio)[0]
                probs = F.softmax(logits, dim=1)
                current_real_prob = probs[0, 0].item()

                if current_real_prob > best_real_prob:
                    best_real_prob = current_real_prob
                    best_adv = adv_audio.clone()

                if verbose and i % 10 == 0:
                    print(f"  Audio iter {i}: Real_prob={current_real_prob:.4f}")

        # Compute stats
        delta = best_adv - orig_audio
        l2_norm = torch.norm(delta).item()

        info = {
            'audio_original_real_prob': orig_real_prob,
            'audio_final_real_prob': best_real_prob,
            'audio_confidence_change': best_real_prob - orig_real_prob,
            'audio_l2_norm': l2_norm,
        }

        return best_adv, info


# Joint Attack (Video + Audio)

class JointAttack:
    """
    Combined video flickering + audio PGD attack.

    Attacks both modalities simultaneously for maximum effectiveness
    against Referee's multimodal architecture.
    """

    def __init__(
        self,
        video_eps: float = 0.2,       # Increased from 0.1
        audio_eps: float = 5.0,       # Increased from 3.0
        video_lr: float = 0.02,       # Increased from 0.01
        audio_step: float = 1.0,      # Increased from 0.5
        num_iterations: int = 100,    # Increased from 50
        flicker_freq: float = 5.0,    # Temporal flicker frequency
        spatial_freq: int = 4,        # Spatial frequency (lower = larger patterns)
        num_basis: int = 8,           # Number of basis patterns
        smoothness_weight: float = 1.0,  # Temporal smoothness (lower = more aggressive)
        device: str = 'cuda'
    ):
        self.video_eps = video_eps
        self.audio_eps = audio_eps
        self.video_lr = video_lr
        self.audio_step = audio_step
        self.num_iterations = num_iterations
        self.flicker_freq = flicker_freq
        self.spatial_freq = spatial_freq
        self.num_basis = num_basis
        self.smoothness_weight = smoothness_weight
        self.device = device

    def attack(
        self,
        model: nn.Module,
        target_video: torch.Tensor,
        target_audio: torch.Tensor,
        ref_video: torch.Tensor,
        ref_audio: torch.Tensor,
        labels: torch.Tensor,
        verbose: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """
        Run joint attack on both video and audio.

        Returns:
            adv_video: Adversarial video (1, S, T, C, H, W)
            adv_audio: Adversarial audio (1, S, 1, F, T)
            info: Attack statistics
        """
        model.eval()

        # Get initial prediction
        with torch.no_grad():
            logits = model(target_video, target_audio, ref_video, ref_audio)[0]
            probs = F.softmax(logits, dim=1)
            orig_real_prob = probs[0, 0].item()
            orig_fake_prob = probs[0, 1].item()

        if verbose:
            print(f"Initial: Real={orig_real_prob:.4f}, Fake={orig_fake_prob:.4f}")

        # Initialize video flickering module
        video_shape = target_video.shape[1:]  # (S, T, C, H, W)
        video_attack = VideoFlickeringAttack(
            video_shape=video_shape,
            num_basis=self.num_basis,
            spatial_freq=self.spatial_freq,
            flicker_freq=self.flicker_freq,
            device=self.device
        )

        # Initialize audio perturbation (direct tensor)
        audio_delta = torch.zeros_like(target_audio, requires_grad=True)
        orig_audio = target_audio.clone().detach()

        # Optimizer for video (flickering parameters)
        video_optimizer = torch.optim.Adam(video_attack.parameters(), lr=self.video_lr)

        # Track best results
        best_adv_video = target_video.clone()
        best_adv_audio = target_audio.clone()
        best_real_prob = orig_real_prob

        start_time = time.time()

        for i in range(self.num_iterations):
            # === Video Update ===
            video_optimizer.zero_grad()

            # Generate video perturbation
            video_pert = video_attack(eps=self.video_eps)
            adv_video = torch.clamp(target_video + video_pert, -1.0, 1.0)

            # === Audio Update ===
            audio_delta.requires_grad_(True)
            adv_audio = orig_audio + audio_delta

            # Forward pass with both perturbations
            logits = model(adv_video, adv_audio, ref_video, ref_audio)[0]
            probs = F.softmax(logits, dim=1)

            # Loss: maximize real probability (= minimize fake)
            loss = F.cross_entropy(logits, labels)

            # Add temporal smoothness for video
            if video_pert.shape[2] > 1:
                temporal_diff = video_pert[:, :, 1:] - video_pert[:, :, :-1]
                smoothness = torch.mean(temporal_diff ** 2) * self.smoothness_weight
                loss = loss + smoothness

            # Backward
            loss.backward()

            # Update video
            video_optimizer.step()

            # Update audio (manual PGD step)
            with torch.no_grad():
                if audio_delta.grad is not None:
                    grad = audio_delta.grad
                    grad_flat = grad.reshape(1, -1)
                    grad_norm = torch.norm(grad_flat, dim=1, keepdim=True) + 1e-10
                    grad_normalized = grad / grad_norm.reshape(-1, 1, 1, 1, 1)

                    # Gradient ascent
                    audio_delta.data = audio_delta.data + self.audio_step * grad_normalized

                    # Project to L2 ball
                    delta_flat = audio_delta.data.reshape(1, -1)
                    delta_norm = torch.norm(delta_flat, dim=1, keepdim=True)
                    if delta_norm > self.audio_eps:
                        audio_delta.data = audio_delta.data * (self.audio_eps / delta_norm.reshape(-1, 1, 1, 1, 1))

            audio_delta.grad = None

            # Track best
            current_real_prob = probs[0, 0].item()
            if current_real_prob > best_real_prob:
                best_real_prob = current_real_prob
                best_adv_video = adv_video.detach().clone()
                best_adv_audio = adv_audio.detach().clone()

            if verbose and (i % 10 == 0 or i == self.num_iterations - 1):
                print(f"Iter {i:3d}: Loss={loss.item():.4f}, Real={current_real_prob:.4f}")

            # Early stopping
            if current_real_prob > 0.95:
                if verbose:
                    print(f"  Early stop at iter {i}!")
                break

        attack_time = time.time() - start_time

        # Final evaluation
        with torch.no_grad():
            logits = model(best_adv_video, best_adv_audio, ref_video, ref_audio)[0]
            probs = F.softmax(logits, dim=1)
            final_real_prob = probs[0, 0].item()
            final_fake_prob = probs[0, 1].item()

        # Compute stats
        video_delta = best_adv_video - target_video
        audio_delta_final = best_adv_audio - target_audio

        info = {
            'method': 'JointAttack_Flickering+PGD',
            'original_real_prob': orig_real_prob,
            'original_fake_prob': orig_fake_prob,
            'final_real_prob': final_real_prob,
            'final_fake_prob': final_fake_prob,
            'confidence_change': final_real_prob - orig_real_prob,
            'attack_success': final_real_prob > 0.5,
            'video_l2_norm': torch.norm(video_delta).item(),
            'video_linf_norm': torch.max(torch.abs(video_delta)).item(),
            'audio_l2_norm': torch.norm(audio_delta_final).item(),
            'video_eps': self.video_eps,
            'audio_eps': self.audio_eps,
            'num_iterations': self.num_iterations,
            'attack_time_seconds': attack_time
        }

        if verbose:
            print(f"Final: Real={final_real_prob:.4f}, Fake={final_fake_prob:.4f}")
            print(f"Confidence change: {info['confidence_change']:+.4f}")

        return best_adv_video, best_adv_audio, info


# Audio Reconstruction (Mel-spectrogram -> WAV)

def mel_spectrogram_to_audio(
    mel_spec: torch.Tensor,
    sample_rate: int = 16000,
    n_fft: int = 1024,
    hop_length: int = 160,
    win_length: int = 400,
    n_mels: int = 128,
    audio_mean: float = -4.2677393,
    audio_std: float = 4.5689974,
) -> np.ndarray:
    """
    Convert mel-spectrogram back to audio waveform using Griffin-Lim.

    Args:
        mel_spec: (1, S, 1, F, T) normalized mel-spectrogram tensor
        Other args: Parameters matching the forward transform

    Returns:
        audio: 1D numpy array of audio samples
    """
    if not HAVE_LIBROSA:
        raise ImportError("librosa required for audio reconstruction")

    # Denormalize: reverse the AST normalization
    # Original: mel = (mel - mean) / (2 * std)
    # Reverse: mel = mel * (2 * std) + mean
    mel_denorm = mel_spec.cpu().numpy() * (2 * audio_std) + audio_mean

    # Shape: (1, S, 1, F, T) -> need to combine segments
    # Remove batch and channel dims: (S, F, T)
    mel_denorm = mel_denorm[0, :, 0, :, :]  # (S, F, T)

    S, F, T = mel_denorm.shape

    # Concatenate segments (with overlap handling)
    # For simplicity, just concatenate (some overlap is okay for audio)
    all_audio = []

    for s in range(S):
        segment_mel = mel_denorm[s]  # (F, T)

        # Convert from log scale back to linear
        segment_mel_linear = np.exp(segment_mel)

        # Inverse mel spectrogram using librosa
        try:
            audio_segment = librosa.feature.inverse.mel_to_audio(
                segment_mel_linear,
                sr=sample_rate,
                n_fft=n_fft,
                hop_length=hop_length,
                win_length=win_length,
                n_iter=32  # Griffin-Lim iterations
            )
            all_audio.append(audio_segment)
        except Exception as e:
            print(f"Warning: Failed to reconstruct segment {s}: {e}")
            # Create silence as fallback
            segment_length = (T - 1) * hop_length + win_length
            all_audio.append(np.zeros(segment_length))

    # Concatenate all segments
    full_audio = np.concatenate(all_audio)

    # Normalize to prevent clipping
    max_val = np.max(np.abs(full_audio))
    if max_val > 0:
        full_audio = full_audio / max_val * 0.95

    return full_audio.astype(np.float32)


def save_audio_wav(audio: np.ndarray, save_path: Path, sample_rate: int = 16000):
    """Save audio array to WAV file."""
    if not HAVE_LIBROSA:
        print("  Skipping WAV save (librosa not available)")
        return

    sf.write(str(save_path), audio, sample_rate)
    print(f"  Saved audio WAV: {save_path}")


def extract_audio_from_video(video_path: str, output_path: Path, sample_rate: int = 16000):
    """
    Extract audio directly from video file using ffmpeg.

    This produces clean, listenable audio (not reconstructed from mel-spectrograms).

    Args:
        video_path: Path to source video file
        output_path: Path to save WAV file
        sample_rate: Audio sample rate
    """
    import subprocess

    video_path = Path(video_path)
    if not video_path.exists():
        print(f"  Warning: Video file not found: {video_path}")
        return False

    try:
        # Use ffmpeg to extract audio
        cmd = [
            'ffmpeg', '-y',  # Overwrite output
            '-i', str(video_path),
            '-vn',  # No video
            '-acodec', 'pcm_s16le',  # PCM 16-bit
            '-ar', str(sample_rate),  # Sample rate
            '-ac', '1',  # Mono
            str(output_path)
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30
        )

        if result.returncode == 0 and output_path.exists():
            print(f"  Extracted audio: {output_path}")
            return True
        else:
            print(f"  Warning: ffmpeg failed: {result.stderr[:200] if result.stderr else 'unknown error'}")
            return False

    except FileNotFoundError:
        print("  Warning: ffmpeg not found. Install ffmpeg for audio extraction.")
        return False
    except subprocess.TimeoutExpired:
        print("  Warning: ffmpeg timed out")
        return False
    except Exception as e:
        print(f"  Warning: Audio extraction failed: {e}")
        return False


# Video Saving (Fixed overlap handling)

def save_video_mp4(
    video_tensor: torch.Tensor,
    save_path: Path,
    fps: int = 25,
    segment_overlap: float = 0.5
):
    """Save video tensor as MP4 with proper segment handling."""
    if not HAVE_CV2:
        print("  Skipping video save (OpenCV not available)")
        return

    # Denormalize
    mean = torch.tensor([0.5, 0.5, 0.5], device=video_tensor.device).view(1, 1, 1, 3, 1, 1)
    std = torch.tensor([0.5, 0.5, 0.5], device=video_tensor.device).view(1, 1, 1, 3, 1, 1)

    video = torch.clamp(video_tensor * std + mean, 0, 1)

    if video.dim() == 6:
        video = video[0]  # Remove batch

    S, T, C, H, W = video.shape

    # Handle overlapping segments - only use non-overlapping portion
    non_overlap_frames = int(T * (1 - segment_overlap))
    if non_overlap_frames < 1:
        non_overlap_frames = 1

    frames = []
    for s in range(S):
        if s < S - 1:
            segment_frames = video[s, :non_overlap_frames]
        else:
            segment_frames = video[s]

        for t in range(segment_frames.shape[0]):
            frame = segment_frames[t].permute(1, 2, 0).cpu().numpy()
            frame = (frame * 255).astype(np.uint8)
            frame = frame[:, :, ::-1]  # RGB to BGR
            frames.append(frame)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(save_path), fourcc, fps, (W, H))

    for frame in frames:
        out.write(frame)

    out.release()
    print(f"  Saved video ({len(frames)} frames): {save_path}")


# Visualization

def save_comparison(orig_video, adv_video, save_path: Path, num_frames: int = 8):
    """Save video frame comparison."""
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

    plt.suptitle('Joint Attack: Video Comparison', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved comparison: {save_path}")


def save_audio_comparison(orig_audio, adv_audio, save_path: Path):
    """Save audio spectrogram comparison."""
    # Denormalize
    audio_mean, audio_std = -4.2677393, 4.5689974
    orig = orig_audio.cpu().numpy() * (2 * audio_std) + audio_mean
    adv = adv_audio.cpu().numpy() * (2 * audio_std) + audio_mean

    # Get first segment: (F, T)
    orig_spec = orig[0, 0, 0]
    adv_spec = adv[0, 0, 0]
    diff = adv_spec - orig_spec

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    im0 = axes[0].imshow(orig_spec, aspect='auto', origin='lower', cmap='viridis')
    axes[0].set_title('Original Audio')
    axes[0].set_xlabel('Time')
    axes[0].set_ylabel('Mel Bins')
    plt.colorbar(im0, ax=axes[0])

    im1 = axes[1].imshow(adv_spec, aspect='auto', origin='lower', cmap='viridis')
    axes[1].set_title('Adversarial Audio')
    axes[1].set_xlabel('Time')
    plt.colorbar(im1, ax=axes[1])

    im2 = axes[2].imshow(diff, aspect='auto', origin='lower', cmap='RdBu_r')
    axes[2].set_title('Perturbation')
    axes[2].set_xlabel('Time')
    plt.colorbar(im2, ax=axes[2])

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved audio comparison: {save_path}")


def save_stats(info: Dict, save_path: Path):
    """Save attack statistics."""
    with open(save_path, 'w') as f:
        f.write("Joint Attack Results\n")
        f.write("=" * 50 + "\n\n")
        for k, v in info.items():
            if isinstance(v, float):
                f.write(f"{k}: {v:.6f}\n")
            else:
                f.write(f"{k}: {v}\n")
    print(f"  Saved stats: {save_path}")


# Model and Data Loading

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


def load_sample(dataset, idx: int, device: str):
    """Load a sample from dataset."""
    sample = dataset[idx]
    return (
        sample['target_audio'].unsqueeze(0).to(device),
        sample['target_video'].unsqueeze(0).to(device),
        sample['reference_audio'].unsqueeze(0).to(device),
        sample['reference_video'].unsqueeze(0).to(device),
        torch.tensor([sample['fake_label']], device=device),
        sample['sample_info']
    )


# Main

def main():
    parser = argparse.ArgumentParser(description="Joint Video+Audio Adversarial Attack")
    parser.add_argument("--num-samples", type=int, default=3,
                        help="Number of samples to test")
    parser.add_argument("--output-dir", type=str, default="./joint-results",
                        help="Output directory")
    parser.add_argument("--video-eps", type=float, default=0.2,
                        help="Video perturbation budget (default: 0.2)")
    parser.add_argument("--audio-eps", type=float, default=1.0,
                        help="Audio perturbation budget L2 (default: 1.0, reduced for human-like audio). "
                             "Use 0.5-1.0 for human-like speech, 3.0-5.0 for maximum attack strength")
    parser.add_argument("--max-iter", type=int, default=100,
                        help="Max iterations (default: 100)")
    parser.add_argument("--video-lr", type=float, default=0.03,
                        help="Video learning rate (default: 0.03)")
    parser.add_argument("--audio-step", type=float, default=0.5,
                        help="Audio PGD step size (default: 0.5)")
    parser.add_argument("--flicker-freq", type=float, default=7.0,
                        help="Temporal flicker frequency in Hz (default: 7.0, higher = faster flicker)")
    parser.add_argument("--spatial-freq", type=int, default=3,
                        help="Spatial pattern frequency (default: 3, lower = larger patterns = more effective)")
    parser.add_argument("--num-basis", type=int, default=12,
                        help="Number of basis patterns (default: 12, more = more expressive)")
    parser.add_argument("--smoothness-weight", type=float, default=0.5,
                        help="Temporal smoothness weight (default: 0.5, lower = more aggressive)")
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()

    print("=" * 70)
    print("Joint Adversarial Attack: Video Flickering + Audio PGD")
    print("=" * 70)
    print(f"Video eps: {args.video_eps}, Audio eps: {args.audio_eps}")
    print(f"Video LR: {args.video_lr}, Audio step: {args.audio_step}")
    print(f"Flicker freq: {args.flicker_freq}Hz, Spatial freq: {args.spatial_freq}")
    print(f"Num basis: {args.num_basis}, Smoothness: {args.smoothness_weight}")
    print(f"Max iterations: {args.max_iter}")
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

        target_audio, target_video, ref_audio, ref_video, labels, info = \
            load_sample(dataset, sample_idx, args.device)

        print(f"Source: {info.get('target_path', 'unknown')}")
        print(f"Video shape: {target_video.shape}")
        print(f"Audio shape: {target_audio.shape}")

        try:
            # Run joint attack with aggressive hyperparameters
            joint_attack = JointAttack(
                video_eps=args.video_eps,
                audio_eps=args.audio_eps,
                video_lr=args.video_lr,
                audio_step=args.audio_step,
                num_iterations=args.max_iter,
                flicker_freq=args.flicker_freq,
                spatial_freq=args.spatial_freq,
                num_basis=args.num_basis,
                smoothness_weight=args.smoothness_weight,
                device=args.device
            )

            adv_video, adv_audio, attack_info = joint_attack.attack(
                model, target_video, target_audio, ref_video, ref_audio, labels
            )

            attack_info['sample_index'] = sample_idx
            attack_info['source_path'] = info.get('target_path', 'unknown')
            results.append(attack_info)

            # Save outputs
            print(f"\nSaving to {sample_out}...")

            # Copy original video file
            source_video_path = info.get('target_path')
            if source_video_path and Path(source_video_path).exists():
                shutil.copy2(source_video_path, sample_out / "original_video_file.mp4")
                print(f"  Copied original video file")

            # Save processed videos
            save_video_mp4(adv_video, sample_out / "adversarial_video.mp4")
            save_video_mp4(target_video, sample_out / "original_processed.mp4")

            # Save video comparison
            save_comparison(target_video, adv_video, sample_out / "video_comparison.png")

            # Save audio spectrograms
            save_audio_comparison(target_audio, adv_audio, sample_out / "audio_comparison.png")

            # Save audio WAV files
            # ORIGINAL: Extract directly from video file (clean audio)
            if source_video_path:
                extract_audio_from_video(
                    source_video_path,
                    sample_out / "original_audio.wav",
                    sample_rate=16000
                )

            # ADVERSARIAL: Use Griffin-Lim reconstruction (will sound noisy/robotic)
            # This is expected - mel-spectrogram perturbations can't be perfectly inverted
            if HAVE_LIBROSA:
                try:
                    print("  Note: Adversarial audio uses Griffin-Lim reconstruction (may sound robotic)")
                    adv_wav = mel_spectrogram_to_audio(adv_audio)
                    save_audio_wav(adv_wav, sample_out / "adversarial_audio.wav")
                except Exception as e:
                    print(f"  Warning: Adversarial audio WAV conversion failed: {e}")

            # Save stats
            save_stats(attack_info, sample_out / "stats.txt")

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
        avg_time = np.mean([r.get('attack_time_seconds', 0) for r in valid])

        print(f"Samples tested: {len(valid)}")
        print(f"Successful attacks: {successes}/{len(valid)}")
        print(f"Average confidence change: {avg_change:+.4f}")
        print(f"Average attack time: {avg_time:.1f}s")
        print()

        print("Per-sample results:")
        for r in valid:
            status = "SUCCESS" if r.get('attack_success') else "FAILED"
            change = r.get('confidence_change', 0)
            print(f"  Sample: {change:+.4f} [{status}]")

    # Save summary
    summary_path = output_path / "summary.txt"
    with open(summary_path, 'w') as f:
        f.write("Joint Attack Summary\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Video eps: {args.video_eps}\n")
        f.write(f"Audio eps: {args.audio_eps}\n")
        f.write(f"Video LR: {args.video_lr}\n")
        f.write(f"Audio step: {args.audio_step}\n")
        f.write(f"Flicker freq: {args.flicker_freq}Hz\n")
        f.write(f"Spatial freq: {args.spatial_freq}\n")
        f.write(f"Num basis: {args.num_basis}\n")
        f.write(f"Smoothness weight: {args.smoothness_weight}\n")
        f.write(f"Max iterations: {args.max_iter}\n\n")

        if valid:
            f.write(f"Samples: {len(valid)}\n")
            f.write(f"Successes: {successes}/{len(valid)}\n")
            f.write(f"Avg confidence change: {avg_change:+.4f}\n")
            f.write(f"Avg time: {avg_time:.1f}s\n\n")

            for r in valid:
                status = "SUCCESS" if r.get('attack_success') else "FAILED"
                f.write(f"  {r.get('confidence_change', 0):+.4f} [{status}]\n")

    print(f"\nOutputs saved to: {output_path.absolute()}")
    print()
    print("=" * 70)
    print("Done!")
    print("=" * 70)


if __name__ == "__main__":
    main()
