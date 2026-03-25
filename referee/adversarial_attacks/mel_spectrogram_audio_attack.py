"""
Mel-Spectrogram Space Audio Attack with Torchaudio Reconstruction

This module implements adversarial audio attacks in mel-spectrogram space,
using torchaudio's InverseMelScale + GriffinLim for high-quality reconstruction.

Key insight from testing: torchaudio reconstruction is MUCH better than librosa's.
This attack:
1. Converts audio to mel-spectrogram
2. Applies PGD perturbations in mel-spectrogram space
3. Reconstructs using torchaudio (high quality)
4. Outputs playable WAV files

Usage:
    python mel_spectrogram_audio_attack.py --num-samples 3 --output-dir ./mel-attack-results
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

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

try:
    import soundfile as sf
    HAVE_SOUNDFILE = True
except ImportError:
    HAVE_SOUNDFILE = False
    print("Warning: soundfile not found. Install with: pip install soundfile")


class TorchaudioMelReconstructor:
    """
    High-quality mel-spectrogram to waveform reconstruction using torchaudio.

    This uses InverseMelScale + GriffinLim which produces much better results
    than librosa's implementation.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        n_fft: int = 1024,
        hop_length: int = 160,
        win_length: int = 400,
        n_mels: int = 128,
        n_iter: int = 64,
        audio_mean: float = -4.2677393,
        audio_std: float = 4.5689974,
        device: str = 'cuda'
    ):
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.n_mels = n_mels
        self.n_iter = n_iter
        self.audio_mean = audio_mean
        self.audio_std = audio_std
        self.device = device

        # Forward transform
        self.mel_spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            n_mels=n_mels
        ).to(device)

        # Inverse transforms
        self.inverse_mel = torchaudio.transforms.InverseMelScale(
            n_stft=n_fft // 2 + 1,
            n_mels=n_mels,
            sample_rate=sample_rate
        ).to(device)

        self.griffin_lim = torchaudio.transforms.GriffinLim(
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            n_iter=n_iter
        ).to(device)

    def waveform_to_mel(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Convert waveform to normalized log mel-spectrogram.

        Args:
            waveform: (T,) or (B, T) audio waveform

        Returns:
            mel: (F, Ta) or (B, F, Ta) normalized log mel-spectrogram
        """
        # Ensure 2D
        squeeze = False
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
            squeeze = True

        # Mel spectrogram
        mel = self.mel_spec(waveform)  # (B, n_mels, T)

        # Log transform
        mel = torch.log(mel + 1e-6)

        # Normalize (AST normalization)
        mel = (mel - self.audio_mean) / (2 * self.audio_std)

        if squeeze:
            mel = mel.squeeze(0)

        return mel

    def mel_to_waveform(self, mel: torch.Tensor) -> torch.Tensor:
        """
        Convert normalized log mel-spectrogram back to waveform using torchaudio.

        Args:
            mel: (F, Ta) or (B, F, Ta) normalized log mel-spectrogram

        Returns:
            waveform: (T,) or (B, T) reconstructed audio
        """
        # Ensure 3D
        squeeze = False
        if mel.dim() == 2:
            mel = mel.unsqueeze(0)
            squeeze = True

        # Denormalize
        mel_denorm = mel * (2 * self.audio_std) + self.audio_mean

        # Inverse log (exp)
        mel_linear = torch.exp(mel_denorm)

        # Inverse mel scale -> linear spectrogram
        spec = self.inverse_mel(mel_linear)  # (B, n_fft//2+1, T)

        # Griffin-Lim reconstruction
        waveform = self.griffin_lim(spec)  # (B, T)

        if squeeze:
            waveform = waveform.squeeze(0)

        return waveform


class MelSpectrogramAudioAttack:
    """
    PGD attack in mel-spectrogram space with high-quality torchaudio reconstruction.

    This attack:
    1. Takes original waveform
    2. Converts to mel-spectrogram
    3. Applies PGD perturbations in mel space
    4. Reconstructs using torchaudio (high quality)
    """

    def __init__(
        self,
        model: nn.Module,
        eps: float = 2.0,  # L2 perturbation budget in mel space
        step_size: float = 0.3,
        num_iterations: int = 100,
        device: str = 'cuda'
    ):
        """
        Args:
            model: Referee model
            eps: L2 perturbation budget in mel-spectrogram space
            step_size: PGD step size
            num_iterations: Number of attack iterations
            device: Device to use
        """
        self.model = model
        self.eps = eps
        self.step_size = step_size
        self.num_iterations = num_iterations
        self.device = device

        # Mel reconstruction module
        self.reconstructor = TorchaudioMelReconstructor(device=device)

        # Segment parameters (matching model preprocessing)
        self.n_segments = 8
        self.max_spec_t = 66
        self.segment_size_vframes = 16
        self.v_fps = 25
        self.sample_rate = 16000
        self.seg_size_aframes = int(self.segment_size_vframes / self.v_fps * self.sample_rate)
        self.stride_aframes = int(0.5 * self.seg_size_aframes)

    def _segment_waveform(self, waveform: torch.Tensor) -> Tuple[torch.Tensor, int]:
        """
        Segment waveform into overlapping segments matching model preprocessing.

        Returns:
            segments: (S, seg_len) audio segments
            start_idx: Starting index in original waveform
        """
        T = waveform.shape[0]

        # Calculate required length
        required_len = (self.n_segments - 1) * self.stride_aframes + self.seg_size_aframes

        # Pad if necessary
        if T < required_len:
            pad_len = required_len - T
            waveform = F.pad(waveform.unsqueeze(0), (0, pad_len)).squeeze(0)
            T = required_len

        # Center crop
        seg_seq_len = self.n_segments * 0.5 + (1 - 0.5)
        aframes_seg_seq_len = int(seg_seq_len * self.seg_size_aframes)
        max_a_start = max(T - aframes_seg_seq_len, 0)
        a_start = max_a_start // 2

        # Extract segments
        segments = []
        for s in range(self.n_segments):
            start = a_start + s * self.stride_aframes
            end = start + self.seg_size_aframes
            segments.append(waveform[start:end])

        return torch.stack(segments), a_start

    def _segments_to_model_mel(self, segments: torch.Tensor) -> torch.Tensor:
        """
        Convert audio segments to model input format.

        Args:
            segments: (S, seg_len) audio segments

        Returns:
            mel: (1, S, 1, F, Ta) normalized log mel-spectrogram
        """
        S = segments.shape[0]
        mels = []

        for s in range(S):
            mel = self.reconstructor.waveform_to_mel(segments[s])  # (F, Ta)

            # Pad/truncate time dimension
            Ta = mel.shape[-1]
            if Ta < self.max_spec_t:
                mel = F.pad(mel, (0, self.max_spec_t - Ta))
            elif Ta > self.max_spec_t:
                mel = mel[..., :self.max_spec_t]

            mels.append(mel)

        # Stack: (S, F, Ta) -> (1, S, 1, F, Ta)
        mel_stacked = torch.stack(mels, dim=0)  # (S, F, Ta)
        mel_stacked = mel_stacked.unsqueeze(0).unsqueeze(2)  # (1, S, 1, F, Ta)

        return mel_stacked

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
        Run mel-spectrogram space attack with torchaudio reconstruction.

        Args:
            original_waveform: (T,) raw audio waveform
            target_video: (1, S, Tv, C, H, W) preprocessed video
            ref_audio: (1, S, 1, F, Ta) reference audio
            ref_video: (1, S, Tv, C, H, W) reference video
            labels: (1,) ground truth labels

        Returns:
            adv_waveform: (T,) adversarial waveform (reconstructed, playable)
            info: Attack statistics
        """
        self.model.eval()

        # Segment the waveform
        segments, start_idx = self._segment_waveform(original_waveform)
        orig_segments = segments.clone()

        # Convert to mel-spectrogram
        orig_mel = self._segments_to_model_mel(segments)

        # Initialize perturbation in mel space
        delta = torch.zeros_like(orig_mel, requires_grad=True, device=self.device)

        # Get initial prediction
        with torch.no_grad():
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
            adv_mel = orig_mel + delta

            # Forward pass
            logits = self.model(target_video, adv_mel, ref_video, ref_audio)[0]
            probs = F.softmax(logits, dim=1)

            # Loss: maximize real probability
            loss = F.cross_entropy(logits, labels)

            # Backward
            self.model.zero_grad()
            loss.backward()

            # Get gradient
            grad = delta.grad.detach()

            # Normalize gradient (L2)
            grad_flat = grad.view(-1)
            grad_norm = torch.norm(grad_flat) + 1e-10
            grad_normalized = grad / grad_norm

            # PGD update (gradient ascent)
            with torch.no_grad():
                delta = delta + self.step_size * grad_normalized

                # Project to L2 ball
                delta_flat = delta.view(-1)
                delta_norm = torch.norm(delta_flat)
                if delta_norm > self.eps:
                    delta = delta * (self.eps / delta_norm)

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

        # Final adversarial mel-spectrogram
        adv_mel = orig_mel + best_delta

        # Final evaluation
        with torch.no_grad():
            logits = self.model(target_video, adv_mel, ref_video, ref_audio)[0]
            probs = F.softmax(logits, dim=1)
            final_real_prob = probs[0, 0].item()
            final_fake_prob = probs[0, 1].item()

        # Reconstruct audio from adversarial mel-spectrogram
        # We need to reconstruct each segment and stitch them together
        adv_mel_segments = adv_mel[0, :, 0, :, :]  # (S, F, Ta)

        reconstructed_segments = []
        for s in range(self.n_segments):
            segment_mel = adv_mel_segments[s]  # (F, Ta)
            segment_audio = self.reconstructor.mel_to_waveform(segment_mel)
            reconstructed_segments.append(segment_audio)

        # Stitch segments with overlap-add
        adv_waveform = self._stitch_segments(reconstructed_segments)

        # Compute stats
        mel_delta = best_delta
        mel_l2_norm = torch.norm(mel_delta).item()

        info = {
            'method': 'MelSpectrogramPGD_TorchaudioReconstruction',
            'original_real_prob': orig_real_prob,
            'original_fake_prob': orig_fake_prob,
            'adversarial_real_prob': final_real_prob,
            'adversarial_fake_prob': final_fake_prob,
            'confidence_change': final_real_prob - orig_real_prob,
            'attack_success': final_real_prob > 0.5,
            'mel_perturbation_l2': mel_l2_norm,
            'eps': self.eps,
            'num_iterations': self.num_iterations,
            'attack_time_seconds': attack_time,
        }

        if verbose:
            print(f"Final: Real={final_real_prob:.4f}, Fake={final_fake_prob:.4f}")
            print(f"Confidence change: {info['confidence_change']:+.4f}")

        return adv_waveform, info

    def _stitch_segments(self, segments: list) -> torch.Tensor:
        """
        Stitch overlapping audio segments using overlap-add.

        Args:
            segments: List of (T,) audio segments

        Returns:
            waveform: (T,) stitched audio
        """
        if not segments:
            return torch.zeros(1)

        # Calculate output length
        total_len = (len(segments) - 1) * self.stride_aframes + len(segments[0])
        output = torch.zeros(total_len, device=self.device)
        weights = torch.zeros(total_len, device=self.device)

        # Create crossfade window
        seg_len = len(segments[0])
        fade_len = seg_len - self.stride_aframes

        for i, seg in enumerate(segments):
            start = i * self.stride_aframes
            end = start + len(seg)

            # Create window for this segment
            window = torch.ones(len(seg), device=self.device)

            # Apply fade-in for non-first segments
            if i > 0 and fade_len > 0:
                fade_in = torch.linspace(0, 1, fade_len, device=self.device)
                window[:fade_len] = fade_in

            # Apply fade-out for non-last segments
            if i < len(segments) - 1 and fade_len > 0:
                fade_out = torch.linspace(1, 0, fade_len, device=self.device)
                window[-fade_len:] = window[-fade_len:] * fade_out

            output[start:end] += seg * window
            weights[start:end] += window

        # Normalize by weights
        weights = torch.clamp(weights, min=1e-8)
        output = output / weights

        return output


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
    parser = argparse.ArgumentParser(description="Mel-spectrogram audio attack with torchaudio reconstruction")
    parser.add_argument("--num-samples", type=int, default=3)
    parser.add_argument("--output-dir", type=str, default="./mel-attack-results")
    parser.add_argument("--eps", type=float, default=2.0,
                        help="L2 perturbation budget in mel space (1.0-5.0 typical)")
    parser.add_argument("--max-iter", type=int, default=100)
    parser.add_argument("--step-size", type=float, default=0.3)
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()

    print("=" * 70)
    print("Mel-Spectrogram Audio Attack (Torchaudio Reconstruction)")
    print("=" * 70)
    print()
    print("This attack operates in mel-spectrogram space and uses torchaudio's")
    print("high-quality Griffin-Lim for reconstruction (better than librosa).")
    print()

    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("Loading model...")
    model = load_model(args.device)

    print("Loading dataset...")
    from adversarial_attacks.real_data_loader import AdversarialTestDataset

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

        # Load preprocessed data
        sample = dataset[sample_idx]
        target_video = sample['target_video'].unsqueeze(0)
        ref_audio = sample['reference_audio'].unsqueeze(0)
        ref_video = sample['reference_video'].unsqueeze(0)
        labels = sample['fake_label'].unsqueeze(0)
        info = sample['sample_info']

        print(f"Source: {info.get('target_path', 'unknown')}")

        try:
            # Extract original waveform
            video_path = info.get('target_path')
            if not video_path or not Path(video_path).exists():
                print(f"  Skipping - video file not found: {video_path}")
                continue

            print("  Extracting audio waveform from video...")
            original_waveform = extract_audio_from_video(video_path)
            print(f"  Waveform: {len(original_waveform)} samples ({len(original_waveform)/16000:.2f}s)")

            # Convert to tensor
            waveform_tensor = torch.from_numpy(original_waveform).float().to(args.device)

            # Run attack
            attack = MelSpectrogramAudioAttack(
                model=model,
                eps=args.eps,
                step_size=args.step_size,
                num_iterations=args.max_iter,
                device=args.device
            )

            adv_waveform, attack_info = attack.attack(
                waveform_tensor, target_video, ref_audio, ref_video, labels
            )

            results.append(attack_info)

            # Save outputs
            print(f"\nSaving to {sample_out}...")

            # Save original
            orig_wav = original_waveform.copy()
            max_val = np.max(np.abs(orig_wav))
            if max_val > 0:
                orig_wav = orig_wav / max_val * 0.95
            sf.write(str(sample_out / "original_audio.wav"), orig_wav, 16000)
            print(f"  Saved: original_audio.wav")

            # Save adversarial (torchaudio reconstructed)
            adv_wav = adv_waveform.cpu().numpy()
            max_val = np.max(np.abs(adv_wav))
            if max_val > 0:
                adv_wav = adv_wav / max_val * 0.95
            sf.write(str(sample_out / "adversarial_audio.wav"), adv_wav, 16000)
            print(f"  Saved: adversarial_audio.wav (torchaudio reconstructed)")

            # Save stats
            with open(sample_out / "stats.txt", 'w') as f:
                f.write("Mel-Spectrogram Audio Attack Results\n")
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

        print(f"Samples tested: {len(valid)}")
        print(f"Successful attacks: {successes}/{len(valid)}")
        print(f"Average confidence change: {avg_change:+.4f}")
        print()

        print("Per-sample results:")
        for r in valid:
            status = "SUCCESS" if r.get('attack_success') else "FAILED"
            change = r.get('confidence_change', 0)
            print(f"  {change:+.4f} [{status}]")

    print()
    print(f"Outputs saved to: {output_path.absolute()}")


if __name__ == "__main__":
    main()
