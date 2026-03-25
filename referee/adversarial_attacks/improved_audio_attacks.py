"""
Audio adversarial attacks for the Referee deepfake detection model.

Includes:
- ImprovedPsychoacousticAttack: Waveform-space attack with psychoacoustic masking
- MelSpacePGDAttack: Direct perturbation in mel-spectrogram space

Usage:
    python improved_audio_attacks.py --method improved-psychoacoustic --num-samples 5
    python improved_audio_attacks.py --method mel-pgd --num-samples 5
    python improved_audio_attacks.py --preset quality --num-samples 5
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
    print("Warning: soundfile not found.")


# Differentiable Mel Transform

class DifferentiableMelTransform(nn.Module):
    """
    Differentiable mel-spectrogram transform matching Referee's preprocessing.
    Allows gradients to flow from the model loss back to the waveform.
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

        # Segment sizes (matching model preprocessing)
        segment_size_vframes = 16
        v_fps = 25
        self.seg_size_aframes = int(segment_size_vframes / v_fps * sample_rate)
        self.stride_aframes = int(0.5 * self.seg_size_aframes)

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


# Psychoacoustic Attack

class ImprovedPsychoacousticAttack:
    """
    Waveform-space audio attack using psychoacoustic masking.

    Perturbations are hidden in loud parts of the audio where they are
    less audible. Includes SNR regularization for quality control.

    Presets:
        quality:   eps=0.08, target_snr=40dB, snr_weight=0.3, masking_strength=0.3
        balanced:  eps=0.15, target_snr=35dB, snr_weight=0.1, masking_strength=0.5
        effective: eps=0.30, target_snr=25dB, snr_weight=0.0, masking_strength=0.8
    """

    def __init__(
        self,
        model: nn.Module,
        eps: float = 0.15,
        step_size: float = 0.01,
        num_iterations: int = 300,
        momentum: float = 0.9,
        adaptive_step: bool = True,
        target_snr_db: float = 35.0,
        snr_weight: float = 0.1,
        masking_strength: float = 0.5,
        device: str = 'cuda'
    ):
        """
        Args:
            model: Referee model
            eps: Maximum L-inf perturbation (0.05-0.3 recommended)
            step_size: Initial PGD step size
            num_iterations: Attack iterations
            momentum: Momentum coefficient
            adaptive_step: Decay step size over time
            target_snr_db: Target SNR in dB (higher = better quality)
            snr_weight: SNR regularization (0 = ignore, 1 = prioritize quality)
            masking_strength: Masking curve strength (0 = uniform, 1 = full masking)
            device: cuda or cpu
        """
        self.model = model
        self.eps = eps
        self.step_size = step_size
        self.num_iterations = num_iterations
        self.momentum = momentum
        self.adaptive_step = adaptive_step
        self.target_snr_db = target_snr_db
        self.snr_weight = snr_weight
        self.masking_strength = masking_strength
        self.device = device

        self.mel_transform = DifferentiableMelTransform(device=device).to(device)

    def compute_masking_curve(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Compute psychoacoustic masking curve based on signal energy.
        Louder parts can hide more perturbation.
        """
        B, T = waveform.shape

        n_fft = 1024
        hop_length = 256
        window = torch.hann_window(n_fft, device=waveform.device)

        spec = torch.stft(
            waveform, n_fft=n_fft, hop_length=hop_length,
            window=window, return_complex=True
        )
        magnitude = torch.abs(spec)

        # Energy per frame
        frame_energy = torch.sum(magnitude ** 2, dim=1)

        # Spectral flatness (tonal sounds mask better)
        geometric_mean = torch.exp(torch.mean(torch.log(magnitude + 1e-10), dim=1))
        arithmetic_mean = torch.mean(magnitude, dim=1)
        spectral_flatness = geometric_mean / (arithmetic_mean + 1e-10)

        # High energy + low flatness = good masking
        masking = frame_energy * (1 - 0.5 * spectral_flatness)
        masking = masking / (masking.max() + 1e-10)

        # Interpolate to waveform length
        mask = F.interpolate(
            masking.unsqueeze(1), size=T, mode='linear', align_corners=False
        ).squeeze(1)

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
        """Run the psychoacoustic attack."""
        self.model.eval()

        if original_waveform.dim() == 1:
            original_waveform = original_waveform.unsqueeze(0)

        B, T = original_waveform.shape
        orig_waveform = original_waveform.clone().detach()

        active_start, active_end = self.mel_transform.get_active_region(T)

        if verbose:
            print(f"  Audio: {T} samples ({T/16000:.2f}s)")
            print(f"  Active region: [{active_start}:{active_end}] ({(active_end-active_start)/16000:.2f}s)")
            print(f"  Attack: eps={self.eps}, target_snr={self.target_snr_db}dB, snr_weight={self.snr_weight}")
            print(f"  Masking strength: {self.masking_strength}, iterations: {self.num_iterations}")

        # Compute masking curve
        with torch.no_grad():
            raw_masking_curve = self.compute_masking_curve(orig_waveform)
            masking_curve = 1.0 - self.masking_strength * (1.0 - raw_masking_curve)
            masking_curve[:, :active_start] = 0
            masking_curve[:, active_end:] = 0

        delta = torch.zeros_like(orig_waveform)
        velocity = torch.zeros_like(orig_waveform)

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

            if self.adaptive_step:
                current_step = self.step_size * (1 - 0.5 * i / self.num_iterations)
            else:
                current_step = self.step_size

            adv_waveform = orig_waveform + delta
            adv_mel = self.mel_transform(adv_waveform)
            logits = self.model(target_video, adv_mel, ref_video, ref_audio)[0]
            probs = F.softmax(logits, dim=1)

            classification_loss = F.cross_entropy(logits, labels)

            # SNR regularization
            if self.snr_weight > 0:
                active_delta = delta[:, active_start:active_end]
                active_orig = orig_waveform[:, active_start:active_end]
                signal_power = torch.sum(active_orig ** 2)
                noise_power = torch.sum(active_delta ** 2) + 1e-10
                current_snr = 10 * torch.log10(signal_power / noise_power)
                snr_penalty = F.relu(self.target_snr_db - current_snr)
                loss = classification_loss - self.snr_weight * snr_penalty
            else:
                loss = classification_loss

            # Backpropagation
            self.model.zero_grad()
            loss.backward()
            grad = delta.grad.detach()

            # Update perturbation with masking and momentum
            with torch.no_grad():
                grad_masked = grad * masking_curve
                grad_norm = torch.norm(grad_masked.view(B, -1), dim=1, keepdim=True).view(B, 1) + 1e-10
                grad_normalized = grad_masked / grad_norm

                velocity = self.momentum * velocity + grad_normalized
                delta = delta + current_step * velocity

                eps_adaptive = self.eps * masking_curve
                delta = torch.clamp(delta, -eps_adaptive, eps_adaptive)

                adv_waveform = torch.clamp(orig_waveform + delta, -1.0, 1.0)
                delta = adv_waveform - orig_waveform
                delta = delta.detach()

            current_real_prob = probs[0, 0].item()
            if current_real_prob > best_real_prob:
                best_real_prob = current_real_prob
                best_delta = delta.clone()

            if verbose and (i % 50 == 0 or i == self.num_iterations - 1):
                print(f"Iter {i:3d}: Real={current_real_prob:.4f}, step={current_step:.4f}")

            if current_real_prob > 0.7:
                if verbose:
                    print(f"  Early stop at iter {i}!")
                break

        attack_time = time.time() - start_time
        adv_waveform = orig_waveform + best_delta

        # Evaluation
        with torch.no_grad():
            adv_mel = self.mel_transform(adv_waveform)
            logits = self.model(target_video, adv_mel, ref_video, ref_audio)[0]
            probs = F.softmax(logits, dim=1)
            final_real_prob = probs[0, 0].item()

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
            'target_snr_db': self.target_snr_db,
            'snr_weight': self.snr_weight,
            'masking_strength': self.masking_strength,
            'num_iterations': self.num_iterations,
            'momentum': self.momentum,
            'attack_time_seconds': attack_time,
        }

        if verbose:
            print(f"Final: Real={final_real_prob:.4f} (change: {info['confidence_change']:+.4f})")
            print(f"SNR: {snr:.1f} dB | L-inf: {linf_norm:.4f}")

        return adv_waveform.detach(), info


# =============================================================================
# Mel-Space Attack
# =============================================================================

class MelSpacePGDAttack:
    """
    PGD attack directly in mel-spectrogram space.

    Pros: Directly manipulates what the model sees, often more effective.
    Cons: Requires reconstruction (Griffin-Lim), which introduces artifacts.
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
        target_audio: torch.Tensor,
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
                grad_flat = grad.view(1, -1)
                grad_norm = torch.norm(grad_flat, dim=1, keepdim=True) + 1e-10
                grad_normalized = grad / grad_norm.view(-1, 1, 1, 1, 1)

                velocity = 0.9 * velocity + grad_normalized
                delta = delta + self.step_size * velocity

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


# Utilities

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
    """Load the Referee model with pretrained weights."""
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


# Main

def main():
    parser = argparse.ArgumentParser(description="Audio adversarial attacks for Referee")
    parser.add_argument("--method", type=str, default="improved-psychoacoustic",
                        choices=["improved-psychoacoustic", "mel-pgd"],
                        help="Attack method")
    parser.add_argument("--num-samples", type=int, default=5)
    parser.add_argument("--output-dir", type=str, default="./improved-audio-results")
    parser.add_argument("--eps", type=float, default=0.15,
                        help="Perturbation budget (0.05-0.3 for waveform)")
    parser.add_argument("--max-iter", type=int, default=300)
    parser.add_argument("--target-snr", type=float, default=35.0,
                        help="Target SNR in dB (higher = better quality)")
    parser.add_argument("--snr-weight", type=float, default=0.1,
                        help="SNR regularization weight (0-1)")
    parser.add_argument("--masking-strength", type=float, default=0.5,
                        help="Masking curve strength (0-1)")
    parser.add_argument("--preset", type=str, default=None,
                        choices=["quality", "balanced", "effective"],
                        help="Use a preset configuration")
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()

    # Apply presets
    if args.preset == "quality":
        args.eps = 0.08
        args.target_snr = 40.0
        args.snr_weight = 0.3
        args.masking_strength = 0.3
        print("Using QUALITY preset: prioritizes audio quality")
    elif args.preset == "balanced":
        args.eps = 0.05
        args.target_snr = 35.0
        args.snr_weight = 0.02
        args.masking_strength = 0.1
        print("Using BALANCED preset")
    elif args.preset == "effective":
        args.eps = 0.30
        args.target_snr = 25.0
        args.snr_weight = 0.0
        args.masking_strength = 0.8
        print("Using EFFECTIVE preset: prioritizes attack success")

    print("=" * 70)
    print("Audio Adversarial Attacks")
    print("=" * 70)
    print()
    print(f"Method: {args.method}")
    print(f"Eps: {args.eps}, Target SNR: {args.target_snr}dB, SNR weight: {args.snr_weight}")
    print(f"Masking strength: {args.masking_strength}, Max iterations: {args.max_iter}")
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
        target_audio_mel = sample['target_audio'].unsqueeze(0)
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
                original_waveform = extract_audio_from_video(video_path)
                waveform_tensor = torch.from_numpy(original_waveform).float().to(args.device)

                attack = ImprovedPsychoacousticAttack(
                    model=model,
                    eps=args.eps,
                    num_iterations=args.max_iter,
                    target_snr_db=args.target_snr,
                    snr_weight=args.snr_weight,
                    masking_strength=args.masking_strength,
                    device=args.device
                )

                adv_waveform, attack_info = attack.attack(
                    waveform_tensor.unsqueeze(0),
                    target_video, ref_audio, ref_video, labels
                )

                adv_wav = adv_waveform[0].cpu().numpy()
                max_val = np.max(np.abs(adv_wav))
                if max_val > 0:
                    adv_wav = adv_wav / max_val * 0.95
                sf.write(str(sample_out / "adversarial.wav"), adv_wav, 16000)

                orig_wav = original_waveform.copy()
                max_val = np.max(np.abs(orig_wav))
                if max_val > 0:
                    orig_wav = orig_wav / max_val * 0.95
                sf.write(str(sample_out / "original.wav"), orig_wav, 16000)

            elif args.method == "mel-pgd":
                attack = MelSpacePGDAttack(
                    model=model,
                    eps=args.eps if args.eps > 1 else 3.0,
                    num_iterations=args.max_iter,
                    device=args.device
                )

                adv_mel, attack_info = attack.attack(
                    target_audio_mel, target_video, ref_audio, ref_video, labels
                )

                print("  Note: Mel-space attack - no waveform output (would need reconstruction)")

            results.append(attack_info)

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
