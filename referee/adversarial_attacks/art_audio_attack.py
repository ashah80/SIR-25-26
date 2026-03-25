"""
Audio Attack Library Wrapper for Referee Model

This module provides wrappers around popular adversarial audio attack libraries
to enable their use with the Referee multimodal deepfake detector.

Supported backends:
1. ART (Adversarial Robustness Toolbox) - industry standard
2. Custom implementations optimized for mel-spectrogram models

The key challenge: Most audio attack libraries target ASR (speech recognition) models
that take raw waveforms. Referee takes mel-spectrograms. This wrapper handles the
translation between waveform-space attacks and mel-spectrogram-space evaluation.

Usage:
    python art_audio_attack.py --method art-pgd --num-samples 3
    python art_audio_attack.py --method psychoacoustic --num-samples 3
    python art_audio_attack.py --method imperceptible --num-samples 3
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
import warnings

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

try:
    import soundfile as sf
    HAVE_SOUNDFILE = True
except ImportError:
    HAVE_SOUNDFILE = False

# Check for ART availability
try:
    import art
    from art.estimators.classification import PyTorchClassifier
    from art.attacks.evasion import ProjectedGradientDescent
    HAVE_ART = True
    ART_VERSION = art.__version__
except ImportError:
    HAVE_ART = False
    ART_VERSION = None
    print("Warning: ART not installed. Install with: pip install adversarial-robustness-toolbox")


class RefereeAudioWrapper(nn.Module):
    """
    Wrapper that makes Referee model compatible with ART's expected interface.

    ART expects: model(x) -> logits where x is the input
    Referee expects: model(target_vis, target_aud, ref_vis, ref_aud) -> logits

    This wrapper:
    1. Takes waveform input
    2. Converts to mel-spectrogram
    3. Passes through Referee with fixed video/reference inputs
    4. Returns logits for ART to compute gradients
    """

    def __init__(
        self,
        referee_model: nn.Module,
        target_video: torch.Tensor,
        ref_video: torch.Tensor,
        ref_audio: torch.Tensor,
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
        self.referee_model = referee_model
        self.target_video = target_video
        self.ref_video = ref_video
        self.ref_audio = ref_audio
        self.device = device

        # Audio processing parameters
        self.sample_rate = sample_rate
        self.audio_mean = audio_mean
        self.audio_std = audio_std
        self.max_spec_t = max_spec_t
        self.n_segments = n_segments
        self.n_mels = n_mels

        # Segment parameters
        segment_size_vframes = 16
        v_fps = 25
        self.seg_size_aframes = int(segment_size_vframes / v_fps * sample_rate)
        self.stride_aframes = int(0.5 * self.seg_size_aframes)

        # Mel spectrogram transform
        self.mel_spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            n_mels=n_mels
        ).to(device)

    def waveform_to_model_input(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Convert waveform to the mel-spectrogram format expected by Referee.

        Args:
            waveform: (B, T) raw audio

        Returns:
            mel: (B, S, 1, F, Ta) normalized log mel-spectrogram
        """
        B, T = waveform.shape

        # Calculate required length
        required_len = (self.n_segments - 1) * self.stride_aframes + self.seg_size_aframes

        # Pad if necessary
        if T < required_len:
            pad_len = required_len - T
            waveform = F.pad(waveform, (0, pad_len), mode='constant', value=0)
            T = required_len

        # Center crop
        seg_seq_len = self.n_segments * 0.5 + 0.5
        aframes_seg_seq_len = int(seg_seq_len * self.seg_size_aframes)
        max_a_start = max(T - aframes_seg_seq_len, 0)
        a_start = max_a_start // 2

        # Extract segments
        segments = []
        for s in range(self.n_segments):
            start = a_start + s * self.stride_aframes
            end = start + self.seg_size_aframes
            segments.append(waveform[:, start:end])

        # Stack and process
        audio_segments = torch.stack(segments, dim=1)  # (B, S, seg_len)
        B, S, seg_len = audio_segments.shape
        audio_flat = audio_segments.view(B * S, seg_len)

        # Mel spectrogram
        mel = self.mel_spec(audio_flat)
        mel = torch.log(mel + 1e-6)

        # Pad/truncate time
        Ta = mel.shape[-1]
        if Ta < self.max_spec_t:
            mel = F.pad(mel, (0, self.max_spec_t - Ta))
        elif Ta > self.max_spec_t:
            mel = mel[..., :self.max_spec_t]

        # Normalize
        mel = (mel - self.audio_mean) / (2 * self.audio_std)

        # Reshape
        mel = mel.view(B, S, self.n_mels, self.max_spec_t)
        mel = mel.unsqueeze(2)  # (B, S, 1, F, Ta)

        return mel

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: waveform -> mel-spectrogram -> Referee -> logits

        Args:
            waveform: (B, T) input waveform

        Returns:
            logits: (B, 2) classification logits
        """
        # Convert to mel-spectrogram
        mel = self.waveform_to_model_input(waveform)

        # Expand fixed inputs to batch size
        B = waveform.shape[0]
        target_video = self.target_video.expand(B, -1, -1, -1, -1, -1)
        ref_video = self.ref_video.expand(B, -1, -1, -1, -1, -1)
        ref_audio = self.ref_audio.expand(B, -1, -1, -1, -1)

        # Forward through Referee
        logits_rf, _ = self.referee_model(target_video, mel, ref_video, ref_audio)

        return logits_rf


class ARTAudioAttack:
    """
    ART-based audio adversarial attack wrapper.

    Uses ART's ProjectedGradientDescent with the RefereeAudioWrapper
    to generate adversarial audio samples.
    """

    def __init__(
        self,
        model: nn.Module,
        eps: float = 0.03,
        eps_step: float = 0.003,
        max_iter: int = 100,
        targeted: bool = False,
        device: str = 'cuda'
    ):
        if not HAVE_ART:
            raise ImportError("ART not installed. Run: pip install adversarial-robustness-toolbox")

        self.model = model
        self.eps = eps
        self.eps_step = eps_step
        self.max_iter = max_iter
        self.targeted = targeted
        self.device = device

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
        Run ART PGD attack on audio.

        Args:
            original_waveform: (B, T) or (T,) raw audio
            target_video, ref_audio, ref_video: Fixed model inputs
            labels: Ground truth labels

        Returns:
            adv_waveform: Adversarial audio
            info: Attack statistics
        """
        # Ensure batch dimension
        if original_waveform.dim() == 1:
            original_waveform = original_waveform.unsqueeze(0)

        B, T = original_waveform.shape

        # Create wrapper model
        wrapper = RefereeAudioWrapper(
            self.model,
            target_video=target_video,
            ref_video=ref_video,
            ref_audio=ref_audio,
            device=self.device
        ).to(self.device)
        wrapper.eval()

        # Get initial prediction
        with torch.no_grad():
            orig_logits = wrapper(original_waveform)
            orig_probs = F.softmax(orig_logits, dim=1)
            orig_real_prob = orig_probs[0, 0].item()

        if verbose:
            print(f"Initial: Real={orig_real_prob:.4f}")

        # Create ART classifier
        # ART expects numpy arrays and channels-first format
        art_classifier = PyTorchClassifier(
            model=wrapper,
            loss=nn.CrossEntropyLoss(),
            input_shape=(T,),
            nb_classes=2,
            clip_values=(-1.0, 1.0),
            device_type='gpu' if 'cuda' in self.device else 'cpu'
        )

        # Create PGD attack
        attack = ProjectedGradientDescent(
            estimator=art_classifier,
            eps=self.eps,
            eps_step=self.eps_step,
            max_iter=self.max_iter,
            targeted=self.targeted,
            verbose=verbose
        )

        start_time = time.time()

        # Run attack (ART works with numpy)
        x_np = original_waveform.cpu().numpy()
        y_np = labels.cpu().numpy()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            adv_np = attack.generate(x=x_np, y=y_np)

        attack_time = time.time() - start_time

        # Convert back to tensor
        adv_waveform = torch.from_numpy(adv_np).float().to(self.device)

        # Evaluate
        with torch.no_grad():
            adv_logits = wrapper(adv_waveform)
            adv_probs = F.softmax(adv_logits, dim=1)
            final_real_prob = adv_probs[0, 0].item()

        # Compute stats
        delta = adv_waveform - original_waveform
        linf_norm = torch.max(torch.abs(delta)).item()
        l2_norm = torch.norm(delta).item()
        snr = 10 * torch.log10(
            torch.sum(original_waveform ** 2) / (torch.sum(delta ** 2) + 1e-10)
        ).item()

        info = {
            'method': 'ART_PGD',
            'original_real_prob': orig_real_prob,
            'adversarial_real_prob': final_real_prob,
            'confidence_change': final_real_prob - orig_real_prob,
            'attack_success': final_real_prob > 0.5,
            'perturbation_linf': linf_norm,
            'perturbation_l2': l2_norm,
            'perturbation_snr_db': snr,
            'eps': self.eps,
            'num_iterations': self.max_iter,
            'attack_time_seconds': attack_time,
        }

        if verbose:
            print(f"Final: Real={final_real_prob:.4f}")
            print(f"Confidence change: {info['confidence_change']:+.4f}")

        return adv_waveform, info


class PsychoacousticAudioAttack:
    """
    Psychoacoustic-aware audio attack.

    Uses frequency masking principles to make perturbations less audible
    by concentrating them in frequencies where they're masked by the original signal.

    Based on concepts from "Imperceptible, Robust, and Targeted Adversarial Examples
    for Automatic Speech Recognition" (Qin et al., 2019).
    """

    def __init__(
        self,
        model: nn.Module,
        eps: float = 0.05,
        step_size: float = 0.005,
        num_iterations: int = 100,
        masking_threshold: float = 0.1,
        device: str = 'cuda'
    ):
        self.model = model
        self.eps = eps
        self.step_size = step_size
        self.num_iterations = num_iterations
        self.masking_threshold = masking_threshold
        self.device = device

        # Import mel transform from waveform attack
        from adversarial_attacks.waveform_audio_attack import DifferentiableMelSpectrogram
        self.mel_transform = DifferentiableMelSpectrogram().to(device)

    def compute_masking_curve(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Compute a simple frequency masking curve based on the signal's spectrum.

        Frequencies with high energy can mask perturbations better.

        Args:
            waveform: (B, T) audio

        Returns:
            mask: (B, T) masking weights in time domain
        """
        B, T = waveform.shape

        # Compute STFT
        n_fft = 1024
        hop_length = 256

        # Windowed FFT
        window = torch.hann_window(n_fft, device=waveform.device)

        # Manual STFT (for gradient flow)
        spec = torch.stft(
            waveform, n_fft=n_fft, hop_length=hop_length,
            window=window, return_complex=True
        )
        magnitude = torch.abs(spec)  # (B, n_fft//2+1, T_frames)

        # Compute masking threshold per frame
        # Higher energy = higher masking = can hide more perturbation
        frame_energy = torch.sum(magnitude ** 2, dim=1)  # (B, T_frames)
        frame_energy = frame_energy / (frame_energy.max() + 1e-10)

        # Map back to time domain (upsample)
        mask = F.interpolate(
            frame_energy.unsqueeze(1),
            size=T,
            mode='linear',
            align_corners=False
        ).squeeze(1)

        # Scale: sqrt to make masking less aggressive
        mask = torch.sqrt(mask)

        # Ensure minimum masking
        mask = torch.clamp(mask, min=self.masking_threshold)

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
        Run psychoacoustic attack.
        """
        self.model.eval()

        if original_waveform.dim() == 1:
            original_waveform = original_waveform.unsqueeze(0)

        B, T = original_waveform.shape
        orig_waveform = original_waveform.clone().detach()

        # Compute masking curve
        with torch.no_grad():
            masking_curve = self.compute_masking_curve(orig_waveform)

        # Initialize perturbation
        delta = torch.zeros_like(orig_waveform, requires_grad=True)

        # Get initial prediction
        with torch.no_grad():
            orig_mel = self.mel_transform(orig_waveform)
            logits = self.model(target_video, orig_mel, ref_video, ref_audio)[0]
            probs = F.softmax(logits, dim=1)
            orig_real_prob = probs[0, 0].item()

        if verbose:
            print(f"Initial: Real={orig_real_prob:.4f}")

        best_delta = delta.clone()
        best_real_prob = orig_real_prob

        start_time = time.time()

        for i in range(self.num_iterations):
            delta.requires_grad_(True)

            # Apply perturbation scaled by masking
            adv_waveform = orig_waveform + delta

            # Forward
            adv_mel = self.mel_transform(adv_waveform)
            logits = self.model(target_video, adv_mel, ref_video, ref_audio)[0]
            probs = F.softmax(logits, dim=1)

            # Loss
            loss = F.cross_entropy(logits, labels)

            # Backward
            self.model.zero_grad()
            loss.backward()

            grad = delta.grad.detach()

            # Update with masking-aware step
            with torch.no_grad():
                # Scale gradient by masking curve (perturb more where masking is higher)
                grad_scaled = grad * masking_curve

                # Normalize
                grad_norm = torch.norm(grad_scaled, dim=1, keepdim=True) + 1e-10
                grad_normalized = grad_scaled / grad_norm

                # Update
                delta = delta + self.step_size * grad_normalized

                # Project with masking-aware epsilon
                # Allow larger perturbations where masking is higher
                eps_per_sample = self.eps * masking_curve
                delta = torch.clamp(delta, -eps_per_sample, eps_per_sample)

                # Valid range
                adv_waveform = torch.clamp(orig_waveform + delta, -1.0, 1.0)
                delta = adv_waveform - orig_waveform
                delta = delta.detach()

            # Track best
            current_real_prob = probs[0, 0].item()
            if current_real_prob > best_real_prob:
                best_real_prob = current_real_prob
                best_delta = delta.clone()

            if verbose and (i % 20 == 0 or i == self.num_iterations - 1):
                print(f"Iter {i:3d}: Real={current_real_prob:.4f}")

            if current_real_prob > 0.95:
                if verbose:
                    print(f"  Early stop at iter {i}!")
                break

        attack_time = time.time() - start_time

        adv_waveform = orig_waveform + best_delta

        # Evaluate
        with torch.no_grad():
            adv_mel = self.mel_transform(adv_waveform)
            logits = self.model(target_video, adv_mel, ref_video, ref_audio)[0]
            probs = F.softmax(logits, dim=1)
            final_real_prob = probs[0, 0].item()

        # Stats
        delta_final = adv_waveform - orig_waveform
        linf_norm = torch.max(torch.abs(delta_final)).item()
        l2_norm = torch.norm(delta_final).item()
        snr = 10 * torch.log10(
            torch.sum(orig_waveform ** 2) / (torch.sum(delta_final ** 2) + 1e-10)
        ).item()

        info = {
            'method': 'Psychoacoustic',
            'original_real_prob': orig_real_prob,
            'adversarial_real_prob': final_real_prob,
            'confidence_change': final_real_prob - orig_real_prob,
            'attack_success': final_real_prob > 0.5,
            'perturbation_linf': linf_norm,
            'perturbation_l2': l2_norm,
            'perturbation_snr_db': snr,
            'eps': self.eps,
            'num_iterations': self.num_iterations,
            'attack_time_seconds': attack_time,
        }

        if verbose:
            print(f"Final: Real={final_real_prob:.4f}")
            print(f"Confidence change: {info['confidence_change']:+.4f}")
            print(f"SNR: {snr:.1f} dB")

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
    parser = argparse.ArgumentParser(description="Audio attack library wrapper")
    parser.add_argument("--method", type=str, default="psychoacoustic",
                        choices=["art-pgd", "psychoacoustic", "all"],
                        help="Attack method to use")
    parser.add_argument("--num-samples", type=int, default=3)
    parser.add_argument("--output-dir", type=str, default="./art-audio-results")
    parser.add_argument("--eps", type=float, default=0.2)
    parser.add_argument("--max-iter", type=int, default=200)
    parser.add_argument("--device", type=str, default="cuda")

    print("=" * 70)
    print("Audio Attack Library Wrapper")
    print("=" * 70)
    print()
    print(f"Method: {args.method}")
    if args.method == "art-pgd" and not HAVE_ART:
        print("ERROR: ART not installed. Run: pip install adversarial-robustness-toolbox")
        return
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
        ref_audio = sample['reference_audio'].unsqueeze(0)
        ref_video = sample['reference_video'].unsqueeze(0)
        labels = sample['fake_label'].unsqueeze(0)
        info = sample['sample_info']

        video_path = info.get('target_path')
        if not video_path or not Path(video_path).exists():
            print(f"  Skipping - video not found")
            continue

        try:
            original_waveform = extract_audio_from_video(video_path)
            waveform_tensor = torch.from_numpy(original_waveform).float().to(args.device)

            # Run selected attack(s)
            methods_to_run = []
            if args.method == "all":
                methods_to_run = ["psychoacoustic"]
                if HAVE_ART:
                    methods_to_run.append("art-pgd")
            else:
                methods_to_run = [args.method]

            for method in methods_to_run:
                print(f"\n  Running {method}...")

                if method == "art-pgd":
                    attack = ARTAudioAttack(
                        model=model,
                        eps=args.eps,
                        max_iter=args.max_iter,
                        device=args.device
                    )
                elif method == "psychoacoustic":
                    attack = PsychoacousticAudioAttack(
                        model=model,
                        eps=args.eps,
                        num_iterations=args.max_iter,
                        device=args.device
                    )

                adv_waveform, attack_info = attack.attack(
                    waveform_tensor.unsqueeze(0) if waveform_tensor.dim() == 1 else waveform_tensor,
                    target_video, ref_audio, ref_video, labels
                )

                results.append(attack_info)

                # Save
                adv_wav = adv_waveform[0].cpu().numpy()
                max_val = np.max(np.abs(adv_wav))
                if max_val > 0:
                    adv_wav = adv_wav / max_val * 0.95
                sf.write(str(sample_out / f"adversarial_{method}.wav"), adv_wav, 16000)
                print(f"  Saved: adversarial_{method}.wav")

            # Save original
            orig_wav = original_waveform.copy()
            max_val = np.max(np.abs(orig_wav))
            if max_val > 0:
                orig_wav = orig_wav / max_val * 0.95
            sf.write(str(sample_out / "original.wav"), orig_wav, 16000)

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
        for method in set(r['method'] for r in valid):
            method_results = [r for r in valid if r['method'] == method]
            avg_change = np.mean([r['confidence_change'] for r in method_results])
            successes = sum(1 for r in method_results if r.get('attack_success', False))
            avg_snr = np.mean([r.get('perturbation_snr_db', 0) for r in method_results])

            print(f"\n{method}:")
            print(f"  Samples: {len(method_results)}")
            print(f"  Successes: {successes}/{len(method_results)}")
            print(f"  Avg confidence change: {avg_change:+.4f}")
            print(f"  Avg SNR: {avg_snr:.1f} dB")

    print()
    print(f"Outputs saved to: {output_path.absolute()}")


if __name__ == "__main__":
    main()
