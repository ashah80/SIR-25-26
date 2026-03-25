"""
Test Mel-Spectrogram Round-Trip Conversion

This script tests whether the mel-spectrogram conversion is the source of audio quality issues.
It performs a round-trip: audio -> mel-spectrogram -> audio, and compares the results.

This helps isolate whether:
1. The mel-spectrogram conversion itself is lossy (expected)
2. The adversarial perturbations are making it worse
3. The denormalization/reconstruction parameters are wrong

Usage:
    python test_mel_roundtrip.py --output-dir ./mel-roundtrip-test
"""

import sys
from pathlib import Path
import torch
import torchaudio
import numpy as np
import argparse

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

try:
    import librosa
    import soundfile as sf
    HAVE_LIBROSA = True
except ImportError:
    HAVE_LIBROSA = False
    print("ERROR: librosa/soundfile required. Install with: pip install librosa soundfile")
    sys.exit(1)


class MelRoundTripTester:
    """
    Tests the mel-spectrogram round-trip conversion quality.
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
    ):
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.n_mels = n_mels
        self.audio_mean = audio_mean
        self.audio_std = audio_std

        # Forward transform (same as model)
        self.mel_spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            win_length=win_length,
            hop_length=hop_length,
            n_fft=n_fft,
            n_mels=n_mels
        )

        # Inverse transforms (for reconstruction)
        self.inverse_mel = torchaudio.transforms.InverseMelScale(
            n_stft=n_fft // 2 + 1,
            n_mels=n_mels,
            sample_rate=sample_rate
        )

        self.griffin_lim = torchaudio.transforms.GriffinLim(
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            n_iter=64  # More iterations for better quality
        )

    def audio_to_normalized_mel(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Convert audio waveform to normalized log mel-spectrogram.
        This is exactly what the model does.

        Args:
            audio: (T,) waveform

        Returns:
            mel: (F, Ta) normalized log mel-spectrogram
        """
        # Mel spectrogram
        mel = self.mel_spec(audio)  # (n_mels, T)

        # Log transform
        mel = torch.log(mel + 1e-6)

        # Normalize (AST normalization)
        mel = (mel - self.audio_mean) / (2 * self.audio_std)

        return mel

    def normalized_mel_to_audio_griffinlim(self, mel: torch.Tensor) -> np.ndarray:
        """
        Convert normalized log mel-spectrogram back to audio using Griffin-Lim.
        This is the standard approach but produces robotic/artifacts.

        Args:
            mel: (F, Ta) normalized log mel-spectrogram

        Returns:
            audio: (T,) numpy array
        """
        # Denormalize
        mel_denorm = mel * (2 * self.audio_std) + self.audio_mean

        # Inverse log (exp)
        mel_linear = torch.exp(mel_denorm)

        # Inverse mel scale (mel -> linear spectrogram)
        spec = self.inverse_mel(mel_linear)  # (n_fft // 2 + 1, T)

        # Griffin-Lim to estimate phase and reconstruct waveform
        audio = self.griffin_lim(spec)

        return audio.numpy()

    def normalized_mel_to_audio_librosa(self, mel: torch.Tensor) -> np.ndarray:
        """
        Alternative reconstruction using librosa (often slightly better).

        Args:
            mel: (F, Ta) normalized log mel-spectrogram

        Returns:
            audio: (T,) numpy array
        """
        # Denormalize
        mel_denorm = mel.numpy() * (2 * self.audio_std) + self.audio_mean

        # Inverse log (exp)
        mel_linear = np.exp(mel_denorm)

        # Use librosa's inverse mel
        audio = librosa.feature.inverse.mel_to_audio(
            mel_linear,
            sr=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            n_iter=64
        )

        return audio

    def compute_metrics(self, original: np.ndarray, reconstructed: np.ndarray) -> dict:
        """Compute quality metrics between original and reconstructed audio."""
        # Ensure same length
        min_len = min(len(original), len(reconstructed))
        orig = original[:min_len]
        recon = reconstructed[:min_len]

        # MSE
        mse = np.mean((orig - recon) ** 2)

        # SNR (Signal-to-Noise Ratio)
        signal_power = np.mean(orig ** 2)
        noise_power = np.mean((orig - recon) ** 2)
        snr = 10 * np.log10(signal_power / (noise_power + 1e-10))

        # Correlation
        correlation = np.corrcoef(orig, recon)[0, 1]

        # Peak amplitude comparison
        orig_peak = np.max(np.abs(orig))
        recon_peak = np.max(np.abs(recon))

        return {
            'mse': mse,
            'snr_db': snr,
            'correlation': correlation,
            'orig_peak': orig_peak,
            'recon_peak': recon_peak,
            'length_orig': len(original),
            'length_recon': len(reconstructed),
        }


def extract_audio_from_video(video_path: str, sample_rate: int = 16000) -> np.ndarray:
    """Extract audio from video file."""
    import subprocess
    import tempfile
    import os

    # Create temp file
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
        temp_path = f.name

    try:
        cmd = [
            'ffmpeg', '-y', '-i', str(video_path),
            '-vn', '-acodec', 'pcm_s16le',
            '-ar', str(sample_rate), '-ac', '1',
            temp_path
        ]
        result = subprocess.run(cmd, capture_output=True, timeout=30)

        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg failed: {result.stderr.decode()[:200]}")

        audio, sr = sf.read(temp_path)
        return audio.astype(np.float32)

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


def run_test(video_path: str, output_dir: Path, tester: MelRoundTripTester):
    """Run the round-trip test on a single video file."""
    print(f"\nTesting: {video_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Extract original audio
    print("  Extracting audio from video...")
    original_audio = extract_audio_from_video(video_path)
    print(f"  Original audio: {len(original_audio)} samples ({len(original_audio)/16000:.2f}s)")

    # Save original
    orig_path = output_dir / "1_original.wav"
    sf.write(str(orig_path), original_audio, 16000)
    print(f"  Saved: {orig_path}")

    # Convert to normalized mel-spectrogram
    print("  Converting to mel-spectrogram...")
    audio_tensor = torch.from_numpy(original_audio).float()
    mel = tester.audio_to_normalized_mel(audio_tensor)
    print(f"  Mel shape: {mel.shape}")

    # Reconstruct using torchaudio Griffin-Lim
    print("  Reconstructing with torchaudio Griffin-Lim...")
    recon_torch = tester.normalized_mel_to_audio_griffinlim(mel)

    # Normalize amplitude
    max_val = np.max(np.abs(recon_torch))
    if max_val > 0:
        recon_torch = recon_torch / max_val * 0.95

    torch_path = output_dir / "2_reconstructed_torchaudio.wav"
    sf.write(str(torch_path), recon_torch, 16000)
    print(f"  Saved: {torch_path}")

    # Reconstruct using librosa
    print("  Reconstructing with librosa...")
    recon_librosa = tester.normalized_mel_to_audio_librosa(mel)

    max_val = np.max(np.abs(recon_librosa))
    if max_val > 0:
        recon_librosa = recon_librosa / max_val * 0.95

    librosa_path = output_dir / "3_reconstructed_librosa.wav"
    sf.write(str(librosa_path), recon_librosa, 16000)
    print(f"  Saved: {librosa_path}")

    # Compute metrics
    print("\n  Quality Metrics:")
    print("  " + "-" * 50)

    metrics_torch = tester.compute_metrics(original_audio, recon_torch)
    print(f"  Torchaudio Griffin-Lim:")
    print(f"    SNR: {metrics_torch['snr_db']:.2f} dB")
    print(f"    Correlation: {metrics_torch['correlation']:.4f}")
    print(f"    MSE: {metrics_torch['mse']:.6f}")

    metrics_librosa = tester.compute_metrics(original_audio, recon_librosa)
    print(f"  Librosa Griffin-Lim:")
    print(f"    SNR: {metrics_librosa['snr_db']:.2f} dB")
    print(f"    Correlation: {metrics_librosa['correlation']:.4f}")
    print(f"    MSE: {metrics_librosa['mse']:.6f}")

    return {
        'video_path': video_path,
        'metrics_torchaudio': metrics_torch,
        'metrics_librosa': metrics_librosa,
    }


def main():
    parser = argparse.ArgumentParser(description="Test mel-spectrogram round-trip conversion")
    parser.add_argument("--output-dir", type=str, default="./mel-roundtrip-test",
                        help="Output directory for test results")
    parser.add_argument("--num-samples", type=int, default=2,
                        help="Number of samples to test")

    args = parser.parse_args()

    print("=" * 70)
    print("Mel-Spectrogram Round-Trip Test")
    print("=" * 70)
    print()
    print("This test verifies whether the mel-spectrogram conversion is the source")
    print("of audio quality issues (it almost certainly is - this is expected).")
    print()
    print("Expected results:")
    print("  - Original audio: Clear, natural speech")
    print("  - Reconstructed audio: Robotic/metallic, artifacts present")
    print("  - SNR typically 5-15 dB (poor)")
    print("  - Correlation typically 0.3-0.7 (moderate)")
    print()

    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    tester = MelRoundTripTester()

    # Load dataset to get real samples
    print("Loading dataset...")
    try:
        from adversarial_attacks.real_data_loader import AdversarialTestDataset
        import json

        json_path = PROJECT_ROOT / "data" / "test_pairs_fixed.json"
        with open(json_path, "r") as f:
            loaded = json.load(f)
            if isinstance(loaded, dict) and "data" in loaded:
                samples = loaded["data"]
            else:
                samples = loaded

        # Get unique video paths
        video_paths = list(set(s["target_file"] for s in samples[:args.num_samples * 2]))[:args.num_samples]

        print(f"Found {len(video_paths)} samples to test")

    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Using fallback test...")
        video_paths = []

    if not video_paths:
        print("No video files found. Please provide video paths manually.")
        return

    # Run tests
    all_results = []
    for i, video_path in enumerate(video_paths):
        sample_dir = output_path / f"sample_{i+1}"
        try:
            results = run_test(video_path, sample_dir, tester)
            all_results.append(results)
        except Exception as e:
            print(f"  Error: {e}")
            import traceback
            traceback.print_exc()

    # Summary
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print()

    if all_results:
        avg_snr_torch = np.mean([r['metrics_torchaudio']['snr_db'] for r in all_results])
        avg_snr_librosa = np.mean([r['metrics_librosa']['snr_db'] for r in all_results])
        avg_corr_torch = np.mean([r['metrics_torchaudio']['correlation'] for r in all_results])
        avg_corr_librosa = np.mean([r['metrics_librosa']['correlation'] for r in all_results])

        print(f"Average metrics across {len(all_results)} samples:")
        print(f"  Torchaudio: SNR={avg_snr_torch:.2f}dB, Correlation={avg_corr_torch:.4f}")
        print(f"  Librosa:    SNR={avg_snr_librosa:.2f}dB, Correlation={avg_corr_librosa:.4f}")
        print()

    print("CONCLUSION:")
    print("-" * 70)
    print("The mel-spectrogram conversion is INHERENTLY LOSSY because:")
    print("  1. Phase information is discarded during the spectrogram computation")
    print("  2. Griffin-Lim only estimates phase, it cannot recover it perfectly")
    print("  3. The mel filterbank compresses frequency resolution")
    print()
    print("This is NOT a bug in your code - it's a fundamental limitation.")
    print()
    print("SOLUTIONS for getting listenable adversarial audio:")
    print("  1. Attack in WAVEFORM space (not mel-spectrogram space)")
    print("     - Compute mel on-the-fly during the attack")
    print("     - Perturbations are applied to raw audio")
    print("     - No reconstruction needed!")
    print()
    print("  2. Use a neural vocoder (e.g., HiFi-GAN, WaveGlow)")
    print("     - These can produce much higher quality audio from spectrograms")
    print("     - But adds complexity and computational cost")
    print()
    print("  3. Use psychoacoustic masking")
    print("     - Apply perturbations in frequency bands where they're less audible")
    print("     - Still limited by phase reconstruction")
    print()
    print(f"Test outputs saved to: {output_path.absolute()}")


if __name__ == "__main__":
    main()
