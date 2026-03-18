"""
Real Data Loader for Adversarial Attacks

This module provides data loading functionality that uses real samples from
the FakeAVCeleb dataset, matching exactly how Referee loads and preprocesses data.

Usage:
    from adversarial_attacks.real_data_loader import load_real_sample, load_test_dataset

    # Load a single sample
    target_audio, target_video, ref_audio, ref_video, labels_rf, sample_info = load_real_sample()

    # Or load multiple samples
    dataset = load_test_dataset(num_samples=10)
"""

import sys
from pathlib import Path
import json
import torch
import torchaudio
import random
from typing import Dict, Any, Tuple, Optional, List

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

from dataset.dataset_utils import get_video_and_audio


class RefereeTestTransform:
    """
    Applies the exact same transforms as Referee's test transform sequence.

    Converts raw video (Tv, C, H, W) and audio (Ta,) to model-ready format:
    - Video: (S, T, C, H, W) = (8 segments, 16 frames, 3 channels, 224, 224)
    - Audio: (S, 1, F, T) = (8 segments, 1 channel, 128 mel bins, 66 time frames)
    """

    def __init__(
        self,
        input_size: int = 224,
        segment_size_vframes: int = 16,
        n_segments: int = 8,
        step_size_seg: float = 0.5,
        sample_rate: int = 16000,
        n_mels: int = 128,
        max_spec_t: int = 66,
        audio_mean: float = -4.2677393,
        audio_std: float = 4.5689974,
        video_mean: Tuple[float, float, float] = (0.5, 0.5, 0.5),
        video_std: Tuple[float, float, float] = (0.5, 0.5, 0.5),
    ):
        self.input_size = input_size
        self.segment_size_vframes = segment_size_vframes
        self.n_segments = n_segments
        self.step_size_seg = step_size_seg
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.max_spec_t = max_spec_t
        self.audio_mean = audio_mean
        self.audio_std = audio_std
        self.video_mean = torch.tensor(video_mean).view(1, 3, 1, 1)
        self.video_std = torch.tensor(video_std).view(1, 3, 1, 1)

        # Mel spectrogram transform
        self.mel_spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            win_length=400,
            hop_length=160,
            n_fft=1024,
            n_mels=n_mels
        )

    def center_crop_video(self, video: torch.Tensor) -> torch.Tensor:
        """Center crop video to input_size x input_size."""
        # video: (Tv, C, H, W)
        h, w = video.shape[-2:]
        th, tw = self.input_size, self.input_size

        i = int(round((h - th) / 2.))
        j = int(round((w - tw) / 2.))

        return video[..., i:(i + th), j:(j + tw)]

    def generate_segments(self, video: torch.Tensor, audio: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate n_segments from video and audio.

        Args:
            video: (Tv, C, H, W)
            audio: (Ta,)

        Returns:
            video: (S, T, C, H, W)
            audio: (S, Ta_seg)
        """
        v_len_frames, C, H, W = video.shape
        a_len_frames = audio.shape[0]

        v_fps = 25
        a_fps = self.sample_rate

        # Segment sizes
        seg_size_vframes = self.segment_size_vframes
        seg_size_aframes = int(seg_size_vframes / v_fps * a_fps)

        # Step sizes (stride)
        stride_vframes = int(self.step_size_seg * seg_size_vframes)
        stride_aframes = int(self.step_size_seg * seg_size_aframes)

        # Calculate required length
        required_v_len = (self.n_segments - 1) * stride_vframes + seg_size_vframes
        required_a_len = (self.n_segments - 1) * stride_aframes + seg_size_aframes

        # Pad if necessary
        if v_len_frames < required_v_len:
            pad_v = required_v_len - v_len_frames
            pad_tensor = torch.zeros(pad_v, C, H, W, dtype=video.dtype)
            video = torch.cat([video, pad_tensor], dim=0)
            v_len_frames = required_v_len

        if a_len_frames < required_a_len:
            pad_a = required_a_len - a_len_frames
            pad_tensor = torch.zeros(pad_a, dtype=audio.dtype)
            audio = torch.cat([audio, pad_tensor], dim=0)
            a_len_frames = required_a_len

        # Calculate starting positions (center crop temporally)
        seg_seq_len = self.n_segments * self.step_size_seg + (1 - self.step_size_seg)
        vframes_seg_seq_len = int(seg_seq_len * seg_size_vframes)
        aframes_seg_seq_len = int(seg_seq_len * seg_size_aframes)

        max_v_start = v_len_frames - vframes_seg_seq_len
        v_start = max(0, max_v_start // 2)
        a_start = int(v_start / v_fps * a_fps)

        max_a_start = max(a_len_frames - aframes_seg_seq_len, 0)
        a_start = min(a_start, max_a_start)

        # Create segment ranges
        v_starts = [v_start + i * stride_vframes for i in range(self.n_segments)]
        a_starts = [a_start + i * stride_aframes for i in range(self.n_segments)]

        # Extract segments
        video_segments = []
        audio_segments = []

        for vs, as_ in zip(v_starts, a_starts):
            video_segments.append(video[vs:vs + seg_size_vframes])
            audio_segments.append(audio[as_:as_ + seg_size_aframes])

        video = torch.stack(video_segments, dim=0)  # (S, T, C, H, W)
        audio = torch.stack(audio_segments, dim=0)  # (S, Ta_seg)

        return video, audio

    def process_audio(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Convert audio waveform to normalized log mel spectrogram.

        Args:
            audio: (S, Ta_seg) waveform segments

        Returns:
            audio: (S, 1, F, T) normalized log mel spectrogram
        """
        # Apply mel spectrogram: (S, Ta_seg) -> (S, F, T)
        mel = self.mel_spec(audio)

        # Log transform
        mel = torch.log(mel + 1e-6)

        # Pad or truncate to max_spec_t
        T = mel.shape[-1]
        if T < self.max_spec_t:
            pad_size = self.max_spec_t - T
            mel = torch.nn.functional.pad(mel, (0, pad_size), mode='constant', value=0)
        elif T > self.max_spec_t:
            mel = mel[..., :self.max_spec_t]

        # Normalize (AST normalization)
        mel = (mel - self.audio_mean) / (2 * self.audio_std)

        # Add channel dimension: (S, F, T) -> (S, 1, F, T)
        mel = mel.unsqueeze(1)

        return mel

    def process_video(self, video: torch.Tensor) -> torch.Tensor:
        """
        Normalize video to [0, 1] and then to mean/std.

        Args:
            video: (S, T, C, H, W) uint8 [0, 255]

        Returns:
            video: (S, T, C, H, W) float normalized
        """
        # Convert to float and [0, 1]
        video = video.float() / 255.0

        # Normalize with mean and std
        # Shape: (S, T, C, H, W), mean/std: (1, 1, C, 1, 1)
        mean = self.video_mean.view(1, 1, 3, 1, 1)
        std = self.video_std.view(1, 1, 3, 1, 1)
        video = (video - mean) / std

        return video

    def __call__(self, video: torch.Tensor, audio: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply full transform pipeline.

        Args:
            video: (Tv, C, H, W) raw video frames
            audio: (Ta,) raw audio waveform

        Returns:
            video: (S, T, C, H, W) processed video
            audio: (S, 1, F, T) processed audio
        """
        # Center crop video spatially
        video = self.center_crop_video(video)

        # Generate segments
        video, audio = self.generate_segments(video, audio)

        # Process audio to mel spectrogram
        audio = self.process_audio(audio)

        # Process video normalization
        video = self.process_video(video)

        return video, audio


class AdversarialTestDataset:
    """
    Dataset for loading real test samples for adversarial attacks.

    Uses the test_pairs_fixed.json file which contains target-reference pairs
    with labels.
    """

    def __init__(
        self,
        json_path: Optional[str] = None,
        transform: Optional[RefereeTestTransform] = None,
        device: str = 'cuda',
        max_samples: Optional[int] = None,
    ):
        """
        Args:
            json_path: Path to test_pairs JSON file. If None, uses default.
            transform: Transform to apply. If None, creates default.
            device: Device to load tensors to.
            max_samples: Maximum number of samples to load (for testing).
        """
        if json_path is None:
            json_path = PROJECT_ROOT / "data" / "test_pairs_fixed.json"

        with open(json_path, "r") as f:
            loaded = json.load(f)
            if isinstance(loaded, dict) and "data" in loaded:
                self.samples = loaded["data"]
            else:
                self.samples = loaded

        if max_samples is not None:
            self.samples = self.samples[:max_samples]

        self.transform = transform if transform is not None else RefereeTestTransform()
        self.device = device

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Load and transform a single sample.

        Returns:
            Dictionary with:
                - target_video: (S, T, C, H, W) tensor
                - target_audio: (S, 1, F, T) tensor
                - reference_video: (S, T, C, H, W) tensor
                - reference_audio: (S, 1, F, T) tensor
                - fake_label: tensor (1 for fake, 0 for real)
                - sample_info: dict with file paths and metadata
        """
        item = self.samples[idx]

        target_path = item["target_file"]
        ref_path = item["reference_file"]

        # Load raw data
        tgt_video, tgt_audio, _ = get_video_and_audio(target_path, get_meta=True)
        ref_video, ref_audio, _ = get_video_and_audio(ref_path, get_meta=True)

        # Apply transforms
        tgt_video, tgt_audio = self.transform(tgt_video, tgt_audio)
        ref_video, ref_audio = self.transform(ref_video, ref_audio)

        # Get labels
        fake_label = int(item.get("fake_label", 0))
        same_identity = int(item.get("same_identity", 0))

        return {
            "target_video": tgt_video.to(self.device),
            "target_audio": tgt_audio.to(self.device),
            "reference_video": ref_video.to(self.device),
            "reference_audio": ref_audio.to(self.device),
            "fake_label": torch.tensor(fake_label, dtype=torch.long, device=self.device),
            "id_label": torch.tensor(same_identity, dtype=torch.long, device=self.device),
            "sample_info": {
                "target_path": target_path,
                "reference_path": ref_path,
                "fake_label": fake_label,
                "same_identity": same_identity,
                "modify_type": item.get("modify_type", "unknown"),
                "idx": idx,
            }
        }

    def get_random_sample(self) -> Dict[str, Any]:
        """Get a random sample from the dataset."""
        idx = random.randint(0, len(self) - 1)
        return self[idx]

    def get_fake_sample(self) -> Dict[str, Any]:
        """Get a random fake sample (fake_label=1)."""
        fake_indices = [i for i, s in enumerate(self.samples) if s.get("fake_label", 0) == 1]
        if not fake_indices:
            raise ValueError("No fake samples found in dataset")
        idx = random.choice(fake_indices)
        return self[idx]

    def get_real_sample(self) -> Dict[str, Any]:
        """Get a random real sample (fake_label=0)."""
        real_indices = [i for i, s in enumerate(self.samples) if s.get("fake_label", 0) == 0]
        if not real_indices:
            raise ValueError("No real samples found in dataset")
        idx = random.choice(real_indices)
        return self[idx]


def load_real_sample(
    device: str = 'cuda',
    sample_type: str = 'fake',
    json_path: Optional[str] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
    """
    Convenience function to load a single real sample.

    Args:
        device: Device to load tensors to.
        sample_type: 'fake', 'real', or 'random'.
        json_path: Optional path to JSON file.

    Returns:
        target_audio: (1, S, 1, F, T) tensor
        target_video: (1, S, T, C, H, W) tensor
        ref_audio: (1, S, 1, F, T) tensor
        ref_video: (1, S, T, C, H, W) tensor
        labels_rf: (1,) tensor
        sample_info: dict with metadata
    """
    dataset = AdversarialTestDataset(json_path=json_path, device=device)

    if sample_type == 'fake':
        sample = dataset.get_fake_sample()
    elif sample_type == 'real':
        sample = dataset.get_real_sample()
    else:
        sample = dataset.get_random_sample()

    # Add batch dimension
    target_audio = sample['target_audio'].unsqueeze(0)
    target_video = sample['target_video'].unsqueeze(0)
    ref_audio = sample['reference_audio'].unsqueeze(0)
    ref_video = sample['reference_video'].unsqueeze(0)
    labels_rf = sample['fake_label'].unsqueeze(0)

    return target_audio, target_video, ref_audio, ref_video, labels_rf, sample['sample_info']


def load_test_dataset(
    num_samples: Optional[int] = None,
    device: str = 'cuda',
    json_path: Optional[str] = None,
) -> AdversarialTestDataset:
    """
    Load the test dataset.

    Args:
        num_samples: Maximum number of samples to include.
        device: Device to load tensors to.
        json_path: Optional path to JSON file.

    Returns:
        AdversarialTestDataset instance.
    """
    return AdversarialTestDataset(
        json_path=json_path,
        device=device,
        max_samples=num_samples,
    )


if __name__ == "__main__":
    # Test the data loader
    print("Testing real data loader...")

    try:
        dataset = AdversarialTestDataset(device='cpu', max_samples=5)
        print(f"Dataset size: {len(dataset)}")

        sample = dataset[0]
        print(f"\nSample shapes:")
        print(f"  Target video: {sample['target_video'].shape}")
        print(f"  Target audio: {sample['target_audio'].shape}")
        print(f"  Reference video: {sample['reference_video'].shape}")
        print(f"  Reference audio: {sample['reference_audio'].shape}")
        print(f"  Fake label: {sample['fake_label']}")
        print(f"\nSample info:")
        for k, v in sample['sample_info'].items():
            print(f"  {k}: {v}")

    except Exception as e:
        print(f"Error loading data: {e}")
        print("This is expected if running on a machine without the dataset.")
        print("The loader will work correctly on the target server.")
