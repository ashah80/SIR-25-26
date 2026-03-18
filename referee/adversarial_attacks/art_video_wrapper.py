"""
ART-compatible wrapper for video-only attacks on Referee model.

NOTE: The main implementation is in test_art_video_attack.py
This file provides the wrapper class for potential reuse.

Usage:
    # See test_art_video_attack.py for full usage example
    from art_video_wrapper import ARTRefereeVideoWrapper, create_art_classifier
"""
import torch
import torch.nn as nn
import numpy as np
from typing import Tuple


class ARTRefereeVideoWrapper(nn.Module):
    """
    Wrapper to make Referee model compatible with ART for video-only attacks.

    This wrapper:
    - Takes only video input (flattened by ART)
    - Holds target_audio, ref_video, ref_audio fixed
    - Returns real/fake logits for ART to optimize against
    """

    def __init__(self, referee_model: nn.Module,
                 target_audio: torch.Tensor,
                 ref_video: torch.Tensor,
                 ref_audio: torch.Tensor,
                 device: str = 'cuda'):
        super().__init__()
        self.referee = referee_model
        self.device = device

        # Store fixed inputs (these won't be attacked)
        self.register_buffer('target_audio', target_audio.to(device))
        self.register_buffer('ref_video', ref_video.to(device))
        self.register_buffer('ref_audio', ref_audio.to(device))

        # Video shape for reshaping: (S=8, T=16, C=3, H=224, W=224)
        self.video_shape = (8, 16, 3, 224, 224)
        self.flattened_size = int(np.prod(self.video_shape))

        print(f"ARTRefereeVideoWrapper initialized:")
        print(f"  Video shape: {self.video_shape}")
        print(f"  Flattened size: {self.flattened_size:,} elements")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for ART compatibility.

        Args:
            x: Flattened video tensor from ART (B, flattened_size)
               Normalized to [-1, 1] range

        Returns:
            logits: Real/fake classification logits (B, 2)
        """
        batch_size = x.shape[0]

        # Reshape flattened input back to video tensor
        video = x.view(batch_size, *self.video_shape)

        # Expand fixed inputs to match batch size
        if batch_size != 1:
            target_audio = self.target_audio.expand(batch_size, -1, -1, -1, -1)
            ref_video = self.ref_video.expand(batch_size, -1, -1, -1, -1, -1)
            ref_audio = self.ref_audio.expand(batch_size, -1, -1, -1, -1)
        else:
            target_audio = self.target_audio
            ref_video = self.ref_video
            ref_audio = self.ref_audio

        # Call Referee model
        logits_rf, _ = self.referee(
            target_vis=video,
            target_aud=target_audio,
            ref_vis=ref_video,
            ref_aud=ref_audio
        )

        return logits_rf

    @property
    def input_shape(self) -> Tuple[int, ...]:
        """Input shape expected by ART (flattened video)."""
        return (self.flattened_size,)

    @property
    def nb_classes(self) -> int:
        """Number of classes for ART."""
        return 2


def create_art_classifier(referee_model: nn.Module,
                         target_audio: torch.Tensor,
                         ref_video: torch.Tensor,
                         ref_audio: torch.Tensor,
                         device: str = 'cuda'):
    """
    Create ART PyTorchClassifier for video-only attacks on Referee.

    Args:
        referee_model: Trained Referee model
        target_audio: Fixed target audio (1, 8, 1, 128, 66)
        ref_video: Fixed reference video (1, 8, 16, 3, 224, 224)
        ref_audio: Fixed reference audio (1, 8, 1, 128, 66)
        device: Device to run on

    Returns:
        classifier: ART PyTorchClassifier ready for attacks
        wrapper: ARTRefereeVideoWrapper instance
    """
    try:
        from art.estimators.classification import PyTorchClassifier
    except ImportError as e:
        raise ImportError("ART not installed. Run: pip install adversarial-robustness-toolbox") from e

    # Create wrapper
    wrapper = ARTRefereeVideoWrapper(
        referee_model=referee_model,
        target_audio=target_audio,
        ref_video=ref_video,
        ref_audio=ref_audio,
        device=device
    )
    wrapper.eval()

    # Create ART classifier
    classifier = PyTorchClassifier(
        model=wrapper,
        loss=nn.CrossEntropyLoss(),
        input_shape=wrapper.input_shape,
        nb_classes=wrapper.nb_classes,
        clip_values=(-1.0, 1.0),  # Video is normalized to [-1, 1]
        device_type='gpu' if device == 'cuda' else 'cpu'
    )

    print("ART PyTorchClassifier created successfully!")

    return classifier, wrapper
