"""
Multimodal Input Wrapper for Referee Adversarial Attacks

This module provides utilities to handle combined audio-video inputs for adversarial attacks
on the Referee deepfake detection model.
"""

import torch
import torch.nn as nn
from typing import Tuple, Dict, Any


class MultiModalInput:
    """
    Wrapper for combined audio/video tensors to enable unified adversarial attacks
    while maintaining separate modality constraints.

    Handles the complexity of attacking both audio and video simultaneously
    while respecting their different tensor shapes and constraint requirements.
    """

    def __init__(self, audio_tensor: torch.Tensor, video_tensor: torch.Tensor):
        """
        Initialize multimodal input wrapper.

        Args:
            audio_tensor: Audio tensor of shape (B, S, 1, F, Ta) = (B, 8, 1, 128, 66)
            video_tensor: Video tensor of shape (B, S, Tv, C, H, W) = (B, 8, 16, 3, 224, 224)
        """
        self.audio = audio_tensor
        self.video = video_tensor
        self.device = audio_tensor.device
        self.dtype = audio_tensor.dtype

        # Store original shapes for reconstruction
        self.audio_shape = audio_tensor.shape
        self.video_shape = video_tensor.shape

        # Validate inputs
        self._validate_inputs()

    def _validate_inputs(self):
        """Validate that audio and video tensors have compatible batch dimensions."""
        assert self.audio.shape[0] == self.video.shape[0], \
            f"Batch dimension mismatch: audio {self.audio.shape[0]} vs video {self.video.shape[0]}"

        assert self.audio.device == self.video.device, \
            f"Device mismatch: audio on {self.audio.device} vs video on {self.video.device}"

    def split_tensors(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return separate audio and video tensors.

        Returns:
            Tuple of (audio_tensor, video_tensor)
        """
        return self.audio, self.video

    def enable_gradients(self, audio: bool = True, video: bool = True):
        """
        Enable gradient computation for specified modalities.

        Args:
            audio: Whether to enable gradients for audio
            video: Whether to enable gradients for video
        """
        if audio:
            self.audio.requires_grad_(True)
        if video:
            self.video.requires_grad_(True)

    def detach_gradients(self):
        """Detach gradients from both audio and video tensors."""
        self.audio = self.audio.detach()
        self.video = self.video.detach()

    def clone(self):
        """Create a deep copy of the multimodal input."""
        return MultiModalInput(self.audio.clone(), self.video.clone())

    def to(self, device):
        """Move both tensors to specified device."""
        return MultiModalInput(self.audio.to(device), self.video.to(device))

    @property
    def batch_size(self) -> int:
        """Return batch size."""
        return self.audio.shape[0]

    def __repr__(self):
        return (f"MultiModalInput(audio_shape={self.audio_shape}, "
                f"video_shape={self.video_shape}, device={self.device})")


class RefereeAttackWrapper(nn.Module):
    """
    Wrapper around the Referee model to enable adversarial attacks.

    This wrapper:
    1. Holds fixed reference audio/video pairs
    2. Takes only target audio/video as input (for attacking)
    3. Provides clean loss computation for gradient-based attacks
    4. Focuses on RF (real/fake) classification head only
    """

    def __init__(self, referee_model: nn.Module):
        """
        Initialize attack wrapper.

        Args:
            referee_model: The Referee model instance
        """
        super().__init__()
        self.referee_model = referee_model

        # Will be set when preparing for attack
        self.ref_audio = None
        self.ref_video = None
        self.attack_labels = None  # Expected ground truth labels for loss computation

    def set_reference_pair(self, ref_audio: torch.Tensor, ref_video: torch.Tensor):
        """
        Set the reference audio/video pair for comparison.

        Args:
            ref_audio: Reference audio tensor (B, S, 1, F, Ta)
            ref_video: Reference video tensor (B, S, Tv, C, H, W)
        """
        self.ref_audio = ref_audio.detach()  # Don't attack reference
        self.ref_video = ref_video.detach()

    def set_attack_labels(self, labels_rf: torch.Tensor):
        """
        Set the ground truth labels for loss computation during attacks.

        Args:
            labels_rf: Real/fake labels (0=real, 1=fake)
        """
        self.attack_labels = labels_rf

    def forward(self, target_audio: torch.Tensor, target_video: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for attack - only RF head loss.

        Args:
            target_audio: Target audio tensor to be attacked
            target_video: Target video tensor to be attacked

        Returns:
            RF classification loss (scalar tensor for gradient computation)
        """
        assert self.ref_audio is not None, "Must set reference pair before attack"
        assert self.ref_video is not None, "Must set reference pair before attack"
        assert self.attack_labels is not None, "Must set attack labels before attack"

        # Ensure model is in training mode for gradient computation
        was_training = self.referee_model.training
        if not was_training:
            self.referee_model.train()

        try:
            # Forward pass through Referee (inference mode - no labels)
            logits_rf, logits_id = self.referee_model(
                target_vis=target_video,
                target_aud=target_audio,
                ref_vis=self.ref_video,
                ref_aud=self.ref_audio
            )

            # Compute RF loss only (this is what we want to attack)
            loss_rf = torch.nn.functional.cross_entropy(logits_rf, self.attack_labels)

        finally:
            # Restore original model mode
            if not was_training:
                self.referee_model.eval()

        return loss_rf

    def predict(self, target_audio: torch.Tensor, target_video: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get model predictions without loss computation.

        Args:
            target_audio: Target audio tensor
            target_video: Target video tensor

        Returns:
            Tuple of (rf_logits, id_logits)
        """
        assert self.ref_audio is not None, "Must set reference pair before prediction"
        assert self.ref_video is not None, "Must set reference pair before prediction"

        with torch.no_grad():
            logits_rf, logits_id = self.referee_model(
                target_vis=target_video,
                target_aud=target_audio,
                ref_vis=self.ref_video,
                ref_aud=self.ref_audio
            )

        return logits_rf, logits_id

    def get_confidence(self, target_audio: torch.Tensor, target_video: torch.Tensor) -> Dict[str, float]:
        """
        Get model confidence scores.

        Returns:
            Dictionary with 'rf_confidence' and 'id_confidence'
        """
        try:
            logits_rf, logits_id = self.predict(target_audio, target_video)

            rf_probs = torch.softmax(logits_rf, dim=1)
            id_probs = torch.softmax(logits_id, dim=1)

            return {
                'rf_confidence': rf_probs.max(dim=1)[0].mean().item(),
                'id_confidence': id_probs.max(dim=1)[0].mean().item(),
                'rf_fake_prob': rf_probs[:, 1].mean().item(),  # Probability of being fake
                'rf_real_prob': rf_probs[:, 0].mean().item()   # Probability of being real
            }
        except Exception as e:
            # Return neutral values if computation fails
            return {
                'rf_confidence': 0.5,
                'id_confidence': 0.5,
                'rf_fake_prob': 0.5,
                'rf_real_prob': 0.5,
                'error': str(e)
            }


def create_attack_wrapper(referee_model: nn.Module,
                         ref_audio: torch.Tensor,
                         ref_video: torch.Tensor,
                         labels_rf: torch.Tensor) -> RefereeAttackWrapper:
    """
    Convenience function to create and setup a RefereeAttackWrapper.

    Args:
        referee_model: The Referee model
        ref_audio: Reference audio tensor
        ref_video: Reference video tensor
        labels_rf: Ground truth RF labels

    Returns:
        Configured RefereeAttackWrapper ready for attacks
    """
    wrapper = RefereeAttackWrapper(referee_model)
    wrapper.set_reference_pair(ref_audio, ref_video)
    wrapper.set_attack_labels(labels_rf)
    return wrapper