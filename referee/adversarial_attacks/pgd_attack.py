"""
PGD Attack Implementation for Referee Multimodal Deepfake Detection

This module implements Projected Gradient Descent adversarial attacks specifically
designed for the Referee model's dual-input (target + reference) architecture.

Features:
- L2 norm constraints for both audio and video
- Temporal smoothness regularization
- Individual modality attacks (audio-only, video-only)
- Joint multimodal attacks
- Proper handling of reference-aware architecture
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Dict, Optional, Union
from .multimodal_wrapper import MultiModalInput, RefereeAttackWrapper


class RefereeMultiModalPGD:
    """
    Projected Gradient Descent attack for the Referee deepfake detection model.

    This attack is designed specifically for the reference-aware, multimodal architecture
    of the Referee model, supporting separate constraints and regularization for
    audio and video modalities.
    """

    def __init__(self,
                 referee_model: nn.Module,
                 eps_audio: float = 0.05,          # 5% audio perturbation
                 eps_video: float = 0.3,           # L2 norm bound for video (scaled to [0,1])
                 eps_step_audio: float = 0.01,     # Audio step size
                 eps_step_video: float = 0.05,     # Video step size
                 max_iter: int = 100,              # Maximum iterations
                 norm: str = 'L2',                 # L2 norm (as requested)
                 targeted: bool = False,           # Untargeted attack (make fake -> real)
                 temporal_weight: float = 0.5,     # Temporal smoothness weight
                 attack_mode: str = 'joint',       # 'audio', 'video', or 'joint'
                 random_init: bool = True,         # Random initialization within eps ball
                 verbose: bool = True):
        """
        Initialize the PGD attack.

        Args:
            referee_model: The Referee model to attack
            eps_audio: L2 norm bound for audio perturbations
            eps_video: L2 norm bound for video perturbations
            eps_step_audio: Step size for audio updates
            eps_step_video: Step size for video updates
            max_iter: Maximum number of iterations
            norm: Norm type (only 'L2' supported currently)
            targeted: Whether to perform targeted attack (not typically used for deepfake detection)
            temporal_weight: Weight for temporal smoothness regularization
            attack_mode: Which modalities to attack ('audio', 'video', 'joint')
            random_init: Whether to initialize perturbations randomly within eps ball
            verbose: Whether to show progress
        """
        self.referee_model = referee_model.eval()  # Set to eval mode for consistent behavior
        self.eps_audio = eps_audio
        self.eps_video = eps_video
        self.eps_step_audio = eps_step_audio
        self.eps_step_video = eps_step_video
        self.max_iter = max_iter
        self.norm = norm
        self.targeted = targeted
        self.temporal_weight = temporal_weight
        self.attack_mode = attack_mode.lower()
        self.random_init = random_init
        self.verbose = verbose

        # Validate inputs
        assert self.norm == 'L2', f"Only L2 norm supported, got {self.norm}"
        assert self.attack_mode in ['audio', 'video', 'joint'], \
            f"Invalid attack mode: {self.attack_mode}"

        if self.verbose:
            print(f"Initialized RefereeMultiModalPGD:")
            print(f"  Mode: {self.attack_mode}")
            print(f"  Audio eps: {self.eps_audio}, step: {self.eps_step_audio}")
            print(f"  Video eps: {self.eps_video}, step: {self.eps_step_video}")
            print(f"  Temporal weight: {self.temporal_weight}")
            print(f"  Max iterations: {self.max_iter}")

    def generate(self,
                 target_audio: torch.Tensor,
                 target_video: torch.Tensor,
                 ref_audio: torch.Tensor,
                 ref_video: torch.Tensor,
                 labels_rf: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        Generate adversarial examples using PGD.

        Args:
            target_audio: Original target audio (B, S, 1, F, Ta)
            target_video: Original target video (B, S, Tv, C, H, W)
            ref_audio: Reference audio (fixed, not attacked)
            ref_video: Reference video (fixed, not attacked)
            labels_rf: Ground truth RF labels for loss computation

        Returns:
            Tuple of (adversarial_audio, adversarial_video, attack_info)
        """
        if self.verbose:
            print(f"Starting PGD attack on {target_audio.shape[0]} samples...")

        # Create attack wrapper
        attack_wrapper = RefereeAttackWrapper(self.referee_model)
        attack_wrapper.set_reference_pair(ref_audio, ref_video)
        attack_wrapper.set_attack_labels(labels_rf)

        # Initialize adversarial examples (copy originals)
        adv_audio = target_audio.clone().detach()
        adv_video = target_video.clone().detach()

        # Initialize perturbations
        if self.random_init:
            delta_audio, delta_video = self._random_init_perturbations(target_audio, target_video)
        else:
            delta_audio = torch.zeros_like(target_audio)
            delta_video = torch.zeros_like(target_video)

        # Apply initial perturbations
        adv_audio = target_audio + delta_audio
        adv_video = target_video + delta_video

        # Track attack progress
        attack_info = {
            'losses': [],
            'temporal_losses': [],
            'success_iterations': [],
            'confidence_scores': []
        }

        # Get initial predictions
        initial_confidence = attack_wrapper.get_confidence(target_audio, target_video)
        if self.verbose:
            print(f"Initial confidence - Real: {initial_confidence['rf_real_prob']:.3f}, "
                  f"Fake: {initial_confidence['rf_fake_prob']:.3f}")

        # PGD attack loop
        for iteration in range(self.max_iter):
            # Enable gradients for attacked modalities
            if self.attack_mode in ['audio', 'joint']:
                adv_audio.requires_grad_(True)
            if self.attack_mode in ['video', 'joint']:
                adv_video.requires_grad_(True)

            # Forward pass and loss computation
            classification_loss = attack_wrapper(adv_audio, adv_video)

            # Temporal regularization
            temporal_loss = self._compute_temporal_loss(adv_audio, adv_video,
                                                      target_audio, target_video)

            # Total loss
            total_loss = classification_loss + self.temporal_weight * temporal_loss

            # Backward pass
            total_loss.backward()

            # Extract gradients
            grad_audio = adv_audio.grad if adv_audio.requires_grad else None
            grad_video = adv_video.grad if adv_video.requires_grad else None

            # Update perturbations
            with torch.no_grad():
                if grad_audio is not None:
                    # L2 normalized gradient step for audio
                    grad_norm = torch.norm(grad_audio.view(grad_audio.size(0), -1), dim=1, keepdim=True)
                    grad_norm = grad_norm.view(-1, 1, 1, 1, 1)
                    grad_normalized = grad_audio / (grad_norm + 1e-8)

                    if self.targeted:
                        adv_audio = adv_audio + self.eps_step_audio * grad_normalized  # Targeted
                    else:
                        adv_audio = adv_audio - self.eps_step_audio * grad_normalized  # Untargeted

                if grad_video is not None:
                    # L2 normalized gradient step for video
                    grad_norm = torch.norm(grad_video.view(grad_video.size(0), -1), dim=1, keepdim=True)
                    grad_norm = grad_norm.view(-1, 1, 1, 1, 1, 1)
                    grad_normalized = grad_video / (grad_norm + 1e-8)

                    if self.targeted:
                        adv_video = adv_video + self.eps_step_video * grad_normalized
                    else:
                        adv_video = adv_video - self.eps_step_video * grad_normalized

                # Project back to valid space (L2 ball + input bounds)
                adv_audio, adv_video = self._project_perturbations(
                    adv_audio, adv_video, target_audio, target_video)

            # Clean up gradients
            if adv_audio.requires_grad:
                adv_audio.grad = None
            if adv_video.requires_grad:
                adv_video.grad = None

            # Track progress
            attack_info['losses'].append(classification_loss.item())
            attack_info['temporal_losses'].append(temporal_loss.item())

            # Check attack success periodically
            if iteration % 10 == 0 or iteration == self.max_iter - 1:
                current_confidence = attack_wrapper.get_confidence(adv_audio, adv_video)
                attack_info['confidence_scores'].append(current_confidence)

                if self.verbose and iteration % 20 == 0:
                    print(f"Iter {iteration:3d}: Loss={classification_loss.item():.4f}, "
                          f"Temporal={temporal_loss.item():.4f}, "
                          f"Real_prob={current_confidence['rf_real_prob']:.3f}")

                # Check if attack succeeded (fake classified as real)
                if not self.targeted and current_confidence['rf_real_prob'] > 0.5:
                    attack_info['success_iterations'].append(iteration)
                    if self.verbose:
                        print(f"  ✓ Attack succeeded at iteration {iteration}!")

        if self.verbose:
            final_confidence = attack_wrapper.get_confidence(adv_audio, adv_video)
            print(f"Final confidence - Real: {final_confidence['rf_real_prob']:.3f}, "
                  f"Fake: {final_confidence['rf_fake_prob']:.3f}")

        return adv_audio.detach(), adv_video.detach(), attack_info

    def _random_init_perturbations(self, target_audio: torch.Tensor,
                                 target_video: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Initialize perturbations randomly within epsilon ball."""
        delta_audio = torch.zeros_like(target_audio)
        delta_video = torch.zeros_like(target_video)

        if self.attack_mode in ['audio', 'joint']:
            # Random audio perturbation within L2 ball
            delta_audio = torch.randn_like(target_audio)
            delta_norm = torch.norm(delta_audio.view(delta_audio.size(0), -1), dim=1, keepdim=True)
            delta_norm = delta_norm.view(-1, 1, 1, 1, 1)
            delta_audio = delta_audio / (delta_norm + 1e-8) * self.eps_audio * torch.rand_like(delta_norm)

        if self.attack_mode in ['video', 'joint']:
            # Random video perturbation within L2 ball
            delta_video = torch.randn_like(target_video)
            delta_norm = torch.norm(delta_video.view(delta_video.size(0), -1), dim=1, keepdim=True)
            delta_norm = delta_norm.view(-1, 1, 1, 1, 1, 1)
            delta_video = delta_video / (delta_norm + 1e-8) * self.eps_video * torch.rand_like(delta_norm)

        return delta_audio, delta_video

    def _compute_temporal_loss(self, adv_audio: torch.Tensor, adv_video: torch.Tensor,
                             orig_audio: torch.Tensor, orig_video: torch.Tensor) -> torch.Tensor:
        """Compute temporal smoothness regularization loss."""
        temporal_loss = 0.0

        if self.attack_mode in ['audio', 'joint']:
            # Audio temporal smoothness (across time frames)
            diff = adv_audio[:, :, :, :, 1:] - adv_audio[:, :, :, :, :-1]
            temporal_loss += torch.mean(diff ** 2)

        if self.attack_mode in ['video', 'joint']:
            # Video temporal smoothness (across frames)
            diff = adv_video[:, :, 1:] - adv_video[:, :, :-1]
            temporal_loss += torch.mean(diff ** 2)

        return temporal_loss

    def _project_perturbations(self, adv_audio: torch.Tensor, adv_video: torch.Tensor,
                             orig_audio: torch.Tensor, orig_video: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Project perturbations back to valid L2 ball and input bounds."""

        if self.attack_mode in ['audio', 'joint']:
            # Project audio perturbation to L2 ball
            delta_audio = adv_audio - orig_audio
            delta_norm = torch.norm(delta_audio.view(delta_audio.size(0), -1), dim=1, keepdim=True)
            delta_norm = delta_norm.view(-1, 1, 1, 1, 1)

            # Clip to epsilon ball
            scale_factor = torch.min(torch.ones_like(delta_norm),
                                   self.eps_audio / (delta_norm + 1e-8))
            delta_audio = delta_audio * scale_factor
            adv_audio = orig_audio + delta_audio

            # Additional bounds for audio (if needed - e.g., log mel-spectrograms)
            # adv_audio = torch.clamp(adv_audio, min_val, max_val)  # Uncomment if bounds needed

        if self.attack_mode in ['video', 'joint']:
            # Project video perturbation to L2 ball
            delta_video = adv_video - orig_video
            delta_norm = torch.norm(delta_video.view(delta_video.size(0), -1), dim=1, keepdim=True)
            delta_norm = delta_norm.view(-1, 1, 1, 1, 1, 1)

            # Clip to epsilon ball
            scale_factor = torch.min(torch.ones_like(delta_norm),
                                   self.eps_video / (delta_norm + 1e-8))
            delta_video = delta_video * scale_factor
            adv_video = orig_video + delta_video

            # Clamp to valid pixel range [0, 1] (assuming preprocessed inputs)
            adv_video = torch.clamp(adv_video, 0.0, 1.0)

        return adv_audio, adv_video

    def compute_perturbation_norms(self, adv_audio: torch.Tensor, adv_video: torch.Tensor,
                                 orig_audio: torch.Tensor, orig_video: torch.Tensor) -> Dict[str, float]:
        """Compute L2 norms of perturbations for analysis."""
        norms = {}

        if self.attack_mode in ['audio', 'joint']:
            delta_audio = adv_audio - orig_audio
            audio_norm = torch.norm(delta_audio.view(delta_audio.size(0), -1), dim=1)
            norms['audio_l2_norm'] = audio_norm.mean().item()
            norms['audio_l2_max'] = audio_norm.max().item()

        if self.attack_mode in ['video', 'joint']:
            delta_video = adv_video - orig_video
            video_norm = torch.norm(delta_video.view(delta_video.size(0), -1), dim=1)
            norms['video_l2_norm'] = video_norm.mean().item()
            norms['video_l2_max'] = video_norm.max().item()

        return norms