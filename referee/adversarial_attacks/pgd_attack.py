"""
PGD Attack Implementation for Referee Multimodal Deepfake Detection

This module implements Projected Gradient Descent adversarial attacks specifically
designed for the Referee model's dual-input (target + reference) architecture.

Based on standard PGD implementations (Madry et al., 2017) and ART library parameters.

Features:
- L2 norm constraints for both audio and video
- Temporal smoothness regularization
- Individual modality attacks (audio-only, video-only)
- Joint multimodal attacks
- Early stopping for efficiency
- Proper gradient direction (ascent for untargeted attacks)
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

    IMPORTANT: For untargeted attacks on fake samples, we use gradient ASCENT
    (add gradients) to MAXIMIZE the loss, which makes the model misclassify
    the fake sample as real.
    """

    def __init__(self,
                 referee_model: nn.Module,
                 eps_audio: float = 0.3,          # L2 norm bound for audio (larger for mel-spectrograms)
                 eps_video: float = 8.0,          # L2 norm bound for video (ART default for L2)
                 eps_step_audio: float = 0.1,     # Audio step size (~eps/3, ART-style)
                 eps_step_video: float = 2.0,     # Video step size (~eps/4, ART-style)
                 max_iter: int = 40,              # ART default
                 norm: str = 'L2',                # L2 norm
                 targeted: bool = False,          # Untargeted attack (make fake -> real)
                 temporal_weight: float = 0.0,    # No temporal regularization by default for stronger attacks
                 attack_mode: str = 'joint',      # 'audio', 'video', or 'joint'
                 random_init: bool = True,        # Random initialization within eps ball
                 early_stop: bool = True,         # Stop when attack succeeds
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
            targeted: Whether to perform targeted attack
            temporal_weight: Weight for temporal smoothness regularization (0 = disabled)
            attack_mode: Which modalities to attack ('audio', 'video', 'joint')
            random_init: Whether to initialize perturbations randomly within eps ball
            early_stop: Whether to stop early when attack succeeds
            verbose: Whether to show progress
        """
        self.referee_model = referee_model
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
        self.early_stop = early_stop
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
            print(f"  Early stop: {self.early_stop}")

    def generate(self,
                 target_audio: torch.Tensor,
                 target_video: torch.Tensor,
                 ref_audio: torch.Tensor,
                 ref_video: torch.Tensor,
                 labels_rf: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        Generate adversarial examples using PGD.

        For UNTARGETED attacks on FAKE samples (label=1):
        - Goal: Make model classify fake as real
        - Method: MAXIMIZE cross-entropy loss with true label
        - Implementation: Gradient ASCENT (add gradients to input)

        Args:
            target_audio: Original target audio (B, S, 1, F, Ta)
            target_video: Original target video (B, S, Tv, C, H, W)
            ref_audio: Reference audio (fixed, not attacked)
            ref_video: Reference video (fixed, not attacked)
            labels_rf: Ground truth RF labels (1=fake, 0=real)

        Returns:
            Tuple of (adversarial_audio, adversarial_video, attack_info)
        """
        device = target_audio.device
        batch_size = target_audio.shape[0]

        if self.verbose:
            print(f"Starting PGD attack on {batch_size} samples...")

        # Create attack wrapper
        attack_wrapper = RefereeAttackWrapper(self.referee_model)
        attack_wrapper.set_reference_pair(ref_audio, ref_video)
        attack_wrapper.set_attack_labels(labels_rf)

        # Store original inputs
        orig_audio = target_audio.clone().detach()
        orig_video = target_video.clone().detach()

        # Initialize adversarial examples
        if self.random_init:
            adv_audio, adv_video = self._random_init(orig_audio, orig_video)
        else:
            adv_audio = orig_audio.clone()
            adv_video = orig_video.clone()

        # Track attack progress
        attack_info = {
            'losses': [],
            'temporal_losses': [],
            'success_iterations': [],
            'confidence_scores': [],
            'converged': False
        }

        # Get initial predictions
        initial_confidence = attack_wrapper.get_confidence(orig_audio, orig_video)
        attack_info['initial_confidence'] = initial_confidence
        if self.verbose:
            print(f"Initial confidence - Real: {initial_confidence['rf_real_prob']:.3f}, "
                  f"Fake: {initial_confidence['rf_fake_prob']:.3f}")

        best_adv_audio = adv_audio.clone()
        best_adv_video = adv_video.clone()
        best_real_prob = initial_confidence['rf_real_prob']

        # PGD attack loop
        for iteration in range(self.max_iter):
            # Ensure we're working with fresh tensors that can have gradients
            adv_audio = adv_audio.detach().clone()
            adv_video = adv_video.detach().clone()

            # Enable gradients for attacked modalities
            if self.attack_mode in ['audio', 'joint']:
                adv_audio.requires_grad_(True)
            if self.attack_mode in ['video', 'joint']:
                adv_video.requires_grad_(True)

            # Forward pass and loss computation
            try:
                # Compute classification loss
                classification_loss = attack_wrapper(adv_audio, adv_video)

                # Temporal regularization (optional)
                if self.temporal_weight > 0:
                    temporal_loss = self._compute_temporal_loss(adv_audio, adv_video)
                    # For untargeted: we want to maximize classification loss but minimize temporal variance
                    # So we subtract temporal loss (which we minimize)
                    total_loss = classification_loss - self.temporal_weight * temporal_loss
                else:
                    temporal_loss = torch.tensor(0.0, device=device)
                    total_loss = classification_loss

                # Backward pass
                total_loss.backward()

                # Get gradients
                grad_audio = adv_audio.grad if adv_audio.requires_grad else None
                grad_video = adv_video.grad if adv_video.requires_grad else None

                # Update perturbations using gradient ASCENT for untargeted attacks
                with torch.no_grad():
                    if self.attack_mode in ['audio', 'joint'] and grad_audio is not None:
                        # Normalize gradient (use reshape for non-contiguous tensors)
                        grad_flat = grad_audio.reshape(batch_size, -1)
                        grad_norm = torch.norm(grad_flat, dim=1, keepdim=True) + 1e-10
                        grad_normalized = grad_audio / grad_norm.reshape(-1, 1, 1, 1, 1)

                        # GRADIENT ASCENT for untargeted (maximize loss)
                        # GRADIENT DESCENT for targeted (minimize loss toward target)
                        if self.targeted:
                            adv_audio = adv_audio - self.eps_step_audio * grad_normalized
                        else:
                            adv_audio = adv_audio + self.eps_step_audio * grad_normalized

                    if self.attack_mode in ['video', 'joint'] and grad_video is not None:
                        # Normalize gradient (use reshape for non-contiguous tensors)
                        grad_flat = grad_video.reshape(batch_size, -1)
                        grad_norm = torch.norm(grad_flat, dim=1, keepdim=True) + 1e-10
                        grad_normalized = grad_video / grad_norm.reshape(-1, 1, 1, 1, 1, 1)

                        # GRADIENT ASCENT for untargeted (maximize loss)
                        # GRADIENT DESCENT for targeted (minimize loss toward target)
                        if self.targeted:
                            adv_video = adv_video - self.eps_step_video * grad_normalized
                        else:
                            adv_video = adv_video + self.eps_step_video * grad_normalized

                    # Project back to epsilon ball and valid input range
                    adv_audio, adv_video = self._project(adv_audio, adv_video, orig_audio, orig_video)

                # Track progress
                attack_info['losses'].append(classification_loss.item())
                attack_info['temporal_losses'].append(temporal_loss.item() if isinstance(temporal_loss, torch.Tensor) else temporal_loss)

                # Check progress every few iterations
                if iteration % 5 == 0 or iteration == self.max_iter - 1:
                    current_confidence = attack_wrapper.get_confidence(adv_audio.detach(), adv_video.detach())
                    attack_info['confidence_scores'].append({
                        'iteration': iteration,
                        **current_confidence
                    })

                    current_real_prob = current_confidence['rf_real_prob']

                    # Track best adversarial example
                    if current_real_prob > best_real_prob:
                        best_real_prob = current_real_prob
                        best_adv_audio = adv_audio.detach().clone()
                        best_adv_video = adv_video.detach().clone()

                    if self.verbose and iteration % 10 == 0:
                        print(f"Iter {iteration:3d}: Loss={classification_loss.item():.4f}, "
                              f"Real_prob={current_real_prob:.3f}")

                    # Early stopping: attack succeeded if fake classified as real
                    if self.early_stop and not self.targeted and current_real_prob > 0.5:
                        attack_info['success_iterations'].append(iteration)
                        attack_info['converged'] = True
                        if self.verbose:
                            print(f"  ✓ Attack succeeded at iteration {iteration}! Real_prob={current_real_prob:.3f}")
                        break

            except Exception as e:
                if self.verbose:
                    print(f"  Error in iteration {iteration}: {e}")
                import traceback
                traceback.print_exc()
                break

        # Use best adversarial example found
        final_adv_audio = best_adv_audio.detach()
        final_adv_video = best_adv_video.detach()

        # Final evaluation
        if self.verbose:
            final_confidence = attack_wrapper.get_confidence(final_adv_audio, final_adv_video)
            print(f"Final confidence - Real: {final_confidence['rf_real_prob']:.3f}, "
                  f"Fake: {final_confidence['rf_fake_prob']:.3f}")
            print(f"Confidence change: {final_confidence['rf_real_prob'] - initial_confidence['rf_real_prob']:+.3f}")

        return final_adv_audio, final_adv_video, attack_info

    def _random_init(self, orig_audio: torch.Tensor, orig_video: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Initialize adversarial examples randomly within epsilon ball."""
        adv_audio = orig_audio.clone()
        adv_video = orig_video.clone()

        if self.attack_mode in ['audio', 'joint']:
            # Random perturbation with L2 norm = eps_audio * random_factor
            delta = torch.randn_like(orig_audio)
            delta_flat = delta.reshape(orig_audio.size(0), -1)
            delta_norm = torch.norm(delta_flat, dim=1, keepdim=True) + 1e-10
            # Random magnitude between 0 and eps
            random_factor = torch.rand(orig_audio.size(0), 1, device=orig_audio.device)
            delta = delta / delta_norm.reshape(-1, 1, 1, 1, 1) * self.eps_audio * random_factor.reshape(-1, 1, 1, 1, 1)
            adv_audio = orig_audio + delta

        if self.attack_mode in ['video', 'joint']:
            # Random perturbation with L2 norm = eps_video * random_factor
            delta = torch.randn_like(orig_video)
            delta_flat = delta.reshape(orig_video.size(0), -1)
            delta_norm = torch.norm(delta_flat, dim=1, keepdim=True) + 1e-10
            # Random magnitude between 0 and eps
            random_factor = torch.rand(orig_video.size(0), 1, device=orig_video.device)
            delta = delta / delta_norm.reshape(-1, 1, 1, 1, 1, 1) * self.eps_video * random_factor.reshape(-1, 1, 1, 1, 1, 1)
            adv_video = orig_video + delta
            # Clamp to valid range (video is normalized to [-1, 1])
            adv_video = torch.clamp(adv_video, -1.0, 1.0)

        return adv_audio, adv_video

    def _compute_temporal_loss(self, adv_audio: torch.Tensor, adv_video: torch.Tensor) -> torch.Tensor:
        """Compute temporal smoothness loss (variance of frame differences)."""
        temporal_loss = torch.tensor(0.0, device=adv_audio.device)

        if self.attack_mode in ['audio', 'joint']:
            # Audio shape: (B, S, 1, F, Ta) - smooth across time dimension
            if adv_audio.shape[-1] > 1:
                diff = adv_audio[:, :, :, :, 1:] - adv_audio[:, :, :, :, :-1]
                temporal_loss = temporal_loss + torch.mean(diff ** 2)

        if self.attack_mode in ['video', 'joint']:
            # Video shape: (B, S, Tv, C, H, W) - smooth across frames
            if adv_video.shape[2] > 1:
                diff = adv_video[:, :, 1:, :, :, :] - adv_video[:, :, :-1, :, :, :]
                temporal_loss = temporal_loss + torch.mean(diff ** 2)

        return temporal_loss

    def _project(self, adv_audio: torch.Tensor, adv_video: torch.Tensor,
                 orig_audio: torch.Tensor, orig_video: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Project perturbations to epsilon ball and valid input range."""

        if self.attack_mode in ['audio', 'joint']:
            # Compute perturbation
            delta = adv_audio - orig_audio
            delta_flat = delta.reshape(delta.size(0), -1)
            delta_norm = torch.norm(delta_flat, dim=1, keepdim=True)

            # Project to L2 ball
            factor = torch.clamp(self.eps_audio / (delta_norm + 1e-10), max=1.0)
            delta = delta * factor.reshape(-1, 1, 1, 1, 1)
            adv_audio = orig_audio + delta

        if self.attack_mode in ['video', 'joint']:
            # Compute perturbation
            delta = adv_video - orig_video
            delta_flat = delta.reshape(delta.size(0), -1)
            delta_norm = torch.norm(delta_flat, dim=1, keepdim=True)

            # Project to L2 ball
            factor = torch.clamp(self.eps_video / (delta_norm + 1e-10), max=1.0)
            delta = delta * factor.reshape(-1, 1, 1, 1, 1, 1)
            adv_video = orig_video + delta

            # Clamp to valid pixel range (video is normalized to [-1, 1])
            adv_video = torch.clamp(adv_video, -1.0, 1.0)

        return adv_audio, adv_video

    def compute_perturbation_norms(self, adv_audio: torch.Tensor, adv_video: torch.Tensor,
                                   orig_audio: torch.Tensor, orig_video: torch.Tensor) -> Dict[str, float]:
        """Compute L2 norms of perturbations for analysis."""
        norms = {}

        if self.attack_mode in ['audio', 'joint']:
            delta = adv_audio - orig_audio
            audio_norm = torch.norm(delta.reshape(delta.size(0), -1), dim=1)
            norms['audio_l2_norm'] = audio_norm.mean().item()
            norms['audio_l2_max'] = audio_norm.max().item()

        if self.attack_mode in ['video', 'joint']:
            delta = adv_video - orig_video
            video_norm = torch.norm(delta.reshape(delta.size(0), -1), dim=1)
            norms['video_l2_norm'] = video_norm.mean().item()
            norms['video_l2_max'] = video_norm.max().item()

        return norms
