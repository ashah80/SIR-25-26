"""
Testing Suite for Referee Multimodal PGD Attacks

This module provides comprehensive testing utilities to validate the correctness
of adversarial attacks on the Referee deepfake detection model.

Tests include:
- Gradient flow validation
- Attack bound verification
- Model compatibility checks
- Temporal coherence validation
- Attack success measurement
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional
from .pgd_attack import RefereeMultiModalPGD
from .multimodal_wrapper import RefereeAttackWrapper, create_attack_wrapper


class RefereeAttackTester:
    """
    Comprehensive testing suite for validating Referee adversarial attacks.
    """

    def __init__(self, referee_model: nn.Module, device: str = 'cuda'):
        """
        Initialize the testing suite.

        Args:
            referee_model: The Referee model to test attacks against
            device: Device to run tests on
        """
        self.referee_model = referee_model.eval()
        self.device = device

    def create_dummy_batch(self, batch_size: int = 2) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Create dummy input tensors for testing.

        Args:
            batch_size: Size of the batch to create

        Returns:
            Tuple of (target_audio, target_video, ref_audio, ref_video, labels_rf)
        """
        # Audio: (B, S, 1, F, Ta) = (B, 8, 1, 128, 66)
        target_audio = torch.randn(batch_size, 8, 1, 128, 66, device=self.device)
        ref_audio = torch.randn(batch_size, 8, 1, 128, 66, device=self.device)

        # Video: (B, S, Tv, C, H, W) = (B, 8, 16, 3, 224, 224)
        target_video = torch.rand(batch_size, 8, 16, 3, 224, 224, device=self.device)  # [0,1] range
        ref_video = torch.rand(batch_size, 8, 16, 3, 224, 224, device=self.device)

        # Labels: fake samples (1) - we want to attack these to be classified as real (0)
        labels_rf = torch.ones(batch_size, dtype=torch.long, device=self.device)

        return target_audio, target_video, ref_audio, ref_video, labels_rf

    def test_gradient_flow(self, batch_size: int = 2) -> Dict[str, bool]:
        """
        Test that gradients flow correctly through the model architecture.

        Returns:
            Dictionary with gradient flow test results
        """
        print(" Testing gradient flow...")

        target_audio, target_video, ref_audio, ref_video, labels_rf = self.create_dummy_batch(batch_size)

        # Create attack wrapper
        wrapper = create_attack_wrapper(self.referee_model, ref_audio, ref_video, labels_rf)

        # Enable gradients for both modalities
        target_audio.requires_grad_(True)
        target_video.requires_grad_(True)

        # Forward pass
        loss = wrapper(target_audio, target_video)

        # Backward pass
        loss.backward()

        # Check gradients
        results = {
            'audio_gradients_exist': target_audio.grad is not None,
            'video_gradients_exist': target_video.grad is not None,
            'audio_gradients_nonzero': False,
            'video_gradients_nonzero': False,
            'loss_is_scalar': loss.dim() == 0,
            'loss_is_finite': torch.isfinite(loss).item()
        }

        if target_audio.grad is not None:
            results['audio_gradients_nonzero'] = torch.any(torch.abs(target_audio.grad) > 1e-8).item()

        if target_video.grad is not None:
            results['video_gradients_nonzero'] = torch.any(torch.abs(target_video.grad) > 1e-8).item()

        # Print results
        for test_name, passed in results.items():
            status = "✓" if passed else "✗"
            print(f"  {status} {test_name}: {passed}")

        print(f"  Gradient flow test: {'PASSED' if all(results.values()) else 'FAILED'}")
        return results

    def test_attack_bounds(self, eps_audio: float = 0.05, eps_video: float = 0.3) -> Dict[str, bool]:
        """
        Test that adversarial perturbations respect epsilon bounds.

        Args:
            eps_audio: Audio epsilon bound to test
            eps_video: Video epsilon bound to test

        Returns:
            Dictionary with bounds test results
        """
        print(f" Testing attack bounds (audio_eps={eps_audio}, video_eps={eps_video})...")

        target_audio, target_video, ref_audio, ref_video, labels_rf = self.create_dummy_batch(1)

        # Run short attack
        attacker = RefereeMultiModalPGD(
            self.referee_model,
            eps_audio=eps_audio,
            eps_video=eps_video,
            max_iter=10,  # Short test
            verbose=False
        )

        adv_audio, adv_video, _ = attacker.generate(
            target_audio, target_video, ref_audio, ref_video, labels_rf
        )

        # Compute perturbation norms
        audio_delta = adv_audio - target_audio
        video_delta = adv_video - target_video

        audio_norm = torch.norm(audio_delta.view(audio_delta.size(0), -1), dim=1).max().item()
        video_norm = torch.norm(video_delta.view(video_delta.size(0), -1), dim=1).max().item()

        results = {
            'audio_bounds_respected': audio_norm <= eps_audio + 1e-6,
            'video_bounds_respected': video_norm <= eps_video + 1e-6,
            'video_pixel_range_valid': (adv_video >= 0).all().item() and (adv_video <= 1).all().item(),
            'audio_finite': torch.isfinite(adv_audio).all().item(),
            'video_finite': torch.isfinite(adv_video).all().item()
        }

        # Print results
        print(f"  Audio L2 norm: {audio_norm:.6f} (bound: {eps_audio})")
        print(f"  Video L2 norm: {video_norm:.6f} (bound: {eps_video})")

        for test_name, passed in results.items():
            status = "✓" if passed else "✗"
            print(f"  {status} {test_name}: {passed}")

        print(f"  Bounds test: {'PASSED' if all(results.values()) else 'FAILED'}")
        return results

    def test_temporal_coherence(self, temporal_weight: float = 0.5) -> Dict[str, float]:
        """
        Test that temporal regularization improves smoothness.

        Args:
            temporal_weight: Temporal weight to test

        Returns:
            Dictionary with temporal coherence metrics
        """
        print(f" Testing temporal coherence (temporal_weight={temporal_weight})...")

        target_audio, target_video, ref_audio, ref_video, labels_rf = self.create_dummy_batch(1)

        # Attack without temporal regularization
        attacker_no_temp = RefereeMultiModalPGD(
            self.referee_model,
            temporal_weight=0.0,
            max_iter=20,
            verbose=False
        )
        adv_audio_no_temp, adv_video_no_temp, _ = attacker_no_temp.generate(
            target_audio, target_video, ref_audio, ref_video, labels_rf
        )

        # Attack with temporal regularization
        attacker_with_temp = RefereeMultiModalPGD(
            self.referee_model,
            temporal_weight=temporal_weight,
            max_iter=20,
            verbose=False
        )
        adv_audio_with_temp, adv_video_with_temp, _ = attacker_with_temp.generate(
            target_audio, target_video, ref_audio, ref_video, labels_rf
        )

        # Compute temporal variance (lower is smoother)
        def compute_temporal_variance(audio, video):
            audio_var = torch.var(audio[:, :, :, :, 1:] - audio[:, :, :, :, :-1]).item()
            video_var = torch.var(video[:, :, 1:] - video[:, :, :-1]).item()
            return audio_var, video_var

        audio_var_no_temp, video_var_no_temp = compute_temporal_variance(adv_audio_no_temp, adv_video_no_temp)
        audio_var_with_temp, video_var_with_temp = compute_temporal_variance(adv_audio_with_temp, adv_video_with_temp)

        results = {
            'audio_variance_without_temp': audio_var_no_temp,
            'audio_variance_with_temp': audio_var_with_temp,
            'video_variance_without_temp': video_var_no_temp,
            'video_variance_with_temp': video_var_with_temp,
            'audio_smoothness_improved': audio_var_with_temp < audio_var_no_temp,
            'video_smoothness_improved': video_var_with_temp < video_var_no_temp
        }

        # Print results
        print(f"  Audio - Without temp: {audio_var_no_temp:.6f}, With temp: {audio_var_with_temp:.6f}")
        print(f"  Video - Without temp: {video_var_no_temp:.6f}, With temp: {video_var_with_temp:.6f}")

        improvement_audio = "✓" if results['audio_smoothness_improved'] else "✗"
        improvement_video = "✓" if results['video_smoothness_improved'] else "✗"
        print(f"  {improvement_audio} Audio smoothness improved")
        print(f"  {improvement_video} Video smoothness improved")

        return results

    def test_model_compatibility(self) -> Dict[str, bool]:
        """
        Test that attacked samples are compatible with the model.

        Returns:
            Dictionary with compatibility test results
        """
        print("🧪 Testing model compatibility...")

        target_audio, target_video, ref_audio, ref_video, labels_rf = self.create_dummy_batch(2)

        # Run attack
        attacker = RefereeMultiModalPGD(
            self.referee_model,
            max_iter=5,  # Short test
            verbose=False
        )

        adv_audio, adv_video, _ = attacker.generate(
            target_audio, target_video, ref_audio, ref_video, labels_rf
        )

        results = {
            'shapes_preserved': True,
            'model_accepts_input': True,
            'output_shapes_correct': True,
            'no_exceptions': True
        }

        try:
            # Check shapes
            if adv_audio.shape != target_audio.shape:
                results['shapes_preserved'] = False
            if adv_video.shape != target_video.shape:
                results['shapes_preserved'] = False

            # Test model forward pass
            with torch.no_grad():
                logits_rf, logits_id = self.referee_model(
                    target_vis=adv_video,
                    target_aud=adv_audio,
                    ref_vis=ref_video,
                    ref_aud=ref_audio
                )

                # Check output shapes
                expected_batch_size = adv_audio.shape[0]
                if logits_rf.shape != (expected_batch_size, 2):
                    results['output_shapes_correct'] = False
                if logits_id.shape != (expected_batch_size, 2):
                    results['output_shapes_correct'] = False

        except Exception as e:
            print(f"  Exception occurred: {e}")
            results['model_accepts_input'] = False
            results['no_exceptions'] = False

        # Print results
        for test_name, passed in results.items():
            status = "✓" if passed else "✗"
            print(f"  {status} {test_name}: {passed}")

        print(f"  Compatibility test: {'PASSED' if all(results.values()) else 'FAILED'}")
        return results

    def test_attack_modes(self) -> Dict[str, Dict[str, bool]]:
        """
        Test that different attack modes (audio, video, joint) work correctly.

        Returns:
            Dictionary with results for each attack mode
        """
        print("🧪 Testing attack modes...")

        target_audio, target_video, ref_audio, ref_video, labels_rf = self.create_dummy_batch(1)
        modes_results = {}

        for mode in ['audio', 'video', 'joint']:
            print(f"  Testing mode: {mode}")

            attacker = RefereeMultiModalPGD(
                self.referee_model,
                attack_mode=mode,
                max_iter=10,
                verbose=False
            )

            try:
                adv_audio, adv_video, attack_info = attacker.generate(
                    target_audio, target_video, ref_audio, ref_video, labels_rf
                )

                # Check that only attacked modalities are modified
                audio_changed = not torch.allclose(adv_audio, target_audio, atol=1e-6)
                video_changed = not torch.allclose(adv_video, target_video, atol=1e-6)

                mode_results = {
                    'execution_successful': True,
                    'audio_changed': audio_changed,
                    'video_changed': video_changed,
                    'correct_modality_attacked': True
                }

                # Verify correct modalities were attacked
                if mode == 'audio':
                    mode_results['correct_modality_attacked'] = audio_changed and not video_changed
                elif mode == 'video':
                    mode_results['correct_modality_attacked'] = not audio_changed and video_changed
                elif mode == 'joint':
                    mode_results['correct_modality_attacked'] = audio_changed and video_changed

            except Exception as e:
                print(f"    Exception in mode {mode}: {e}")
                mode_results = {
                    'execution_successful': False,
                    'audio_changed': False,
                    'video_changed': False,
                    'correct_modality_attacked': False
                }

            modes_results[mode] = mode_results

            # Print mode results
            for test_name, passed in mode_results.items():
                status = "✓" if passed else "✗"
                print(f"    {status} {test_name}: {passed}")

        return modes_results

    def run_all_tests(self) -> Dict[str, bool]:
        """
        Run all tests in sequence.

        Returns:
            Dictionary with overall test results
        """
        print("🧪 Running all Referee attack tests...\n")

        all_results = {}

        # Test 1: Gradient flow
        gradient_results = self.test_gradient_flow()
        all_results['gradient_flow'] = all(gradient_results.values())
        print()

        # Test 2: Attack bounds
        bounds_results = self.test_attack_bounds()
        all_results['attack_bounds'] = all(bounds_results.values())
        print()

        # Test 3: Temporal coherence
        temporal_results = self.test_temporal_coherence()
        all_results['temporal_coherence'] = (
            temporal_results['audio_smoothness_improved'] or temporal_results['video_smoothness_improved']
        )
        print()

        # Test 4: Model compatibility
        compatibility_results = self.test_model_compatibility()
        all_results['model_compatibility'] = all(compatibility_results.values())
        print()

        # Test 5: Attack modes
        modes_results = self.test_attack_modes()
        all_results['attack_modes'] = all(
            all(mode_results.values()) for mode_results in modes_results.values()
        )
        print()

        # Overall results
        all_passed = all(all_results.values())
        print("=" * 60)
        print(f"🧪 Overall test results: {'ALL PASSED ✓' if all_passed else 'SOME FAILED ✗'}")

        for test_name, passed in all_results.items():
            status = "✓" if passed else "✗"
            print(f"  {status} {test_name}")

        print("=" * 60)

        return all_results


def quick_test_attack_installation(referee_model: nn.Module, device: str = 'cuda') -> bool:
    """
    Quick test to verify the attack implementation is working correctly.

    Args:
        referee_model: Referee model to test
        device: Device to run test on

    Returns:
        True if basic functionality works, False otherwise
    """
    print("🚀 Quick test of attack installation...")

    try:
        tester = RefereeAttackTester(referee_model, device)

        # Run minimal tests
        gradient_results = tester.test_gradient_flow(batch_size=1)
        compatibility_results = tester.test_model_compatibility()

        basic_working = (
            all(gradient_results.values()) and
            all(compatibility_results.values())
        )

        if basic_working:
            print("✓ Quick test PASSED - Attack implementation is working!")
        else:
            print("✗ Quick test FAILED - Check installation")

        return basic_working

    except Exception as e:
        print(f"✗ Quick test FAILED with exception: {e}")
        return False