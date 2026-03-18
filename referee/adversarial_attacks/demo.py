"""
Demo script for Referee Adversarial Attacks

This script demonstrates how to use the multimodal PGD attack implementation
for the Referee deepfake detection model.

Usage examples:
1. Quick test of installation
2. Individual modality attacks (audio-only, video-only)
3. Joint multimodal attacks
4. Comprehensive evaluation with different parameters

Run this script after training/loading a Referee model to test adversarial attacks.
"""

import sys
from pathlib import Path
import torch
import torch.nn as nn
from omegaconf import OmegaConf

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

from model.referee import Referee
from adversarial_attacks.pgd_attack import RefereeMultiModalPGD
from adversarial_attacks.testing_suite import RefereeAttackTester, quick_test_attack_installation
from adversarial_attacks.evaluation_pipeline import RefereeAttackEvaluator


def load_referee_model(config_path: str, checkpoint_path: str = None) -> Referee:
    """
    Load a Referee model from config and optional checkpoint.

    Args:
        config_path: Path to configuration file
        checkpoint_path: Path to model checkpoint (optional)

    Returns:
        Loaded Referee model
    """
    # Load configuration
    cfg = OmegaConf.load(config_path)

    # Create model
    model = Referee(cfg, ckpt_path=checkpoint_path)
    model.eval()

    print(f"✅ Loaded Referee model from {checkpoint_path or 'random weights'}")
    return model


def create_dummy_data(batch_size: int = 1, device: str = 'cuda'):
    """Create dummy input data for testing."""
    # Audio: (B, S, 1, F, Ta) = (B, 8, 1, 128, 66)
    target_audio = torch.randn(batch_size, 8, 1, 128, 66, device=device)
    ref_audio = torch.randn(batch_size, 8, 1, 128, 66, device=device)

    # Video: (B, S, Tv, C, H, W) = (B, 8, 16, 3, 224, 224)
    target_video = torch.rand(batch_size, 8, 16, 3, 224, 224, device=device)
    ref_video = torch.rand(batch_size, 8, 16, 3, 224, 224, device=device)

    # Labels (fake samples - we want to make them appear real)
    labels_rf = torch.ones(batch_size, dtype=torch.long, device=device)

    return target_audio, target_video, ref_audio, ref_video, labels_rf


def demo_quick_test(model: nn.Module, device: str = 'cuda'):
    """Demonstrate quick installation test."""
    print("=" * 60)
    print("DEMO 1: Quick Installation Test")
    print("=" * 60)

    success = quick_test_attack_installation(model, device)

    if success:
        print("🎉 Adversarial attack implementation is working correctly!")
    else:
        print("❌ There are issues with the implementation. Check logs above.")

    return success


def demo_individual_attacks(model: nn.Module, device: str = 'cuda'):
    """Demonstrate individual modality attacks."""
    print("\n" + "=" * 60)
    print("DEMO 2: Individual Modality Attacks")
    print("=" * 60)

    # Create test data
    target_audio, target_video, ref_audio, ref_video, labels_rf = create_dummy_data(1, device)

    # Test each attack mode
    modes = ['audio', 'video', 'joint']

    for mode in modes:
        print(f"\n🎯 Testing {mode.upper()} attack...")

        attacker = RefereeMultiModalPGD(
            model,
            attack_mode=mode,
            eps_audio=0.05,
            eps_video=0.3,
            max_iter=20,
            verbose=True
        )

        try:
            adv_audio, adv_video, attack_info = attacker.generate(
                target_audio, target_video, ref_audio, ref_video, labels_rf
            )

            # Check what changed
            audio_changed = not torch.allclose(adv_audio, target_audio, atol=1e-6)
            video_changed = not torch.allclose(adv_video, target_video, atol=1e-6)

            print(f"  ✅ {mode} attack completed")
            print(f"  Audio modified: {audio_changed}")
            print(f"  Video modified: {video_changed}")
            print(f"  Attack iterations: {len(attack_info['losses'])}")

        except Exception as e:
            print(f"  ❌ {mode} attack failed: {e}")


def demo_comprehensive_evaluation(model: nn.Module, device: str = 'cuda'):
    """Demonstrate comprehensive evaluation pipeline."""
    print("\n" + "=" * 60)
    print("DEMO 3: Comprehensive Evaluation")
    print("=" * 60)

    # Create test data
    target_audio, target_video, ref_audio, ref_video, labels_rf = create_dummy_data(2, device)

    # Create evaluator
    evaluator = RefereeAttackEvaluator(model, device)

    # Run comprehensive evaluation
    results = evaluator.run_comprehensive_evaluation(
        target_audio, target_video, ref_audio, ref_video, labels_rf,
        output_dir="./attack_evaluation_results"
    )

    print("\n📊 Evaluation Summary:")
    print("-" * 40)

    # Show mode comparison results
    if 'mode_results' in results:
        for mode, mode_results in results['mode_results'].items():
            if 'error' not in mode_results:
                success = mode_results['attack_success']
                conf_change = mode_results['confidence_change']
                print(f"{mode:6s}: Success={success}, Δ Confidence={conf_change:.3f}")

    return results


def demo_parameter_tuning(model: nn.Module, device: str = 'cuda'):
    """Demonstrate parameter tuning effects."""
    print("\n" + "=" * 60)
    print("DEMO 4: Parameter Tuning Effects")
    print("=" * 60)

    # Create test data
    target_audio, target_video, ref_audio, ref_video, labels_rf = create_dummy_data(1, device)

    # Test different temporal weights
    temporal_weights = [0.0, 0.5, 1.0]

    print("🔧 Testing temporal regularization weights...")
    for weight in temporal_weights:
        print(f"\n  Testing temporal_weight = {weight}")

        attacker = RefereeMultiModalPGD(
            model,
            temporal_weight=weight,
            max_iter=15,
            verbose=False
        )

        try:
            adv_audio, adv_video, attack_info = attacker.generate(
                target_audio, target_video, ref_audio, ref_video, labels_rf
            )

            # Compute temporal variance as smoothness measure
            def compute_temporal_variance(audio, video):
                audio_var = torch.var(audio[:, :, :, :, 1:] - audio[:, :, :, :, :-1]).item()
                video_var = torch.var(video[:, :, 1:] - video[:, :, :-1]).item()
                return audio_var, video_var

            audio_var, video_var = compute_temporal_variance(adv_audio, adv_video)

            print(f"    Final loss: {attack_info['losses'][-1]:.4f}")
            print(f"    Audio temporal variance: {audio_var:.6f}")
            print(f"    Video temporal variance: {video_var:.6f}")

        except Exception as e:
            print(f"    Error: {e}")


def demo_full_testing_suite(model: nn.Module, device: str = 'cuda'):
    """Demonstrate full testing suite."""
    print("\n" + "=" * 60)
    print("DEMO 5: Full Testing Suite")
    print("=" * 60)

    tester = RefereeAttackTester(model, device)
    test_results = tester.run_all_tests()

    all_passed = all(test_results.values())
    print(f"\n🧪 Overall test status: {'ALL PASSED ✅' if all_passed else 'SOME FAILED ❌'}")

    return test_results


def main():
    """Main demo function."""
    print("🚀 Referee Adversarial Attack Demo")
    print("=" * 60)

    # Configuration
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Load model (you'll need to provide your own config and checkpoint)
    config_path = "./configs/pair_sync.yaml"
    checkpoint_path = None  # Set to your checkpoint path if available

    try:
        model = load_referee_model(config_path, checkpoint_path)
        model = model.to(device)
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        print("Creating dummy model for demo purposes...")

        # Create a minimal dummy model for demo
        class DummyReferee(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(100, 2)

            def forward(self, target_vis, target_aud, ref_vis, ref_aud, **kwargs):
                batch_size = target_vis.shape[0]
                # Return dummy logits
                logits_rf = torch.randn(batch_size, 2, device=target_vis.device)
                logits_id = torch.randn(batch_size, 2, device=target_vis.device)
                return logits_rf, logits_id

        model = DummyReferee().to(device)

    # Run demos
    try:
        # Demo 1: Quick test
        success = demo_quick_test(model, device)
        if not success:
            print("❌ Basic functionality failed. Stopping demos.")
            return

        # Demo 2: Individual attacks
        demo_individual_attacks(model, device)

        # Demo 3: Comprehensive evaluation
        demo_comprehensive_evaluation(model, device)

        # Demo 4: Parameter tuning
        demo_parameter_tuning(model, device)

        # Demo 5: Full testing suite
        demo_full_testing_suite(model, device)

        print("\n" + "=" * 60)
        print("🎉 All demos completed successfully!")
        print("=" * 60)

    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()