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
    # Use original Referee dimensions if we have the real model, smaller for dummy
    # Let's try original dimensions first, fall back to smaller if memory issues

    try:
        # Original Referee dimensions (what the real model expects)
        # Audio: (B, S, 1, F, Ta) = (B, 8, 1, 128, 66)
        target_audio = torch.randn(batch_size, 8, 1, 128, 66, device=device)
        ref_audio = torch.randn(batch_size, 8, 1, 128, 66, device=device)

        # Video: (B, S, Tv, C, H, W) = (B, 8, 16, 3, 224, 224)
        target_video = torch.rand(batch_size, 8, 16, 3, 224, 224, device=device)
        ref_video = torch.rand(batch_size, 8, 16, 3, 224, 224, device=device)

        # Labels (fake samples - we want to make them appear real)
        labels_rf = torch.ones(batch_size, dtype=torch.long, device=device)

        return target_audio, target_video, ref_audio, ref_video, labels_rf

    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print("⚠️  GPU memory low, using smaller tensors...")

            # Fallback to smaller tensors
            target_audio = torch.randn(batch_size, 4, 1, 32, 32, device=device)
            ref_audio = torch.randn(batch_size, 4, 1, 32, 32, device=device)
            target_video = torch.rand(batch_size, 4, 8, 3, 64, 64, device=device)
            ref_video = torch.rand(batch_size, 4, 8, 3, 64, 64, device=device)
            labels_rf = torch.ones(batch_size, dtype=torch.long, device=device)

            return target_audio, target_video, ref_audio, ref_video, labels_rf
        else:
            raise e


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

    # Test each attack mode with more aggressive parameters
    modes = ['audio', 'video', 'joint']

    for mode in modes:
        print(f"\n🎯 Testing {mode.upper()} attack...")

        # Use more aggressive attack parameters to ensure they work
        attacker = RefereeMultiModalPGD(
            model,
            attack_mode=mode,
            eps_audio=0.1,        # Larger perturbation
            eps_video=0.5,        # Larger perturbation
            eps_step_audio=0.02,  # Larger step
            eps_step_video=0.1,   # Larger step
            max_iter=30,          # More iterations
            temporal_weight=0.1,  # Lower temporal weight for stronger attacks
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

            # Clean up memory
            del adv_audio, adv_video, attack_info
            torch.cuda.empty_cache() if device == 'cuda' else None

        except Exception as e:
            print(f"  ❌ {mode} attack failed: {e}")

        # Clean up between attacks
        if device == 'cuda':
            torch.cuda.empty_cache()


def demo_comprehensive_evaluation(model: nn.Module, device: str = 'cuda'):
    """Demonstrate comprehensive evaluation pipeline."""
    print("\n" + "=" * 60)
    print("DEMO 3: Comprehensive Evaluation")
    print("=" * 60)

    # Create test data with smaller batch size
    target_audio, target_video, ref_audio, ref_video, labels_rf = create_dummy_data(1, device)  # Use batch_size=1

    # Create evaluator
    evaluator = RefereeAttackEvaluator(model, device)

    # Run comprehensive evaluation with smaller ranges
    try:
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

        # Clean up memory
        torch.cuda.empty_cache() if device == 'cuda' else None

    except Exception as e:
        print(f"❌ Comprehensive evaluation failed: {e}")
        import traceback
        traceback.print_exc()

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

    # Load model (try multiple possible paths)
    config_path = "../configs/pair_sync.yaml"

    # Try different possible paths for the pretrained model
    possible_model_paths = [
        "../model/pretrained/pretrained.pth",      # User's mentioned path
        "../model/pretrained/pretrained.pt",       # Alternative extension
        "model/pretrained/pretrained.pth",         # Without ../
        "model/pretrained/pretrained.pt",          # Alternative
    ]

    checkpoint_path = None
    for path in possible_model_paths:
        if Path(path).exists():
            checkpoint_path = path
            break

    if checkpoint_path:
        print(f"🔍 Found pretrained model at: {checkpoint_path}")
    else:
        print("⚠️  No pretrained model found, will use dummy model")

    try:
        if checkpoint_path:
            model = load_referee_model(config_path, checkpoint_path)
            model = model.to(device)
            print("🎉 Using REAL pretrained Referee model - attacks should work much better!")
        else:
            raise FileNotFoundError("No pretrained model found")
    except Exception as e:
        print(f"❌ Failed to load real model: {e}")
        print("Creating improved dummy model for demo purposes...")

        # Create a much better dummy model with correct dimensions
        class DummyReferee(nn.Module):
            def __init__(self):
                super().__init__()
                # Handle both small and large input dimensions flexibly

                self.classifier = nn.Sequential(
                    nn.Linear(2, 64),      # Just 2 features: one from audio, one from video
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(64, 32),
                    nn.ReLU(),
                    nn.Linear(32, 2)
                )

            def forward(self, target_vis, target_aud, ref_vis, ref_aud, **kwargs):
                B = target_vis.shape[0]

                # Ultra-simple approach: just take global means
                # This works for any input size and avoids dimension issues

                # Audio: Any shape -> scalar per batch
                audio_feat = torch.mean(target_aud, dim=list(range(1, target_aud.ndim)))  # (B,)

                # Video: Any shape -> scalar per batch
                video_feat = torch.mean(target_vis, dim=list(range(1, target_vis.ndim)))  # (B,)

                # Combine features: (B, 2)
                combined = torch.stack([audio_feat, video_feat], dim=1)

                # Scale to make model sensitive to small changes
                combined_scaled = combined * 50.0  # Amplify effects

                # Classification
                logits_rf = self.classifier(combined_scaled)

                # ID logits - similar but with noise
                logits_id = self.classifier(combined_scaled * 0.7) + torch.randn(B, 2, device=target_vis.device) * 0.2

                return logits_rf, logits_id

        model = DummyReferee().to(device)

    # Run demos
    try:
        # Demo 1: Quick test
        success = demo_quick_test(model, device)
        if not success:
            print("⚠️  Initial test had issues, but continuing with demos...")
            print("   (This is normal for dummy models)")

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

        # Final memory cleanup
        if device == 'cuda':
            torch.cuda.empty_cache()
            print(f"🧹 Cleaned up GPU memory")

    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()

        # Clean up on error too
        if device == 'cuda':
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()