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
    # Clear memory first
    if device == 'cuda':
        torch.cuda.empty_cache()

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
            torch.cuda.empty_cache()

            # Fallback to smaller tensors
            target_audio = torch.randn(batch_size, 4, 1, 64, 33, device=device)
            ref_audio = torch.randn(batch_size, 4, 1, 64, 33, device=device)
            target_video = torch.rand(batch_size, 4, 8, 3, 112, 112, device=device)
            ref_video = torch.rand(batch_size, 4, 8, 3, 112, 112, device=device)
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

    # Clean up
    if device == 'cuda':
        torch.cuda.empty_cache()

    return success


def demo_individual_attacks(model: nn.Module, device: str = 'cuda'):
    """Demonstrate individual modality attacks."""
    print("\n" + "=" * 60)
    print("DEMO 2: Individual Modality Attacks")
    print("=" * 60)

    # Create test data
    target_audio, target_video, ref_audio, ref_video, labels_rf = create_dummy_data(1, device)

    # Test each attack mode with aggressive parameters (ART-style)
    modes = ['audio', 'video', 'joint']

    for mode in modes:
        print(f"\n🎯 Testing {mode.upper()} attack...")

        # Aggressive attack parameters based on ART defaults
        attacker = RefereeMultiModalPGD(
            model,
            attack_mode=mode,
            eps_audio=0.3,        # L2 norm bound for audio
            eps_video=8.0,        # L2 norm bound for video (ART default)
            eps_step_audio=0.1,   # ~eps/3
            eps_step_video=2.0,   # ~eps/4
            max_iter=40,          # ART default
            temporal_weight=0.0,  # Disable for maximum attack strength
            early_stop=True,      # Stop when attack succeeds
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
            if attack_info.get('converged'):
                print(f"  ✓ Attack converged early!")

            # Clean up memory
            del adv_audio, adv_video, attack_info, attacker
            if device == 'cuda':
                torch.cuda.empty_cache()

        except Exception as e:
            print(f"  ❌ {mode} attack failed: {e}")
            import traceback
            traceback.print_exc()

    # Final cleanup
    del target_audio, target_video, ref_audio, ref_video, labels_rf
    if device == 'cuda':
        torch.cuda.empty_cache()


def demo_comprehensive_evaluation(model: nn.Module, device: str = 'cuda'):
    """Demonstrate comprehensive evaluation pipeline."""
    print("\n" + "=" * 60)
    print("DEMO 3: Comprehensive Evaluation")
    print("=" * 60)

    # Create test data
    target_audio, target_video, ref_audio, ref_video, labels_rf = create_dummy_data(1, device)

    # Create evaluator
    evaluator = RefereeAttackEvaluator(model, device)

    # Run comprehensive evaluation
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
                    print(f"{mode:6s}: Success={success}, Δ Confidence={conf_change:+.3f}")
                else:
                    print(f"{mode:6s}: Error - {mode_results['error']}")

    except Exception as e:
        print(f"❌ Comprehensive evaluation failed: {e}")
        import traceback
        traceback.print_exc()

    # Cleanup
    del target_audio, target_video, ref_audio, ref_video, labels_rf
    if device == 'cuda':
        torch.cuda.empty_cache()

    return results if 'results' in dir() else {}


def demo_parameter_tuning(model: nn.Module, device: str = 'cuda'):
    """Demonstrate parameter tuning effects."""
    print("\n" + "=" * 60)
    print("DEMO 4: Parameter Tuning Effects")
    print("=" * 60)

    # Create test data
    target_audio, target_video, ref_audio, ref_video, labels_rf = create_dummy_data(1, device)

    # Test different epsilon values
    eps_values = [4.0, 8.0, 16.0]

    print("🔧 Testing different epsilon values (video)...")
    for eps in eps_values:
        print(f"\n  Testing eps_video = {eps}")

        attacker = RefereeMultiModalPGD(
            model,
            attack_mode='video',
            eps_video=eps,
            eps_step_video=eps / 4,
            max_iter=30,
            temporal_weight=0.0,
            early_stop=True,
            verbose=False
        )

        try:
            adv_audio, adv_video, attack_info = attacker.generate(
                target_audio, target_video, ref_audio, ref_video, labels_rf
            )

            # Get confidence scores
            from adversarial_attacks.multimodal_wrapper import create_attack_wrapper
            wrapper = create_attack_wrapper(model, ref_audio, ref_video, labels_rf)
            initial_conf = wrapper.get_confidence(target_audio, target_video)
            final_conf = wrapper.get_confidence(adv_audio, adv_video)

            conf_change = final_conf['rf_real_prob'] - initial_conf['rf_real_prob']
            print(f"    Initial real_prob: {initial_conf['rf_real_prob']:.3f}")
            print(f"    Final real_prob: {final_conf['rf_real_prob']:.3f}")
            print(f"    Confidence change: {conf_change:+.3f}")
            print(f"    Attack success: {final_conf['rf_real_prob'] > 0.5}")

            del adv_audio, adv_video, attack_info, attacker, wrapper

        except Exception as e:
            print(f"    Error: {e}")

        if device == 'cuda':
            torch.cuda.empty_cache()

    # Cleanup
    del target_audio, target_video, ref_audio, ref_video, labels_rf
    if device == 'cuda':
        torch.cuda.empty_cache()


def demo_full_testing_suite(model: nn.Module, device: str = 'cuda'):
    """Demonstrate full testing suite."""
    print("\n" + "=" * 60)
    print("DEMO 5: Full Testing Suite")
    print("=" * 60)

    # Clean memory before running full suite
    if device == 'cuda':
        torch.cuda.empty_cache()

    tester = RefereeAttackTester(model, device)
    test_results = tester.run_all_tests()

    all_passed = all(test_results.values())
    print(f"\n🧪 Overall test status: {'ALL PASSED ✅' if all_passed else 'SOME FAILED ❌'}")

    # Cleanup
    if device == 'cuda':
        torch.cuda.empty_cache()

    return test_results


def main():
    """Main demo function."""
    print("🚀 Referee Adversarial Attack Demo")
    print("=" * 60)

    # Configuration
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Clean GPU memory at start
    if device == 'cuda':
        torch.cuda.empty_cache()

    # Load model (try multiple possible paths)
    config_path = "../configs/pair_sync.yaml"

    # Try different possible paths for the pretrained model
    possible_model_paths = [
        "../model/pretrained/pretrained.pth",
        "../model/pretrained/pretrained.pt",
        "model/pretrained/pretrained.pth",
        "model/pretrained/pretrained.pt",
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
                # More complex classifier for better gradient flow
                self.classifier = nn.Sequential(
                    nn.Linear(2, 256),
                    nn.ReLU(),
                    nn.Linear(256, 128),
                    nn.ReLU(),
                    nn.Linear(128, 64),
                    nn.ReLU(),
                    nn.Linear(64, 2)
                )

            def forward(self, target_vis, target_aud, ref_vis, ref_aud, **kwargs):
                B = target_vis.shape[0]

                # Extract features using global mean
                audio_feat = torch.mean(target_aud, dim=list(range(1, target_aud.ndim)))
                video_feat = torch.mean(target_vis, dim=list(range(1, target_vis.ndim)))

                # Combine and scale
                combined = torch.stack([audio_feat, video_feat], dim=1)
                combined_scaled = combined * 100.0  # Amplify for sensitivity

                # Classification
                logits_rf = self.classifier(combined_scaled)
                logits_id = self.classifier(combined_scaled * 0.8)

                return logits_rf, logits_id

        model = DummyReferee().to(device)

    # Run demos
    try:
        # Demo 1: Quick test
        success = demo_quick_test(model, device)
        if not success:
            print("⚠️  Initial test had issues, but continuing with demos...")

        # Demo 2: Individual attacks
        demo_individual_attacks(model, device)

        # Demo 3: Comprehensive evaluation
        demo_comprehensive_evaluation(model, device)

        # Demo 4: Parameter tuning
        demo_parameter_tuning(model, device)

        # Demo 5: Full testing suite
        demo_full_testing_suite(model, device)

        print("\n" + "=" * 60)
        print("🎉 All demos completed!")
        print("=" * 60)

    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()

    finally:
        # Final memory cleanup
        if device == 'cuda':
            torch.cuda.empty_cache()
            print(f"🧹 Cleaned up GPU memory")


if __name__ == "__main__":
    main()
