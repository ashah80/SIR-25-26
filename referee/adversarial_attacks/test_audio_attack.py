"""
Audio Attack Hyperparameter Testing Script

This script tests various hyperparameter configurations specifically for audio attacks
to find the most effective settings for fooling the Referee model.

Tests different:
- Epsilon values (perturbation budget)
- Step sizes
- Number of iterations
- Momentum settings

Uses REAL samples from the FakeAVCeleb dataset.

Run this to optimize audio attack effectiveness.
"""

import sys
from pathlib import Path
import torch
import torch.nn as nn
from typing import Dict, List, Optional
import time
import argparse

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

from adversarial_attacks.pgd_attack import RefereeMultiModalPGD
from adversarial_attacks.multimodal_wrapper import create_attack_wrapper


def load_real_model(checkpoint_path: Optional[str] = None, device: str = 'cuda'):
    """
    Load the real Referee model.

    Args:
        checkpoint_path: Path to model checkpoint. If None, uses default location.
        device: Device to load model to.

    Returns:
        Loaded Referee model in eval mode.
    """
    from model.referee import Referee
    from omegaconf import OmegaConf

    # Load config
    config_path = PROJECT_ROOT / "configs" / "pair_sync.yaml"
    cfg = OmegaConf.load(config_path)

    # Create model
    model = Referee(cfg.model.params)

    # Load checkpoint
    if checkpoint_path is None:
        checkpoint_path = PROJECT_ROOT / "model" / "pretrained" / "pretrained.pth"

    if Path(checkpoint_path).exists():
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)
        print("Checkpoint loaded successfully!")
    else:
        print(f"Warning: Checkpoint not found at {checkpoint_path}")
        print("Using randomly initialized model for testing.")

    model = model.to(device)
    model.eval()
    return model


def create_dummy_model(device: str = 'cuda'):
    """Create a dummy model for testing when real model is unavailable."""
    class DummyReferee(nn.Module):
        def __init__(self):
            super().__init__()
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
            audio_feat = torch.mean(target_aud, dim=list(range(1, target_aud.ndim)))
            video_feat = torch.mean(target_vis, dim=list(range(1, target_vis.ndim)))
            combined = torch.stack([audio_feat, video_feat], dim=1) * 100.0
            logits_rf = self.classifier(combined)
            logits_id = self.classifier(combined * 0.8)
            return logits_rf, logits_id

    return DummyReferee().to(device)


def load_real_data(device: str = 'cuda', sample_type: str = 'fake'):
    """
    Load real data from the FakeAVCeleb dataset.

    Args:
        device: Device to load tensors to.
        sample_type: 'fake' or 'real' - which type of sample to load.

    Returns:
        target_audio, target_video, ref_audio, ref_video, labels_rf, sample_info
    """
    from adversarial_attacks.real_data_loader import load_real_sample

    return load_real_sample(device=device, sample_type=sample_type)


def create_dummy_data(batch_size: int = 1, device: str = 'cuda'):
    """Create dummy data for testing when real data is unavailable."""
    if device == 'cuda':
        torch.cuda.empty_cache()

    # Referee dimensions: Audio (B, S, 1, F, Ta), Video (B, S, Tv, C, H, W)
    target_audio = torch.randn(batch_size, 8, 1, 128, 66, device=device)
    ref_audio = torch.randn(batch_size, 8, 1, 128, 66, device=device)
    target_video = torch.rand(batch_size, 8, 16, 3, 224, 224, device=device)
    ref_video = torch.rand(batch_size, 8, 16, 3, 224, 224, device=device)
    labels_rf = torch.ones(batch_size, dtype=torch.long, device=device)

    sample_info = {"source": "dummy", "target_path": "dummy", "reference_path": "dummy"}
    return target_audio, target_video, ref_audio, ref_video, labels_rf, sample_info


def test_audio_config(
    model: nn.Module,
    target_audio: torch.Tensor,
    target_video: torch.Tensor,
    ref_audio: torch.Tensor,
    ref_video: torch.Tensor,
    labels_rf: torch.Tensor,
    eps_audio: float,
    eps_step_audio: float,
    max_iter: int,
    device: str,
    config_name: str = ""
) -> Dict:
    """Test a specific audio attack configuration."""

    # Create wrapper for initial/final confidence
    wrapper = create_attack_wrapper(model, ref_audio, ref_video, labels_rf)

    # Get initial confidence
    initial_conf = wrapper.get_confidence(target_audio, target_video)

    # Run attack
    start_time = time.time()

    attacker = RefereeMultiModalPGD(
        model,
        attack_mode='audio',
        eps_audio=eps_audio,
        eps_step_audio=eps_step_audio,
        max_iter=max_iter,
        temporal_weight=0.0,
        early_stop=True,
        verbose=False
    )

    adv_audio, adv_video, attack_info = attacker.generate(
        target_audio, target_video, ref_audio, ref_video, labels_rf
    )

    elapsed_time = time.time() - start_time

    # Get final confidence
    final_conf = wrapper.get_confidence(adv_audio, adv_video)

    # Calculate metrics
    confidence_change = final_conf['rf_real_prob'] - initial_conf['rf_real_prob']
    attack_success = final_conf['rf_real_prob'] > 0.5
    iterations_used = len(attack_info['losses'])

    # Compute perturbation norm
    delta = adv_audio - target_audio
    perturbation_norm = torch.norm(delta.view(delta.size(0), -1), dim=1).mean().item()

    # Clean up
    del attacker, adv_audio, adv_video, attack_info, wrapper
    if device == 'cuda':
        torch.cuda.empty_cache()

    return {
        'config_name': config_name,
        'eps_audio': eps_audio,
        'eps_step_audio': eps_step_audio,
        'max_iter': max_iter,
        'initial_real_prob': initial_conf['rf_real_prob'],
        'final_real_prob': final_conf['rf_real_prob'],
        'confidence_change': confidence_change,
        'attack_success': attack_success,
        'iterations_used': iterations_used,
        'elapsed_time': elapsed_time,
        'perturbation_norm': perturbation_norm,
    }


def run_audio_hyperparameter_sweep(
    use_real_data: bool = True,
    use_real_model: bool = True,
    checkpoint_path: Optional[str] = None,
    num_samples: int = 1,
):
    """
    Run comprehensive audio hyperparameter sweep.

    Args:
        use_real_data: Whether to use real dataset samples.
        use_real_model: Whether to use the real Referee model.
        checkpoint_path: Path to model checkpoint.
        num_samples: Number of samples to test each config on.
    """
    print("=" * 70)
    print("AUDIO ATTACK HYPERPARAMETER TESTING")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    print(f"Using real data: {use_real_data}")
    print(f"Using real model: {use_real_model}")
    print()

    # Load model
    print("Loading model...")
    if use_real_model:
        try:
            model = load_real_model(checkpoint_path, device)
        except Exception as e:
            print(f"Failed to load real model: {e}")
            print("Falling back to dummy model.")
            model = create_dummy_model(device)
    else:
        model = create_dummy_model(device)
    model.eval()

    # Load data
    print("Loading data...")
    if use_real_data:
        try:
            target_audio, target_video, ref_audio, ref_video, labels_rf, sample_info = load_real_data(device)
            print(f"Loaded sample: {sample_info.get('target_path', 'unknown')}")
        except Exception as e:
            print(f"Failed to load real data: {e}")
            print("Falling back to dummy data.")
            target_audio, target_video, ref_audio, ref_video, labels_rf, sample_info = create_dummy_data(1, device)
    else:
        target_audio, target_video, ref_audio, ref_video, labels_rf, sample_info = create_dummy_data(1, device)

    print(f"Audio shape: {target_audio.shape}")
    print(f"Video shape: {target_video.shape}")
    print()

    # Define test configurations
    test_configs = [
        # Baseline (current settings)
        {"name": "Baseline (0.3)", "eps": 0.3, "step": 0.1, "iter": 40},

        # Epsilon variations (moderate step size)
        {"name": "Eps=1.0", "eps": 1.0, "step": 0.33, "iter": 40},
        {"name": "Eps=2.0", "eps": 2.0, "step": 0.66, "iter": 40},
        {"name": "Eps=3.0", "eps": 3.0, "step": 1.0, "iter": 40},
        {"name": "Eps=5.0", "eps": 5.0, "step": 1.66, "iter": 40},
        {"name": "Eps=10.0", "eps": 10.0, "step": 3.33, "iter": 40},

        # Step size variations (fixed eps=3.0)
        {"name": "Eps=3.0, SmallStep", "eps": 3.0, "step": 0.5, "iter": 40},
        {"name": "Eps=3.0, LargeStep", "eps": 3.0, "step": 2.0, "iter": 40},

        # Iteration variations (fixed eps=3.0, step=1.0)
        {"name": "Eps=3.0, Iter=100", "eps": 3.0, "step": 1.0, "iter": 100},
        {"name": "Eps=3.0, Iter=200", "eps": 3.0, "step": 1.0, "iter": 200},
    ]

    results = []

    print("Running tests...")
    print("-" * 70)

    for config in test_configs:
        print(f"Testing: {config['name']:<25} ", end="", flush=True)

        result = test_audio_config(
            model,
            target_audio.clone(),
            target_video.clone(),
            ref_audio,
            ref_video,
            labels_rf,
            eps_audio=config['eps'],
            eps_step_audio=config['step'],
            max_iter=config['iter'],
            device=device,
            config_name=config['name']
        )

        results.append(result)

        # Print compact result
        success_icon = "SUCCESS" if result['attack_success'] else "FAILED"
        print(f"{success_icon} | Change={result['confidence_change']:+.3f} | "
              f"Time={result['elapsed_time']:.1f}s | "
              f"Iters={result['iterations_used']}")

    print("-" * 70)
    print()

    # Analysis and recommendations
    print("=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    # Sort by confidence change
    results_sorted = sorted(results, key=lambda x: x['confidence_change'], reverse=True)

    print("\nTop 5 Configurations (by confidence change):\n")
    print(f"{'Rank':<6} {'Config':<25} {'Change':<12} {'Success':<10} {'Time':<10}")
    print("-" * 70)

    for i, result in enumerate(results_sorted[:5], 1):
        success_str = "Yes" if result['attack_success'] else "No"
        print(f"{i:<6} {result['config_name']:<25} {result['confidence_change']:+.4f}{' '*6} "
              f"{success_str:<10} {result['elapsed_time']:.1f}s")

    print("\n" + "=" * 70)

    # Best configuration
    best_result = results_sorted[0]
    print("RECOMMENDED CONFIGURATION:")
    print("-" * 70)
    print(f"  Name:               {best_result['config_name']}")
    print(f"  eps_audio:          {best_result['eps_audio']}")
    print(f"  eps_step_audio:     {best_result['eps_step_audio']}")
    print(f"  max_iter:           {best_result['max_iter']}")
    print(f"  Confidence change:  {best_result['confidence_change']:+.4f}")
    print(f"  Attack success:     {'Yes' if best_result['attack_success'] else 'No'}")
    print(f"  Time:               {best_result['elapsed_time']:.1f}s")
    print(f"  Iterations used:    {best_result['iterations_used']}")
    print("=" * 70)

    # Sample info
    print("\nSAMPLE INFO:")
    print("-" * 70)
    for k, v in sample_info.items():
        print(f"  {k}: {v}")
    print("=" * 70)

    # Cleanup
    del target_audio, target_video, ref_audio, ref_video, labels_rf, model
    if device == 'cuda':
        torch.cuda.empty_cache()

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test audio attack hyperparameters')
    parser.add_argument('--dummy-data', action='store_true',
                       help='Use dummy data instead of real dataset')
    parser.add_argument('--dummy-model', action='store_true',
                       help='Use dummy model instead of real Referee model')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Path to model checkpoint')

    args = parser.parse_args()

    results = run_audio_hyperparameter_sweep(
        use_real_data=not args.dummy_data,
        use_real_model=not args.dummy_model,
        checkpoint_path=args.checkpoint,
    )

    print("\nAudio hyperparameter testing complete!")
    print("You can now update pgd_attack.py or demo.py with the recommended configuration.")
