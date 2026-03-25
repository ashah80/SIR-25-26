"""
Video adversarial attacks for the Referee deepfake detection model.

Includes:
- FlickeringAttack: Temporally smooth perturbations using learnable basis patterns
- ART-based PGD attack with L2 norm constraints

Usage:
    python improved_video_attack.py --attack flickering --num-samples 3
    python improved_video_attack.py --attack art-improved --num-samples 3
"""

import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Dict, Any, List
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import shutil
import argparse
import time

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

try:
    import cv2
    HAVE_CV2 = True
except ImportError:
    HAVE_CV2 = False

try:
    from art.estimators.classification import PyTorchClassifier
    from art.attacks.evasion import ProjectedGradientDescent
    HAVE_ART = True
except ImportError:
    HAVE_ART = False


# Video Saving

def save_video_non_overlapping(
    video_tensor: torch.Tensor,
    save_path: Path,
    original_fps: int = 25,
    segment_overlap: float = 0.5,
    mean: Tuple[float, ...] = (0.5, 0.5, 0.5),
    std: Tuple[float, ...] = (0.5, 0.5, 0.5)
):
    """
    Save video tensor as MP4.

    The model uses 8 segments with 50% overlap. We only use the non-overlapping
    portion of each segment to avoid frame repetition.

    Args:
        video_tensor: (B, S, T, C, H, W) or (S, T, C, H, W) normalized video
        save_path: Output path for MP4 file
        original_fps: Frame rate for output video
        segment_overlap: Overlap ratio between segments (0.5 = 50%)
        mean, std: Normalization parameters to reverse
    """
    if not HAVE_CV2:
        print("  Skipping video save (OpenCV not available)")
        return

    # Denormalize
    mean_t = torch.tensor(mean, device=video_tensor.device).view(1, 1, 1, 3, 1, 1)
    std_t = torch.tensor(std, device=video_tensor.device).view(1, 1, 1, 3, 1, 1)

    if video_tensor.dim() == 5:
        video_tensor = video_tensor.unsqueeze(0)
        mean_t = mean_t.squeeze(0)
        std_t = std_t.squeeze(0)

    video_denorm = video_tensor * std_t + mean_t
    video_denorm = torch.clamp(video_denorm, 0, 1)
    video_denorm = video_denorm[0]  # Remove batch dim: (S, T, C, H, W)

    S, T, C, H, W = video_denorm.shape

    # Only use first half of each segment (non-overlapping portion), except last segment
    non_overlap_frames = int(T * (1 - segment_overlap))
    if non_overlap_frames < 1:
        non_overlap_frames = 1

    frames = []
    for s in range(S):
        if s < S - 1:
            segment_frames = video_denorm[s, :non_overlap_frames]
        else:
            segment_frames = video_denorm[s]

        for t in range(segment_frames.shape[0]):
            frame = segment_frames[t].permute(1, 2, 0).cpu().numpy()
            frame = (frame * 255).astype(np.uint8)
            frame = frame[:, :, ::-1]  # RGB to BGR for OpenCV
            frames.append(frame)

    # Write video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(save_path), fourcc, original_fps, (W, H))
    for frame in frames:
        out.write(frame)
    out.release()

    print(f"  Saved video ({len(frames)} frames): {save_path}")


# Flickering Attack

class FlickeringAttack:
    """
    Video attack using temporally smooth, learnable basis patterns. Inspired by ART.

    Creates perturbations that:
    - Flicker at specific frequencies (often imperceptible to humans)
    - Use smooth spatial patterns (not pixel-level noise)
    - Are designed to survive physical capture (display -> camera)
    """

    def __init__(
        self,
        model: nn.Module,
        eps: float = 0.1,
        flicker_freq: float = 5.0,
        spatial_freq: int = 4,
        num_basis: int = 8,
        num_iterations: int = 50,
        step_size: float = 0.01,
        smoothness_weight: float = 1.0,
        targeted: bool = False,
        device: str = 'cuda'
    ):
        """
        Args:
            model: Target Referee model
            eps: Maximum perturbation magnitude
            flicker_freq: Temporal flicker frequency in Hz
            spatial_freq: Spatial pattern frequency (lower = larger patterns)
            num_basis: Number of learnable basis patterns
            num_iterations: Optimization iterations
            step_size: Learning rate for optimizer
            smoothness_weight: Weight for temporal smoothness regularization
            targeted: Whether to do targeted attack
            device: cuda or cpu
        """
        self.model = model
        self.eps = eps
        self.flicker_freq = flicker_freq
        self.spatial_freq = spatial_freq
        self.num_basis = num_basis
        self.num_iterations = num_iterations
        self.step_size = step_size
        self.smoothness_weight = smoothness_weight
        self.targeted = targeted
        self.device = device

    def _create_learnable_flicker(self, shape: Tuple[int, ...]) -> List[torch.Tensor]:
        """
        Create learnable parameters for the flickering perturbation.

        Args:
            shape: (S, T, C, H, W) video shape

        Returns:
            List of learnable parameters [basis_patterns, temporal_coeffs]
        """
        S, T, C, H, W = shape

        # Learnable spatial basis patterns
        basis_size = max(W // self.spatial_freq, 16)
        basis_patterns = torch.randn(
            self.num_basis, C, basis_size, basis_size,
            device=self.device
        ) * 0.01
        basis_patterns = basis_patterns.detach().requires_grad_(True)

        # Learnable temporal coefficients for each basis
        temporal_coeffs = torch.randn(
            self.num_basis, S * T,
            device=self.device
        ) * 0.1
        temporal_coeffs = temporal_coeffs.detach().requires_grad_(True)

        return [basis_patterns, temporal_coeffs]

    def _build_perturbation(self, params: List[torch.Tensor], shape: Tuple[int, ...]) -> torch.Tensor:
        """Build full perturbation tensor from learnable parameters."""
        S, T, C, H, W = shape
        basis_patterns, temporal_coeffs = params

        # Upsample basis patterns to full resolution
        tiled_basis = F.interpolate(
            basis_patterns,
            size=(H, W),
            mode='bilinear',
            align_corners=False
        )

        # Build perturbation frame by frame
        perturbation = torch.zeros(S, T, C, H, W, device=self.device)
        for s in range(S):
            for t in range(T):
                frame_idx = s * T + t
                frame_pert = torch.zeros(C, H, W, device=self.device)
                for b in range(self.num_basis):
                    coeff = torch.tanh(temporal_coeffs[b, frame_idx])
                    frame_pert = frame_pert + coeff * tiled_basis[b]
                perturbation[s, t] = frame_pert

        return perturbation

    def attack(
        self,
        target_video: torch.Tensor,
        target_audio: torch.Tensor,
        ref_video: torch.Tensor,
        ref_audio: torch.Tensor,
        labels: torch.Tensor,
        verbose: bool = True
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Run the flickering attack.

        Args:
            target_video: (1, S, T, C, H, W) input video
            target_audio: (1, S, 1, F, Ta) input audio (unchanged)
            ref_video: (1, S, T, C, H, W) reference video
            ref_audio: (1, S, 1, F, Ta) reference audio
            labels: (1,) ground truth label
            verbose: Print progress

        Returns:
            adv_video: Adversarial video tensor
            info: Dictionary with attack statistics
        """
        self.model.eval()

        # Get initial prediction
        with torch.no_grad():
            logits = self.model(target_video, target_audio, ref_video, ref_audio)[0]
            probs = F.softmax(logits, dim=1)
            orig_real_prob = probs[0, 0].item()
            orig_fake_prob = probs[0, 1].item()

        if verbose:
            print(f"Initial: Real={orig_real_prob:.4f}, Fake={orig_fake_prob:.4f}")

        # Initialize learnable parameters
        shape = target_video.shape[1:]  # (S, T, C, H, W)
        params = self._create_learnable_flicker(shape)
        optimizer = torch.optim.Adam(params, lr=self.step_size)

        best_adv = target_video.clone()
        best_real_prob = orig_real_prob
        start_time = time.time()

        for i in range(self.num_iterations):
            optimizer.zero_grad()

            # Build perturbation and apply
            perturbation = self._build_perturbation(params, shape)
            perturbation = perturbation.unsqueeze(0)
            perturbation = torch.clamp(perturbation, -self.eps, self.eps)

            adv_video = target_video + perturbation
            adv_video = torch.clamp(adv_video, -1.0, 1.0)

            # Forward pass
            logits = self.model(adv_video, target_audio, ref_video, ref_audio)[0]
            probs = F.softmax(logits, dim=1)

            # Loss: maximize real probability
            if self.targeted:
                loss = F.cross_entropy(logits, torch.zeros_like(labels))
            else:
                loss = -F.cross_entropy(logits, labels)

            # Temporal smoothness regularization
            if perturbation.shape[2] > 1:
                temporal_diff = perturbation[:, :, 1:] - perturbation[:, :, :-1]
                smoothness_loss = torch.mean(temporal_diff ** 2) * self.smoothness_weight
                loss = loss + smoothness_loss

            loss.backward()
            optimizer.step()

            # Track best result
            current_real_prob = probs[0, 0].item()
            if current_real_prob > best_real_prob:
                best_real_prob = current_real_prob
                best_adv = adv_video.detach().clone()

            if verbose and (i % 10 == 0 or i == self.num_iterations - 1):
                print(f"Iter {i:3d}: Loss={loss.item():.4f}, Real={current_real_prob:.4f}")

            # Early stopping
            if current_real_prob > 0.95:
                if verbose:
                    print(f"  Early stop at iter {i}!")
                break

        attack_time = time.time() - start_time

        # Final evaluation
        with torch.no_grad():
            logits = self.model(best_adv, target_audio, ref_video, ref_audio)[0]
            probs = F.softmax(logits, dim=1)
            final_real_prob = probs[0, 0].item()
            final_fake_prob = probs[0, 1].item()

        # Compute perturbation stats
        final_pert = best_adv - target_video
        l2_norm = torch.norm(final_pert).item()
        linf_norm = torch.max(torch.abs(final_pert)).item()

        info = {
            'method': 'FlickeringAttack',
            'original_real_prob': orig_real_prob,
            'original_fake_prob': orig_fake_prob,
            'adversarial_real_prob': final_real_prob,
            'adversarial_fake_prob': final_fake_prob,
            'confidence_change': final_real_prob - orig_real_prob,
            'attack_success': final_real_prob > 0.5,
            'perturbation_l2_norm': l2_norm,
            'perturbation_linf_norm': linf_norm,
            'eps': self.eps,
            'flicker_freq': self.flicker_freq,
            'spatial_freq': self.spatial_freq,
            'num_basis': self.num_basis,
            'smoothness_weight': self.smoothness_weight,
            'num_iterations': self.num_iterations,
            'attack_time_seconds': attack_time
        }

        if verbose:
            print(f"Final: Real={final_real_prob:.4f}, Fake={final_fake_prob:.4f}")
            print(f"Confidence change: {info['confidence_change']:+.4f}")

        return best_adv, info


# =============================================================================
# ART-based Attack
# =============================================================================

class TemporallyConsistentARTWrapper(nn.Module):
    """Wrapper to make Referee compatible with ART's interface."""

    def __init__(
        self,
        referee_model: nn.Module,
        target_audio: torch.Tensor,
        ref_video: torch.Tensor,
        ref_audio: torch.Tensor,
        temporal_weight: float = 0.1,
        device: str = 'cuda'
    ):
        super().__init__()
        self.referee = referee_model
        self.device = device
        self.temporal_weight = temporal_weight

        self.register_buffer('target_audio', target_audio.to(device))
        self.register_buffer('ref_video', ref_video.to(device))
        self.register_buffer('ref_audio', ref_audio.to(device))

        self.video_shape = (8, 16, 3, 224, 224)
        self.flattened_size = int(np.prod(self.video_shape))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        video = x.view(batch_size, *self.video_shape)

        if batch_size != 1:
            target_audio = self.target_audio.expand(batch_size, -1, -1, -1, -1)
            ref_video = self.ref_video.expand(batch_size, -1, -1, -1, -1, -1)
            ref_audio = self.ref_audio.expand(batch_size, -1, -1, -1, -1)
        else:
            target_audio = self.target_audio
            ref_video = self.ref_video
            ref_audio = self.ref_audio

        logits_rf, _ = self.referee(
            target_vis=video,
            target_aud=target_audio,
            ref_vis=ref_video,
            ref_aud=ref_audio
        )
        return logits_rf


def run_improved_art_attack(
    model: nn.Module,
    target_video: torch.Tensor,
    target_audio: torch.Tensor,
    ref_video: torch.Tensor,
    ref_audio: torch.Tensor,
    labels: torch.Tensor,
    eps: float = 0.1,
    max_iter: int = 30,
    device: str = 'cuda',
    verbose: bool = True
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """
    Run ART PGD attack with L2 norm.

    Note: For L2 norm with video (~19M elements), per-pixel perturbation is tiny.
    We auto-scale epsilon to get visible effect.
    """
    if not HAVE_ART:
        raise ImportError("ART not installed")

    wrapper = TemporallyConsistentARTWrapper(
        referee_model=model,
        target_audio=target_audio,
        ref_video=ref_video,
        ref_audio=ref_audio,
        device=device
    )
    wrapper.eval()

    # Scale epsilon for high-dimensional video
    num_elements = wrapper.flattened_size
    eps_scaled = eps * np.sqrt(num_elements) * 0.1

    if verbose:
        print(f"L2 epsilon scaling: {eps:.2f} -> {eps_scaled:.1f} (for {num_elements:,} elements)")

    # Get initial prediction
    with torch.no_grad():
        logits = model(target_video, target_audio, ref_video, ref_audio)[0]
        probs = F.softmax(logits, dim=1)
        orig_real_prob = probs[0, 0].item()
        orig_fake_prob = probs[0, 1].item()

    if verbose:
        print(f"Initial: Real={orig_real_prob:.4f}, Fake={orig_fake_prob:.4f}")

    # Create ART classifier
    classifier = PyTorchClassifier(
        model=wrapper,
        loss=nn.CrossEntropyLoss(),
        input_shape=(wrapper.flattened_size,),
        nb_classes=2,
        clip_values=(-1.0, 1.0),
        device_type='gpu' if device == 'cuda' else 'cpu'
    )

    # Create PGD attack with L2 norm
    attack = ProjectedGradientDescent(
        estimator=classifier,
        norm=2,
        eps=eps_scaled,
        eps_step=eps_scaled / 10,
        max_iter=max_iter,
        targeted=False,
        num_random_init=0,
        batch_size=1,
        verbose=False
    )

    video_flat = target_video.reshape(1, -1).cpu().numpy()
    labels_np = labels.cpu().numpy()

    if verbose:
        print(f"Running ART attack (L2 norm, eps_scaled={eps_scaled:.1f}, iter={max_iter})...")

    start_time = time.time()
    adv_video_flat = attack.generate(x=video_flat, y=labels_np)
    attack_time = time.time() - start_time

    adv_video = torch.from_numpy(adv_video_flat).float().to(device)
    adv_video = adv_video.reshape(target_video.shape)

    # Final evaluation
    with torch.no_grad():
        logits = model(adv_video, target_audio, ref_video, ref_audio)[0]
        probs = F.softmax(logits, dim=1)
        final_real_prob = probs[0, 0].item()
        final_fake_prob = probs[0, 1].item()

    pert = adv_video - target_video
    l2_norm = torch.norm(pert).item()
    linf_norm = torch.max(torch.abs(pert)).item()

    info = {
        'method': 'ART_PGD_L2_Improved',
        'original_real_prob': orig_real_prob,
        'original_fake_prob': orig_fake_prob,
        'adversarial_real_prob': final_real_prob,
        'adversarial_fake_prob': final_fake_prob,
        'confidence_change': final_real_prob - orig_real_prob,
        'attack_success': final_real_prob > 0.5,
        'perturbation_l2_norm': l2_norm,
        'perturbation_linf_norm': linf_norm,
        'eps': eps,
        'max_iter': max_iter,
        'attack_time_seconds': attack_time
    }

    if verbose:
        print(f"Final: Real={final_real_prob:.4f}, Fake={final_fake_prob:.4f}")
        print(f"Confidence change: {info['confidence_change']:+.4f}")
        print(f"Time: {attack_time:.2f}s")

    return adv_video, info


# Model Loading

def load_model(device: str = 'cuda'):
    """Load the Referee model with pretrained weights."""
    from model.referee import Referee
    from omegaconf import OmegaConf

    config_path = PROJECT_ROOT / "configs" / "pair_sync.yaml"
    cfg = OmegaConf.load(config_path)
    model = Referee(cfg)

    checkpoint_path = PROJECT_ROOT / "model" / "pretrained" / "pretrained.pth"
    if checkpoint_path.exists():
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint

        # Remove 'module.' prefix if present
        new_state_dict = {(k[7:] if k.startswith('module.') else k): v
                         for k, v in state_dict.items()}
        model.load_state_dict(new_state_dict, strict=False)
        print("Checkpoint loaded!")

    model = model.to(device)
    model.eval()
    return model


def load_sample(dataset, idx: int, device: str):
    """Load a single sample from the dataset."""
    sample = dataset[idx]
    return (
        sample['target_audio'].unsqueeze(0).to(device),
        sample['target_video'].unsqueeze(0).to(device),
        sample['reference_audio'].unsqueeze(0).to(device),
        sample['reference_video'].unsqueeze(0).to(device),
        torch.tensor([sample['fake_label']], device=device),
        sample['sample_info']
    )


# Visualization

def save_comparison(orig_video, adv_video, save_path: Path, num_frames: int = 8):
    """Save a visual comparison of original vs adversarial frames."""
    mean = torch.tensor([0.5, 0.5, 0.5], device=orig_video.device).view(1, 1, 1, 3, 1, 1)
    std = torch.tensor([0.5, 0.5, 0.5], device=orig_video.device).view(1, 1, 1, 3, 1, 1)

    orig = torch.clamp(orig_video * std + mean, 0, 1)[0]
    adv = torch.clamp(adv_video * std + mean, 0, 1)[0]

    S, T, C, H, W = orig.shape
    total = S * T
    indices = np.linspace(0, total - 1, num_frames, dtype=int)

    orig_flat = orig.reshape(total, C, H, W)
    adv_flat = adv.reshape(total, C, H, W)

    fig, axes = plt.subplots(3, num_frames, figsize=(num_frames * 2, 6))

    for i, idx in enumerate(indices):
        orig_f = orig_flat[idx].permute(1, 2, 0).cpu().numpy()
        adv_f = adv_flat[idx].permute(1, 2, 0).cpu().numpy()
        diff = np.abs(adv_f - orig_f)

        axes[0, i].imshow(orig_f)
        axes[0, i].axis('off')
        if i == 0:
            axes[0, i].set_ylabel('Original', fontsize=10)

        axes[1, i].imshow(adv_f)
        axes[1, i].axis('off')
        if i == 0:
            axes[1, i].set_ylabel('Adversarial', fontsize=10)

        axes[2, i].imshow(np.clip(diff * 10, 0, 1))
        axes[2, i].axis('off')
        if i == 0:
            axes[2, i].set_ylabel('Diff (10x)', fontsize=10)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved comparison: {save_path}")


def save_stats(info: Dict, save_path: Path):
    """Save attack statistics to a text file."""
    with open(save_path, 'w') as f:
        f.write("Attack Results\n")
        f.write("=" * 40 + "\n\n")
        for k, v in info.items():
            if isinstance(v, float):
                f.write(f"{k}: {v:.6f}\n")
            else:
                f.write(f"{k}: {v}\n")
    print(f"  Saved stats: {save_path}")


# Main

def main():
    parser = argparse.ArgumentParser(description="Video adversarial attacks for Referee")
    parser.add_argument("--attack", type=str, default="flickering",
                        choices=["flickering", "art-improved"],
                        help="Attack type to run")
    parser.add_argument("--num-samples", type=int, default=3,
                        help="Number of samples to attack")
    parser.add_argument("--output-dir", type=str, default="./art-testing",
                        help="Directory to save results")
    parser.add_argument("--eps", type=float, default=0.05,
                        help="Perturbation budget")
    parser.add_argument("--max-iter", type=int, default=200,
                        help="Maximum iterations")
    parser.add_argument("--flicker-freq", type=float, default=2.5,
                        help="Temporal flicker frequency in Hz")
    parser.add_argument("--spatial-freq", type=int, default=8,
                        help="Spatial pattern frequency (lower = larger patterns)")
    parser.add_argument("--num-basis", type=int, default=4,
                        help="Number of basis patterns")
    parser.add_argument("--smoothness-weight", type=float, default=2.0,
                        help="Temporal smoothness regularization weight")
    parser.add_argument("--step-size", type=float, default=0.05,
                        help="Optimizer learning rate")
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()

    print("=" * 70)
    print(f"Video Attack: {args.attack}")
    print("=" * 70)

    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("Loading model...")
    model = load_model(args.device)

    print("Loading dataset...")
    from adversarial_attacks.real_data_loader import AdversarialTestDataset
    dataset = AdversarialTestDataset(device=args.device)

    # Get fake samples to attack
    fake_indices = [i for i, s in enumerate(dataset.samples) if s.get('fake_label', 0) == 1]
    test_indices = fake_indices[:args.num_samples]

    results = []

    for i, sample_idx in enumerate(test_indices):
        print()
        print("=" * 70)
        print(f"Sample {i+1}/{args.num_samples}")
        print("=" * 70)

        sample_out = output_path / f"sample_{i+1}"
        sample_out.mkdir(exist_ok=True)

        target_audio, target_video, ref_audio, ref_video, labels, info = \
            load_sample(dataset, sample_idx, args.device)

        print(f"Source: {info.get('target_path', 'unknown')}")

        try:
            if args.attack == "flickering":
                attack = FlickeringAttack(
                    model=model,
                    eps=args.eps,
                    flicker_freq=args.flicker_freq,
                    spatial_freq=args.spatial_freq,
                    num_basis=args.num_basis,
                    num_iterations=args.max_iter,
                    step_size=args.step_size,
                    smoothness_weight=args.smoothness_weight,
                    device=args.device
                )
                adv_video, attack_info = attack.attack(
                    target_video, target_audio, ref_video, ref_audio, labels
                )
            else:  # art-improved
                adv_video, attack_info = run_improved_art_attack(
                    model, target_video, target_audio, ref_video, ref_audio, labels,
                    eps=args.eps, max_iter=args.max_iter, device=args.device
                )

            results.append(attack_info)

            # Save outputs
            print(f"\nSaving to {sample_out}...")

            if 'target_path' in info and info['target_path'] and Path(info['target_path']).exists():
                shutil.copy2(info['target_path'], sample_out / "original_video_file.mp4")
                print(f"  Copied original: original_video_file.mp4")

            save_video_non_overlapping(adv_video, sample_out / "adversarial_video.mp4")
            save_video_non_overlapping(target_video, sample_out / "original_processed.mp4")
            save_comparison(target_video, adv_video, sample_out / "comparison.png")
            save_stats(attack_info, sample_out / "stats.txt")

        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
            results.append({'error': str(e), 'sample_index': sample_idx})

    # Print summary
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)

    valid = [r for r in results if 'confidence_change' in r]
    if valid:
        avg_change = np.mean([r['confidence_change'] for r in valid])
        successes = sum(1 for r in valid if r.get('attack_success', False))

        print(f"Samples: {len(valid)}")
        print(f"Successes: {successes}/{len(valid)}")
        print(f"Avg confidence change: {avg_change:+.4f}")

        for r in valid:
            status = "OK" if r.get('attack_success') else "FAIL"
            print(f"  {r['confidence_change']:+.4f} [{status}]")

    print(f"\nOutputs saved to: {output_path.absolute()}")


if __name__ == "__main__":
    main()
