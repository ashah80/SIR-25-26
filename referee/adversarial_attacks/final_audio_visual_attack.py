"""
Final Audio-Visual Adversarial Attack for Referee Deepfake Detection Model.

This script runs FlickeringAttack (video), ImprovedPsychoacousticAttack (audio),
and evaluates ALL THREE scenarios:
  1. Video-only attack
  2. Audio-only attack
  3. Combined audio+video attack

This allows direct comparison of attack effectiveness across modalities.

Usage:
    python adversarial_attacks/final_audio_visual_attack.py --num-samples 3
    python adversarial_attacks/final_audio_visual_attack.py --num-samples 5 --output-dir ./final-results
"""

import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import numpy as np
from typing import Tuple, Dict, Any
import argparse
import time
import shutil
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

try:
    import cv2
    HAVE_CV2 = True
except ImportError:
    HAVE_CV2 = False
    print("Warning: OpenCV not found. Video saving will be disabled.")

try:
    import soundfile as sf
    HAVE_SOUNDFILE = True
except ImportError:
    HAVE_SOUNDFILE = False
    print("Warning: soundfile not found. Install with: pip install soundfile")

# Import attack classes
from adversarial_attacks.improved_video_attack import FlickeringAttack, save_video_non_overlapping
from adversarial_attacks.improved_audio_attacks import (
    ImprovedPsychoacousticAttack,
    DifferentiableMelTransform,
    extract_audio_from_video
)
from adversarial_attacks.real_data_loader import AdversarialTestDataset


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

        new_state_dict = {(k[7:] if k.startswith('module.') else k): v
                         for k, v in state_dict.items()}
        model.load_state_dict(new_state_dict, strict=False)
        print("Checkpoint loaded!")

    model = model.to(device)
    model.eval()
    return model


def save_frame_comparison(
    orig_video: torch.Tensor,
    adv_video: torch.Tensor,
    save_path: Path,
    num_frames: int = 8
):
    """Save a visual comparison of original vs adversarial frames with perturbation."""
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

    plt.suptitle('Video Perturbation Visualization', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved frame comparison: {save_path}")


def save_mel_comparison(
    original_waveform: torch.Tensor,
    adversarial_waveform: torch.Tensor,
    save_path: Path,
    sample_rate: int = 16000
):
    """Save mel-spectrogram comparison: original, adversarial, and difference."""
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=1024,
        hop_length=160,
        win_length=400,
        n_mels=128
    )

    if original_waveform.dim() == 2:
        original_waveform = original_waveform[0]
    if adversarial_waveform.dim() == 2:
        adversarial_waveform = adversarial_waveform[0]

    orig_mel = mel_transform(original_waveform.cpu())
    adv_mel = mel_transform(adversarial_waveform.cpu())

    orig_mel_db = 10 * torch.log10(orig_mel + 1e-10)
    adv_mel_db = 10 * torch.log10(adv_mel + 1e-10)
    diff_mel_db = adv_mel_db - orig_mel_db

    fig, axes = plt.subplots(3, 1, figsize=(12, 10))

    im0 = axes[0].imshow(orig_mel_db.numpy(), aspect='auto', origin='lower', cmap='viridis')
    axes[0].set_title('Original Audio Mel-Spectrogram')
    axes[0].set_ylabel('Mel Bin')
    plt.colorbar(im0, ax=axes[0], label='dB')

    im1 = axes[1].imshow(adv_mel_db.numpy(), aspect='auto', origin='lower', cmap='viridis')
    axes[1].set_title('Adversarial Audio Mel-Spectrogram')
    axes[1].set_ylabel('Mel Bin')
    plt.colorbar(im1, ax=axes[1], label='dB')

    vmax = max(abs(diff_mel_db.min().item()), abs(diff_mel_db.max().item()))
    im2 = axes[2].imshow(diff_mel_db.numpy(), aspect='auto', origin='lower',
                         cmap='RdBu_r', vmin=-vmax, vmax=vmax)
    axes[2].set_title('Difference (Adversarial - Original)')
    axes[2].set_ylabel('Mel Bin')
    axes[2].set_xlabel('Time Frame')
    plt.colorbar(im2, ax=axes[2], label='dB Difference')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved mel comparison: {save_path}")


def save_comparison_chart(results: Dict[str, Any], save_path: Path):
    """Save a bar chart comparing all three attack scenarios."""
    scenarios = ['Original', 'Video-Only', 'Audio-Only', 'Combined']
    real_probs = [
        results['original_real_prob'],
        results['video_only_real_prob'],
        results['audio_only_real_prob'],
        results['combined_real_prob']
    ]

    colors = ['gray', 'blue', 'green', 'red']
    success_threshold = 0.5

    fig, ax = plt.subplots(figsize=(10, 6))

    bars = ax.bar(scenarios, real_probs, color=colors, edgecolor='black', linewidth=1.5)

    # Add success threshold line
    ax.axhline(y=success_threshold, color='orange', linestyle='--', linewidth=2, label='Success Threshold (0.5)')

    # Add value labels on bars
    for bar, prob in zip(bars, real_probs):
        height = bar.get_height()
        ax.annotate(f'{prob:.3f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=12, fontweight='bold')

    # Add success/fail labels
    for bar, prob in zip(bars, real_probs):
        if prob > success_threshold:
            status = '✓ SUCCESS'
            color = 'green'
        else:
            status = '✗ FAIL'
            color = 'red'
        ax.annotate(status,
                    xy=(bar.get_x() + bar.get_width() / 2, 0.02),
                    ha='center', va='bottom', fontsize=9, color=color, fontweight='bold')

    ax.set_ylabel('Real Probability (higher = more fooled)', fontsize=12)
    ax.set_title('Attack Effectiveness Comparison', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 1.1)
    ax.legend(loc='upper right')
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved comparison chart: {save_path}")


def combine_video_audio(video_path: Path, audio_path: Path, output_path: Path):
    """Combine video file with audio file using ffmpeg."""
    import subprocess

    cmd = [
        'ffmpeg', '-y',
        '-i', str(video_path),
        '-i', str(audio_path),
        '-c:v', 'copy',
        '-c:a', 'aac',
        '-strict', 'experimental',
        '-shortest',
        str(output_path)
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, timeout=60)
        if result.returncode == 0:
            print(f"  Combined video+audio: {output_path}")
        else:
            print(f"  Warning: ffmpeg failed - {result.stderr.decode()[:200]}")
    except Exception as e:
        print(f"  Warning: Could not combine video+audio: {e}")


def save_stats(info: Dict, save_path: Path):
    """Save attack statistics to a text file."""
    with open(save_path, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("FINAL AUDIO-VISUAL ATTACK RESULTS\n")
        f.write("=" * 60 + "\n\n")

        # Summary section
        f.write("SUMMARY\n")
        f.write("-" * 40 + "\n")
        f.write(f"Original Real Prob:    {info['original_real_prob']:.4f}\n")
        f.write(f"Video-Only Real Prob:  {info['video_only_real_prob']:.4f} (change: {info['video_only_confidence_change']:+.4f})\n")
        f.write(f"Audio-Only Real Prob:  {info['audio_only_real_prob']:.4f} (change: {info['audio_only_confidence_change']:+.4f})\n")
        f.write(f"Combined Real Prob:    {info['combined_real_prob']:.4f} (change: {info['combined_confidence_change']:+.4f})\n")
        f.write("\n")
        f.write(f"Video-Only Success: {'YES' if info['video_only_attack_success'] else 'NO'}\n")
        f.write(f"Audio-Only Success: {'YES' if info['audio_only_attack_success'] else 'NO'}\n")
        f.write(f"Combined Success:   {'YES' if info['combined_attack_success'] else 'NO'}\n")
        f.write("\n")

        # Detailed section
        f.write("DETAILED METRICS\n")
        f.write("-" * 40 + "\n")
        for k, v in info.items():
            if isinstance(v, float):
                f.write(f"{k}: {v:.6f}\n")
            else:
                f.write(f"{k}: {v}\n")

    print(f"  Saved stats: {save_path}")


def save_detailed_results(results: list, output_path: Path, args):
    """
    Save comprehensive results data for visualization and analysis.
    
    This generates a detailed text file with:
    - Experiment configuration
    - Per-sample data in CSV-like format (easy to parse)
    - Aggregate statistics
    - All metrics needed for graphs
    """
    from datetime import datetime
    
    with open(output_path, 'w') as f:
        # ================================================================
        # HEADER
        # ================================================================
        f.write("=" * 80 + "\n")
        f.write("AUDIO-VISUAL ADVERSARIAL ATTACK - DETAILED RESULTS\n")
        f.write("=" * 80 + "\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total Samples: {len(results)}\n")
        f.write("\n")
        
        # ================================================================
        # EXPERIMENT CONFIGURATION
        # ================================================================
        f.write("=" * 80 + "\n")
        f.write("EXPERIMENT CONFIGURATION\n")
        f.write("=" * 80 + "\n")
        f.write("\n[Video Attack Parameters]\n")
        f.write(f"video_eps={args.video_eps}\n")
        f.write(f"video_iterations={args.video_iter}\n")
        f.write(f"video_step_size={args.video_step_size}\n")
        f.write(f"flicker_freq={args.flicker_freq}\n")
        f.write(f"spatial_freq={args.spatial_freq}\n")
        f.write(f"num_basis={args.num_basis}\n")
        f.write(f"smoothness_weight={args.smoothness_weight}\n")
        f.write("\n[Audio Attack Parameters]\n")
        f.write(f"audio_eps={args.audio_eps}\n")
        f.write(f"audio_iterations={args.audio_iter}\n")
        f.write(f"target_snr_db={args.target_snr}\n")
        f.write(f"snr_weight={args.snr_weight}\n")
        f.write(f"masking_strength={args.masking_strength}\n")
        f.write("\n[General]\n")
        f.write(f"device={args.device}\n")
        f.write(f"output_dir={args.output_dir}\n")
        f.write("\n")
        
        # ================================================================
        # AGGREGATE STATISTICS
        # ================================================================
        f.write("=" * 80 + "\n")
        f.write("AGGREGATE STATISTICS\n")
        f.write("=" * 80 + "\n\n")
        
        if results:
            n = len(results)
            
            # Success rates
            video_successes = sum(1 for r in results if r.get('video_only_attack_success', False))
            audio_successes = sum(1 for r in results if r.get('audio_only_attack_success', False))
            combined_successes = sum(1 for r in results if r.get('combined_attack_success', False))
            
            f.write("[Success Rates]\n")
            f.write(f"video_only_success_count={video_successes}\n")
            f.write(f"video_only_success_rate={video_successes/n:.4f}\n")
            f.write(f"audio_only_success_count={audio_successes}\n")
            f.write(f"audio_only_success_rate={audio_successes/n:.4f}\n")
            f.write(f"combined_success_count={combined_successes}\n")
            f.write(f"combined_success_rate={combined_successes/n:.4f}\n")
            f.write("\n")
            
            # Confidence changes
            video_changes = [r['video_only_confidence_change'] for r in results]
            audio_changes = [r['audio_only_confidence_change'] for r in results]
            combined_changes = [r['combined_confidence_change'] for r in results]
            
            f.write("[Confidence Change Statistics]\n")
            f.write(f"video_only_change_mean={np.mean(video_changes):.6f}\n")
            f.write(f"video_only_change_std={np.std(video_changes):.6f}\n")
            f.write(f"video_only_change_min={np.min(video_changes):.6f}\n")
            f.write(f"video_only_change_max={np.max(video_changes):.6f}\n")
            f.write(f"audio_only_change_mean={np.mean(audio_changes):.6f}\n")
            f.write(f"audio_only_change_std={np.std(audio_changes):.6f}\n")
            f.write(f"audio_only_change_min={np.min(audio_changes):.6f}\n")
            f.write(f"audio_only_change_max={np.max(audio_changes):.6f}\n")
            f.write(f"combined_change_mean={np.mean(combined_changes):.6f}\n")
            f.write(f"combined_change_std={np.std(combined_changes):.6f}\n")
            f.write(f"combined_change_min={np.min(combined_changes):.6f}\n")
            f.write(f"combined_change_max={np.max(combined_changes):.6f}\n")
            f.write("\n")
            
            # Real probability statistics
            orig_probs = [r['original_real_prob'] for r in results]
            video_probs = [r['video_only_real_prob'] for r in results]
            audio_probs = [r['audio_only_real_prob'] for r in results]
            combined_probs = [r['combined_real_prob'] for r in results]
            
            f.write("[Real Probability Statistics]\n")
            f.write(f"original_prob_mean={np.mean(orig_probs):.6f}\n")
            f.write(f"original_prob_std={np.std(orig_probs):.6f}\n")
            f.write(f"video_only_prob_mean={np.mean(video_probs):.6f}\n")
            f.write(f"video_only_prob_std={np.std(video_probs):.6f}\n")
            f.write(f"audio_only_prob_mean={np.mean(audio_probs):.6f}\n")
            f.write(f"audio_only_prob_std={np.std(audio_probs):.6f}\n")
            f.write(f"combined_prob_mean={np.mean(combined_probs):.6f}\n")
            f.write(f"combined_prob_std={np.std(combined_probs):.6f}\n")
            f.write("\n")
            
            # Perturbation statistics
            video_linf = [r.get('video_perturbation_linf', 0) for r in results]
            video_l2 = [r.get('video_perturbation_l2', 0) for r in results]
            audio_linf = [r.get('audio_perturbation_linf', 0) for r in results]
            audio_snr = [r.get('audio_perturbation_snr_db', 0) for r in results]
            
            f.write("[Perturbation Statistics]\n")
            f.write(f"video_linf_mean={np.mean(video_linf):.6f}\n")
            f.write(f"video_linf_std={np.std(video_linf):.6f}\n")
            f.write(f"video_l2_mean={np.mean(video_l2):.6f}\n")
            f.write(f"video_l2_std={np.std(video_l2):.6f}\n")
            f.write(f"audio_linf_mean={np.mean(audio_linf):.6f}\n")
            f.write(f"audio_linf_std={np.std(audio_linf):.6f}\n")
            f.write(f"audio_snr_mean={np.mean(audio_snr):.6f}\n")
            f.write(f"audio_snr_std={np.std(audio_snr):.6f}\n")
            f.write("\n")
            
            # Timing statistics
            video_times = [r.get('video_attack_time_seconds', 0) for r in results]
            audio_times = [r.get('audio_attack_time_seconds', 0) for r in results]
            total_times = [r.get('total_attack_time_seconds', 0) for r in results]
            
            f.write("[Timing Statistics (seconds)]\n")
            f.write(f"video_attack_time_mean={np.mean(video_times):.2f}\n")
            f.write(f"video_attack_time_total={np.sum(video_times):.2f}\n")
            f.write(f"audio_attack_time_mean={np.mean(audio_times):.2f}\n")
            f.write(f"audio_attack_time_total={np.sum(audio_times):.2f}\n")
            f.write(f"total_time_mean={np.mean(total_times):.2f}\n")
            f.write(f"total_time_sum={np.sum(total_times):.2f}\n")
            f.write("\n")
            
            # Best attack analysis
            video_best = sum(1 for r in results if r['video_only_confidence_change'] >= r['audio_only_confidence_change'] 
                           and r['video_only_confidence_change'] >= r['combined_confidence_change'])
            audio_best = sum(1 for r in results if r['audio_only_confidence_change'] >= r['video_only_confidence_change']
                           and r['audio_only_confidence_change'] >= r['combined_confidence_change'])
            combined_best = sum(1 for r in results if r['combined_confidence_change'] >= r['video_only_confidence_change']
                              and r['combined_confidence_change'] >= r['audio_only_confidence_change'])
            
            f.write("[Best Attack Per Sample]\n")
            f.write(f"video_best_count={video_best}\n")
            f.write(f"audio_best_count={audio_best}\n")
            f.write(f"combined_best_count={combined_best}\n")
            f.write("\n")
        
        # ================================================================
        # PER-SAMPLE DATA (CSV FORMAT)
        # ================================================================
        f.write("=" * 80 + "\n")
        f.write("PER-SAMPLE DATA (CSV FORMAT)\n")
        f.write("=" * 80 + "\n")
        f.write("# Copy the lines below to create a CSV file for analysis\n\n")
        
        # CSV Header
        csv_columns = [
            "sample_id",
            "sample_index",
            "source_file",
            "original_real_prob",
            "original_fake_prob",
            "video_only_real_prob",
            "video_only_confidence_change",
            "video_only_attack_success",
            "audio_only_real_prob",
            "audio_only_confidence_change",
            "audio_only_attack_success",
            "combined_real_prob",
            "combined_confidence_change",
            "combined_attack_success",
            "video_perturbation_linf",
            "video_perturbation_l2",
            "audio_perturbation_linf",
            "audio_perturbation_snr_db",
            "video_attack_time_seconds",
            "audio_attack_time_seconds",
            "total_attack_time_seconds",
            "best_attack"
        ]
        
        f.write(",".join(csv_columns) + "\n")
        
        # CSV Data rows
        for idx, r in enumerate(results, 1):
            # Determine best attack for this sample
            changes = {
                'video': r.get('video_only_confidence_change', 0),
                'audio': r.get('audio_only_confidence_change', 0),
                'combined': r.get('combined_confidence_change', 0)
            }
            best_attack = max(changes, key=changes.get)
            
            row = [
                str(idx),
                str(r.get('sample_index', '')),
                str(r.get('source_file', '')).replace(',', ';'),  # Escape commas
                f"{r.get('original_real_prob', 0):.6f}",
                f"{r.get('original_fake_prob', 0):.6f}",
                f"{r.get('video_only_real_prob', 0):.6f}",
                f"{r.get('video_only_confidence_change', 0):.6f}",
                str(int(r.get('video_only_attack_success', False))),
                f"{r.get('audio_only_real_prob', 0):.6f}",
                f"{r.get('audio_only_confidence_change', 0):.6f}",
                str(int(r.get('audio_only_attack_success', False))),
                f"{r.get('combined_real_prob', 0):.6f}",
                f"{r.get('combined_confidence_change', 0):.6f}",
                str(int(r.get('combined_attack_success', False))),
                f"{r.get('video_perturbation_linf', 0):.6f}",
                f"{r.get('video_perturbation_l2', 0):.6f}",
                f"{r.get('audio_perturbation_linf', 0):.6f}",
                f"{r.get('audio_perturbation_snr_db', 0):.6f}",
                f"{r.get('video_attack_time_seconds', 0):.2f}",
                f"{r.get('audio_attack_time_seconds', 0):.2f}",
                f"{r.get('total_attack_time_seconds', 0):.2f}",
                best_attack
            ]
            f.write(",".join(row) + "\n")
        
        f.write("\n")
        
        # ================================================================
        # RAW DATA ARRAYS (for easy copy-paste into Python/plotting)
        # ================================================================
        f.write("=" * 80 + "\n")
        f.write("RAW DATA ARRAYS (Python format - copy directly into scripts)\n")
        f.write("=" * 80 + "\n\n")
        
        if results:
            # Original probabilities
            f.write("# Original real probabilities (before any attack)\n")
            f.write(f"original_probs = {[round(r['original_real_prob'], 6) for r in results]}\n\n")
            
            # Video-only results
            f.write("# Video-only attack results\n")
            f.write(f"video_only_probs = {[round(r['video_only_real_prob'], 6) for r in results]}\n")
            f.write(f"video_only_changes = {[round(r['video_only_confidence_change'], 6) for r in results]}\n")
            f.write(f"video_only_successes = {[int(r['video_only_attack_success']) for r in results]}\n\n")
            
            # Audio-only results
            f.write("# Audio-only attack results\n")
            f.write(f"audio_only_probs = {[round(r['audio_only_real_prob'], 6) for r in results]}\n")
            f.write(f"audio_only_changes = {[round(r['audio_only_confidence_change'], 6) for r in results]}\n")
            f.write(f"audio_only_successes = {[int(r['audio_only_attack_success']) for r in results]}\n\n")
            
            # Combined results
            f.write("# Combined attack results\n")
            f.write(f"combined_probs = {[round(r['combined_real_prob'], 6) for r in results]}\n")
            f.write(f"combined_changes = {[round(r['combined_confidence_change'], 6) for r in results]}\n")
            f.write(f"combined_successes = {[int(r['combined_attack_success']) for r in results]}\n\n")
            
            # Perturbation metrics
            f.write("# Perturbation metrics\n")
            f.write(f"video_linf_norms = {[round(r.get('video_perturbation_linf', 0), 6) for r in results]}\n")
            f.write(f"video_l2_norms = {[round(r.get('video_perturbation_l2', 0), 6) for r in results]}\n")
            f.write(f"audio_linf_norms = {[round(r.get('audio_perturbation_linf', 0), 6) for r in results]}\n")
            f.write(f"audio_snr_dbs = {[round(r.get('audio_perturbation_snr_db', 0), 2) for r in results]}\n\n")
            
            # Timing
            f.write("# Attack timing (seconds)\n")
            f.write(f"video_times = {[round(r.get('video_attack_time_seconds', 0), 2) for r in results]}\n")
            f.write(f"audio_times = {[round(r.get('audio_attack_time_seconds', 0), 2) for r in results]}\n")
            f.write(f"total_times = {[round(r.get('total_attack_time_seconds', 0), 2) for r in results]}\n\n")
        
        # ================================================================
        # VISUALIZATION SUGGESTIONS
        # ================================================================
        f.write("=" * 80 + "\n")
        f.write("VISUALIZATION SUGGESTIONS\n")
        f.write("=" * 80 + "\n\n")
        f.write("Recommended graphs to generate from this data:\n\n")
        f.write("1. BAR CHART: Success rates comparison (video vs audio vs combined)\n")
        f.write("2. BOX PLOT: Confidence change distribution by attack type\n")
        f.write("3. SCATTER PLOT: Original prob vs final prob (color by attack type)\n")
        f.write("4. LINE PLOT: Per-sample comparison (sample ID on x-axis)\n")
        f.write("5. HISTOGRAM: Distribution of confidence changes\n")
        f.write("6. STACKED BAR: Best attack type distribution\n")
        f.write("7. SCATTER: Perturbation magnitude vs confidence change\n")
        f.write("8. HEATMAP: Correlation between metrics\n")
        f.write("\n")
        
        f.write("=" * 80 + "\n")
        f.write("END OF RESULTS FILE\n")
        f.write("=" * 80 + "\n")


class FinalAudioVisualAttack:
    """
    Final audio-visual attack that evaluates all three scenarios:
    1. Video-only attack
    2. Audio-only attack
    3. Combined attack
    """

    def __init__(
        self,
        model: nn.Module,
        # Video attack params
        video_eps: float = 0.05,
        video_iterations: int = 200,
        video_step_size: float = 0.05,
        flicker_freq: float = 2.5,
        spatial_freq: int = 8,
        num_basis: int = 4,
        smoothness_weight: float = 2.0,
        # Audio attack params
        audio_eps: float = 0.05,
        audio_iterations: int = 300,
        target_snr_db: float = 35.0,
        snr_weight: float = 0.02,
        masking_strength: float = 0.1,
        device: str = 'cuda'
    ):
        self.model = model
        self.device = device

        # Initialize video attack
        self.video_attack = FlickeringAttack(
            model=model,
            eps=video_eps,
            flicker_freq=flicker_freq,
            spatial_freq=spatial_freq,
            num_basis=num_basis,
            num_iterations=video_iterations,
            step_size=video_step_size,
            smoothness_weight=smoothness_weight,
            device=device
        )

        # Initialize audio attack
        self.audio_attack = ImprovedPsychoacousticAttack(
            model=model,
            eps=audio_eps,
            num_iterations=audio_iterations,
            target_snr_db=target_snr_db,
            snr_weight=snr_weight,
            masking_strength=masking_strength,
            device=device
        )

        # Mel transform for converting adversarial waveform to mel
        self.mel_transform = DifferentiableMelTransform(device=device).to(device)

        # Store params for reporting
        self.video_eps = video_eps
        self.audio_eps = audio_eps

    def attack(
        self,
        target_video: torch.Tensor,
        target_audio_mel: torch.Tensor,
        ref_video: torch.Tensor,
        ref_audio: torch.Tensor,
        labels: torch.Tensor,
        original_waveform: torch.Tensor,
        verbose: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """
        Run all three attack scenarios and compare.

        Args:
            target_video: (1, S, T, C, H, W) input video
            target_audio_mel: (1, S, 1, F, Ta) input audio mel-spectrogram
            ref_video: (1, S, T, C, H, W) reference video
            ref_audio: (1, S, 1, F, Ta) reference audio
            labels: (1,) ground truth label
            original_waveform: (1, T) raw audio waveform

        Returns:
            adv_video: Adversarial video tensor
            adv_waveform: Adversarial audio waveform
            info: Dictionary with all attack statistics
        """
        self.model.eval()
        start_time = time.time()

        # Get initial prediction
        with torch.no_grad():
            logits = self.model(target_video, target_audio_mel, ref_video, ref_audio)[0]
            probs = F.softmax(logits, dim=1)
            orig_real_prob = probs[0, 0].item()
            orig_fake_prob = probs[0, 1].item()

        if verbose:
            print(f"\n{'='*70}")
            print("FINAL AUDIO-VISUAL ATTACK - ALL THREE SCENARIOS")
            print(f"{'='*70}")
            print(f"Initial prediction: Real={orig_real_prob:.4f}, Fake={orig_fake_prob:.4f}")

        # ================================================================
        # SCENARIO 1: Video-Only Attack
        # ================================================================
        if verbose:
            print(f"\n{'='*70}")
            print("SCENARIO 1: VIDEO-ONLY ATTACK (FlickeringAttack)")
            print(f"{'='*70}")

        adv_video, video_info = self.video_attack.attack(
            target_video, target_audio_mel, ref_video, ref_audio, labels, verbose=verbose
        )

        # Evaluate video-only (adversarial video + original audio)
        with torch.no_grad():
            logits_video_only = self.model(adv_video, target_audio_mel, ref_video, ref_audio)[0]
            probs_video_only = F.softmax(logits_video_only, dim=1)
            video_only_real_prob = probs_video_only[0, 0].item()

        # ================================================================
        # SCENARIO 2: Audio-Only Attack
        # ================================================================
        if verbose:
            print(f"\n{'='*70}")
            print("SCENARIO 2: AUDIO-ONLY ATTACK (ImprovedPsychoacousticAttack)")
            print(f"{'='*70}")

        adv_waveform, audio_info = self.audio_attack.attack(
            original_waveform, target_video, ref_audio, ref_video, labels, verbose=verbose
        )

        # Convert adversarial waveform to mel
        adv_audio_mel = self.mel_transform(adv_waveform)

        # Evaluate audio-only (original video + adversarial audio)
        with torch.no_grad():
            logits_audio_only = self.model(target_video, adv_audio_mel, ref_video, ref_audio)[0]
            probs_audio_only = F.softmax(logits_audio_only, dim=1)
            audio_only_real_prob = probs_audio_only[0, 0].item()

        # ================================================================
        # SCENARIO 3: Combined Attack
        # ================================================================
        if verbose:
            print(f"\n{'='*70}")
            print("SCENARIO 3: COMBINED ATTACK (Video + Audio)")
            print(f"{'='*70}")

        # Evaluate combined (adversarial video + adversarial audio)
        with torch.no_grad():
            logits_combined = self.model(adv_video, adv_audio_mel, ref_video, ref_audio)[0]
            probs_combined = F.softmax(logits_combined, dim=1)
            combined_real_prob = probs_combined[0, 0].item()

        total_time = time.time() - start_time

        # ================================================================
        # Compile Results
        # ================================================================
        info = {
            'method': 'FinalAudioVisualAttack',
            # Original
            'original_real_prob': orig_real_prob,
            'original_fake_prob': orig_fake_prob,
            # Video-only results
            'video_only_real_prob': video_only_real_prob,
            'video_only_confidence_change': video_only_real_prob - orig_real_prob,
            'video_only_attack_success': video_only_real_prob > 0.5,
            # Audio-only results
            'audio_only_real_prob': audio_only_real_prob,
            'audio_only_confidence_change': audio_only_real_prob - orig_real_prob,
            'audio_only_attack_success': audio_only_real_prob > 0.5,
            # Combined results
            'combined_real_prob': combined_real_prob,
            'combined_confidence_change': combined_real_prob - orig_real_prob,
            'combined_attack_success': combined_real_prob > 0.5,
            # Video attack details
            'video_perturbation_linf': video_info['perturbation_linf_norm'],
            'video_perturbation_l2': video_info['perturbation_l2_norm'],
            'video_eps': video_info['eps'],
            'video_attack_time_seconds': video_info['attack_time_seconds'],
            # Audio attack details
            'audio_perturbation_linf': audio_info['perturbation_linf'],
            'audio_perturbation_snr_db': audio_info['perturbation_snr_db'],
            'audio_eps': audio_info['eps'],
            'audio_attack_time_seconds': audio_info['attack_time_seconds'],
            # Total time
            'total_attack_time_seconds': total_time,
        }

        # ================================================================
        # Print Summary
        # ================================================================
        if verbose:
            print(f"\n{'='*70}")
            print("RESULTS COMPARISON")
            print(f"{'='*70}")
            print(f"{'Scenario':<20} {'Real Prob':>12} {'Change':>12} {'Success':>10}")
            print("-" * 54)
            print(f"{'Original':<20} {orig_real_prob:>12.4f} {'---':>12} {'---':>10}")
            print(f"{'Video-Only':<20} {video_only_real_prob:>12.4f} {info['video_only_confidence_change']:>+12.4f} {'YES' if info['video_only_attack_success'] else 'NO':>10}")
            print(f"{'Audio-Only':<20} {audio_only_real_prob:>12.4f} {info['audio_only_confidence_change']:>+12.4f} {'YES' if info['audio_only_attack_success'] else 'NO':>10}")
            print(f"{'Combined':<20} {combined_real_prob:>12.4f} {info['combined_confidence_change']:>+12.4f} {'YES' if info['combined_attack_success'] else 'NO':>10}")
            print("-" * 54)

            # Determine best attack
            changes = {
                'Video-Only': info['video_only_confidence_change'],
                'Audio-Only': info['audio_only_confidence_change'],
                'Combined': info['combined_confidence_change']
            }
            best = max(changes, key=changes.get)
            print(f"\nMost effective: {best} ({changes[best]:+.4f})")
            print(f"Total time: {total_time:.1f}s")

        return adv_video, adv_waveform, info


def main():
    parser = argparse.ArgumentParser(description="Final Audio-Visual Attack - All Three Scenarios")
    parser.add_argument("--num-samples", type=int, default=3,
                        help="Number of samples to attack")
    parser.add_argument("--output-dir", type=str, default="./final_audiovisualattack",
                        help="Directory to save results")
    # Video params
    parser.add_argument("--video-eps", type=float, default=0.05,
                        help="Video perturbation budget")
    parser.add_argument("--video-iter", type=int, default=200,
                        help="Video attack iterations")
    parser.add_argument("--video-step-size", type=float, default=0.05,
                        help="Video optimizer learning rate")
    parser.add_argument("--flicker-freq", type=float, default=2.5,
                        help="Temporal flicker frequency in Hz")
    parser.add_argument("--spatial-freq", type=int, default=8,
                        help="Spatial pattern frequency")
    parser.add_argument("--num-basis", type=int, default=4,
                        help="Number of basis patterns")
    parser.add_argument("--smoothness-weight", type=float, default=2.0,
                        help="Temporal smoothness regularization")
    # Audio params
    parser.add_argument("--audio-eps", type=float, default=0.15,
                        help="Audio perturbation budget")
    parser.add_argument("--audio-iter", type=int, default=300,
                        help="Audio attack iterations")
    parser.add_argument("--target-snr", type=float, default=35.0,
                        help="Target SNR in dB")
    parser.add_argument("--snr-weight", type=float, default=0.1,
                        help="SNR regularization weight")
    parser.add_argument("--masking-strength", type=float, default=0.5,
                        help="Psychoacoustic masking strength")
    # General
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()

    print("=" * 70)
    print("FINAL AUDIO-VISUAL ADVERSARIAL ATTACK")
    print("Comparing: Video-Only vs Audio-Only vs Combined")
    print("=" * 70)
    print(f"\nVideo params: eps={args.video_eps}, iter={args.video_iter}")
    print(f"Audio params: eps={args.audio_eps}, iter={args.audio_iter}, SNR={args.target_snr}dB")
    print()

    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("Loading model...")
    model = load_model(args.device)

    print("Loading dataset...")
    dataset = AdversarialTestDataset(device=args.device)

    # Get fake samples
    fake_indices = [i for i, s in enumerate(dataset.samples) if s.get('fake_label', 0) == 1]
    test_indices = fake_indices[:args.num_samples]

    print(f"Found {len(fake_indices)} fake samples, testing {len(test_indices)}")

    # Initialize attack
    attack = FinalAudioVisualAttack(
        model=model,
        video_eps=args.video_eps,
        video_iterations=args.video_iter,
        video_step_size=args.video_step_size,
        flicker_freq=args.flicker_freq,
        spatial_freq=args.spatial_freq,
        num_basis=args.num_basis,
        smoothness_weight=args.smoothness_weight,
        audio_eps=args.audio_eps,
        audio_iterations=args.audio_iter,
        target_snr_db=args.target_snr,
        snr_weight=args.snr_weight,
        masking_strength=args.masking_strength,
        device=args.device
    )

    results = []

    for i, sample_idx in enumerate(test_indices):
        print()
        print("=" * 70)
        print(f"SAMPLE {i+1}/{args.num_samples}")
        print("=" * 70)

        sample_out = output_path / f"sample_{i+1}"
        sample_out.mkdir(exist_ok=True)

        # Load sample
        sample = dataset[sample_idx]
        target_video = sample['target_video'].unsqueeze(0)
        target_audio_mel = sample['target_audio'].unsqueeze(0)
        ref_audio = sample['reference_audio'].unsqueeze(0)
        ref_video = sample['reference_video'].unsqueeze(0)
        labels = sample['fake_label'].unsqueeze(0)
        info = sample['sample_info']

        video_path = info.get('target_path')
        if not video_path or not Path(video_path).exists():
            print(f"  Skipping - video file not found: {video_path}")
            continue

        print(f"Source: {video_path}")

        try:
            # Extract original audio waveform
            original_waveform = extract_audio_from_video(video_path)
            waveform_tensor = torch.from_numpy(original_waveform).float().to(args.device)
            if waveform_tensor.dim() == 1:
                waveform_tensor = waveform_tensor.unsqueeze(0)

            # Run attack (all three scenarios)
            adv_video, adv_waveform, attack_info = attack.attack(
                target_video, target_audio_mel, ref_video, ref_audio,
                labels, waveform_tensor, verbose=True
            )

            # Save outputs
            print(f"\nSaving outputs to {sample_out}...")

            # 1. Original video file (with audio)
            shutil.copy2(video_path, sample_out / "01_original_video_with_audio.mp4")
            print(f"  Saved: 01_original_video_with_audio.mp4")

            # 2. Processed video file (without audio) - from tensor
            save_video_non_overlapping(target_video, sample_out / "02_original_video_processed.mp4")

            # 3. Original audio file
            if HAVE_SOUNDFILE:
                orig_wav = original_waveform.copy()
                max_val = np.max(np.abs(orig_wav))
                if max_val > 0:
                    orig_wav = orig_wav / max_val * 0.95
                sf.write(str(sample_out / "03_original_audio.wav"), orig_wav, 16000)
                print(f"  Saved: 03_original_audio.wav")

            # 4. Adversarial video file (without audio)
            save_video_non_overlapping(adv_video, sample_out / "04_adversarial_video.mp4")

            # 5. Adversarial audio file
            if HAVE_SOUNDFILE:
                adv_wav = adv_waveform[0].cpu().numpy()
                max_val = np.max(np.abs(adv_wav))
                if max_val > 0:
                    adv_wav = adv_wav / max_val * 0.95
                sf.write(str(sample_out / "05_adversarial_audio.wav"), adv_wav, 16000)
                print(f"  Saved: 05_adversarial_audio.wav")

            # 6. Final adversarial video (with audio)
            if HAVE_SOUNDFILE and HAVE_CV2:
                combine_video_audio(
                    sample_out / "04_adversarial_video.mp4",
                    sample_out / "05_adversarial_audio.wav",
                    sample_out / "06_final_adversarial_video.mp4"
                )

            # 7. Frame-level visualization
            save_frame_comparison(target_video, adv_video, sample_out / "07_video_perturbation.png")

            # 8. Mel-spectrogram visualization
            save_mel_comparison(
                waveform_tensor, adv_waveform,
                sample_out / "08_audio_mel_comparison.png"
            )

            # 9. Comparison chart
            save_comparison_chart(attack_info, sample_out / "09_attack_comparison_chart.png")

            # Save stats
            attack_info['source_file'] = str(video_path)
            attack_info['sample_index'] = sample_idx
            save_stats(attack_info, sample_out / "stats.txt")

            results.append(attack_info)

        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
            results.append({'error': str(e), 'sample_index': sample_idx})

    # ================================================================
    # Print Final Summary
    # ================================================================
    print()
    print("=" * 70)
    print("FINAL SUMMARY - ALL SAMPLES")
    print("=" * 70)

    valid = [r for r in results if 'combined_confidence_change' in r]
    if valid:
        # Compute averages
        avg_video_change = np.mean([r['video_only_confidence_change'] for r in valid])
        avg_audio_change = np.mean([r['audio_only_confidence_change'] for r in valid])
        avg_combined_change = np.mean([r['combined_confidence_change'] for r in valid])

        video_successes = sum(1 for r in valid if r.get('video_only_attack_success', False))
        audio_successes = sum(1 for r in valid if r.get('audio_only_attack_success', False))
        combined_successes = sum(1 for r in valid if r.get('combined_attack_success', False))

        n = len(valid)

        print(f"\nSamples tested: {n}")
        print()
        print(f"{'Attack Type':<15} {'Success Rate':>15} {'Avg Δ Confidence':>20}")
        print("-" * 50)
        print(f"{'Video-Only':<15} {f'{video_successes}/{n} ({100*video_successes/n:.0f}%)':>15} {avg_video_change:>+20.4f}")
        print(f"{'Audio-Only':<15} {f'{audio_successes}/{n} ({100*audio_successes/n:.0f}%)':>15} {avg_audio_change:>+20.4f}")
        print(f"{'Combined':<15} {f'{combined_successes}/{n} ({100*combined_successes/n:.0f}%)':>15} {avg_combined_change:>+20.4f}")
        print("-" * 50)

        # Determine best overall
        avg_changes = {
            'Video-Only': avg_video_change,
            'Audio-Only': avg_audio_change,
            'Combined': avg_combined_change
        }
        best = max(avg_changes, key=avg_changes.get)
        print(f"\nMost effective overall: {best}")

        # Per-sample breakdown
        print("\nPer-sample breakdown:")
        print(f"{'#':<3} {'Original':>10} {'Video':>10} {'Audio':>10} {'Combined':>10} {'Best':>12}")
        print("-" * 60)
        for idx, r in enumerate(valid, 1):
            orig = r['original_real_prob']
            video = r['video_only_real_prob']
            audio = r['audio_only_real_prob']
            combined = r['combined_real_prob']

            # Determine best for this sample
            changes = {'V': r['video_only_confidence_change'],
                       'A': r['audio_only_confidence_change'],
                       'C': r['combined_confidence_change']}
            best_for_sample = max(changes, key=changes.get)
            best_map = {'V': 'Video', 'A': 'Audio', 'C': 'Combined'}

            print(f"{idx:<3} {orig:>10.4f} {video:>10.4f} {audio:>10.4f} {combined:>10.4f} {best_map[best_for_sample]:>12}")

    # Save overall summary
    summary_path = output_path / "overall_summary.txt"
    with open(summary_path, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("FINAL AUDIO-VISUAL ATTACK - OVERALL SUMMARY\n")
        f.write("=" * 70 + "\n\n")

        if valid:
            f.write(f"Samples tested: {n}\n\n")
            f.write(f"{'Attack Type':<15} {'Success Rate':>15} {'Avg Δ Confidence':>20}\n")
            f.write("-" * 50 + "\n")
            f.write(f"{'Video-Only':<15} {f'{video_successes}/{n} ({100*video_successes/n:.0f}%)':>15} {avg_video_change:>+20.4f}\n")
            f.write(f"{'Audio-Only':<15} {f'{audio_successes}/{n} ({100*audio_successes/n:.0f}%)':>15} {avg_audio_change:>+20.4f}\n")
            f.write(f"{'Combined':<15} {f'{combined_successes}/{n} ({100*combined_successes/n:.0f}%)':>15} {avg_combined_change:>+20.4f}\n")
            f.write("-" * 50 + "\n")
            f.write(f"\nMost effective overall: {best}\n")

    # ================================================================
    # Save Detailed Results for Visualization (audiovisualresults.txt)
    # ================================================================
    detailed_results_path = output_path / "audiovisualresults.txt"
    save_detailed_results(
        results=valid,
        output_path=detailed_results_path,
        args=args
    )
    print(f"Detailed results saved to: {detailed_results_path}")

    print()
    print(f"Outputs saved to: {output_path.absolute()}")


if __name__ == "__main__":
    main()
