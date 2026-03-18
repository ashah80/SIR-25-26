"""
Evaluation Pipeline for Referee Adversarial Attacks

This module provides comprehensive evaluation utilities for analyzing the effectiveness
and quality of adversarial attacks on the Referee deepfake detection model.

Features:
- Attack success rate measurement
- Perceptual quality metrics (SSIM, SNR)
- Cross-modal attack comparison
- Quantitative and qualitative analysis
- Results visualization and reporting
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import time

from .pgd_attack import RefereeMultiModalPGD
from .multimodal_wrapper import RefereeAttackWrapper, create_attack_wrapper


class PerceptualMetrics:
    """Utility class for computing perceptual quality metrics."""

    @staticmethod
    def compute_ssim(img1: torch.Tensor, img2: torch.Tensor) -> float:
        """
        Compute Structural Similarity Index (SSIM) for videos.
        Simplified version for batch processing.
        """
        # Convert to grayscale if RGB
        if img1.dim() == 6 and img1.shape[-3] == 3:  # (B,S,T,C,H,W)
            img1_gray = 0.299 * img1[..., 0, :, :] + 0.587 * img1[..., 1, :, :] + 0.114 * img1[..., 2, :, :]
            img2_gray = 0.299 * img2[..., 0, :, :] + 0.587 * img2[..., 1, :, :] + 0.114 * img2[..., 2, :, :]
        else:
            img1_gray = img1.squeeze()
            img2_gray = img2.squeeze()

        # Compute local means
        mu1 = F.avg_pool2d(img1_gray, 3, stride=1, padding=1)
        mu2 = F.avg_pool2d(img2_gray, 3, stride=1, padding=1)

        mu1_sq = mu1 * mu1
        mu2_sq = mu2 * mu2
        mu1_mu2 = mu1 * mu2

        # Compute local variances and covariance
        sigma1_sq = F.avg_pool2d(img1_gray * img1_gray, 3, stride=1, padding=1) - mu1_sq
        sigma2_sq = F.avg_pool2d(img2_gray * img2_gray, 3, stride=1, padding=1) - mu2_sq
        sigma12 = F.avg_pool2d(img1_gray * img2_gray, 3, stride=1, padding=1) - mu1_mu2

        # SSIM constants
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        ssim = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        return ssim.mean().item()

    @staticmethod
    def compute_snr(original: torch.Tensor, perturbed: torch.Tensor) -> float:
        """Compute Signal-to-Noise Ratio in dB."""
        signal_power = torch.mean(original ** 2)
        noise_power = torch.mean((perturbed - original) ** 2)

        if noise_power == 0:
            return float('inf')

        snr_db = 10 * torch.log10(signal_power / noise_power)
        return snr_db.item()

    @staticmethod
    def compute_lpips(img1: torch.Tensor, img2: torch.Tensor) -> float:
        """
        Placeholder for LPIPS (perceptual distance).
        Would require importing and using the lpips library.
        """
        # For now, return MSE as a simple perceptual proxy
        mse = F.mse_loss(img1, img2)
        return mse.item()


class RefereeAttackEvaluator:
    """
    Comprehensive evaluation pipeline for Referee adversarial attacks.
    """

    def __init__(self, referee_model: nn.Module, device: str = 'cuda'):
        """
        Initialize the evaluator.

        Args:
            referee_model: The Referee model
            device: Device to run evaluation on
        """
        self.referee_model = referee_model.eval()
        self.device = device
        self.metrics = PerceptualMetrics()

    def evaluate_single_attack(self,
                             target_audio: torch.Tensor,
                             target_video: torch.Tensor,
                             ref_audio: torch.Tensor,
                             ref_video: torch.Tensor,
                             labels_rf: torch.Tensor,
                             attack_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate a single adversarial attack configuration.

        Args:
            target_audio: Original target audio
            target_video: Original target video
            ref_audio: Reference audio
            ref_video: Reference video
            labels_rf: Ground truth labels
            attack_config: Attack configuration parameters

        Returns:
            Dictionary with evaluation results
        """
        # Create attacker with given configuration
        attacker = RefereeMultiModalPGD(
            self.referee_model,
            **attack_config,
            verbose=False
        )

        # Measure attack time
        start_time = time.time()
        adv_audio, adv_video, attack_info = attacker.generate(
            target_audio, target_video, ref_audio, ref_video, labels_rf
        )
        attack_time = time.time() - start_time

        # Create wrapper for predictions
        wrapper = create_attack_wrapper(self.referee_model, ref_audio, ref_video, labels_rf)

        # Get predictions
        original_conf = wrapper.get_confidence(target_audio, target_video)
        adversarial_conf = wrapper.get_confidence(adv_audio, adv_video)

        # Compute success metrics
        attack_success = adversarial_conf['rf_real_prob'] > 0.5  # Fake classified as real
        confidence_change = adversarial_conf['rf_real_prob'] - original_conf['rf_real_prob']

        # Compute perceptual quality metrics
        perceptual_metrics = {}

        if attack_config.get('attack_mode', 'joint') in ['video', 'joint']:
            # Video quality metrics
            perceptual_metrics['video_ssim'] = self.metrics.compute_ssim(target_video, adv_video)
            perceptual_metrics['video_mse'] = F.mse_loss(target_video, adv_video).item()
            perceptual_metrics['video_lpips'] = self.metrics.compute_lpips(target_video, adv_video)

        if attack_config.get('attack_mode', 'joint') in ['audio', 'joint']:
            # Audio quality metrics
            perceptual_metrics['audio_snr'] = self.metrics.compute_snr(target_audio, adv_audio)
            perceptual_metrics['audio_mse'] = F.mse_loss(target_audio, adv_audio).item()

        # Compute perturbation norms
        perturbation_norms = attacker.compute_perturbation_norms(
            adv_audio, adv_video, target_audio, target_video
        )

        # Compile results
        results = {
            'attack_config': attack_config,
            'attack_success': attack_success,
            'confidence_change': confidence_change,
            'original_confidence': original_conf,
            'adversarial_confidence': adversarial_conf,
            'perceptual_metrics': perceptual_metrics,
            'perturbation_norms': perturbation_norms,
            'attack_info': attack_info,
            'attack_time': attack_time,
            'batch_size': target_audio.shape[0]
        }

        return results

    def evaluate_attack_modes(self,
                            target_audio: torch.Tensor,
                            target_video: torch.Tensor,
                            ref_audio: torch.Tensor,
                            ref_video: torch.Tensor,
                            labels_rf: torch.Tensor,
                            base_config: Optional[Dict[str, Any]] = None) -> Dict[str, Dict[str, Any]]:
        """
        Compare different attack modes (audio, video, joint).

        Returns:
            Dictionary with results for each attack mode
        """
        if base_config is None:
            # Use more aggressive parameters to ensure attacks work
            base_config = {
                'eps_audio': 0.1,         # Larger perturbation
                'eps_video': 0.5,         # Larger perturbation
                'eps_step_audio': 0.02,   # Larger step
                'eps_step_video': 0.1,    # Larger step
                'max_iter': 30,           # More iterations
                'temporal_weight': 0.1    # Lower temporal weight for stronger attacks
            }

        modes = ['audio', 'video', 'joint']
        results = {}

        print("🎯 Evaluating attack modes...")

        for mode in modes:
            print(f"  Testing mode: {mode}")

            config = base_config.copy()
            config['attack_mode'] = mode

            try:
                mode_results = self.evaluate_single_attack(
                    target_audio, target_video, ref_audio, ref_video, labels_rf, config
                )
                results[mode] = mode_results

                # Print summary
                success = mode_results['attack_success']
                conf_change = mode_results['confidence_change']
                print(f"    Success: {success}, Confidence change: {conf_change:.3f}")

                # Clean memory between tests
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            except Exception as e:
                print(f"    Error in mode {mode}: {e}")
                results[mode] = {'error': str(e)}

        return results

    def evaluate_epsilon_sweep(self,
                             target_audio: torch.Tensor,
                             target_video: torch.Tensor,
                             ref_audio: torch.Tensor,
                             ref_video: torch.Tensor,
                             labels_rf: torch.Tensor,
                             epsilon_ranges: Optional[Dict[str, List[float]]] = None) -> Dict[str, List[Dict[str, Any]]]:
        """
        Evaluate attack effectiveness across different epsilon values.

        Args:
            epsilon_ranges: Dictionary with epsilon values to test for each modality

        Returns:
            Dictionary with results for each epsilon configuration
        """
        if epsilon_ranges is None:
            # Use smaller ranges to avoid memory issues
            epsilon_ranges = {
                'audio': [0.05, 0.1],      # Reduced range for quick test
                'video': [0.3, 0.5]        # Reduced range for quick test
            }

        base_config = {
            'max_iter': 20,           # Reduced iterations
            'temporal_weight': 0.1,   # Lower temporal weight for stronger attacks
            'attack_mode': 'joint'
        }

        results = {'audio_sweep': [], 'video_sweep': []}

        print("📊 Running epsilon sweep analysis...")

        # Audio epsilon sweep (fixed video eps)
        print("  Testing audio epsilon values...")
        for eps_audio in epsilon_ranges['audio']:
            config = base_config.copy()
            config.update({
                'eps_audio': eps_audio,
                'eps_video': 0.3  # Fixed
            })

            try:
                result = self.evaluate_single_attack(
                    target_audio, target_video, ref_audio, ref_video, labels_rf, config
                )
                results['audio_sweep'].append(result)
                print(f"    eps_audio={eps_audio:.3f}: Success={result['attack_success']}")

                # Clean memory between tests
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            except Exception as e:
                print(f"    eps_audio={eps_audio:.3f}: Error={e}")

        # Video epsilon sweep (fixed audio eps)
        print("  Testing video epsilon values...")
        for eps_video in epsilon_ranges['video']:
            config = base_config.copy()
            config.update({
                'eps_audio': 0.05,  # Fixed
                'eps_video': eps_video
            })

            try:
                result = self.evaluate_single_attack(
                    target_audio, target_video, ref_audio, ref_video, labels_rf, config
                )
                results['video_sweep'].append(result)
                print(f"    eps_video={eps_video:.3f}: Success={result['attack_success']}")

                # Clean memory between tests
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            except Exception as e:
                print(f"    eps_video={eps_video:.3f}: Error={e}")

        return results

    def generate_evaluation_report(self, results: Dict[str, Any], output_dir: Optional[str] = None) -> str:
        """
        Generate a comprehensive evaluation report.

        Args:
            results: Evaluation results from various tests
            output_dir: Directory to save report and plots

        Returns:
            Report text
        """
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(exist_ok=True)

        report_lines = [
            "=" * 80,
            "REFEREE ADVERSARIAL ATTACK EVALUATION REPORT",
            "=" * 80,
            ""
        ]

        # Mode comparison summary
        if 'mode_results' in results:
            report_lines.extend([
                "ATTACK MODE COMPARISON",
                "-" * 40,
                ""
            ])

            for mode, mode_results in results['mode_results'].items():
                if 'error' not in mode_results:
                    success = mode_results['attack_success']
                    conf_change = mode_results['confidence_change']
                    attack_time = mode_results['attack_time']

                    report_lines.extend([
                        f"Mode: {mode.upper()}",
                        f"  Success: {success}",
                        f"  Confidence Change: {conf_change:.3f}",
                        f"  Attack Time: {attack_time:.2f}s",
                        ""
                    ])

                    # Add perceptual metrics
                    if 'perceptual_metrics' in mode_results:
                        report_lines.append("  Perceptual Quality:")
                        for metric, value in mode_results['perceptual_metrics'].items():
                            report_lines.append(f"    {metric}: {value:.4f}")
                        report_lines.append("")

        # Epsilon sweep analysis
        if 'epsilon_results' in results:
            report_lines.extend([
                "EPSILON SWEEP ANALYSIS",
                "-" * 40,
                ""
            ])

            # Audio sweep
            if 'audio_sweep' in results['epsilon_results']:
                report_lines.append("Audio Epsilon Sweep:")
                for result in results['epsilon_results']['audio_sweep']:
                    eps = result['attack_config']['eps_audio']
                    success = result['attack_success']
                    conf_change = result['confidence_change']
                    report_lines.append(f"  eps_audio={eps:.3f}: Success={success}, ΔConf={conf_change:.3f}")
                report_lines.append("")

            # Video sweep
            if 'video_sweep' in results['epsilon_results']:
                report_lines.append("Video Epsilon Sweep:")
                for result in results['epsilon_results']['video_sweep']:
                    eps = result['attack_config']['eps_video']
                    success = result['attack_success']
                    conf_change = result['confidence_change']
                    report_lines.append(f"  eps_video={eps:.3f}: Success={success}, ΔConf={conf_change:.3f}")
                report_lines.append("")

        # Summary statistics
        report_lines.extend([
            "SUMMARY",
            "-" * 40,
            ""
        ])

        # Calculate overall success rates
        all_results = []
        if 'mode_results' in results:
            all_results.extend([r for r in results['mode_results'].values() if 'error' not in r])
        if 'epsilon_results' in results:
            for sweep_results in results['epsilon_results'].values():
                all_results.extend(sweep_results)

        if all_results:
            success_rate = sum(r['attack_success'] for r in all_results) / len(all_results)
            avg_conf_change = np.mean([r['confidence_change'] for r in all_results])
            avg_attack_time = np.mean([r['attack_time'] for r in all_results])

            report_lines.extend([
                f"Overall Attack Success Rate: {success_rate:.2%}",
                f"Average Confidence Change: {avg_conf_change:.3f}",
                f"Average Attack Time: {avg_attack_time:.2f}s",
                ""
            ])

        report_text = "\n".join(report_lines)

        # Save report if output directory provided
        if output_dir:
            report_path = output_path / "evaluation_report.txt"
            with open(report_path, 'w') as f:
                f.write(report_text)

            # Save raw results as JSON
            results_path = output_path / "evaluation_results.json"
            with open(results_path, 'w') as f:
                # Convert tensors to lists for JSON serialization
                json_results = self._serialize_results_for_json(results)
                json.dump(json_results, f, indent=2)

            print(f"📁 Report saved to: {report_path}")
            print(f"📁 Results saved to: {results_path}")

        return report_text

    def _serialize_results_for_json(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Convert tensors and other non-serializable objects to JSON format."""
        def convert_item(item):
            if isinstance(item, torch.Tensor):
                return item.tolist()
            elif isinstance(item, np.ndarray):
                return item.tolist()
            elif isinstance(item, dict):
                return {k: convert_item(v) for k, v in item.items()}
            elif isinstance(item, list):
                return [convert_item(x) for x in item]
            else:
                return item

        return convert_item(results)

    def run_comprehensive_evaluation(self,
                                   target_audio: torch.Tensor,
                                   target_video: torch.Tensor,
                                   ref_audio: torch.Tensor,
                                   ref_video: torch.Tensor,
                                   labels_rf: torch.Tensor,
                                   output_dir: Optional[str] = None) -> Dict[str, Any]:
        """
        Run a comprehensive evaluation of adversarial attacks.

        Returns:
            Complete evaluation results
        """
        print("🚀 Starting comprehensive adversarial attack evaluation...\n")

        results = {}

        # 1. Attack mode comparison
        mode_results = self.evaluate_attack_modes(
            target_audio, target_video, ref_audio, ref_video, labels_rf
        )
        results['mode_results'] = mode_results
        print()

        # 2. Epsilon sweep analysis
        epsilon_results = self.evaluate_epsilon_sweep(
            target_audio, target_video, ref_audio, ref_video, labels_rf
        )
        results['epsilon_results'] = epsilon_results
        print()

        # 3. Generate report
        report = self.generate_evaluation_report(results, output_dir)
        results['report'] = report

        print("✅ Comprehensive evaluation completed!")
        return results