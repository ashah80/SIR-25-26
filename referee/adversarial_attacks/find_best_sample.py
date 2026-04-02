"""
Find Best Quality Adversarial Samples

This script analyzes the results directory and identifies samples with:
- Lowest MSE (least visual distortion)
- Highest SNR (least audio distortion)
- Best combined quality score

Usage:
    python adversarial_attacks/find_best_sample.py --results-dir ./final_audiovisualattack
    python adversarial_attacks/find_best_sample.py --results-dir ./final_audiovisualattack --top 5
"""

import sys
from pathlib import Path
import numpy as np
import argparse
import re
from typing import Dict, List, Any, Optional, Tuple

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    print("Warning: OpenCV (cv2) not available. MSE calculation will be skipped.")


def parse_results_file(filepath: Path) -> Dict[str, Any]:
    """Parse the audiovisualresults.txt file and extract per-sample data."""
    with open(filepath, 'r') as f:
        content = f.read()
    
    data = {
        'samples': [],
        'arrays': {}
    }
    
    # Parse CSV data
    csv_match = re.search(r'PER-SAMPLE DATA.*?\n\n(sample_id.*?)(?=\n\n=)', content, re.DOTALL)
    if csv_match:
        lines = csv_match.group(1).strip().split('\n')
        if lines:
            headers = lines[0].split(',')
            for line in lines[1:]:
                if line.strip():
                    values = line.split(',')
                    sample = {}
                    for h, v in zip(headers, values):
                        try:
                            if '.' in v:
                                sample[h] = float(v)
                            elif v.isdigit():
                                sample[h] = int(v)
                            else:
                                sample[h] = v
                        except:
                            sample[h] = v
                    data['samples'].append(sample)
    
    # Parse raw arrays for SNR
    array_patterns = {
        'audio_snr_dbs': r'audio_snr_dbs\s*=\s*\[(.*?)\]',
        'video_only_successes': r'video_only_successes\s*=\s*\[(.*?)\]',
        'audio_only_successes': r'audio_only_successes\s*=\s*\[(.*?)\]',
        'combined_successes': r'combined_successes\s*=\s*\[(.*?)\]',
        'video_only_changes': r'video_only_changes\s*=\s*\[(.*?)\]',
        'audio_only_changes': r'audio_only_changes\s*=\s*\[(.*?)\]',
        'combined_changes': r'combined_changes\s*=\s*\[(.*?)\]',
    }
    
    for name, pattern in array_patterns.items():
        match = re.search(pattern, content)
        if match:
            values = [float(x.strip()) for x in match.group(1).split(',') if x.strip()]
            data['arrays'][name] = np.array(values)
    
    return data


def calculate_video_mse(original_path: Path, adversarial_path: Path) -> Optional[float]:
    """Calculate Mean Squared Error between original and adversarial video frames."""
    if not CV2_AVAILABLE:
        return None
    
    if not original_path.exists() or not adversarial_path.exists():
        return None
    
    try:
        cap_orig = cv2.VideoCapture(str(original_path))
        cap_adv = cv2.VideoCapture(str(adversarial_path))
        
        mse_values = []
        
        while True:
            ret_orig, frame_orig = cap_orig.read()
            ret_adv, frame_adv = cap_adv.read()
            
            if not ret_orig or not ret_adv:
                break
            
            frame_orig = frame_orig.astype(np.float64)
            frame_adv = frame_adv.astype(np.float64)
            
            mse = np.mean((frame_orig - frame_adv) ** 2)
            mse_values.append(mse)
        
        cap_orig.release()
        cap_adv.release()
        
        if mse_values:
            return np.mean(mse_values)
        return None
        
    except Exception as e:
        print(f"  Warning: Could not calculate MSE: {e}")
        return None


def collect_sample_data(results_dir: Path, n_samples: int, data: Dict) -> List[Dict]:
    """Collect MSE and SNR data for all samples."""
    samples = []
    
    snr_values = data['arrays'].get('audio_snr_dbs', np.array([]))
    video_successes = data['arrays'].get('video_only_successes', np.array([]))
    audio_successes = data['arrays'].get('audio_only_successes', np.array([]))
    combined_successes = data['arrays'].get('combined_successes', np.array([]))
    video_changes = data['arrays'].get('video_only_changes', np.array([]))
    audio_changes = data['arrays'].get('audio_only_changes', np.array([]))
    combined_changes = data['arrays'].get('combined_changes', np.array([]))
    
    for i in range(1, n_samples + 1):
        sample_dir = results_dir / f"sample_{i}"
        original_video = sample_dir / "02_original_video_processed.mp4"
        adversarial_video = sample_dir / "04_adversarial_video.mp4"
        
        sample_data = {
            'sample_id': i,
            'sample_dir': str(sample_dir),
            'mse': None,
            'snr': None,
            'video_success': None,
            'audio_success': None,
            'combined_success': None,
            'video_change': None,
            'audio_change': None,
            'combined_change': None,
        }
        
        # Calculate MSE
        if CV2_AVAILABLE:
            sample_data['mse'] = calculate_video_mse(original_video, adversarial_video)
        
        # Get SNR from parsed data
        idx = i - 1
        if idx < len(snr_values):
            sample_data['snr'] = snr_values[idx]
        if idx < len(video_successes):
            sample_data['video_success'] = bool(video_successes[idx])
        if idx < len(audio_successes):
            sample_data['audio_success'] = bool(audio_successes[idx])
        if idx < len(combined_successes):
            sample_data['combined_success'] = bool(combined_successes[idx])
        if idx < len(video_changes):
            sample_data['video_change'] = video_changes[idx]
        if idx < len(audio_changes):
            sample_data['audio_change'] = audio_changes[idx]
        if idx < len(combined_changes):
            sample_data['combined_change'] = combined_changes[idx]
        
        samples.append(sample_data)
    
    return samples


def print_rankings(samples: List[Dict], top_n: int):
    """Print rankings for best samples."""
    
    print("\n" + "=" * 70)
    print("BEST QUALITY ADVERSARIAL SAMPLES")
    print("=" * 70)
    
    # Filter samples with valid MSE
    samples_with_mse = [s for s in samples if s['mse'] is not None]
    
    # Filter samples with valid SNR
    samples_with_snr = [s for s in samples if s['snr'] is not None]
    
    # Top N by lowest MSE (best video quality)
    if samples_with_mse:
        print(f"\n{'─' * 70}")
        print(f"TOP {top_n} LOWEST MSE (Best Video Quality - Least Visual Distortion)")
        print(f"{'─' * 70}")
        sorted_by_mse = sorted(samples_with_mse, key=lambda x: x['mse'])
        for rank, s in enumerate(sorted_by_mse[:top_n], 1):
            success_str = "✓" if s['video_success'] else "✗"
            change_str = f"{s['video_change']:.4f}" if s['video_change'] is not None else "N/A"
            print(f"  {rank}. Sample {s['sample_id']:2d}  |  MSE: {s['mse']:8.2f}  |  "
                  f"Video Attack: {success_str}  |  Prob Change: {change_str}")
    else:
        print("\n  No MSE data available (OpenCV required)")
    
    # Top N by highest SNR (best audio quality)
    if samples_with_snr:
        print(f"\n{'─' * 70}")
        print(f"TOP {top_n} HIGHEST SNR (Best Audio Quality - Least Audio Distortion)")
        print(f"{'─' * 70}")
        sorted_by_snr = sorted(samples_with_snr, key=lambda x: x['snr'], reverse=True)
        for rank, s in enumerate(sorted_by_snr[:top_n], 1):
            success_str = "✓" if s['audio_success'] else "✗"
            change_str = f"{s['audio_change']:.4f}" if s['audio_change'] is not None else "N/A"
            print(f"  {rank}. Sample {s['sample_id']:2d}  |  SNR: {s['snr']:6.2f} dB  |  "
                  f"Audio Attack: {success_str}  |  Prob Change: {change_str}")
    else:
        print("\n  No SNR data available")
    
    # Combined quality score (normalized MSE + SNR)
    samples_with_both = [s for s in samples if s['mse'] is not None and s['snr'] is not None]
    
    if samples_with_both:
        # Normalize MSE (lower is better, so invert)
        mse_values = [s['mse'] for s in samples_with_both]
        mse_min, mse_max = min(mse_values), max(mse_values)
        
        # Normalize SNR (higher is better)
        snr_values = [s['snr'] for s in samples_with_both]
        snr_min, snr_max = min(snr_values), max(snr_values)
        
        for s in samples_with_both:
            # Normalized scores (0-1, higher is better)
            if mse_max > mse_min:
                mse_score = 1 - (s['mse'] - mse_min) / (mse_max - mse_min)
            else:
                mse_score = 1.0
            
            if snr_max > snr_min:
                snr_score = (s['snr'] - snr_min) / (snr_max - snr_min)
            else:
                snr_score = 1.0
            
            # Combined score (equal weight)
            s['quality_score'] = (mse_score + snr_score) / 2
        
        print(f"\n{'─' * 70}")
        print(f"TOP {top_n} COMBINED QUALITY (Best Overall Fakes)")
        print(f"{'─' * 70}")
        sorted_by_quality = sorted(samples_with_both, key=lambda x: x['quality_score'], reverse=True)
        for rank, s in enumerate(sorted_by_quality[:top_n], 1):
            combined_str = "✓" if s['combined_success'] else "✗"
            print(f"  {rank}. Sample {s['sample_id']:2d}  |  Quality: {s['quality_score']:.3f}  |  "
                  f"MSE: {s['mse']:8.2f}  |  SNR: {s['snr']:6.2f} dB  |  Combined: {combined_str}")
    
    # Successful attacks only
    successful_combined = [s for s in samples_with_both if s['combined_success']]
    if successful_combined:
        print(f"\n{'─' * 70}")
        print(f"TOP {top_n} BEST SUCCESSFUL ATTACKS (Combined Attack Success)")
        print(f"{'─' * 70}")
        sorted_successful = sorted(successful_combined, key=lambda x: x['quality_score'], reverse=True)
        for rank, s in enumerate(sorted_successful[:top_n], 1):
            print(f"  {rank}. Sample {s['sample_id']:2d}  |  Quality: {s['quality_score']:.3f}  |  "
                  f"MSE: {s['mse']:8.2f}  |  SNR: {s['snr']:6.2f} dB  |  "
                  f"Prob Change: {s['combined_change']:.4f}")
    
    print("\n" + "=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Find best quality adversarial samples")
    parser.add_argument("--results-dir", type=str, default="./final_audiovisualattack",
                        help="Directory containing results (default: ./final_audiovisualattack)")
    parser.add_argument("--top", type=int, default=5,
                        help="Number of top samples to show (default: 5)")
    
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    results_file = results_dir / "audiovisualresults.txt"
    
    if not results_file.exists():
        print(f"Error: Results file not found: {results_file}")
        sys.exit(1)
    
    print("=" * 70)
    print("FIND BEST QUALITY SAMPLES")
    print("=" * 70)
    print(f"\nResults directory: {results_dir}")
    
    # Parse results file
    data = parse_results_file(results_file)
    n_samples = len(data['samples'])
    
    if n_samples == 0:
        # Try to count sample directories
        sample_dirs = list(results_dir.glob("sample_*"))
        n_samples = len(sample_dirs)
    
    print(f"Found {n_samples} samples")
    
    if n_samples == 0:
        print("No samples found!")
        sys.exit(1)
    
    # Collect data
    print("\nAnalyzing samples...")
    samples = collect_sample_data(results_dir, n_samples, data)
    
    # Print rankings
    print_rankings(samples, args.top)


if __name__ == "__main__":
    main()
