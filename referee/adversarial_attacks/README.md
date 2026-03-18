# Referee Adversarial Attacks

This package provides comprehensive adversarial attack implementations specifically designed for the **Referee multimodal deepfake detection model**. The implementation supports individual and joint attacks on audio and video modalities with proper handling of the model's reference-aware architecture.

## Features

- ✅ **Multimodal PGD Attacks** with L2 norm constraints
- ✅ **Individual Modality Attacks** (audio-only, video-only)
- ✅ **Joint Multimodal Attacks** (audio + video simultaneously)
- ✅ **Temporal Regularization** for imperceptible perturbations
- ✅ **Reference-Aware Architecture** handling
- ✅ **Comprehensive Testing Suite** with validation
- ✅ **Evaluation Pipeline** with perceptual quality metrics

## Quick Start

### 1. Installation Test

```python
from adversarial_attacks import quick_test_attack_installation

# Test basic functionality
success = quick_test_attack_installation(referee_model)
if success:
    print("✅ Ready to run adversarial attacks!")
```

### 2. Basic Attack Usage

```python
from adversarial_attacks import RefereeMultiModalPGD

# Create attacker
attacker = RefereeMultiModalPGD(
    referee_model,
    attack_mode='joint',      # 'audio', 'video', or 'joint'
    eps_audio=0.05,          # L2 norm bound for audio
    eps_video=0.3,           # L2 norm bound for video
    max_iter=100,            # Attack iterations
    temporal_weight=0.5      # Temporal smoothness weight
)

# Generate adversarial examples
adv_audio, adv_video, attack_info = attacker.generate(
    target_audio,    # Target audio to attack
    target_video,    # Target video to attack
    ref_audio,       # Reference audio (unchanged)
    ref_video,       # Reference video (unchanged)
    labels_rf        # Ground truth labels
)

print(f"Attack success: {attack_info['success_iterations']}")
```

### 3. Comprehensive Evaluation

```python
from adversarial_attacks import RefereeAttackEvaluator

# Create evaluator
evaluator = RefereeAttackEvaluator(referee_model)

# Run comprehensive evaluation
results = evaluator.run_comprehensive_evaluation(
    target_audio, target_video, ref_audio, ref_video, labels_rf,
    output_dir="./evaluation_results"
)

# View results
print(results['report'])
```

## Architecture Overview

### Input Format

The Referee model expects specific tensor shapes:
- **Audio**: `(B, S, 1, F, Ta)` = `(Batch, 8, 1, 128, 66)`
- **Video**: `(B, S, Tv, C, H, W)` = `(Batch, 8, 16, 3, 224, 224)`
- **Labels**: `(B,)` for real/fake classification

### Attack Pipeline

1. **Wrapper Creation**: `RefereeAttackWrapper` handles model interface
2. **Gradient Computation**: Compute gradients w.r.t. target inputs only
3. **Perturbation Update**: L2-normalized gradient steps with projection
4. **Temporal Regularization**: Smooth frame-to-frame transitions
5. **Bound Enforcement**: Project to epsilon ball + valid input ranges

## Attack Modes

### Audio-Only Attack
```python
attacker = RefereeMultiModalPGD(referee_model, attack_mode='audio')
```
- Perturbs only audio spectrograms
- Video remains unchanged
- Good for testing audio pathway importance

### Video-Only Attack
```python
attacker = RefereeMultiModalPGD(referee_model, attack_mode='video')
```
- Perturbs only video frames
- Audio remains unchanged
- Includes temporal smoothness regularization

### Joint Multimodal Attack
```python
attacker = RefereeMultiModalPGD(referee_model, attack_mode='joint')
```
- Attacks both audio and video simultaneously
- Separate epsilon bounds for each modality
- Most comprehensive attack mode

## Key Parameters

| Parameter | Description | Typical Values |
|-----------|-------------|----------------|
| `eps_audio` | L2 norm bound for audio perturbations | 0.01-0.1 |
| `eps_video` | L2 norm bound for video perturbations | 0.1-0.5 |
| `temporal_weight` | Weight for temporal smoothness | 0.1-1.0 |
| `max_iter` | Maximum attack iterations | 50-200 |
| `attack_mode` | Which modalities to attack | 'audio', 'video', 'joint' |

## Testing and Validation

### Run All Tests
```python
from adversarial_attacks import RefereeAttackTester

tester = RefereeAttackTester(referee_model)
results = tester.run_all_tests()
```

**Tests Include:**
- ✅ Gradient flow validation
- ✅ Attack bound verification
- ✅ Model compatibility checks
- ✅ Temporal coherence validation
- ✅ Attack mode functionality

### Individual Tests
```python
# Test gradient flow
tester.test_gradient_flow()

# Test epsilon bounds
tester.test_attack_bounds(eps_audio=0.05, eps_video=0.3)

# Test temporal coherence
tester.test_temporal_coherence(temporal_weight=0.5)
```

## Evaluation Metrics

### Attack Success Metrics
- **Success Rate**: Fraction of fake samples classified as real
- **Confidence Change**: Change in model confidence scores
- **Convergence Speed**: Iterations to successful attack

### Perceptual Quality Metrics
- **SSIM**: Structural similarity for video quality
- **SNR**: Signal-to-noise ratio for audio quality
- **MSE**: Mean squared error for both modalities
- **L2 Norm**: Perturbation magnitude measurement

### Example Evaluation
```python
# Compare attack modes
mode_results = evaluator.evaluate_attack_modes(
    target_audio, target_video, ref_audio, ref_video, labels_rf
)

# Sweep epsilon values
epsilon_results = evaluator.evaluate_epsilon_sweep(
    target_audio, target_video, ref_audio, ref_video, labels_rf
)
```

## File Structure

```
adversarial_attacks/
├── __init__.py                  # Package initialization
├── multimodal_wrapper.py        # Input wrappers and model interface
├── pgd_attack.py               # Core PGD attack implementation
├── testing_suite.py           # Comprehensive testing utilities
├── evaluation_pipeline.py     # Attack evaluation and metrics
├── demo.py                     # Usage examples and demos
└── README.md                   # This file
```

## Implementation Notes

### Temporal Regularization
The implementation includes temporal smoothness constraints to ensure generated adversarial examples are imperceptible:

```python
# Audio temporal loss (across time frames)
audio_temporal_loss = torch.mean((adv_audio[:,:,:,:,1:] - adv_audio[:,:,:,:,:-1]) ** 2)

# Video temporal loss (across frames)
video_temporal_loss = torch.mean((adv_video[:,:,1:] - adv_video[:,:,:-1]) ** 2)
```

### L2 Projection
All perturbations are projected to L2 balls with proper normalization:

```python
# Project to L2 ball
delta_norm = torch.norm(delta.view(delta.size(0), -1), dim=1, keepdim=True)
scale_factor = torch.min(torch.ones_like(delta_norm), eps / (delta_norm + 1e-8))
delta_projected = delta * scale_factor
```

### Reference-Aware Design
The attack wrapper properly handles the Referee model's dual-input architecture:
- **Target**: Audio/video inputs that are attacked (gradients enabled)
- **Reference**: Authentic inputs used for comparison (gradients disabled)

## Usage Examples

### Demo Script
Run the complete demo:
```bash
cd adversarial_attacks
python demo.py
```

### Custom Attack Configuration
```python
# Configure custom attack
custom_attacker = RefereeMultiModalPGD(
    referee_model,
    eps_audio=0.02,          # Smaller audio perturbation
    eps_video=0.5,           # Larger video perturbation
    eps_step_audio=0.005,    # Smaller audio step
    eps_step_video=0.1,      # Larger video step
    max_iter=200,            # More iterations
    temporal_weight=1.0,     # Strong temporal regularization
    attack_mode='joint'      # Attack both modalities
)
```

### Batch Processing
```python
# Process multiple samples
batch_results = []
for i in range(num_batches):
    batch_audio = target_audio[i:i+batch_size]
    batch_video = target_video[i:i+batch_size]
    # ... (ref inputs and labels)

    adv_audio, adv_video, info = attacker.generate(
        batch_audio, batch_video, batch_ref_audio, batch_ref_video, batch_labels
    )
    batch_results.append((adv_audio, adv_video, info))
```

## Troubleshooting

### Common Issues

1. **CUDA Memory Error**: Reduce batch size or max_iter
2. **Gradient is None**: Check model is in correct mode, inputs require_grad=True
3. **Attack Not Converging**: Try larger epsilon values or more iterations
4. **Temporal Artifacts**: Increase temporal_weight parameter

### Debug Mode
```python
# Enable verbose output for debugging
attacker = RefereeMultiModalPGD(referee_model, verbose=True)
```

### Performance Tips
- Use smaller batch sizes for memory efficiency
- Start with fewer iterations for quick testing
- Use audio-only or video-only modes for faster attacks
- Cache reference inputs when attacking multiple targets

## Citation

If you use this adversarial attack implementation in your research, please cite the original Referee paper and this implementation.

---

**Note**: This implementation is designed for research purposes to evaluate the robustness of deepfake detection models. Please use responsibly and in accordance with applicable laws and regulations.