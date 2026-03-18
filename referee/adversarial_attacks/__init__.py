"""
Adversarial Attacks for Referee Deepfake Detection Model

This package provides comprehensive adversarial attack implementations specifically
designed for the Referee multimodal deepfake detection model.

Components:
- MultiModalInput: Wrapper for combined audio/video tensors
- RefereeAttackWrapper: Model wrapper for adversarial attacks
- RefereeMultiModalPGD: PGD attack implementation with L2 norm and temporal regularization
- RefereeAttackTester: Testing suite for validating attack implementations
- RefereeAttackEvaluator: Evaluation pipeline for measuring attack effectiveness

Quick Usage:
    from adversarial_attacks import RefereeMultiModalPGD, quick_test_attack_installation

    # Quick test
    success = quick_test_attack_installation(referee_model)

    # Run attack
    attacker = RefereeMultiModalPGD(referee_model, attack_mode='joint')
    adv_audio, adv_video, info = attacker.generate(
        target_audio, target_video, ref_audio, ref_video, labels_rf
    )
"""

# Core attack implementations
from .pgd_attack import RefereeMultiModalPGD

# Wrapper utilities
from .multimodal_wrapper import (
    MultiModalInput,
    RefereeAttackWrapper,
    create_attack_wrapper
)

# Testing and evaluation
from .testing_suite import (
    RefereeAttackTester,
    quick_test_attack_installation
)

from .evaluation_pipeline import (
    RefereeAttackEvaluator,
    PerceptualMetrics
)

# Version info
__version__ = "1.0.0"
__author__ = "Referee Attack Team"

# Main exports
__all__ = [
    # Core attack
    'RefereeMultiModalPGD',

    # Wrappers
    'MultiModalInput',
    'RefereeAttackWrapper',
    'create_attack_wrapper',

    # Testing
    'RefereeAttackTester',
    'quick_test_attack_installation',

    # Evaluation
    'RefereeAttackEvaluator',
    'PerceptualMetrics',
]