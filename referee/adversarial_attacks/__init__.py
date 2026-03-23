"""
Adversarial Attacks for Referee Deepfake Detection Model

This package provides adversarial attack implementations for the Referee
multimodal deepfake detection model.

Components:
- RefereeMultiModalPGD: PGD attack implementation with L2 norm constraints
- MultiModalInput: Wrapper for combined audio/video tensors
- RefereeAttackWrapper: Model wrapper for adversarial attacks

Usage:
    from adversarial_attacks import RefereeMultiModalPGD

    attacker = RefereeMultiModalPGD(referee_model, attack_mode='joint')
    adv_audio, adv_video, info = attacker.generate(...)
"""

# Core attack implementation
from .pgd_attack import RefereeMultiModalPGD

# Wrapper utilities
from .multimodal_wrapper import (
    MultiModalInput,
    RefereeAttackWrapper,
    create_attack_wrapper
)

# Data loading
from .real_data_loader import AdversarialTestDataset, load_real_sample

__version__ = "1.0.0"

__all__ = [
    'RefereeMultiModalPGD',
    'MultiModalInput',
    'RefereeAttackWrapper',
    'create_attack_wrapper',
    'AdversarialTestDataset',
    'load_real_sample',
]