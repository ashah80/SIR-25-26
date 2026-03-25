"""
Adversarial Attacks for Referee Deepfake Detection Model

This package provides adversarial attack implementations for the Referee
multimodal deepfake detection model.

Main scripts:
- improved_video_attack.py: Video attacks (FlickeringAttack)
- improved_audio_attacks.py: Audio attacks (ImprovedPsychoacousticAttack, MelSpacePGDAttack)
- art_audio_attack.py: ART-based audio attacks

Usage:
    python adversarial_attacks/improved_video_attack.py --attack flickering --num-samples 3
    python adversarial_attacks/improved_audio_attacks.py --method improved-psychoacoustic --num-samples 5
"""

from .real_data_loader import AdversarialTestDataset, load_real_sample

__version__ = "1.0.0"

__all__ = [
    'AdversarialTestDataset',
    'load_real_sample',
]
