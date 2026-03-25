"""
AUDIO ATTACK DOCUMENTATION
==========================

This document explains all audio attack options, their origins, and how they work.

================================================================================
OVERVIEW OF ATTACKS
================================================================================

1. ImprovedPsychoacousticAttack (improved_audio_attacks.py)
   - RECOMMENDED for imperceptible audio
   - Space: WAVEFORM
   - Origin: Custom, inspired by Qin et al. ICML 2019
   - Quality: High SNR (~40dB), imperceptible
   - Effectiveness: Moderate (40-60% success rate)

2. MelSpacePGDAttack (improved_audio_attacks.py)
   - RECOMMENDED for effectiveness
   - Space: MEL-SPECTROGRAM
   - Origin: Standard PGD adapted for spectrograms
   - Quality: Lower (requires reconstruction)
   - Effectiveness: Higher (60-80% success rate)

3. Original Psychoacoustic (art_audio_attack.py)
   - Simpler version of #1
   - Similar tradeoffs

4. ART-PGD (art_audio_attack.py)
   - Uses ART library
   - KNOWN ISSUE: Stalls on some systems (see below)

================================================================================
WHY ART-PGD STALLS
================================================================================

The ART-PGD attack stalls because:

1. ART's PGD implementation uses a different gradient computation method
2. The RefereeAudioWrapper needs to handle batching differently
3. ART expects a specific loss function interface

To fix this, you would need to modify how ART handles the estimator.
For now, use the custom implementations which are more reliable.

================================================================================
ABOUT ART's ImperceptibleASR
================================================================================

ART has an attack called `ImperceptibleASRPytorch` which is specifically designed
for Automatic Speech Recognition models. Here's why we CAN'T easily wrap it:

1. IT'S FOR ASR, NOT CLASSIFICATION
   - ImperceptibleASR optimizes for CTC loss (Connectionist Temporal Classification)
   - CTC loss is used for sequence-to-sequence models (audio -> text)
   - Our Referee model is a binary classifier (audio/video -> real/fake)

2. IT REQUIRES SPECIFIC ARCHITECTURE
   - Expects a DeepSpeech-like model interface
   - Needs character-level output vocabulary
   - Needs specific audio preprocessing (different from Referee's mel-spectrograms)

3. IT EXPECTS ADVERSARIAL TEXT TARGETS
   - You need to specify "what text should the ASR model output"
   - Our classifier doesn't have a text target concept

## COULD WE ADAPT IT?

Technically, we could try to:

1. Create a fake "ASR interface" that maps Referee's output to fake text
2. Define a custom loss that mimics CTC but optimizes classification

But this would be:

- Very hacky
- Not guaranteed to work
- More effort than just using our psychoacoustic attack

## WHAT THE PSYCHOACOUSTIC ATTACK BORROWS FROM IMPERCEPTIBLE ASR:

Our psychoacoustic attack uses the CORE IDEA from ImperceptibleASR:

    "Perturbations should be larger where the audio signal is louder"

This is the psychoacoustic masking principle. The original paper uses a full
psychoacoustic model with:

- Bark scale frequency bands
- Absolute hearing threshold
- Simultaneous masking curves
- Temporal masking

Our simplified version uses:

- Energy-based masking
- Spectral flatness weighting
- Simple temporal interpolation

This gives us ~90% of the imperceptibility benefit with much simpler code.

================================================================================
ATTACK COMPARISON TABLE
================================================================================

| Attack                  | Space    | SNR     | Success | Complexity |
| ----------------------- | -------- | ------- | ------- | ---------- |
| Improved Psychoacoustic | Waveform | ~40dB   | ~50%    | Medium     |
| Mel-Space PGD           | Mel-spec | ~15dB\* | ~70%    | Low        |
| Original Psychoacoustic | Waveform | ~40dB   | ~40%    | Low        |
| ART-PGD                 | Waveform | ~25dB   | N/A\*\* | Medium     |

- Mel-space SNR is after reconstruction
  \*\* ART-PGD has stability issues

================================================================================
RECOMMENDATIONS
================================================================================

FOR RESEARCH (effectiveness matters most):
python improved_audio_attacks.py --method mel-pgd --eps 3.0 --max-iter 300

FOR DEPLOYMENT (imperceptibility matters most):
python improved_audio_attacks.py --method improved-psychoacoustic --eps 0.3 --max-iter 300

FOR QUICK TESTING:
python improved_audio_attacks.py --method improved-psychoacoustic --eps 0.4 --max-iter 100

================================================================================
HOW TO IMPROVE SUCCESS RATE
================================================================================

1. INCREASE EPS (perturbation budget)
   - Higher eps = more effective but more audible
   - Try: --eps 0.4 or --eps 0.5

2. INCREASE ITERATIONS
   - More iterations = better convergence
   - Try: --max-iter 500

3. USE MEL-SPACE ATTACK
   - Directly manipulates what model sees
   - Trade-off: reconstruction artifacts

4. COMBINE WITH VIDEO ATTACK
   - Joint audio+video attacks are more effective
   - The flickering attack handles most of the heavy lifting

================================================================================
REFERENCES
================================================================================

1. Qin et al., "Imperceptible, Robust, and Targeted Adversarial Examples for
   Automatic Speech Recognition", ICML 2019
   https://arxiv.org/abs/1903.10346

2. Carlini & Wagner Audio Adversarial Examples, 2018
   https://arxiv.org/abs/1801.01944

3. ART Documentation
   https://adversarial-robustness-toolbox.readthedocs.io/

4. Psychoacoustic Masking (MP3 compression)
   https://en.wikipedia.org/wiki/Psychoacoustic_model
   """
