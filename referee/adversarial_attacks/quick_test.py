#!/usr/bin/env python3
"""
Quick test script to verify adversarial attack installation is working.
Run this from the adversarial_attacks directory.
"""

import sys
from pathlib import Path
import torch
import torch.nn as nn

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

def main():
    print("🧪 Quick Adversarial Attack Test")
    print("=" * 50)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    try:
        # Import our modules
        from adversarial_attacks import (
            RefereeMultiModalPGD,
            quick_test_attack_installation
        )
        print("✅ Imports successful")

        # Create simple dummy model
        class TestModel(nn.Module):
            def __init__(self):
                super().__init__()
                # Ultra-simple model that works with any input dimensions
                self.classifier = nn.Sequential(
                    nn.Linear(2, 32),
                    nn.ReLU(),
                    nn.Linear(32, 2)
                )

            def forward(self, target_vis, target_aud, ref_vis, ref_aud, **kwargs):
                B = target_vis.shape[0]

                # Take global means - works for any dimensions
                audio_feat = torch.mean(target_aud, dim=list(range(1, target_aud.ndim)))
                video_feat = torch.mean(target_vis, dim=list(range(1, target_vis.ndim)))

                combined = torch.stack([audio_feat, video_feat], dim=1)
                combined_scaled = combined * 50.0  # Make sensitive to changes

                logits_rf = self.classifier(combined_scaled)
                logits_id = torch.randn(B, 2, device=target_vis.device)
                return logits_rf, logits_id

        model = TestModel().to(device)
        print("✅ Test model created")

        # Test installation
        success = quick_test_attack_installation(model, device)

        if success:
            print("\n🎉 ALL TESTS PASSED!")
        else:
            print("\n⚠️  Some tests failed, but attacks might still work")

        # Try a quick attack
        print("\n🎯 Testing quick attack...")

        # Create small test data to avoid memory issues
        try:
            # Try original dimensions first
            target_audio = torch.randn(1, 8, 1, 128, 66, device=device)
            target_video = torch.rand(1, 8, 16, 3, 224, 224, device=device)
            ref_audio = torch.randn(1, 8, 1, 128, 66, device=device)
            ref_video = torch.rand(1, 8, 16, 3, 224, 224, device=device)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print("⚠️  Using smaller tensors due to memory constraints")
                target_audio = torch.randn(1, 4, 1, 32, 32, device=device)
                target_video = torch.rand(1, 4, 8, 3, 64, 64, device=device)
                ref_audio = torch.randn(1, 4, 1, 32, 32, device=device)
                ref_video = torch.rand(1, 4, 8, 3, 64, 64, device=device)
            else:
                raise e
        labels_rf = torch.ones(1, dtype=torch.long, device=device)

        attacker = RefereeMultiModalPGD(
            model,
            attack_mode='joint',
            eps_audio=0.1,     # More aggressive for testing
            eps_video=0.5,     # More aggressive for testing
            max_iter=10,       # Fewer iterations for quick test
            verbose=False
        )

        adv_audio, adv_video, info = attacker.generate(
            target_audio, target_video, ref_audio, ref_video, labels_rf
        )

        print("✅ Attack completed successfully!")
        print(f"   Final loss: {info['losses'][-1]:.4f}")

        audio_changed = not torch.allclose(adv_audio, target_audio, atol=1e-6)
        video_changed = not torch.allclose(adv_video, target_video, atol=1e-6)

        print(f"   Audio modified: {audio_changed}")
        print(f"   Video modified: {video_changed}")

        print("\n🚀 Ready to run full demo!")
        print("   Run: python demo.py")

        # Clean up memory
        if device == 'cuda':
            torch.cuda.empty_cache()

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

        # Clean up memory on error too
        if device == 'cuda':
            torch.cuda.empty_cache()
        return False

    return True

if __name__ == "__main__":
    main()