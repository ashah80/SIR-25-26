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
                self.audio_proj = nn.Linear(128*66, 128)
                self.video_proj = nn.Linear(16*3*224*224, 128)
                self.classifier = nn.Linear(256, 2)

            def forward(self, target_vis, target_aud, ref_vis, ref_aud, **kwargs):
                B = target_vis.shape[0]
                audio_feat = self.audio_proj(target_aud.view(B, -1))
                video_feat = self.video_proj(target_vis.view(B, -1))
                combined = torch.cat([audio_feat, video_feat], dim=1)
                logits_rf = self.classifier(combined)
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

        # Create small test data
        target_audio = torch.randn(1, 8, 1, 128, 66, device=device)
        target_video = torch.rand(1, 8, 16, 3, 224, 224, device=device)
        ref_audio = torch.randn(1, 8, 1, 128, 66, device=device)
        ref_video = torch.rand(1, 8, 16, 3, 224, 224, device=device)
        labels_rf = torch.ones(1, dtype=torch.long, device=device)

        attacker = RefereeMultiModalPGD(
            model,
            attack_mode='joint',
            max_iter=5,
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

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True

if __name__ == "__main__":
    main()