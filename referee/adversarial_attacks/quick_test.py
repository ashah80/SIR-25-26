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
                # Use pooling to reduce size like in demo.py
                self.audio_pool = nn.AdaptiveAvgPool2d((8, 8))
                self.video_pool = nn.AdaptiveAvgPool3d((4, 16, 16))

                self.audio_proj = nn.Linear(8 * 8, 32)
                self.video_proj = nn.Linear(4 * 16 * 16 * 3, 32)
                self.classifier = nn.Sequential(
                    nn.Linear(64, 32),
                    nn.ReLU(),
                    nn.Linear(32, 2)
                )

            def forward(self, target_vis, target_aud, ref_vis, ref_aud, **kwargs):
                B = target_vis.shape[0]

                # Pool inputs to manageable size
                audio_pooled = self.audio_pool(target_aud.view(-1, target_aud.shape[-2], target_aud.shape[-1]))
                audio_pooled = audio_pooled.view(B, -1)

                video_reshaped = target_vis.view(-1, target_vis.shape[-3], target_vis.shape[-2], target_vis.shape[-1])
                video_pooled = self.video_pool(video_reshaped.permute(0, 2, 1, 3, 4))
                video_pooled = video_pooled.view(B, -1)

                # Project features
                audio_feat = self.audio_proj(audio_pooled)
                video_feat = self.video_proj(video_pooled)
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

        # Create small test data to avoid memory issues
        target_audio = torch.randn(1, 4, 1, 32, 32, device=device)
        target_video = torch.rand(1, 4, 8, 3, 64, 64, device=device)
        ref_audio = torch.randn(1, 4, 1, 32, 32, device=device)
        ref_video = torch.rand(1, 4, 8, 3, 64, 64, device=device)
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