#!/usr/bin/env python3
"""
Debug test to verify the fixed adversarial attack implementation.
"""

import sys
from pathlib import Path
import torch
import torch.nn as nn

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

def test_model_and_data():
    """Test model and data compatibility"""
    print("🔍 Debug Test: Model and Data Compatibility")
    print("=" * 60)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    try:
        # Import and run basic compatibility test
        from adversarial_attacks.demo import create_dummy_data, main

        # Test data creation
        print("\n📊 Testing data creation...")
        target_audio, target_video, ref_audio, ref_video, labels_rf = create_dummy_data(1, device)

        print(f"✅ Data created successfully:")
        print(f"  Audio shape: {target_audio.shape}")
        print(f"  Video shape: {target_video.shape}")
        print(f"  Labels: {labels_rf}")

        # Test simple dummy model
        print("\n🤖 Testing ultra-simple dummy model...")

        class UltraSimpleDummy(nn.Module):
            def __init__(self):
                super().__init__()
                self.classifier = nn.Linear(2, 2)

            def forward(self, target_vis, target_aud, ref_vis, ref_aud, **kwargs):
                B = target_vis.shape[0]
                audio_feat = torch.mean(target_aud)
                video_feat = torch.mean(target_vis)
                combined = torch.stack([audio_feat, video_feat]).unsqueeze(0).repeat(B, 1)
                return self.classifier(combined), torch.randn(B, 2, device=target_vis.device)

        model = UltraSimpleDummy().to(device)

        # Test forward pass
        print("\n⚡ Testing forward pass...")
        logits_rf, logits_id = model(target_video, target_audio, ref_video, ref_audio)
        print(f"✅ Forward pass successful:")
        print(f"  RF logits shape: {logits_rf.shape}")
        print(f"  ID logits shape: {logits_id.shape}")

        # Test gradient flow
        print("\n🌊 Testing gradient flow...")
        target_audio.requires_grad_(True)
        target_video.requires_grad_(True)

        loss = nn.functional.cross_entropy(logits_rf, labels_rf)
        print(f"  Loss: {loss.item():.4f}")

        loss.backward()

        print(f"✅ Gradient flow successful:")
        print(f"  Audio grad exists: {target_audio.grad is not None}")
        print(f"  Video grad exists: {target_video.grad is not None}")

        if target_audio.grad is not None:
            print(f"  Audio grad max: {target_audio.grad.max().item():.6f}")
        if target_video.grad is not None:
            print(f"  Video grad max: {target_video.grad.max().item():.6f}")

        # Test attack wrapper
        print("\n🎯 Testing attack wrapper...")
        from adversarial_attacks import create_attack_wrapper

        wrapper = create_attack_wrapper(model, ref_audio, ref_video, labels_rf)

        # Reset gradients
        target_audio.grad = None
        target_video.grad = None
        target_audio.requires_grad_(True)
        target_video.requires_grad_(True)

        attack_loss = wrapper(target_audio, target_video)
        print(f"  Attack loss: {attack_loss.item():.4f}")

        attack_loss.backward()
        print(f"✅ Attack wrapper working:")
        print(f"  Gradients after attack wrapper: Audio={target_audio.grad is not None}, Video={target_video.grad is not None}")

        # Test confidence computation
        print("\n📈 Testing confidence computation...")
        confidence = wrapper.get_confidence(target_audio.detach(), target_video.detach())
        print(f"✅ Confidence computation working:")
        for key, value in confidence.items():
            print(f"  {key}: {value}")

        print("\n🎉 ALL BASIC TESTS PASSED!")
        print("=" * 60)
        return True

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_model_and_data()
    if success:
        print("\n🚀 Ready to run full demo with fixed implementation!")
    else:
        print("\n💥 Still has issues - needs more debugging")