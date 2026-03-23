"""
T-Net (STN3d) ablation experiment.

Replaces the input spatial transformer (STN3d) with an identity transform,
then measures classification accuracy on a 50-sample ModelNet40 test subset
to quantify how much the learned input alignment contributes to accuracy.
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import torch
import torch.nn as nn
import numpy as np


# ---------------------------------------------------------------------------
# Patched PointNet forward with identity STN
# ---------------------------------------------------------------------------

class IdentitySTN3d(nn.Module):
    """Drop-in replacement for STN3d that returns the 3×3 identity matrix."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        return torch.eye(3, device=x.device).unsqueeze(0).repeat(batch_size, 1, 1)


def ablate_tnet(model: nn.Module) -> nn.Module:
    """Replace STN3d inside a PointNet model with IdentitySTN3d in-place."""
    from src.models.pointnet_utils import PointNetEncoder

    for module in model.modules():
        # PointNetEncoder holds stn (STN3d) as first submodule
        if hasattr(module, "stn"):
            module.stn = IdentitySTN3d()
    return model


# ---------------------------------------------------------------------------
# Accuracy evaluation
# ---------------------------------------------------------------------------

def evaluate(model: nn.Module, point_clouds: torch.Tensor, labels: torch.Tensor) -> float:
    """Return accuracy (0–1) for a batch of point clouds."""
    model.eval()
    with torch.no_grad():
        logits, _ = model(point_clouds)
        preds = logits.argmax(dim=1)
    return (preds == labels).float().mean().item()


def main():
    n_samples = 50
    checkpoint_path = "checkpoints/pointnet2_ssg/best_model.pth"

    from src.models.pointnet_cls import get_model

    # Load original model
    original = get_model(40, normal_channel=False)
    original.eval()
    state = torch.load(checkpoint_path, map_location="cpu")
    original.load_state_dict(state["model_state_dict"])

    # Load ablated copy
    import copy
    ablated = copy.deepcopy(original)
    ablate_tnet(ablated)
    ablated.eval()

    # Synthetic point clouds (replace with real ModelNet40 test loader)
    rng = np.random.default_rng(0)
    pts = rng.standard_normal((n_samples, 3, 1024)).astype(np.float32)
    labels = torch.randint(0, 40, (n_samples,))
    clouds = torch.from_numpy(pts)

    acc_orig = evaluate(original, clouds, labels)
    acc_abl = evaluate(ablated, clouds, labels)
    drop = acc_orig - acc_abl

    print("\nT-Net Ablation Results")
    print("=" * 40)
    print(f"  Original  : {acc_orig * 100:.1f}%")
    print(f"  Ablated   : {acc_abl * 100:.1f}%")
    print(f"  Drop      : {drop * 100:.1f}%")
    print("=" * 40)
    print("(Run with real ModelNet40 test data for meaningful numbers)")


if __name__ == "__main__":
    main()
