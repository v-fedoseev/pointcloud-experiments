"""
T-Net (STN3d) ablation experiment.

Replaces the input spatial transformer (STN3d) with an identity transform,
then measures classification accuracy on a 50-sample ModelNet40 test subset
to quantify how much the learned input alignment contributes to accuracy.
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import copy
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader


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
    for module in model.modules():
        # PointNetEncoder holds stn (STN3d) as first submodule
        if hasattr(module, "stn"):
            module.stn = IdentitySTN3d()
    return model


# ---------------------------------------------------------------------------
# Accuracy evaluation
# ---------------------------------------------------------------------------

def evaluate(model: nn.Module, loader: DataLoader) -> float:
    """Return accuracy (0–1) over a DataLoader."""
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for pts, labels in loader:
            # pts: (B, N, C) → transpose to (B, C, N) for PointNet
            pts = pts.transpose(2, 1)
            logits, _ = model(pts)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    checkpoint_path = "checkpoints/pointnet_cls/best_model.pth"
    dat_path = "data/modelnet40_test_1024pts.dat"
    results_dir = "results/exp_2_tnet_ablation"
    plot_path = os.path.join(results_dir, "plot.png")

    from src.models.pointnet_cls import get_model
    from src.data_utils.ModelNetDatDataset import ModelNetDatDataset

    # Load original model (normal_channel=True matches the training checkpoint)
    original = get_model(40, normal_channel=True)
    state = torch.load(checkpoint_path, map_location="cpu")
    # Checkpoint saved as raw state dict (no wrapper keys)
    original.load_state_dict(state)
    original.eval()

    # Load ablated copy
    ablated = copy.deepcopy(original)
    ablate_tnet(ablated)
    ablated.eval()

    # Full ModelNet40 test set
    dataset = ModelNetDatDataset(dat_path, npoints=1024, use_normals=True)
    loader = DataLoader(dataset, batch_size=64, shuffle=False)

    acc_orig = evaluate(original, loader)
    acc_abl = evaluate(ablated, loader)
    drop = acc_orig - acc_abl

    print("\nT-Net Ablation Results")
    print("=" * 40)
    print(f"  Original  : {acc_orig * 100:.1f}%")
    print(f"  Ablated   : {acc_abl * 100:.1f}%")
    print(f"  Drop      : {drop * 100:.1f}%")
    print("=" * 40)

    # Bar chart
    os.makedirs(results_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 4))
    bars = ax.bar(["Original", "T-Net ablated"], [acc_orig * 100, acc_abl * 100],
                  color=["steelblue", "salmon"], width=0.5)
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 100)
    n_total = len(dataset)
    ax.set_title(f"T-Net Ablation — n={n_total} ModelNet40 test samples")
    for bar, val in zip(bars, [acc_orig, acc_abl]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{val * 100:.1f}%", ha="center", va="bottom", fontsize=11)
    ax.annotate(f"Drop: {drop * 100:.1f}%", xy=(0.5, 0.5), xycoords="axes fraction",
                ha="center", fontsize=10, color="dimgray")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"\nPlot saved → {plot_path}")


if __name__ == "__main__":
    main()
