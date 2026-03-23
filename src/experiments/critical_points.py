"""
Critical point set extraction and visualization for PointNet.

The "critical point set" is defined in PointNet (Qi et al. 2017, §4.3) as the
minimal subset of input points that produces the same global feature vector —
i.e., the points that survive the global max-pool operation.
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from src.models.pointnet_cls import get_model


def extract_critical_points(model: torch.nn.Module, point_cloud: torch.Tensor) -> torch.Tensor:
    """Return indices of critical points for a single point cloud.

    Args:
        model: PointNet classification model (in eval mode).
        point_cloud: (1, 3, N) tensor — a single point cloud.

    Returns:
        indices: (K,) LongTensor of the N indices that are critical
                 (i.e., each index is the argmax winner of at least one
                 feature dimension after the global max-pool).
    """
    captured = {}

    def hook(module, input, output):
        # input[0]: (1, 1024, N) pre-pool feature map
        captured["pre_pool"] = input[0].detach()

    # Register hook on the global max-pool (torch.max in forward)
    # PointNet's get_model forward: feat = torch.max(x, 2, keepdim=True)[0]
    # We hook the last conv layer before max-pool instead.
    handle = model.feat.conv3.register_forward_hook(hook)

    with torch.no_grad():
        model(point_cloud)

    handle.remove()

    pre_pool = captured["pre_pool"]  # (1, C, N)
    # argmax across the N dimension for each feature channel
    argmax_indices = pre_pool[0].argmax(dim=1)  # (C,)
    critical_indices = argmax_indices.unique()
    return critical_indices


def visualize_critical_points(
    point_cloud: np.ndarray,
    critical_indices: torch.Tensor,
    label: str,
    save_path: str | None = None,
) -> None:
    """Render point cloud with critical points highlighted in red.

    Args:
        point_cloud: (N, 3) numpy array.
        critical_indices: (K,) LongTensor of critical point indices.
        label: class name string for the plot title.
        save_path: if given, save figure here instead of showing.
    """
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    mask = np.zeros(len(point_cloud), dtype=bool)
    mask[critical_indices.numpy()] = True

    ax.scatter(*point_cloud[~mask].T, s=1, c="lightgray", alpha=0.4, label="non-critical")
    ax.scatter(*point_cloud[mask].T, s=8, c="red", alpha=0.9, label=f"critical ({mask.sum()})")

    ax.set_title(f"Critical points — {label}")
    ax.legend(loc="upper right")
    ax.set_box_aspect([1, 1, 1])
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150)
        plt.close()
    else:
        plt.show()


def main():
    """Load pretrained PointNet2-SSG, run on 5 samples per class (10 classes), save figures."""
    import importlib

    checkpoint_path = "checkpoints/pointnet2_ssg/best_model.pth"
    results_dir = "results/critical_points"
    n_samples_per_class = 5
    n_classes = 10

    print(f"Loading checkpoint: {checkpoint_path}")
    classifier = get_model(40, normal_channel=False)
    classifier.eval()

    state = torch.load(checkpoint_path, map_location="cpu")
    classifier.load_state_dict(state["model_state_dict"])

    # Generate synthetic point clouds (replace with real ModelNet40 data loader)
    rng = np.random.default_rng(42)
    class_names = [f"class_{i:02d}" for i in range(n_classes)]

    for cls_idx, cls_name in enumerate(class_names):
        for sample_idx in range(n_samples_per_class):
            pts = rng.standard_normal((1024, 3)).astype(np.float32)
            pts /= np.linalg.norm(pts, axis=1, keepdims=True) + 1e-8  # normalize to unit sphere

            cloud_tensor = torch.from_numpy(pts.T).unsqueeze(0)  # (1, 3, N)
            critical_idx = extract_critical_points(classifier, cloud_tensor)

            save_path = os.path.join(results_dir, f"{cls_name}_sample{sample_idx:02d}.png")
            visualize_critical_points(pts, critical_idx, label=cls_name, save_path=save_path)
            print(f"  {cls_name} sample {sample_idx}: {len(critical_idx)} critical pts → {save_path}")

    print("Done.")


if __name__ == "__main__":
    main()
