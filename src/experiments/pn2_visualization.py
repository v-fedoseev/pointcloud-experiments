"""
PointNet++ Set Abstraction (SA) layer visualization.

Hooks into each PointNetSetAbstraction layer to capture ball-query centroids
and grouped points, then renders groupings at SA level 1 and 2 for a
cylinder-like and a flat-plane-like synthetic shape.
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from typing import List, Dict


# ---------------------------------------------------------------------------
# Hook infrastructure
# ---------------------------------------------------------------------------

class SAHook:
    """Captures (centroids, grouped_xyz) from a PointNetSetAbstraction forward."""

    def __init__(self):
        self.captures: List[Dict] = []
        self._handles = []

    def register(self, sa_module) -> None:
        handle = sa_module.register_forward_hook(self._hook_fn)
        self._handles.append(handle)

    def _hook_fn(self, module, inputs, output):
        # output: (new_xyz, new_features)  — new_xyz is (B, npoint, 3)
        new_xyz = output[0].detach().cpu()
        self.captures.append({"centroids": new_xyz})

    def remove(self):
        for h in self._handles:
            h.remove()
        self._handles.clear()


def register_sa_hooks(model: torch.nn.Module) -> SAHook:
    """Attach hooks to all PointNetSetAbstraction layers in the model."""
    from src.models.pointnet_utils import PointNetSetAbstraction

    hook = SAHook()
    for module in model.modules():
        if isinstance(module, PointNetSetAbstraction):
            hook.register(module)
    return hook


# ---------------------------------------------------------------------------
# Synthetic shapes
# ---------------------------------------------------------------------------

def cylinder_cloud(n: int = 1024, rng=None) -> np.ndarray:
    """Unit cylinder point cloud, (N, 3)."""
    if rng is None:
        rng = np.random.default_rng(1)
    theta = rng.uniform(0, 2 * np.pi, n)
    z = rng.uniform(-1, 1, n)
    x = np.cos(theta)
    y = np.sin(theta)
    return np.stack([x, y, z], axis=1).astype(np.float32)


def flat_plane_cloud(n: int = 1024, rng=None) -> np.ndarray:
    """Flat XY plane point cloud at z≈0, (N, 3)."""
    if rng is None:
        rng = np.random.default_rng(2)
    x = rng.uniform(-1, 1, n)
    y = rng.uniform(-1, 1, n)
    z = rng.normal(0, 0.02, n)
    return np.stack([x, y, z], axis=1).astype(np.float32)


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def visualize_sa_groupings(
    point_cloud: np.ndarray,
    captures: List[Dict],
    shape_name: str,
    save_path: str | None = None,
) -> None:
    """Plot centroids from SA levels 1 and 2 on top of the raw point cloud."""
    n_levels = min(len(captures), 2)
    fig, axes = plt.subplots(1, n_levels, figsize=(6 * n_levels, 5), subplot_kw={"projection": "3d"})
    if n_levels == 1:
        axes = [axes]

    colors = ["dodgerblue", "orange"]
    for i in range(n_levels):
        ax = axes[i]
        centroids = captures[i]["centroids"][0].numpy()  # (npoint, 3)

        ax.scatter(*point_cloud.T, s=0.5, c="lightgray", alpha=0.3)
        ax.scatter(*centroids.T, s=20, c=colors[i], alpha=0.9, label=f"{len(centroids)} centroids")
        ax.set_title(f"{shape_name} — SA level {i + 1}")
        ax.legend(loc="upper right")
        ax.set_box_aspect([1, 1, 1])

    plt.suptitle(f"Ball-query SA groupings: {shape_name}", fontsize=13)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"Saved: {save_path}")
    else:
        plt.show()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    from src.models.pointnet_cls import get_model

    checkpoint_path = "checkpoints/pointnet2_ssg/best_model.pth"
    results_dir = "results/pn2_visualization"

    print(f"Loading checkpoint: {checkpoint_path}")
    model = get_model(40, normal_channel=False)
    model.eval()
    state = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state["model_state_dict"])

    shapes = {
        "cylinder": cylinder_cloud(),
        "flat_plane": flat_plane_cloud(),
    }

    for shape_name, pts in shapes.items():
        cloud_tensor = torch.from_numpy(pts.T).unsqueeze(0)  # (1, 3, N)

        hook = register_sa_hooks(model)
        with torch.no_grad():
            model(cloud_tensor)
        hook.remove()

        print(f"{shape_name}: captured {len(hook.captures)} SA layers")
        for lvl, cap in enumerate(hook.captures):
            print(f"  SA level {lvl + 1}: {cap['centroids'].shape[1]} centroids")

        save_path = os.path.join(results_dir, f"{shape_name}_sa_groupings.png")
        visualize_sa_groupings(pts, hook.captures, shape_name=shape_name, save_path=save_path)

    print("Done.")


if __name__ == "__main__":
    main()
