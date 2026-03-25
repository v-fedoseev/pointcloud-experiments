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
        # output: (1, 1024, N) — bn3 output is the exact pre-pool feature map
        # (PointNetEncoder forward: x = self.bn3(self.conv3(x)); x = torch.max(x, 2))
        captured["pre_pool"] = output.detach()

    # Hook bn3 (not conv3) to get the full bn3(conv3(x)) output, shape (1, 1024, N).
    # Hooking conv3's input[0] would give (1, 128, N) — wrong dimension.
    handle = model.feat.bn3.register_forward_hook(hook)

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
    """Load trained PointNet v1, run critical point extraction on real ModelNet40 test samples,
    save a single grid plot to results/exp_1_critical_points/plot.png."""
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
    from src.data_utils.ModelNetDatDataset import ModelNetDatDataset

    checkpoint_path = "checkpoints/pointnet_cls/best_model.pth"
    dat_path = "data/modelnet40_test_1024pts.dat"
    results_dir = "results/exp_1_critical_points"
    plot_path = os.path.join(results_dir, "plot.png")

    # Classes chosen to stress-test the hypothesis: structural variety
    # (thin structures, flat planes, curved surfaces, legs/edges)
    # Two lamp samples to compare variation within the same class.
    plot_items = [
        (0,  "airplane",  0),  # (class_idx, label, sample_rank)
        (8,  "chair",     0),
        (17, "guitar",    0),
        (19, "lamp",      0),
        (33, "table",     0),
        (19, "lamp",      1),  # second lamp
    ]
    n_cols = 3
    n_rows = 2  # 6 panels in a 2×3 grid

    print(f"Loading checkpoint: {checkpoint_path}")
    classifier = get_model(40, normal_channel=True)
    classifier.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    classifier.eval()

    print(f"Loading test data: {dat_path}")
    dataset = ModelNetDatDataset(dat_path, npoints=1024, use_normals=True)

    # Collect up to 2 samples per class
    needed = {}  # cls_idx → how many samples needed
    for cls_idx, _, _ in plot_items:
        needed[cls_idx] = needed.get(cls_idx, 0) + 1

    collected = {}  # cls_idx → list of (pts_xyz, cloud_tensor)
    for pts_tensor, label in dataset:
        if label in needed and len(collected.get(label, [])) < needed[label]:
            pts_xyz = pts_tensor[:, :3].numpy()
            cloud_tensor = pts_tensor.T.unsqueeze(0)
            collected.setdefault(label, []).append((pts_xyz, cloud_tensor))
        if all(len(collected.get(c, [])) >= n for c, n in needed.items()):
            break

    os.makedirs(results_dir, exist_ok=True)

    fig = plt.figure(figsize=(n_cols * 4, n_rows * 4))
    for plot_idx, (cls_idx, cls_name, sample_rank) in enumerate(plot_items, start=1):
        pts_xyz, cloud_tensor = collected[cls_idx][sample_rank]
        critical_idx = extract_critical_points(classifier, cloud_tensor)
        n_critical = len(critical_idx)
        suffix = f" #{sample_rank + 1}" if needed[cls_idx] > 1 else ""
        print(f"  {cls_name}{suffix}: {n_critical} critical / 1024 points")

        ax = fig.add_subplot(n_rows, n_cols, plot_idx, projection="3d")
        mask = np.zeros(len(pts_xyz), dtype=bool)
        mask[critical_idx.numpy()] = True
        ax.scatter(*pts_xyz[~mask].T, s=1, c="dimgray", alpha=0.5)
        ax.scatter(*pts_xyz[mask].T, s=6, c="red", alpha=0.9)
        ax.set_title(f"{cls_name}{suffix}\n{n_critical} critical / 1024", fontsize=9)
        ax.set_box_aspect([1, 1, 1])
        ax.axis("off")

    fig.suptitle("PointNet v1 — Critical Point Sets (ModelNet40 test)", fontsize=12)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"\nPlot saved → {plot_path}")


if __name__ == "__main__":
    main()
