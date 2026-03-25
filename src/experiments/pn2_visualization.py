"""
PointNet++ SA-level visualization — Exp 3.

Two panels per shape, 2 rows × 3 columns:
  Row 0 — groupings:     SA1 centroids | SA2 centroids | SA3 (all SA2 used)
  Row 1 — critical pts:  SA1 local winners (original input pts)
                       | SA2 local winners (SA1 centroid pts)
                       | SA3 global winners traced back to original input pts

Results saved to results/exp_3_pn2_sa_visualization/
"""
import sys
import os
import types

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from src.models.pointnet2_utils import (
    farthest_point_sample,
    query_ball_point,
    index_points,
)


# ---------------------------------------------------------------------------
# SA layer instrumentation
# ---------------------------------------------------------------------------

def instrument_sa(sa) -> None:
    """Monkey-patch sa.forward to capture fps_idx, ball_idx, and pool_argmax.

    After each forward call the module gains three attributes:
      _fps_idx    [B, npoint] or None (group_all)   — which parent points are centroids
      _ball_idx   [B, npoint, nsample] or None       — which parent points are in each ball
      _pool_argmax [B, C_out, npoint]                — which nsample neighbor won each channel
    """
    def forward(self, xyz, points):
        xyz = xyz.permute(0, 2, 1)              # [B, N, 3]
        if points is not None:
            points = points.permute(0, 2, 1)   # [B, N, D]

        if self.group_all:
            B, N, C = xyz.shape
            new_xyz = torch.zeros(B, 1, C, device=xyz.device)
            grouped_xyz = xyz.view(B, 1, N, C)
            if points is not None:
                new_points = torch.cat([grouped_xyz, points.view(B, 1, N, -1)], dim=-1)
            else:
                new_points = grouped_xyz
            self._fps_idx = None
            self._ball_idx = None
        else:
            B, N, C = xyz.shape
            fps_idx = farthest_point_sample(xyz, self.npoint)            # [B, npoint]
            new_xyz = index_points(xyz, fps_idx)                         # [B, npoint, 3]
            ball_idx = query_ball_point(                                 # [B, npoint, nsample]
                self.radius, self.nsample, xyz, new_xyz
            )
            grouped_xyz = index_points(xyz, ball_idx)                    # [B, npoint, nsample, 3]
            grouped_xyz_norm = grouped_xyz - new_xyz.view(B, self.npoint, 1, C)
            if points is not None:
                grouped_points = index_points(points, ball_idx)
                new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1)
            else:
                new_points = grouped_xyz_norm
            self._fps_idx = fps_idx.detach().cpu()    # [B, npoint]
            self._ball_idx = ball_idx.detach().cpu()  # [B, npoint, nsample]

        # MLP stack
        new_points = new_points.permute(0, 3, 2, 1)  # [B, C+D, nsample, npoint]
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))

        # Local max-pool — capture argmax
        pool_out, pool_argmax = torch.max(new_points, 2)  # [B, C_out, npoint]
        self._pool_argmax = pool_argmax.detach().cpu()

        new_xyz = new_xyz.permute(0, 2, 1)  # [B, 3, npoint]
        return new_xyz, pool_out

    sa.forward = types.MethodType(forward, sa)


# ---------------------------------------------------------------------------
# Critical point extraction
# ---------------------------------------------------------------------------

def sa1_critical_input_pts(sa1) -> torch.Tensor:
    """Original input-point indices that won at least one SA1 local pool channel.

    Returns: 1-D LongTensor, values in 0..N-1 (N = number of original input points).
    """
    argmax = sa1._pool_argmax[0]  # [C_out, npoint=512]
    ball = sa1._ball_idx[0]       # [npoint=512, nsample=32]
    npoint = argmax.shape[1]
    # For centroid i and channel c: ball[i, argmax[c, i]] = original input index
    argmax_t = argmax.t()         # [512, C_out]
    orig = ball[torch.arange(npoint).unsqueeze(1), argmax_t]  # [512, C_out]
    return orig.reshape(-1).unique()


def sa2_critical_sa1_centroids(sa2) -> torch.Tensor:
    """SA1-centroid indices that won at least one SA2 local pool channel.

    Returns: 1-D LongTensor, values in 0..511 (indexing l1_xyz).
    """
    argmax = sa2._pool_argmax[0]  # [C_out, npoint=128]
    ball = sa2._ball_idx[0]       # [npoint=128, nsample=64] — indexes into SA1 centroids
    npoint = argmax.shape[1]
    argmax_t = argmax.t()         # [128, C_out]
    sa1_idx = ball[torch.arange(npoint).unsqueeze(1), argmax_t]  # [128, C_out]
    return sa1_idx.reshape(-1).unique()


def sa3_critical_original_pts(sa1, sa2, sa3) -> torch.Tensor:
    """SA3 global pool winners traced all the way back to original input-point indices.

    SA3 pool_argmax[c] → SA2 centroid → SA1 centroid (via sa2._fps_idx) →
        original input point (via sa1._fps_idx).

    Returns: 1-D LongTensor, values in 0..N-1.
    """
    # SA3: group_all over 128 SA2 centroids; argmax indexes directly into those 128
    argmax = sa3._pool_argmax[0, :, 0]          # [1024] — which SA2 centroid won each channel
    critical_sa2 = argmax.unique()              # subset of 0..127

    # Trace SA2 centroid → SA1 centroid
    sa2_fps = sa2._fps_idx[0]                   # [128] — which SA1 centroid became each SA2 centroid
    critical_sa1 = sa2_fps[critical_sa2]        # subset of 0..511

    # Trace SA1 centroid → original input point
    sa1_fps = sa1._fps_idx[0]                   # [512] — which original pt became each SA1 centroid
    critical_orig = sa1_fps[critical_sa1]       # subset of 0..N-1
    return critical_orig.unique()


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def _depth_colors(pts: np.ndarray, cmap: str = "plasma") -> np.ndarray:
    """Return RGBA colors for pts based on z coordinate (depth cueing)."""
    z = pts[:, 2]
    z_min, z_max = z.min(), z.max()
    norm = (z - z_min) / (z_max - z_min + 1e-8)
    return plt.get_cmap(cmap)(norm)


def _scatter(ax, pts, s, c, alpha, label=None):
    """Scatter with flat color (used for background gray points)."""
    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=s, c=c, alpha=alpha, label=label)


def _scatter_depth(ax, pts, s, alpha, cmap="plasma", label=None):
    """Scatter with per-point z-based depth coloring."""
    colors = _depth_colors(pts, cmap=cmap)
    colors[:, 3] = alpha  # override alpha uniformly
    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=s, c=colors, label=label)


def visualize_shape(
    original_pts: np.ndarray,
    model,
    shape_name: str,
    save_path: str,
) -> None:
    """Run one forward pass and produce a 2×3 figure for this shape."""
    sa1, sa2, sa3 = model.sa1, model.sa2, model.sa3

    cloud_t = torch.from_numpy(original_pts.T).unsqueeze(0)  # [1, 3, N]
    with torch.no_grad():
        model(cloud_t)

    # --- centroid xyz for plotting ---
    # Recover centroid coords from fps_idx
    sa1_centroids = original_pts[sa1._fps_idx[0].numpy()]   # [512, 3]
    # SA2 centroids are FPS-selected SA1 centroids
    sa2_centroids = sa1_centroids[sa2._fps_idx[0].numpy()]  # [128, 3]

    # --- critical point indices ---
    crit_sa1 = sa1_critical_input_pts(sa1).numpy()        # original input pts
    crit_sa2 = sa2_critical_sa1_centroids(sa2).numpy()    # SA1 centroid indices
    crit_sa3 = sa3_critical_original_pts(sa1, sa2, sa3).numpy()  # original input pts

    n1 = len(crit_sa1)
    n2 = len(crit_sa2)
    n3 = len(crit_sa3)
    print(f"  {shape_name}: SA1 critical={n1}/1024  SA2 critical SA1-centroids={n2}/512  SA3 traced={n3}/1024")

    fig, axes = plt.subplots(
        2, 3, figsize=(15, 9), subplot_kw={"projection": "3d"}
    )

    # ---- Row 0: groupings ----
    for col, (title, cdata, sz) in enumerate([
        ("SA1 groupings — 512 FPS centroids",  sa1_centroids, 15),
        ("SA2 groupings — 128 FPS centroids",  sa2_centroids, 25),
    ]):
        ax = axes[0, col]
        _scatter_depth(ax, original_pts, s=1, alpha=0.15)
        _scatter_depth(ax, cdata, s=sz, alpha=0.85)
        ax.set_title(title, fontsize=9)
        ax.set_box_aspect([1, 1, 1])
        ax.axis("off")

    axes[0, 2].set_visible(False)

    # ---- Row 1: critical points ----

    # Row 1, Col 0: SA1 winners = original input pts
    crit_mask_sa1 = np.zeros(len(original_pts), dtype=bool)
    crit_mask_sa1[crit_sa1] = True

    ax = axes[1, 0]
    _scatter_depth(ax, original_pts[~crit_mask_sa1], s=1, alpha=0.15)
    _scatter_depth(ax, original_pts[crit_mask_sa1], s=8, alpha=0.9)
    ax.set_title(f"SA1 local max-pool winners\n{n1} / 1024 input pts", fontsize=9)
    ax.set_box_aspect([1, 1, 1])
    ax.axis("off")

    # Row 1, Col 1: SA2 winners = SA1 centroids (by index)
    crit_mask_sa2 = np.zeros(len(sa1_centroids), dtype=bool)
    crit_mask_sa2[crit_sa2] = True

    ax = axes[1, 1]
    _scatter_depth(ax, original_pts, s=1, alpha=0.1)
    _scatter_depth(ax, sa1_centroids[~crit_mask_sa2], s=2, alpha=0.2)
    _scatter_depth(ax, sa1_centroids[crit_mask_sa2], s=10, alpha=0.9)
    ax.set_title(f"SA2 local max-pool winners\n{n2} / 512 SA1 centroids", fontsize=9)
    ax.set_box_aspect([1, 1, 1])
    ax.axis("off")

    # Row 1, Col 2: SA3 winners traced back to original input pts
    crit_mask_sa3 = np.zeros(len(original_pts), dtype=bool)
    crit_mask_sa3[crit_sa3] = True

    ax = axes[1, 2]
    _scatter_depth(ax, original_pts[~crit_mask_sa3], s=1, alpha=0.15)
    _scatter_depth(ax, original_pts[crit_mask_sa3], s=8, alpha=0.9)
    ax.set_title(f"SA3 global max-pool winners → input pts\n{n3} / 1024 input pts", fontsize=9)
    ax.set_box_aspect([1, 1, 1])
    ax.axis("off")

    fig.suptitle(
        f"PointNet++ SSG — SA groupings + critical points: {shape_name}",
        fontsize=12,
    )
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved: {save_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    from src.models.pointnet2_cls_ssg import get_model
    from src.data_utils.ModelNetDatDataset import ModelNetDatDataset

    checkpoint_path = "checkpoints/pointnet2_ssg/best_model.pth"
    dat_path = "data/modelnet40_test_1024pts.dat"
    results_dir = "results/exp_3_pn2_sa_visualization"

    # Same instances as exp 1 for direct comparison
    plot_items = [
        (0,  "airplane", 0),
        (8,  "chair",    0),
        (17, "guitar",   0),
        (19, "lamp",     0),
        (33, "table",    0),
        (19, "lamp",     1),
    ]

    print(f"Loading checkpoint: {checkpoint_path}")
    model = get_model(num_class=40, normal_channel=False)
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model_state_dict"])
    model.eval()

    instrument_sa(model.sa1)
    instrument_sa(model.sa2)
    instrument_sa(model.sa3)

    print(f"Loading test data: {dat_path}")
    dataset = ModelNetDatDataset(dat_path, npoints=1024, use_normals=False)

    needed = {}
    for cls_idx, _, _ in plot_items:
        needed[cls_idx] = needed.get(cls_idx, 0) + 1

    collected = {}  # cls_idx → list of (pts_xyz [N,3], cloud_tensor [1,3,N])
    for pts_tensor, label in dataset:
        if label in needed and len(collected.get(label, [])) < needed[label]:
            pts_xyz = pts_tensor[:, :3].numpy()
            cloud_tensor = pts_tensor[:, :3].T.unsqueeze(0)
            collected.setdefault(label, []).append((pts_xyz, cloud_tensor))
        if all(len(collected.get(c, [])) >= n for c, n in needed.items()):
            break

    for cls_idx, cls_name, sample_rank in plot_items:
        pts_xyz, cloud_tensor = collected[cls_idx][sample_rank]
        suffix = f"_{sample_rank + 1}" if needed[cls_idx] > 1 else ""
        shape_name = f"{cls_name}{suffix}"
        print(f"\nProcessing: {shape_name}")
        save_path = os.path.join(results_dir, f"{shape_name}.png")
        visualize_shape(pts_xyz, model, shape_name=shape_name, save_path=save_path)


if __name__ == "__main__":
    main()
