---
source: https://github.com/GitBoSun/PointNet_vis
date_captured: 2026-03-24
author: GitBoSun
type: documentation
tags: [pointnet, critical-points, upper-bound-shape, visualization, max-pooling, implementation-reference]
---

# PointNet Critical Points & Upper Bound Shape Visualization (GitBoSun/PointNet_vis)

> **Source Link:** [https://github.com/GitBoSun/PointNet_vis](https://github.com/GitBoSun/PointNet_vis)
> **Gist:** Prior implementation of exactly the critical point + upper bound shape visualization in experiment deliverable 1 — confirms the hook-the-max-pool approach and reveals the feature dimensionality used.

## Core Gist / Summary

This repository implements visualization of PointNet's critical points and upper-bound shape — the two constructs from Section 4.3 of the paper. It is the closest existing reference implementation to experiment deliverable 1 in this repo, and its code structure (model modification → data extraction → visualization) mirrors the approach we will take.

### Critical Point Extraction: The Method

The key insight is that critical points are identified by **comparing per-point feature activations to the global max-pool values**:

```
For each point p_i and each feature dimension k:
    p_i is a critical point  iff  hx[i, k] == maxpool[k]  for any k
```

Where:
- `hx[i, k]` = the activation of point `p_i` at feature dimension `k` (output of shared MLP before global pool)
- `maxpool[k]` = the maximum value across all points for dimension `k` (the global descriptor entry)

A point is critical if it achieves the maximum in **at least one** feature dimension. Since max-pool has 1024 dimensions, there can be at most 1024 critical points (in practice far fewer due to ties and the learned embedding clustering).

### Upper Bound Shape: The Method

The upper-bound shape is computed by evaluating **all points in the inscribed unit sphere**, not just the input cloud:

```
1. Generate a dense grid of points within the unit sphere
2. Run forward pass through the MLP (hx) for all grid points
3. Keep only points where hx[i, k] <= maxpool[k] for ALL k
   (i.e., adding this point would not displace any existing maximum)
4. The surviving set = upper bound shape N_S
```

This requires two separate forward passes (the `--vis_mode critical` and `--vis_mode all` split in the code) because the input sizes differ between the original cloud and the dense grid.

### Code Structure

| File | Role |
|------|------|
| `pointnet_cls.py` | Modified classification network that returns `(hx, maxpool)` alongside predictions |
| `get_file.py` | Dual-mode data extraction: `--vis_mode critical` for input cloud, `--vis_mode all` for unit sphere grid |
| `vis.py` | Interactive 3D viewer that loads saved `.npz` files |

**Key model modification** (in `pointnet_cls.py`):
```python
# Standard PointNet returns only class logits
# Modified version also returns:
#   hx:      [B, N, 1024] per-point features before max-pool
#   maxpool: [B, 1024]    global max-pool vector
return pred, hx, maxpool
```

### Plain-Language Explanation of Critical Points

> **Analogy:** Imagine 1024 judges each looking for one specific geometric feature. Each judge picks their single favorite point from the entire cloud — the point that best exhibits "their" feature. The set of all points chosen by at least one judge = the critical point set. Every other point is invisible to the final descriptor.

This is why critical points tend to be:
- **Object extremities** (endpoints of legs, tips of wings, tops of chairs) — these are geometrically distinctive
- **Corners and edges** — high feature variance = higher chance of being the max in some dimension
- **Sparse** — typically 100–300 points out of 1024 input, concentrated on the shape skeleton

### Plain-Language Explanation of Upper/Lower Bound Shapes

> The **lower bound** (critical point set): the *smallest* input that produces the same classification. Remove any critical point and the result changes.
>
> The **upper bound**: the *largest* input that produces the same classification. Add any point *outside* the upper bound and some max-pool entry flips, potentially changing the result.
>
> Everything between these two shapes is "equivalent" from the network's perspective. The network is not learning a precise surface — it's learning an equivalence class of shapes.

### Feature Dimensionality Note

GitBoSun uses `hx` from concatenated intermediate MLP layers (total 10,224 dimensions) rather than just the final 1024-d max-pool. For this experiment, we use the standard 1024-d global max-pool output as described in the paper, giving ≤1024 critical points. The approach is identical; only the dimensionality differs.

## Key Takeaways for the Framework

- **Forward-hook approach confirmed.** Register a hook on the global max-pool layer to capture `hx` (pre-pool per-point features) and `maxpool` (global descriptor). No model surgery needed beyond adding `return_intermediates=True` or using `register_forward_hook`.
- **Two-mode extraction is required.** Critical points and upper-bound shape need separate passes (different input shapes). Plan for this in the experiment script structure.
- **`.npz` is the natural storage format.** Save `(critical_point_indices, input_xyz, class_label)` per sample; visualize in a separate step. Avoids re-running inference for visualization tweaks.
- **This repo trains from scratch.** We use pretrained weights instead — the extraction logic in `get_file.py` is reusable but `train.py` is not needed.
- **Sparsity is visually striking.** If the experiment reproduces results similar to the original paper's Fig. 7, critical points should form a clean skeleton of ~200 points — this is the "wow" visual for the README.

## References

- Repository: https://github.com/GitBoSun/PointNet_vis
- PointNet paper Section 4.3: https://arxiv.org/abs/1612.00593
- Patrick's paper notes (concise critical point summary): https://patrick-llgc.github.io/Learning-Deep-Learning/paper_notes/pointnet.html
- DataScienceUB visual explainer: https://datascienceub.medium.com/pointnet-implementation-explained-visually-c7e300139698
