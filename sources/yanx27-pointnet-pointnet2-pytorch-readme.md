---
source: https://github.com/yanx27/Pointnet_Pointnet2_pytorch
date_captured: 2026-03-23
author: Xu Yan (yanx27)
type: documentation
tags: [pointnet, pointnet++, pytorch, pretrained-weights, cpu-inference, modelnet40, checkpoint]
---

# yanx27/Pointnet_Pointnet2_pytorch README

> **Source Link:** [https://github.com/yanx27/Pointnet_Pointnet2_pytorch](https://github.com/yanx27/Pointnet_Pointnet2_pytorch)
> **Gist:** Base repo providing pretrained checkpoints, CPU-compatible inference via `--use_cpu`, and model implementations that are the direct source of all model files used in this experiment.

## Core Gist / Summary

This PyTorch reimplementation of PointNet and PointNet++ (~4.8k stars) provides production-quality model implementations and pretrained checkpoints for ModelNet40 classification, ShapeNet part segmentation, and S3DIS semantic segmentation. The `--use_cpu` flag (added 2021-03-20) makes all models runnable on CPU-only hardware.

### Classification Accuracy Table (ModelNet40)

| Model | Accuracy |
|-------|----------|
| PointNet (Official paper) | 89.2% |
| PointNet2 (Official paper) | 91.9% |
| PointNet (Pytorch, no normals) | 90.6% |
| PointNet (Pytorch, with normals) | 91.4% |
| PointNet2_SSG (no normals) | 92.2% |
| PointNet2_SSG (with normals) | 92.4% |
| **PointNet2_MSG (with normals)** | **92.8%** |

### Pretrained Checkpoint Locations (in repo)

Checkpoints are stored in `log/` and committed directly to the repository:

```
log/
├── classification/
│   ├── pointnet2_msg_normals/       ← PointNet++ MSG best checkpoint
│   └── pointnet2_ssg_wo_normals/    ← PointNet++ SSG best checkpoint
├── part_seg/
└── sem_seg/
    ├── pointnet_sem_seg/            ← 40.7 MB
    └── pointnet2_sem_seg/           ← 11.2 MB
```

**Note:** The PointNet (not PointNet++) classification checkpoint is **not committed** to the repo — only SSG and MSG checkpoints are present. PointNet classification weights must be downloaded separately or trained.

### Key CLI Flags

```bash
# CPU-only inference (critical for this experiment's hardware)
python test_classification.py --log_dir pointnet2_cls_ssg --use_cpu

# With normal features
python test_classification.py --log_dir pointnet2_cls_ssg --use_normals

# ModelNet10 (10-class subset, faster)
python test_classification.py --log_dir pointnet2_cls_ssg --num_category 10
```

### Data Preparation

- **ModelNet40:** Download `modelnet40_normal_resampled.zip` from ShapeNet's Stanford CDN; save to `data/modelnet40_normal_resampled/`
- **Pre-processed data:** Available on Google Drive (avoids re-processing on first run with `--process_data`)
- **Default input:** 1024 points, XYZ coordinates (optionally + normals)

### Model File Locations

All model definitions live in `./models/`:

| File | Model |
|------|-------|
| `pointnet_cls.py` | PointNet classification |
| `pointnet2_cls_ssg.py` | PointNet++ SSG classification |
| `pointnet2_cls_msg.py` | PointNet++ MSG classification |
| `pointnet_part_seg.py` | PointNet part segmentation |
| `pointnet2_part_seg_msg.py` | PointNet++ MSG part seg |
| `pointnet_sem_seg.py` | PointNet semantic seg |
| `pointnet2_sem_seg.py` | PointNet++ semantic seg |

### Part Segmentation Performance (ShapeNet)

| Model | Instance avg IoU | Class avg IoU |
|-------|-----------------|---------------|
| PointNet (Official) | 83.7% | 80.4% |
| PointNet2 (Official) | 85.1% | 81.9% |
| PointNet (Pytorch) | 84.3% | 81.1% |
| PointNet2_MSG (Pytorch) | **85.4%** | **82.5%** |

## Key Takeaways for the Framework

- **`--use_cpu` is the critical flag.** All experiments in this repo must pass `--use_cpu` to the test scripts; without it the code attempts CUDA initialization and fails on CPU-only hardware.
- **Checkpoints in `log/classification/` are the starting point.** `pointnet2_msg_normals/` and `pointnet2_ssg_wo_normals/` contain committed `.pth` files ready for forward-pass inference — no download step needed for PointNet++.
- **PointNet classification checkpoint is missing.** The T-Net ablation experiment needs a PointNet classification checkpoint; this must be either trained (fast, ~10 min CPU for a partial run) or sourced from the original charlesq34 Caffe repo and converted.
- **Model files are self-contained.** Each `models/*.py` file imports only standard PyTorch; they can be copied directly into the experiment directory and modified (e.g., adding hooks, removing T-Net) without breaking the rest of the repo.
- **`test_classification.py` is the inference entry point.** It loads a checkpoint from `--log_dir`, runs evaluation on the test split, and reports per-class and overall accuracy — the right script for the T-Net ablation accuracy measurement.

## References

- Repository: https://github.com/yanx27/Pointnet_Pointnet2_pytorch
- Original PointNet (Caffe): https://github.com/charlesq34/pointnet
- Original PointNet++ (TF): https://github.com/charlesq34/pointnet2
- ModelNet40 data: https://shapenet.cs.stanford.edu/media/modelnet40_normal_resampled.zip
