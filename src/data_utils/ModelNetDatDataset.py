"""
Lightweight dataset that loads pre-processed ModelNet40 .dat files directly.
Use when the raw modelnet40_normal_resampled/ directory is not available.

Expected files (from yanx27 Google Drive):
  data/modelnet40_train_1024pts.dat
  data/modelnet40_test_1024pts.dat

Each .dat file is a pickle of (list_of_points, list_of_labels) where
  list_of_points[i]: (N, 6) float32 array  — xyz + normals
  list_of_labels[i]: (1,)   int32 array
"""
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset


class ModelNetDatDataset(Dataset):
    def __init__(self, dat_path: str, npoints: int = 1024, use_normals: bool = False):
        with open(dat_path, "rb") as f:
            self.points, self.labels = pickle.load(f)
        self.npoints = npoints
        self.use_normals = use_normals
        print(f"Loaded {len(self.points)} samples from {dat_path}")

    def __len__(self):
        return len(self.points)

    def __getitem__(self, idx):
        pts = self.points[idx][:self.npoints, :]   # (N, 6)
        label = int(self.labels[idx][0])

        if not self.use_normals:
            pts = pts[:, :3]                        # xyz only

        # Normalize to unit sphere
        pts[:, :3] -= pts[:, :3].mean(axis=0)
        dist = np.linalg.norm(pts[:, :3], axis=1).max()
        pts[:, :3] /= (dist + 1e-8)

        return torch.from_numpy(pts).float(), label
