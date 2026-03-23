# Pretrained Checkpoints

## Available (committed to repo)

| Model | Source | Accuracy |
|-------|--------|----------|
| `pointnet2_ssg/best_model.pth` | yanx27/Pointnet_Pointnet2_pytorch `log/classification/pointnet2_ssg_wo_normals` | 91.9% |
| `pointnet2_msg/best_model.pth` | yanx27/Pointnet_Pointnet2_pytorch `log/classification/pointnet2_msg_normals` | 92.8% |

## Not Available Upstream

`pointnet_cls/best_model.pth` — yanx27 does not provide a pretrained PointNet v1 classification checkpoint.
The T-Net ablation experiment in `src/experiments/tnet_ablation.py` uses `pointnet2_ssg` as the base model instead.

To train PointNet v1 yourself (~1h on GPU):
```bash
python train_classification.py --model pointnet_cls --log_dir pointnet_cls
```
Then copy `log/classification/pointnet_cls/checkpoints/best_model.pth` here.
