# Exp 2: T-Net (STN3d) Ablation

## Hypothesis

My hypothesis is that classification accuracy would drop, since the network is trained with T-Net. If ablated, mis-oriented objects would not be recognized. However, I assumed the dataset was mostly orientation-normalized, so I predicted ~10% drop at most.

## Setup

| Parameter | Value |
|-----------|-------|
| Model | PointNet v1 |
| Checkpoint | `checkpoints/pointnet_cls/best_model.pth` (85.7% test acc) |
| Dataset | ModelNet40 test (full, 2468 samples) |
| Ablation | STN3d replaced with identity (3×3 I) |

## Results

| Condition | Accuracy |
|-----------|---------|
| Original | 85.7% |
| T-Net ablated | 20.7% |
| Drop | 65.0% |

Plot: `results/exp_2_tnet_ablation/plot.png`

## Observations

65-point drop — way larger than expected. The ablated model isn't completely random (20.7% vs 2.5% chance for 40 classes), but it's close to useless.

## Analysis

The MLP layers were trained jointly with the T-Net and learned to assume the input is already aligned.

The residual 20.7% is probably the feature-space T-Net (fstn, STN64d) still doing some work: only the input transform was ablated here. But it was given the features from the misoriented shapes, which are likely out of training distribution of features, or were linked to wrong classes during training.

## Conclusion

Directionally right, wrong on magnitude. The T-Net isn't just a minor preprocessing step, it's key to the whole model. The rest of the network is brittle without it.
