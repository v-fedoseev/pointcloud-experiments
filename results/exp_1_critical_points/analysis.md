# Exp 1: Critical Point Set Visualization in PointNet

## Hypothesis

Critical points are the points that have at least one feature dimension that "won" the maxpool layer (the maxpool at the end pools every dimension across all input points). On the other hand, the upper bound set is the set of all points that will not change the output of the maxpool layer, if added to the input. That means they would not win any pooling dimension, so they would be not used by the classification and segmentation heads. Therefore my hypothesis is that the non-critical points are the ones that are irrelevant for those tasks. These should be the points that are: 1) internal to the volume, 2) some of the flat plane "duplicates" - points that are near each other on a flat plane, if not all the plane points except the edges.

Given this, the critical points should be: 1) corner points, 2) outer edge points, 3) points along thin structures that are relevant for the class (e.g. table legs), 4) perhaps some points on flat planes and lines that reinforce the signal of the flat structure (not all "flat" points might be eliminated).

This aligns with the visualizations in the original paper. This also sounds like edge extraction with gradients in 2D images, where non-edge data like low-frequency image zones does not contribute to the edge image.

## Setup

| Parameter | Value |
|-----------|-------|
| Model | PointNet v1 |
| Checkpoint | `checkpoints/pointnet_cls/best_model.pth` (85.7% test acc) |
| Dataset | ModelNet40 |
| Samples | 1 per class (2 for lamp); airplane, chair, guitar, lamp, table |
| Hook target | `model.feat.bn3` (bn3 output = exact pre-pool feature map, shape 1×1024×N) |

## Results


| Class | Sample | # Critical points | Notes |
|-------|--------|-------------------|-------|
| airplane | #1 | 244 / 1024 (24%) | |
| chair | #1 | 304 / 1024 (30%) | |
| guitar | #1 | 225 / 1024 (22%) | |
| lamp | #1 | 371 / 1024 (36%) | |
| lamp | #2 | 291 / 1024 (28%) | |
| table | #1 | 299 / 1024 (29%) | |

## Observations

The points are clustered in high-frequency regions, and located densely along edges. However, if an edge or a surface is smooth, there can be gaps (e.g. nose of a plane). However, if it a thin structure, like the leg of a lamp, then the critical points are very dense along it. The flat surfaces, or conical surfaces have very few points, but still some are preserved.

## Analysis

This reveals that PointNet looks at edges and thin structures, like non-maximum point suppression, and barely looks at homogeneous surfaces. This is not exactly "skeleton" or surface of an object as the Section 4.3 of the paper describes, as that would also include all surface points.

## Conclusion

The results largely confirm the hypothesis. Critical points concentrate on edges, corners, and thin structures — consistent   with the prediction that interior and flat-plane duplicates would be eliminated.                
                                                                                                                                
The partial contradiction with the paper's §4.3 framing is worth noting: the paper describes the critical set as capturing the
"skeleton or boundary" of the shape, which implies broader surface coverage. What we observe is more selective than that —     
closer to an edge detector than a surface sampler. This may reflect that our checkpoint (85.7%, trained for 10 epochs) has not
fully converged, so its feature space may not be as spatially uniform as a fully-trained model's. Alternatively, it may simply 
be that "skeleton" in the paper is used loosely and the edge-like pattern is the true behaviour. That is supported by the visualzations in the paper.

The lamp-to-lamp variation (371 vs 291 critical points) shows the critical set is instance-specific, not just class-specific — 
the exact geometry of a sample determines how many points the max-pool "needs".
