# Exp 3: PointNet++ SA-Level Critical Points Visualization

## Hypothesis

SA-level max-pooling should produce progressively sparser winner sets as depth increases. Winners at SA3 should concentrate on geometrically distinctive regions - thin structures, edges, structurally separating features - mirroring PointNet v1's critical point preference toward edges and corners. Flat surfaces should be present at SA1 but drop out by SA3.

## Setup

| Parameter | Value |
|-----------|-------|
| Model | PointNet++ SSG |
| Checkpoint | `checkpoints/pointnet2_ssg/best_model.pth` (91.9% test acc) |
| Dataset | ModelNet40 test set |
| Samples | Same instances as exp 1: airplane, chair, guitar, lamp (×2), table |
| SA1 | npoint=512, radius=0.2, nsample=32 |
| SA2 | npoint=128, radius=0.4, nsample=64 |
| SA3 | group_all (all 128 SA2 centroids → 1 global descriptor) |

SA1 critical points = original input points that won at least one SA1 local max-pool channel.
SA2 critical points = SA1 centroids that won at least one SA2 local max-pool channel.
SA3 critical points = SA3 global pool winners traced back through SA2 FPS → SA1 FPS → original input indices.

## Results

All percentages are relative to the original 1024 input points. SA2 has a ≤50% ceiling since only the 512 FPS-selected SA1 centroids are eligible.

| Shape | PN v1 / 1024 | SA1 / 1024 | SA2 / 1024 (≤50%) | SA3 / 1024 | SA3 / PN v1 |
|-------|--------------|------------|-------------------|------------|-------------|
| airplane | 244 (24%) | 609 (59%) | 416 (41%) | 114 (11%) | 0.47× |
| chair    | 304 (30%) | 648 (63%) | 427 (42%) | 119 (12%) | 0.39× |
| guitar   | 225 (22%) | 285 (28%) | 219 (21%) |  91  (9%) | 0.40× |
| lamp_1   | 371 (36%) | 319 (31%) | 302 (29%) |  86  (8%) | 0.23× |
| lamp_2   | 291 (28%) | 778 (76%) | 458 (45%) | 119 (12%) | 0.41× |
| table    | 299 (29%) | 865 (84%) | 504 (49%) | 116 (11%) | 0.39× |

## Observations

SA1 rates (28–84%) drop to SA2 (21–49%), then collapse to SA3 (8–12%). SA3 converges to 86–119 across all shapes despite ~3× variation at SA1.

Notable cases:

- Guitar: most selective at SA1 (28%) - thin body has minimal ball-query overlap. SA3 shows roughly uniform sparse coverage across body, neck, and headstock.
- Lamp_1: SA1→SA2 barely compresses (31% → 29%). SA3 yields two clusters at the head and base; the arm disappears.
- Table: SA1 near-saturated (84%) - flat top maximizes ball overlap. Still converges to 116 at SA3.

## Analysis

SA1 win rate tracks local surface density - dense flat surfaces win more, thin structures win less. SA3 converges all shapes to a narrow band regardless of SA1 count.

Lamp_1's flat SA1→SA2 (31% → 29%): the elongated geometry means SA2 centroids along the arm have few neighbors within radius 0.4. Ball query pads underpopulated neighborhoods by repeating the nearest point, so a nominally 64-slot ball may hold only ~5–10 unique entries - nearly all of which win some channel. The arm's SA1 centroids almost entirely survive to SA2, and the head/base clusters are too spatially separated to compete with each other.

SA3 traced winners are 2-3x sparser than PN v1 critical points for the same shapes (ratio 0.23-0.47). PN v1's single global max-pool works directly on raw 3D coordinates, so its winners are tied to geometric distinctiveness. PN++'s SA3 pool works on learned local features after two stages of aggregation - the smaller count reflects prior compression, not sharper geometric focus.

## Conclusion

Results confirm the hypothesis. SA1 win rates track local surface density; SA3 converges to a narrow count range; SA3 spatial distribution favors disconnected or structurally distinct regions (lamp head/base, chair seat/back, guitar body/headstock) over connecting structures.

PN v1's flat architecture resolves the whole shape in one max-pool over raw geometry, biasing its critical set toward edges and extremities. PN++'s staged pooling distributes that compression across levels - SA3 winners encode learned local features rather than raw geometry, making them sparser but less directly readable as geometric markers.

So the conclusion contradicts the hypothesis. SA1 is like PN v1, but the next stages do feature aggregation, not geometric refinement.
