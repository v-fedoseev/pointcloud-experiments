# Exp 0: PointNet v1 Baseline Training

## Hypothesis

Small flat LR training for 5 epochs would produce a quick reasonable checkpoint for further experiments

## Runs

| Run | Optimizer | lr | Epochs | Train acc @ best test ep | Best test acc | Notes |
|-----|-----------|-----|--------|--------------------------|--------------|-------|
| 1 | Adam (flat LR) | 0.001 | 5 | 76.5% (ep4) | 81.2% (ep4) | ep5 drops to 74.4% — oscillation |
| 2 | AdamWScheduleFree | 0.001 | 10 | 90.1% (ep7) | 77.8% (ep7) | fast overfit; noisy test curve |
| 3 | AdamWScheduleFree | 0.0003 | 10 | 94.0% (ep9) | 85.7% (ep9) | stable convergence; best checkpoint |

## Observations

Run 1 (Adam flat LR) hit 81.2% test at ep4 then dropped sharply to 74.4% at ep5 - sign of a too large step size. Run 2 resulted in train acc 90% but test peaked at 77.8%, a 12-point gap. Run 3 (ScheduleFree, lr=0.0003) was the only run that converged stably: test acc climbed monotonically to 85.7% with a tighter 8-point train/test gap.

## Analysis

The oscillation in Run 1: flat LR with Adam means the optimizer keeps taking large steps even once loss is low, so it overshoots. ScheduleFree removes the need to tune a schedule, but it doesn't fix an aggressive learning rate: Run 2 quickly overfit on training data. Dropping lr to 0.0003 gave the optimizer smaller steps, which slowed overfit and let the model generalise better. The 85.7% result is still below ~89% (what's in the paper for ModelNet40), which is expected given only 10 epochs and no data augmentation.

## Conclusion

Taking the Run 3 checkpoint (AdamWScheduleFree, lr=0.0003, ep9, 85.7% test acc). Below paper quality, but sufficient as a quick baseline for the critical points and T-Net ablation experiments.
