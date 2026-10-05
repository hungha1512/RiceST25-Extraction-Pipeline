# Weights

Checkpoints are not committed (see `.gitignore`). Place them as:

| Path | Used by | Trained on |
|---|---|---|
| `weights/lopo/DGCNN_<pot>.pth` (13 files: C1.1-C1.4, C2.1-C2.5, C3.1-C3.4) | `reproduce`, `analyze` | 11 pots; `<pot>` held out for testing, the next pot in sorted order used for validation |
| `weights/potwise/DGCNN_seed42.pth` | `eval-seg` | pot-wise split in `configs/split_potwise.yaml` |

Format: `torch.save(model.state_dict(), path)` of `ricest25.models.DGCNNSeg(num_classes=2, in_channel=7, k=20)`,
i.e. keys `ec1.net.0.weight ... seg.7.bias`. A `module.` prefix (DataParallel) is accepted. The notebook saved
them as `runs/lopo/DGCNN_<pot>.pth` and `runs/DGCNN_seed42.pth`, so they can be copied unchanged.

`python -m ricest25 inspect <file>` prints input channels, classes, k and parameter count (1,539,714 for the
7D model). A checkpoint with a different layout fails to load (strict loading).
