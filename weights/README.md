# Weights

Checkpoints are not tracked by git. Place them as

| Path | Used by | Training data |
|---|---|---|
| `lopo/DGCNN_<pot>.pth` (13 files, C1.1 … C3.4) | `reproduce.py`, `extract.py` | 11 pots; `<pot>` held out for test, next pot in sorted order used for validation |
| `potwise/DGCNN_seed42.pth` | `eval_seg.py` | pot-wise split of `configs/split_potwise.yaml` |

Each file is `torch.save(model.state_dict(), path)` of `models.dgcnn.DGCNNSeg(num_classes=2, in_channel=7, k=20)`.
A `module.` key prefix (DataParallel) is accepted; loading is strict. The notebook's `runs/lopo/DGCNN_<pot>.pth` and
`runs/DGCNN_seed42.pth` can be copied unchanged.
