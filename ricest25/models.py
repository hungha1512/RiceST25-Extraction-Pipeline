"""DGCNN segmentation network (layer names match the training notebook's checkpoints)."""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


def knn_graph(x: torch.Tensor, k: int) -> torch.Tensor:
    inner = -2 * torch.bmm(x.permute(0, 2, 1), x)
    sq = (x ** 2).sum(1, keepdim=True)
    return (inner + sq + sq.permute(0, 2, 1)).topk(k + 1, largest=False)[1][:, :, 1:]


def edge_feat(x: torch.Tensor, k: int) -> torch.Tensor:
    b, c, n = x.shape
    idx = knn_graph(x, k)                                                     # (B, N, k)
    xt = x.permute(0, 2, 1).contiguous()
    bi = torch.arange(b, device=x.device).view(b, 1, 1).expand(b, n, k)
    nb = xt[bi.reshape(-1), idx.reshape(-1)].reshape(b, n, k, c).permute(0, 3, 1, 2)
    ctr = x.unsqueeze(-1).expand_as(nb)
    return torch.cat([ctr, nb - ctr], dim=1)                                  # (B, 2C, N, k)


class EdgeConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, k: int = 20):
        super().__init__()
        self.k = k
        self.net = nn.Sequential(
            nn.Conv2d(in_ch * 2, out_ch, 1, bias=False), nn.BatchNorm2d(out_ch), nn.LeakyReLU(0.2, True)
        )

    def forward(self, x):
        return self.net(edge_feat(x, self.k)).max(-1)[0]


class DGCNNSeg(nn.Module):
    """Input (B, C, N) -> per-point log-probabilities (B, N, num_classes)."""

    def __init__(self, num_classes: int = 2, in_channel: int = 7, k: int = 20):
        super().__init__()
        self.ec1, self.ec2 = EdgeConv(in_channel, 64, k), EdgeConv(64, 64, k)
        self.ec3, self.ec4 = EdgeConv(64, 128, k), EdgeConv(128, 256, k)
        self.cg = nn.Sequential(nn.Conv1d(512, 1024, 1), nn.BatchNorm1d(1024), nn.LeakyReLU(0.2))
        self.seg = nn.Sequential(
            nn.Conv1d(1536, 512, 1), nn.BatchNorm1d(512), nn.LeakyReLU(0.2),
            nn.Conv1d(512, 256, 1), nn.BatchNorm1d(256), nn.LeakyReLU(0.2),
            nn.Dropout(0.5), nn.Conv1d(256, num_classes, 1),
        )

    def forward(self, x):
        x1 = self.ec1(x); x2 = self.ec2(x1); x3 = self.ec3(x2); x4 = self.ec4(x3)
        local = torch.cat([x1, x2, x3, x4], 1)                                # (B, 512, N)
        glob = self.cg(local).max(-1, keepdim=True)[0].expand(-1, -1, local.shape[-1])
        return F.log_softmax(self.seg(torch.cat([local, glob], 1)).permute(0, 2, 1), dim=-1)


def resolve_device(name: str = "auto") -> torch.device:
    return torch.device("cuda" if name == "auto" and torch.cuda.is_available() else
                        "cpu" if name == "auto" else name)


def load_dgcnn(checkpoint: str | Path, device: str | torch.device = "cpu", k: int = 20) -> DGCNNSeg:
    """Load a ``state_dict`` checkpoint; input channels / classes are read from the weights."""
    path = Path(checkpoint)
    if not path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    for key in ("state_dict", "model_state_dict"):
        if isinstance(payload, dict) and key in payload:
            payload = payload[key]
    state = {(key[7:] if key.startswith("module.") else key): v for key, v in payload.items()}
    in_ch = state["ec1.net.0.weight"].shape[1] // 2
    n_cls = state["seg.7.weight"].shape[0]
    model = DGCNNSeg(n_cls, in_ch, k)
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()
