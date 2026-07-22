from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def knn_graph(x: torch.Tensor, k: int) -> torch.Tensor:
    """Return indices of the k nearest neighbours for every point."""
    _, _, n_points = x.shape
    if n_points <= k:
        raise ValueError(f"DGCNN requires N > k, got N={n_points}, k={k}")
    inner = -2.0 * torch.bmm(x.permute(0, 2, 1), x)
    squared = (x**2).sum(dim=1, keepdim=True)
    distance = inner + squared + squared.permute(0, 2, 1)
    return distance.topk(k + 1, largest=False, dim=-1).indices[:, :, 1:]


def edge_features(x: torch.Tensor, k: int) -> torch.Tensor:
    batch, channels, n_points = x.shape
    indices = knn_graph(x, k)
    transposed = x.permute(0, 2, 1).contiguous()
    batch_indices = (
        torch.arange(batch, device=x.device).view(batch, 1, 1).expand(batch, n_points, k)
    )
    neighbors = transposed[
        batch_indices.reshape(-1), indices.reshape(-1)
    ].reshape(batch, n_points, k, channels)
    neighbors = neighbors.permute(0, 3, 1, 2)
    center = x.unsqueeze(-1).expand_as(neighbors)
    return torch.cat([center, neighbors - center], dim=1)


class EdgeConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, k: int = 20) -> None:
        super().__init__()
        self.k = int(k)
        # Keep the notebook attribute name so its raw state_dict keys load strictly.
        self.net = nn.Sequential(
            nn.Conv2d(in_channels * 2, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(edge_features(x, self.k)).max(dim=-1).values


class DGCNNSeg(nn.Module):
    """DGCNN segmentation architecture copied from the research notebook."""

    def __init__(self, num_classes: int = 2, in_channels: int = 7, k: int = 20) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.num_classes = int(num_classes)
        self.k = int(k)

        self.ec1 = EdgeConv(in_channels, 64, k)
        self.ec2 = EdgeConv(64, 64, k)
        self.ec3 = EdgeConv(64, 128, k)
        self.ec4 = EdgeConv(128, 256, k)
        self.cg = nn.Sequential(
            nn.Conv1d(512, 1024, 1),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2),
        )
        self.seg = nn.Sequential(
            nn.Conv1d(1536, 512, 1),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),
            nn.Conv1d(512, 256, 1),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5),
            nn.Conv1d(256, num_classes, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.ec1(x)
        x2 = self.ec2(x1)
        x3 = self.ec3(x2)
        x4 = self.ec4(x3)
        local = torch.cat([x1, x2, x3, x4], dim=1)
        global_features = self.cg(local).max(dim=-1, keepdim=True).values
        global_features = global_features.expand(-1, -1, local.shape[-1])
        logits = self.seg(torch.cat([local, global_features], dim=1)).permute(0, 2, 1)
        return F.log_softmax(logits, dim=-1)
