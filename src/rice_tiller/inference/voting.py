from __future__ import annotations

import numpy as np

from rice_tiller.inference.chunking import batched_chunks, make_covering_chunks


def full_cloud_voting(
    model,
    features: np.ndarray,
    *,
    num_classes: int,
    chunk_size: int = 4096,
    batch_size: int = 1,
    passes: int = 3,
    seed: int = 42,
    device: str = "cpu",
) -> tuple[np.ndarray, np.ndarray]:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("Full-cloud inference requires PyTorch") from exc

    n_points = len(features)
    votes = np.zeros((n_points, num_classes), dtype=np.float64)
    counts = np.zeros(n_points, dtype=np.int32)
    rng = np.random.default_rng(seed)

    model.eval()
    with torch.inference_mode():
        for _ in range(passes):
            chunks = make_covering_chunks(n_points, chunk_size, rng)
            for chunk_batch in batched_chunks(chunks, batch_size):
                batch = np.stack([features[chunk.indices].T for chunk in chunk_batch])
                tensor = torch.from_numpy(batch).float().to(device)
                log_probabilities = model(tensor)
                probabilities = log_probabilities.exp().cpu().numpy()
                for batch_index, chunk in enumerate(chunk_batch):
                    valid_length = chunk.valid_length
                    valid_indices = chunk.valid_indices
                    votes[valid_indices] += probabilities[batch_index, :valid_length]
                    counts[valid_indices] += 1

    if np.any(counts == 0):
        raise RuntimeError("Voting invariant failed: at least one point received no prediction")
    probabilities = votes / counts[:, None]
    return probabilities.astype(np.float32), counts

