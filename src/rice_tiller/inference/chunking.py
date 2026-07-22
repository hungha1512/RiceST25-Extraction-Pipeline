from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class InferenceChunk:
    indices: np.ndarray
    valid_length: int

    @property
    def valid_indices(self) -> np.ndarray:
        return self.indices[: self.valid_length]


def make_covering_chunks(
    n_points: int,
    chunk_size: int,
    rng: np.random.Generator,
) -> list[InferenceChunk]:
    if n_points <= 0:
        raise ValueError("n_points must be positive")
    if chunk_size <= 1:
        raise ValueError("chunk_size must be greater than one")

    permutation = rng.permutation(n_points)
    chunks: list[InferenceChunk] = []
    for start in range(0, n_points, chunk_size):
        valid = permutation[start : start + chunk_size]
        valid_length = len(valid)
        if valid_length < chunk_size:
            padding = rng.choice(valid, size=chunk_size - valid_length, replace=True)
            indices = np.concatenate([valid, padding])
        else:
            indices = valid
        chunks.append(InferenceChunk(indices=indices.astype(np.int64), valid_length=valid_length))
    return chunks


def batched_chunks(chunks: list[InferenceChunk], batch_size: int):
    if batch_size < 1:
        raise ValueError("batch_size must be at least one")
    for start in range(0, len(chunks), batch_size):
        yield chunks[start : start + batch_size]

