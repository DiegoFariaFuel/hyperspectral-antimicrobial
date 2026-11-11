from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class CubeSet:
    cubes: np.ndarray  # (N, H, W, B), float32
    labels: np.ndarray  # (N,), integer class ids
    class_names: tuple[str, ...]


def _as_class_strings(labels: np.ndarray) -> list[str]:
    if labels.ndim != 1:
        raise ValueError("labels must be a one-dimensional array")
    if np.issubdtype(labels.dtype, np.number) and not np.isfinite(labels).all():
        raise ValueError("labels contain non-finite numeric values")
    return [str(value) for value in labels.tolist()]


def load_npz(path: str | Path, class_names: Sequence[str] | None = None) -> CubeSet:
    """Load a hyperspectral dataset without enabling pickle deserialization."""
    with np.load(path, allow_pickle=False) as archive:
        if "cubes" not in archive or "labels" not in archive:
            raise ValueError("NPZ must contain 'cubes' and 'labels' arrays")
        cubes = np.asarray(archive["cubes"], dtype=np.float32)
        raw_labels = np.asarray(archive["labels"])

    if cubes.ndim != 4:
        raise ValueError("cubes must have shape (samples, height, width, bands)")
    if min(cubes.shape) <= 0:
        raise ValueError("cubes must not contain empty dimensions")
    if cubes.shape[0] != raw_labels.shape[0]:
        raise ValueError("cubes and labels must contain the same number of samples")
    if not np.isfinite(cubes).all():
        raise ValueError("cubes contain NaN or infinite values")

    label_strings = _as_class_strings(raw_labels)
    names = tuple(class_names) if class_names is not None else tuple(sorted(set(label_strings)))
    label_to_id = {name: i for i, name in enumerate(names)}
    unknown = sorted(set(label_strings) - set(label_to_id))
    if unknown:
        raise ValueError(f"dataset contains classes not present in training: {unknown}")
    if len(names) < 2:
        raise ValueError("at least two classes are required")
    encoded = np.asarray([label_to_id[name] for name in label_strings], dtype=np.int64)
    return CubeSet(cubes=cubes, labels=encoded, class_names=names)


def fit_band_normalization(cubes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fit per-band mean/std on training cubes only."""
    mean = cubes.mean(axis=(0, 1, 2), dtype=np.float64).astype(np.float32)
    std = cubes.std(axis=(0, 1, 2), dtype=np.float64).astype(np.float32)
    std = np.where(std < 1e-8, 1.0, std).astype(np.float32)
    return mean, std


def normalize_cubes(cubes: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    if cubes.shape[-1] != mean.shape[0] or mean.shape != std.shape:
        raise ValueError("band statistics do not match the cube's band count")
    return ((cubes - mean.reshape(1, 1, 1, -1)) / std.reshape(1, 1, 1, -1)).astype(np.float32)

