"""Small dependency-free color maps for diagnostic images."""

import numpy as np


def signed_potential_rgba(values, percentile=99.0):
    """Map a signed 2D potential to blue-white-red RGBA and return its scale."""
    array = np.asarray(values, dtype=np.float32)
    finite = np.isfinite(array)
    if not finite.any():
        return np.zeros((*array.shape, 4), dtype=np.uint8), 0.0
    finite_absolute = np.abs(array[finite])
    scale = float(np.percentile(finite_absolute, percentile))
    if not np.isfinite(scale) or scale <= 0:
        scale = float(finite_absolute.max()) if finite_absolute.size else 0.0
    if scale <= 0:
        scale = 1.0
    normalized = np.clip(np.nan_to_num(array) / scale, -1.0, 1.0)
    magnitude = np.abs(normalized)
    channel = np.round(255 * (1 - magnitude)).astype(np.uint8)
    rgba = np.empty((*array.shape, 4), dtype=np.uint8)
    positive = normalized >= 0
    rgba[..., 0] = np.where(positive, 255, channel)
    rgba[..., 1] = channel
    rgba[..., 2] = np.where(positive, channel, 255)
    rgba[..., 3] = 255
    return rgba, scale
