"""Array backend selection: numpy on CPU, cupy on GPU when available.

Every module imports `xp` from here instead of numpy directly, so the exact
same from-scratch code runs on a laptop CPU or on a free Colab GPU.
"""
from __future__ import annotations

import numpy as _np

try:  # pragma: no cover - exercised only on machines with a CUDA GPU
    import cupy as _cp

    _cp.cuda.runtime.getDeviceCount()
    xp = _cp
    _DEVICE = "cuda"
except Exception:
    xp = _np
    _DEVICE = "cpu"


def device_name() -> str:
    """Return the active compute device: "cpu" or "cuda"."""
    return _DEVICE


def asnumpy(array) -> _np.ndarray:
    """Bring an xp array back to host memory as a numpy array."""
    if xp is _np:
        return _np.asarray(array)
    return xp.asnumpy(array)  # pragma: no cover - GPU only


def scatter_add(target, index, values) -> None:
    """In-place `target[index] += values`, accumulating on repeated indices.

    Plain `target[index] += values` does NOT accumulate when an index appears
    more than once (e.g. the same token used twice), which is exactly what
    gradient backprop through indexing requires. numpy needs np.add.at; cupy
    provides cupyx.scatter_add.
    """
    if xp is _np:
        _np.add.at(target, index, values)
    else:  # pragma: no cover - GPU only
        import cupyx

        cupyx.scatter_add(target, index, values)
