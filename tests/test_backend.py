"""The backend exposes `xp`, a numpy-compatible array module.

On a machine without cupy (like this laptop), xp must be numpy itself so the
whole library runs on CPU. On Colab with cupy installed, xp becomes cupy and
the same code runs on GPU.
"""
import numpy as np

from ownai.backend import device_name, xp


def test_xp_is_numpy_when_cupy_missing():
    assert xp is np


def test_device_name_reports_cpu():
    assert device_name() == "cpu"


def test_asnumpy_roundtrip():
    from ownai.backend import asnumpy

    a = xp.array([1.0, 2.0])
    out = asnumpy(a)
    assert isinstance(out, np.ndarray)
    assert out.tolist() == [1.0, 2.0]
