"""Gradient checks for the hand-written autograd engine.

Every analytic gradient produced by Tensor.backward() is compared against a
numerical gradient computed with central finite differences. If these match,
the backpropagation implementation is mathematically correct.
"""
import numpy as np
import pytest

from ownai.autograd import Tensor

RNG = np.random.default_rng(0)
EPS = 1e-6
TOL = 1e-4


def numeric_grad(f, x: np.ndarray) -> np.ndarray:
    """Central finite-difference gradient of scalar function f at x."""
    grad = np.zeros_like(x)
    it = np.nditer(x, flags=["multi_index"])
    for _ in it:
        idx = it.multi_index
        orig = x[idx]
        x[idx] = orig + EPS
        f_plus = f(x)
        x[idx] = orig - EPS
        f_minus = f(x)
        x[idx] = orig
        grad[idx] = (f_plus - f_minus) / (2 * EPS)
    return grad


def check_grad(f_tensor, *shapes):
    """Build float64 leaf tensors, run backward, compare with finite diff."""
    arrays = [RNG.standard_normal(s).astype(np.float64) for s in shapes]
    tensors = [Tensor(a.copy(), requires_grad=True) for a in arrays]
    out = f_tensor(*tensors)
    out.backward()
    for i, (a, t) in enumerate(zip(arrays, tensors)):
        def f_num(x, i=i):
            args = [Tensor(arr.copy()) for arr in arrays]
            args[i] = Tensor(x.copy())
            return float(f_tensor(*args).data)
        expected = numeric_grad(f_num, a.copy())
        got = np.asarray(t.grad)
        assert got.shape == a.shape
        np.testing.assert_allclose(got, expected, rtol=TOL, atol=TOL)


def test_add_mul_chain():
    check_grad(lambda a, b: ((a * b + a) * 2.0).sum(), (3, 4), (3, 4))


def test_broadcast_add():
    check_grad(lambda a, b: (a + b).sum(), (3, 1), (1, 4))


def test_broadcast_mul_scalar_tensor():
    check_grad(lambda a, b: (a * b).sum(), (2, 3), (3,))


def test_sub_div_pow():
    check_grad(lambda a, b: ((a - b) / 2.0 + (a**2)).sum(), (2, 3), (2, 3))


def test_matmul():
    check_grad(lambda a, b: (a @ b).sum(), (3, 4), (4, 5))


def test_batched_matmul():
    check_grad(lambda a, b: (a @ b).sum(), (2, 3, 4), (2, 4, 5))


def test_exp_log_tanh_relu():
    check_grad(lambda a: (a.exp() + (a**2 + 1.0).log() + a.tanh() + a.relu()).sum(), (3, 3))


def test_sum_axis_and_mean():
    check_grad(lambda a: a.sum(axis=1).mean(), (3, 4))
    check_grad(lambda a: a.mean(axis=-1, keepdims=True).sum(), (2, 5))


def test_reshape_transpose():
    check_grad(lambda a: (a.reshape(6, 2).transpose(1, 0) * 3.0).sum(), (3, 4))


def test_softmax_matches_numeric():
    check_grad(lambda a: (a.softmax(axis=-1) * a).sum(), (3, 5))


def test_log_softmax():
    check_grad(lambda a: a.log_softmax(axis=-1).sum(), (2, 6))


def test_getitem_slice():
    check_grad(lambda a: a[1:, :2].sum(), (3, 4))


def test_getitem_integer_array_like_embedding():
    idx = np.array([0, 2, 2, 1])

    def f(table):
        return table[idx].sum()

    check_grad(f, (4, 3))


def test_getitem_tuple_arrays_like_cross_entropy():
    rows = np.arange(3)
    cols = np.array([1, 0, 2])
    check_grad(lambda a: a[rows, cols].sum(), (3, 4))


def test_grad_accumulates_when_tensor_reused():
    a = Tensor(np.array([2.0]), requires_grad=True)
    out = a * a + a  # da = 2a + 1 = 5
    out.backward()
    np.testing.assert_allclose(np.asarray(a.grad), [5.0])


def test_no_grad_tracking_when_not_required():
    a = Tensor(np.ones((2, 2)))
    out = (a * 3.0).sum()
    out.backward()
    assert a.grad is None


def test_backward_requires_scalar():
    a = Tensor(np.ones((2, 2)), requires_grad=True)
    with pytest.raises(ValueError):
        (a * 2.0).backward()
