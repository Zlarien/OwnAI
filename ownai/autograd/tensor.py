"""A reverse-mode automatic differentiation engine written from scratch.

A Tensor wraps an xp (numpy/cupy) array and records the operations applied to
it in a computation graph. Calling backward() on a scalar output walks that
graph in reverse topological order, applying the chain rule by hand for every
operation — this is the same mechanism PyTorch uses, reimplemented here.
"""
from __future__ import annotations

from ownai.backend import scatter_add, xp


def _as_array(data):
    arr = xp.asarray(data)
    if not xp.issubdtype(arr.dtype, xp.floating):
        arr = arr.astype(xp.float32)
    return arr


def _unbroadcast(grad, shape):
    """Reduce a gradient back to `shape` after numpy-style broadcasting."""
    if grad.shape == shape:
        return grad
    while grad.ndim > len(shape):
        grad = grad.sum(axis=0)
    for axis, size in enumerate(shape):
        if size == 1 and grad.shape[axis] != 1:
            grad = grad.sum(axis=axis, keepdims=True)
    return grad


class Tensor:
    __slots__ = ("data", "grad", "requires_grad", "_prev", "_backward")

    def __init__(self, data, requires_grad: bool = False, _prev: tuple = ()):
        self.data = data if isinstance(data, xp.ndarray) else _as_array(data)
        self.requires_grad = requires_grad
        self.grad = None
        self._prev = _prev
        self._backward = None

    # ------------------------------------------------------------- helpers
    @property
    def shape(self):
        return self.data.shape

    @property
    def ndim(self):
        return self.data.ndim

    @property
    def dtype(self):
        return self.data.dtype

    def __repr__(self):
        return f"Tensor(shape={self.data.shape}, requires_grad={self.requires_grad})"

    def _add_grad(self, grad):
        # Gradients are never mutated in place, so plain reassignment is safe.
        self.grad = grad if self.grad is None else self.grad + grad

    def _make(self, data, prev):
        req = any(t.requires_grad for t in prev)
        return Tensor(data, req, tuple(prev) if req else ())

    def detach(self) -> "Tensor":
        return Tensor(self.data)

    # ------------------------------------------------------- arithmetic ops
    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(_as_array(other))
        out = self._make(self.data + other.data, (self, other))

        def backward():
            g = out.grad
            if self.requires_grad:
                self._add_grad(_unbroadcast(g, self.data.shape))
            if other.requires_grad:
                other._add_grad(_unbroadcast(g, other.data.shape))

        out._backward = backward
        return out

    __radd__ = __add__

    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(_as_array(other))
        out = self._make(self.data * other.data, (self, other))

        def backward():
            g = out.grad
            if self.requires_grad:
                self._add_grad(_unbroadcast(g * other.data, self.data.shape))
            if other.requires_grad:
                other._add_grad(_unbroadcast(g * self.data, other.data.shape))

        out._backward = backward
        return out

    __rmul__ = __mul__

    def __neg__(self):
        return self * -1.0

    def __sub__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(_as_array(other))
        return self + (-other)

    def __rsub__(self, other):
        return (-self) + other

    def __truediv__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(_as_array(other))
        out = self._make(self.data / other.data, (self, other))

        def backward():
            g = out.grad
            if self.requires_grad:
                self._add_grad(_unbroadcast(g / other.data, self.data.shape))
            if other.requires_grad:
                gb = -g * self.data / (other.data * other.data)
                other._add_grad(_unbroadcast(gb, other.data.shape))

        out._backward = backward
        return out

    def __pow__(self, power):
        assert isinstance(power, (int, float)), "only scalar exponents supported"
        out = self._make(self.data**power, (self,))

        def backward():
            if self.requires_grad:
                self._add_grad(out.grad * power * self.data ** (power - 1))

        out._backward = backward
        return out

    def __matmul__(self, other):
        assert isinstance(other, Tensor)
        out = self._make(self.data @ other.data, (self, other))

        def backward():
            g = out.grad
            if self.requires_grad:
                ga = g @ xp.swapaxes(other.data, -1, -2)
                self._add_grad(_unbroadcast(ga, self.data.shape))
            if other.requires_grad:
                gb = xp.swapaxes(self.data, -1, -2) @ g
                other._add_grad(_unbroadcast(gb, other.data.shape))

        out._backward = backward
        return out

    # ------------------------------------------------------ element-wise ops
    def exp(self):
        out = self._make(xp.exp(self.data), (self,))

        def backward():
            if self.requires_grad:
                self._add_grad(out.grad * out.data)

        out._backward = backward
        return out

    def log(self):
        out = self._make(xp.log(self.data), (self,))

        def backward():
            if self.requires_grad:
                self._add_grad(out.grad / self.data)

        out._backward = backward
        return out

    def tanh(self):
        out = self._make(xp.tanh(self.data), (self,))

        def backward():
            if self.requires_grad:
                self._add_grad(out.grad * (1.0 - out.data * out.data))

        out._backward = backward
        return out

    def relu(self):
        out = self._make(xp.maximum(self.data, 0.0), (self,))

        def backward():
            if self.requires_grad:
                self._add_grad(out.grad * (self.data > 0))

        out._backward = backward
        return out

    # ---------------------------------------------------------- reductions
    def sum(self, axis=None, keepdims: bool = False):
        out = self._make(self.data.sum(axis=axis, keepdims=keepdims), (self,))

        def backward():
            if not self.requires_grad:
                return
            g = out.grad
            if axis is not None and not keepdims:
                g = xp.expand_dims(g, axis)
            self._add_grad(xp.broadcast_to(g, self.data.shape))

        out._backward = backward
        return out

    def mean(self, axis=None, keepdims: bool = False):
        if axis is None:
            count = self.data.size
        else:
            axes = axis if isinstance(axis, tuple) else (axis,)
            count = 1
            for ax in axes:
                count *= self.data.shape[ax]
        return self.sum(axis=axis, keepdims=keepdims) * (1.0 / count)

    # -------------------------------------------------------- shape ops
    def reshape(self, *shape):
        out = self._make(self.data.reshape(*shape), (self,))

        def backward():
            if self.requires_grad:
                self._add_grad(out.grad.reshape(self.data.shape))

        out._backward = backward
        return out

    def transpose(self, *axes):
        axes = axes or tuple(reversed(range(self.data.ndim)))
        out = self._make(self.data.transpose(axes), (self,))
        inverse = tuple(int(i) for i in xp.argsort(xp.asarray(axes)))

        def backward():
            if self.requires_grad:
                self._add_grad(out.grad.transpose(inverse))

        out._backward = backward
        return out

    def __getitem__(self, idx):
        out = self._make(self.data[idx], (self,))

        def backward():
            if not self.requires_grad:
                return
            g = xp.zeros_like(self.data)
            scatter_add(g, idx, out.grad)
            self._add_grad(g)

        out._backward = backward
        return out

    # ------------------------------------------------- numerically-stable ops
    def softmax(self, axis: int = -1):
        shifted = self.data - self.data.max(axis=axis, keepdims=True)
        e = xp.exp(shifted)
        s = e / e.sum(axis=axis, keepdims=True)
        out = self._make(s, (self,))

        def backward():
            if self.requires_grad:
                g = out.grad
                self._add_grad(s * (g - (g * s).sum(axis=axis, keepdims=True)))

        out._backward = backward
        return out

    def log_softmax(self, axis: int = -1):
        shifted = self.data - self.data.max(axis=axis, keepdims=True)
        logp = shifted - xp.log(xp.exp(shifted).sum(axis=axis, keepdims=True))
        out = self._make(logp, (self,))

        def backward():
            if self.requires_grad:
                g = out.grad
                softmax = xp.exp(logp)
                self._add_grad(g - softmax * g.sum(axis=axis, keepdims=True))

        out._backward = backward
        return out

    # ------------------------------------------------------------- backward
    def backward(self):
        if self.data.size != 1:
            raise ValueError("backward() can only be called on a scalar tensor")

        # Iterative topological sort (graphs are too deep for recursion).
        topo, visited, stack = [], set(), [(self, False)]
        while stack:
            node, processed = stack.pop()
            if processed:
                topo.append(node)
                continue
            if id(node) in visited:
                continue
            visited.add(id(node))
            stack.append((node, True))
            for child in node._prev:
                if id(child) not in visited:
                    stack.append((child, False))

        self.grad = xp.ones_like(self.data)
        for node in reversed(topo):
            if node._backward is not None:
                node._backward()
