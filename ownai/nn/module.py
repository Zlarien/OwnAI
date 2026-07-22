"""Base Module: parameter/child registration, train/eval mode, save/load.

Deliberately mirrors the PyTorch nn.Module mental model so the code reads
familiarly to anyone who has used a framework — but every line is our own.
"""
from __future__ import annotations

from ownai.autograd import Tensor
from ownai.backend import asnumpy, xp


class Parameter(Tensor):
    """A Tensor that is a learnable parameter (always requires grad)."""

    def __init__(self, data):
        super().__init__(data, requires_grad=True)


class Module:
    def __init__(self):
        self._params: dict[str, Parameter] = {}
        self._modules: dict[str, "Module"] = {}
        self.training = True

    def __setattr__(self, name, value):
        if isinstance(value, Parameter):
            self.__dict__.setdefault("_params", {})[name] = value
        elif isinstance(value, Module):
            self.__dict__.setdefault("_modules", {})[name] = value
        super().__setattr__(name, value)

    # ------------------------------------------------------------ traversal
    def parameters(self):
        seen = set()
        for p in self._params.values():
            if id(p) not in seen:
                seen.add(id(p))
                yield p
        for m in self._modules.values():
            for p in m.parameters():
                if id(p) not in seen:
                    seen.add(id(p))
                    yield p

    def named_parameters(self, prefix=""):
        for name, p in self._params.items():
            yield f"{prefix}{name}", p
        for mod_name, m in self._modules.items():
            yield from m.named_parameters(f"{prefix}{mod_name}.")

    # ---------------------------------------------------------------- modes
    def train(self):
        self.training = True
        for m in self._modules.values():
            m.train()
        return self

    def eval(self):
        self.training = False
        for m in self._modules.values():
            m.eval()
        return self

    # -------------------------------------------------------------- forward
    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    # ------------------------------------------------------ serialization
    def state_dict(self) -> dict:
        return {name: asnumpy(p.data) for name, p in self.named_parameters()}

    def load_state_dict(self, state: dict):
        params = dict(self.named_parameters())
        for name, arr in state.items():
            if name not in params:
                raise KeyError(f"unexpected parameter {name!r}")
            params[name].data = xp.asarray(arr)


class ModuleList(Module):
    """An ordered list of submodules, all registered for parameter traversal."""

    def __init__(self, modules=None):
        super().__init__()
        self._items: list[Module] = []
        for m in modules or []:
            self.append(m)

    def append(self, module: "Module"):
        self._modules[str(len(self._items))] = module
        self._items.append(module)
        return self

    def __iter__(self):
        return iter(self._items)

    def __getitem__(self, i):
        return self._items[i]

    def __len__(self):
        return len(self._items)
