"""Cross-entropy loss for language modelling / classification."""
from __future__ import annotations

from ownai.autograd import Tensor
from ownai.backend import xp
from ownai.nn.module import Module


class CrossEntropyLoss(Module):
    """Mean negative log-likelihood over a batch of logits.

    logits: Tensor of shape (N, vocab); targets: integer array of shape (N,).
    Uses log_softmax (numerically stable) then gathers the target log-probs.
    """

    def forward(self, logits: Tensor, targets) -> Tensor:
        targets = xp.asarray(targets).astype(xp.int64)
        n = logits.shape[0]
        logp = logits.log_softmax(axis=-1)
        rows = xp.arange(n)
        chosen = logp[(rows, targets)]
        return -(chosen.sum() * (1.0 / n))
