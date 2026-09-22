"""OwnGPT: a from-scratch GPT sized for real GPUs (PyTorch), trained on Colab or cloud.

Needs the optional extra: ``pip install -e ".[gpt]"``.
"""
from ownai.gpt.model import GPT, PRESETS, GPTConfig, config_from_preset

__all__ = ["GPT", "GPTConfig", "PRESETS", "config_from_preset"]
