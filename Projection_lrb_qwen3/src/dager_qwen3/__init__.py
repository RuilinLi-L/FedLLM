"""None-only, Qwen3/RoPE-adapted DAGER components.

This package deliberately has no Projection-LRB implementation or defended-
gradient branch.  Its public entrypoint identifies every run as
``dager_qwen3_rope_defense_unaware``.
"""

from .model_adapter import ModelAdapterError, Qwen3RoPEDagerAdapter

ATTACK_NAME = "dager_qwen3_rope_defense_unaware"

__all__ = ["ATTACK_NAME", "ModelAdapterError", "Qwen3RoPEDagerAdapter"]
