"""None-only, Qwen3/RoPE-adapted DAGER components.

This package deliberately has no Projection-LRB implementation or defended-
gradient branch.  Its public entrypoint identifies every run as
``dager_qwen3_rope_defense_unaware``.
"""

ATTACK_NAME = "dager_qwen3_rope_defense_unaware"

__all__ = ["ATTACK_NAME", "ModelAdapterError", "Qwen3RoPEDagerAdapter"]


def __getattr__(name: str):
    """Avoid importing Transformers-dependent adapters for metadata-only controls."""
    if name in {"ModelAdapterError", "Qwen3RoPEDagerAdapter"}:
        from .model_adapter import ModelAdapterError, Qwen3RoPEDagerAdapter

        return {
            "ModelAdapterError": ModelAdapterError,
            "Qwen3RoPEDagerAdapter": Qwen3RoPEDagerAdapter,
        }[name]
    raise AttributeError(name)
