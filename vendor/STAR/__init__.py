"""Minimal inference-only vendored architecture for STAR (CVPR 2023)."""

from .stacked_hg import STARWFLWConfig, StackedHGNetV1

__all__ = [
    "STARWFLWConfig",
    "StackedHGNetV1",
]
