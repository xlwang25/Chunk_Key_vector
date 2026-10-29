# SPDX-License-Identifier: Apache-2.0
"""Shared constants/helpers for external KV injector paths."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import os
import re
import time

import torch

_LAYER_RE = re.compile(r"\.layers\.(\d+)\.")
_ROPE_IMPL_NONE = "none"
_ROPE_IMPL_TORCH_TWO_STEP = "torch_two_step"
_ROPE_IMPL_LMCACHE_WRAPPER = "lmcache_wrapper"
_ROPE_IMPL_LMCACHE_COPS_DIRECT = "lmcache_cops_direct"
_ROPE_IMPL_LMCACHE_FUSED_ALIAS = "lmcache_fused"
_DYNAMIC_UPDATE_BACKEND_CPU_REBUILD = "cpu_rebuild"
_DYNAMIC_UPDATE_BACKEND_GPU_POSITIONS_ONLY = "gpu_positions_only"

_KEEPOLD_PHASE_OFF = "off"
_KEEPOLD_PHASE_RECORD = "record"
_KEEPOLD_PHASE_REUSE = "reuse"
_KEEPOLD_FUSED_K_OFF = "off"
_KEEPOLD_FUSED_K_TORCH_CHUNKED = "torch_chunked"
_KEEPOLD_FUSED_K_TRITON = "triton"
_KEEPOLD_FUSED_K_CUDA_OP = "cuda_op"


@dataclass
class _KeepOldLayerSnapshot:
    key_cache: torch.Tensor
    value_cache: torch.Tensor
    slot_mapping: torch.Tensor
    num_actual_tokens: int


@dataclass
class _KeepOldPageBridge:
    prompt_len: int
    switch_prompt_positions_dev: Optional[torch.Tensor]
    layers: dict[int, _KeepOldLayerSnapshot]
    ts_unix: float


@dataclass
class _KeepOldCpuSubsetLayer:
    k_cpu: torch.Tensor
    v_cpu: torch.Tensor
    t_subset_prepare_ms: float
    t_pin_memory_ms: float


_KEEPOLD_PAGE_BRIDGES: dict[str, _KeepOldPageBridge] = {}


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in ("1", "true", "yes", "on")


def _extract_layer_id(layer_name: str) -> Optional[int]:
    match = _LAYER_RE.search(layer_name)
    if match is None:
        return None
    return int(match.group(1))


def _normalize_rope_impl(rope_impl: str) -> str:
    rope_impl = rope_impl.strip().lower()
    if rope_impl == _ROPE_IMPL_LMCACHE_FUSED_ALIAS:
        return _ROPE_IMPL_LMCACHE_WRAPPER
    return rope_impl


def clear_keepold_page_bridge(key: Optional[str] = None) -> None:
    """Clear KeepOld two-request paged-cache bridge snapshots."""
    if key is None:
        _KEEPOLD_PAGE_BRIDGES.clear()
        return
    _KEEPOLD_PAGE_BRIDGES.pop(str(key), None)


def _normalize_keepold_phase(phase: str) -> str:
    phase = phase.strip().lower()
    if phase not in (_KEEPOLD_PHASE_OFF, _KEEPOLD_PHASE_RECORD,
                     _KEEPOLD_PHASE_REUSE):
        raise ValueError(
            "Invalid keepold_phase: "
            f"{phase}. Expected one of "
            f"{_KEEPOLD_PHASE_OFF}, {_KEEPOLD_PHASE_RECORD}, {_KEEPOLD_PHASE_REUSE}."
        )
    return phase


def _normalize_keepold_fused_k_mode(mode: str) -> str:
    mode = mode.strip().lower()
    if mode not in (_KEEPOLD_FUSED_K_OFF, _KEEPOLD_FUSED_K_TORCH_CHUNKED,
                    _KEEPOLD_FUSED_K_TRITON, _KEEPOLD_FUSED_K_CUDA_OP):
        raise ValueError(
            "Invalid keepold_fused_k_mode: "
            f"{mode}. Expected one of "
            f"{_KEEPOLD_FUSED_K_OFF}, {_KEEPOLD_FUSED_K_TORCH_CHUNKED}, "
            f"{_KEEPOLD_FUSED_K_TRITON}, {_KEEPOLD_FUSED_K_CUDA_OP}.")
    return mode
