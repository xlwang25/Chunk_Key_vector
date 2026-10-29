# SPDX-License-Identifier: Apache-2.0
"""Compatibility facade for external KV injector APIs.

This module preserves historical imports while delegating implementation to
smaller internal modules.
"""

from __future__ import annotations

from .external_kv_common import (
    _DYNAMIC_UPDATE_BACKEND_CPU_REBUILD,
    _DYNAMIC_UPDATE_BACKEND_GPU_POSITIONS_ONLY,
    _KEEPOLD_FUSED_K_OFF,
    _KEEPOLD_FUSED_K_CUDA_OP,
    _KEEPOLD_FUSED_K_TRITON,
    _KEEPOLD_FUSED_K_TORCH_CHUNKED,
    _KEEPOLD_PHASE_OFF,
    _KEEPOLD_PHASE_RECORD,
    _KEEPOLD_PHASE_REUSE,
    _ROPE_IMPL_LMCACHE_COPS_DIRECT,
    _ROPE_IMPL_LMCACHE_WRAPPER,
    _ROPE_IMPL_NONE,
    _ROPE_IMPL_TORCH_TWO_STEP,
    clear_keepold_page_bridge,
)
from .external_kv_external import (
    ExternalKVInjector,
    ExternalKVInjectorConfig,
    configure_external_kv_injector,
    get_external_kv_injector,
)
from .external_kv_selective import (
    SelectiveRecomputeConfig,
    SelectiveRecomputeRuntime,
    clear_keepold_chunk_page_index,
    clear_selective_recompute_runtime,
    commit_deferred_chunk_page_entries,
    configure_selective_recompute_runtime,
    get_selective_recompute_runtime,
)

__all__ = [
    "ExternalKVInjector",
    "ExternalKVInjectorConfig",
    "SelectiveRecomputeConfig",
    "SelectiveRecomputeRuntime",
    "get_external_kv_injector",
    "configure_external_kv_injector",
    "get_selective_recompute_runtime",
    "configure_selective_recompute_runtime",
    "clear_selective_recompute_runtime",
    "commit_deferred_chunk_page_entries",
    "clear_keepold_chunk_page_index",
    "clear_keepold_page_bridge",
    "_ROPE_IMPL_NONE",
    "_ROPE_IMPL_TORCH_TWO_STEP",
    "_ROPE_IMPL_LMCACHE_WRAPPER",
    "_ROPE_IMPL_LMCACHE_COPS_DIRECT",
    "_DYNAMIC_UPDATE_BACKEND_CPU_REBUILD",
    "_DYNAMIC_UPDATE_BACKEND_GPU_POSITIONS_ONLY",
    "_KEEPOLD_PHASE_OFF",
    "_KEEPOLD_PHASE_RECORD",
    "_KEEPOLD_PHASE_REUSE",
    "_KEEPOLD_FUSED_K_OFF",
    "_KEEPOLD_FUSED_K_TORCH_CHUNKED",
    "_KEEPOLD_FUSED_K_TRITON",
    "_KEEPOLD_FUSED_K_CUDA_OP",
]
