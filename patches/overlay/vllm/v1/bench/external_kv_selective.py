# SPDX-License-Identifier: Apache-2.0
"""Selective recompute runtime implementation."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Any, Optional, Sequence
import math
import sys
import time

import torch

from vllm import _custom_ops as ops
from vllm.logger import init_logger

from .external_kv_common import (
    _ROPE_IMPL_NONE,
    _ROPE_IMPL_LMCACHE_COPS_DIRECT,
    _DYNAMIC_UPDATE_BACKEND_CPU_REBUILD,
    _DYNAMIC_UPDATE_BACKEND_GPU_POSITIONS_ONLY,
    _KEEPOLD_PHASE_OFF,
    _KEEPOLD_PHASE_RECORD,
    _KEEPOLD_PHASE_REUSE,
    _KEEPOLD_FUSED_K_OFF,
    _KEEPOLD_FUSED_K_TRITON,
    _KEEPOLD_FUSED_K_CUDA_OP,
    _normalize_rope_impl,
    _normalize_keepold_phase,
    _normalize_keepold_fused_k_mode,
    _KeepOldCpuSubsetLayer,
    _KeepOldLayerSnapshot,
    _KeepOldPageBridge,
    _KEEPOLD_PAGE_BRIDGES,
)
from .external_kv_layout import (
    reshape_to_kv_heads,
    extract_tokens_from_paged_cache,
    fused_extract_scatter_k_from_paged_cache,
    fused_extract_scatter_kv_from_paged_cache,
    fused_extract_scatter_kv_from_paged_cache_cuda_op_fast,
    fused_extract_scatter_kv_from_paged_cache_spans_cuda_op_fast,
    fused_scatter_rows_to_reuse_k,
    fused_scatter_rows_to_reuse_kv,
    fused_scatter_rows_to_reuse_kv_cuda_op_fast,
    prewarm_keepold_cuda_op,
)
from .chunk_page_index_hash import normalize_chunk_page_hash_algo
from .chunk_page_index_manager import clear_chunk_page_index, get_chunk_page_index
from .chunk_page_plan import (build_all_miss_page_reuse_plan,
                              build_old_slots_and_dst_idx,
                              build_page_reuse_plan, build_record_entries,
                              build_span_reuse_plan)
from .external_kv_metrics import append_metrics_jsonl
from .external_kv_selective_paths import compose_full_kv as compose_full_kv_impl

logger = init_logger(__name__)

_SELECTIVE_RUNTIME_SINGLETON: Optional["SelectiveRecomputeRuntime"] = None
_SELECTIVE_RUNTIME_INITED = False
_KEEPOLD_INDEX_MODE_SLOT_MAPPING = "slot_mapping"
_KEEPOLD_INDEX_MODE_CHUNK_PAGE = "chunk_page"
_KEEPOLD_SWITCH_SCOPE_ALL_NON_IMPORTANT = "all_non_important"
_KEEPOLD_SWITCH_SCOPE_D_ONLY = "d_only"
_KEEPOLD_TIMING_MODE_LIGHT = "light"
_KEEPOLD_TIMING_MODE_CUDA_EVENT = "cuda_event"
_KEEPOLD_LAYER_METRICS_MODE_FULL = "full"
_KEEPOLD_LAYER_METRICS_MODE_MINIMAL = "minimal"
_KEEPOLD_CPU_FALLBACK_MODE_SUBSET = "subset"
_KEEPOLD_CPU_FALLBACK_MODE_DENSE = "dense"
_KEEPOLD_CPU_FALLBACK_MODE_AUTO = "auto"
_KEEPOLD_CPU_FALLBACK_MODE_SUBSET_SPAN_DIRECT = "subset_span_direct"
_KEEPOLD_CPU_FALLBACK_AUTO_THRESHOLD = 0.5
_CHUNK_PAGE_EXEC_GRANULARITY_TOKEN = "token"
_CHUNK_PAGE_EXEC_GRANULARITY_SPAN = "span"
_KEEPOLD_LRU_UNIT_ENTRY = "entry"
_KEEPOLD_LRU_UNIT_CHUNK_HASH = "chunk_hash"


def _normalize_keepold_cpu_fallback_mode(mode: str) -> str:
    norm = str(mode).strip().lower()
    if norm not in (
            _KEEPOLD_CPU_FALLBACK_MODE_SUBSET,
            _KEEPOLD_CPU_FALLBACK_MODE_DENSE,
            _KEEPOLD_CPU_FALLBACK_MODE_AUTO,
            _KEEPOLD_CPU_FALLBACK_MODE_SUBSET_SPAN_DIRECT):
        raise ValueError(
            "Invalid keepold_cpu_fallback_mode: "
            f"{mode}. Expected one of "
            f"{_KEEPOLD_CPU_FALLBACK_MODE_SUBSET}, "
            f"{_KEEPOLD_CPU_FALLBACK_MODE_DENSE}, "
            f"{_KEEPOLD_CPU_FALLBACK_MODE_AUTO}, "
            f"{_KEEPOLD_CPU_FALLBACK_MODE_SUBSET_SPAN_DIRECT}.")
    return norm


def _normalize_chunk_page_exec_granularity(mode: str) -> str:
    norm = str(mode).strip().lower()
    if norm not in (_CHUNK_PAGE_EXEC_GRANULARITY_TOKEN,
                    _CHUNK_PAGE_EXEC_GRANULARITY_SPAN):
        raise ValueError(
            "Invalid keepold_chunk_page_exec_granularity: "
            f"{mode}. Expected one of "
            f"{_CHUNK_PAGE_EXEC_GRANULARITY_TOKEN}, "
            f"{_CHUNK_PAGE_EXEC_GRANULARITY_SPAN}.")
    return norm


def _normalize_keepold_layer_metrics_mode(mode: str) -> str:
    norm = str(mode).strip().lower()
    if norm not in (_KEEPOLD_LAYER_METRICS_MODE_FULL,
                    _KEEPOLD_LAYER_METRICS_MODE_MINIMAL):
        raise ValueError(
            "Invalid keepold_layer_metrics_mode: "
            f"{mode}. Expected one of "
            f"{_KEEPOLD_LAYER_METRICS_MODE_FULL}, "
            f"{_KEEPOLD_LAYER_METRICS_MODE_MINIMAL}.")
    return norm


def _normalize_keepold_lru_unit(mode: str) -> str:
    norm = str(mode).strip().lower()
    if norm not in (_KEEPOLD_LRU_UNIT_ENTRY, _KEEPOLD_LRU_UNIT_CHUNK_HASH):
        raise ValueError(
            "Invalid keepold_chunk_page_lru_unit: "
            f"{mode}. Expected one of "
            f"{_KEEPOLD_LRU_UNIT_ENTRY}, "
            f"{_KEEPOLD_LRU_UNIT_CHUNK_HASH}.")
    return norm

@dataclass
class SelectiveRecomputeConfig:
    enabled: bool
    pt_path: Optional[Path]
    layer_kv_data: Optional[dict[int, dict[str, torch.Tensor]]]
    prompt_len: int
    recompute_positions: list[int]
    reuse_old_positions: Optional[list[int]]
    reuse_rope_impl: str
    lmcache_root: Optional[Path]
    prefill_only: bool
    metrics_path: Optional[Path]
    dynamic_update_backend: str = _DYNAMIC_UPDATE_BACKEND_CPU_REBUILD
    keepold_bridge_key: Optional[str] = None
    keepold_phase: str = _KEEPOLD_PHASE_OFF
    keepold_switch_prompt_positions: Optional[list[int]] = None
    keepold_fused_k_mode: str = _KEEPOLD_FUSED_K_OFF
    keepold_fused_k_chunk_size: int = 0
    keepold_index_mode: str = _KEEPOLD_INDEX_MODE_SLOT_MAPPING
    keepold_switch_scope: str = _KEEPOLD_SWITCH_SCOPE_ALL_NON_IMPORTANT
    keepold_chunk_infos: Optional[list[dict[str, int]]] = None
    keepold_prompt_token_ids: Optional[list[int]] = None
    keepold_chunk_page_hash_algo: str = "sha256"
    keepold_chunk_page_max_chunks: int = 128
    keepold_chunk_page_max_entries_per_hash: int = 8
    keepold_chunk_page_lru_unit: str = _KEEPOLD_LRU_UNIT_ENTRY
    keepold_chunk_page_lru_evict_batch_chunks: int = 0
    keepold_chunk_page_lru_low_watermark_chunks: int = -1
    keepold_chunk_page_clear_each_sample: bool = True
    keepold_chunk_page_global_lookup: bool = False
    keepold_chunk_page_no_replace_existing_hash: bool = False
    keepold_chunk_page_block_invalidation: bool = False
    keepold_chunk_page_defer_register: bool = False
    keepold_chunk_page_register_hit_chunks_after_sample: bool = False
    keepold_chunk_page_update_entry_on_hit: bool = True
    keepold_model_id: Optional[str] = None
    keepold_kv_layout: Optional[str] = None
    keepold_chunk_page_debug_checks: bool = False
    keepold_chunk_page_execution_enabled: bool = False
    keepold_chunk_page_exec_granularity: str = _CHUNK_PAGE_EXEC_GRANULARITY_TOKEN
    keepold_timing_mode: str = _KEEPOLD_TIMING_MODE_LIGHT
    keepold_layer_metrics_mode: str = _KEEPOLD_LAYER_METRICS_MODE_FULL
    keepold_overlap_cpu_gpu_compose: bool = False
    keepold_token_skip_zero_if_full_coverage: bool = False
    keepold_cpu_fallback_mode: str = _KEEPOLD_CPU_FALLBACK_MODE_SUBSET
    keepold_diagnostic_layer_breakdown: bool = False
    keepold_rank_timing_diagnostic: bool = False
    keepold_nvtx_diagnostic: bool = False
    keepold_tensor_parallel_size: int = 1


class SelectiveRecomputeRuntime:
    """Runtime state for selective recompute + layerwise KV reuse."""

    def __init__(self, config: SelectiveRecomputeConfig):
        self.config = config
        self._source_loaded = False
        self._src_layer_kv: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        self._src_layer_kv_dynamic: dict[int, tuple[torch.Tensor,
                                                    torch.Tensor]] = {}
        self._dynamic_src_ready = False
        self._reuse_positions_cpu = torch.empty(0, dtype=torch.long)
        self._reuse_old_positions_cpu = torch.empty(0, dtype=torch.long)
        self._base_reuse_positions_cpu = torch.empty(0, dtype=torch.long)
        self._base_reuse_old_positions_cpu = torch.empty(0, dtype=torch.long)
        self._reuse_pos_to_old_cpu = torch.empty(0, dtype=torch.long)
        self._recompute_positions_cpu = torch.empty(0, dtype=torch.long)

        self._active = False
        self._forward_prompt_len = 0
        self._full_positions: Optional[torch.Tensor] = None
        self._recompute_positions_dev: Optional[torch.Tensor] = None
        self._reuse_positions_dev: Optional[torch.Tensor] = None
        self._reuse_old_positions_dev: Optional[torch.Tensor] = None
        self._base_reuse_positions_dev: Optional[torch.Tensor] = None
        self._base_reuse_old_positions_dev: Optional[torch.Tensor] = None
        self._lmcache_cops_module: Optional[Any] = None
        self._reuse_k_staging: Optional[torch.Tensor] = None
        self._reuse_v_staging: Optional[torch.Tensor] = None
        self._keepold_bridge_key: Optional[str] = None
        self._keepold_phase: str = _KEEPOLD_PHASE_OFF
        self._keepold_switch_prompt_positions_cpu = torch.empty(
            0, dtype=torch.long)
        self._keepold_switch_prompt_positions_dev: Optional[torch.Tensor] = None
        self._keepold_switch_reuse_idx_dev: Optional[torch.Tensor] = None
        self._keepold_switch_reuse_idx_cpu = torch.empty(0, dtype=torch.long)
        self._keepold_cpu_reuse_idx_dev: Optional[torch.Tensor] = None
        self._keepold_cpu_reuse_idx_cpu = torch.empty(0, dtype=torch.long)
        self._keepold_cpu_subset_layers: dict[int, _KeepOldCpuSubsetLayer] = {}
        self._keepold_cpu_subset_total_prepare_ms = 0.0
        self._keepold_cpu_subset_total_prepare_envelope_ms = 0.0
        self._keepold_cpu_subset_total_pin_ms = 0.0
        self._keepold_cpu_subset_layer_loop_ms = 0.0
        self._keepold_cpu_subset_index_select_ms = 0.0
        self._keepold_cpu_subset_contiguous_ms = 0.0
        self._keepold_cpu_subset_store_ms = 0.0
        self._keepold_cpu_subset_num_layers = 0
        self._keepold_cpu_subset_tokens = 0
        self._keepold_cpu_subset_ready = False
        self._keepold_cpu_fallback_mode_effective = (
            _KEEPOLD_CPU_FALLBACK_MODE_SUBSET)
        self._keepold_cpu_fallback_dense_enabled = False
        self._keepold_cpu_fallback_dense_reason = "inactive"
        self._keepold_cpu_fallback_tokens = 0
        self._keepold_cpu_fallback_ratio = 0.0
        self._keepold_cpu_fallback_subset_skipped = False
        self._keepold_cpu_fallback_span_direct_ready = False
        self._keepold_cpu_fallback_span_direct_reason = "inactive"
        self._keepold_cpu_fallback_span_direct_span_count = 0
        self._keepold_cpu_fallback_span_direct_tokens = 0
        self._keepold_cpu_fallback_span_direct_build_ms = 0.0
        self._keepold_cpu_fallback_span_direct_src_starts_cpu = torch.empty(
            0, dtype=torch.long)
        self._keepold_cpu_fallback_span_direct_dst_starts_cpu = torch.empty(
            0, dtype=torch.long)
        self._keepold_cpu_fallback_span_direct_lengths_cpu = torch.empty(
            0, dtype=torch.long)
        self._keepold_overlap_cpu_gpu_eligible = False
        self._keepold_overlap_cpu_gpu_reason = "disabled"
        self._reuse_cpu_k_staging: Optional[torch.Tensor] = None
        self._reuse_cpu_v_staging: Optional[torch.Tensor] = None
        self._layer_kv_pipeline_starts: dict[int, float] = {}
        self._keepold_diag_layer_breakdown_enabled = bool(
            getattr(self.config, "keepold_diagnostic_layer_breakdown", False))
        self._keepold_diag_layer_components: dict[int, dict[str, Any]] = {}
        self._keepold_diag_layer_pipeline_events: list[dict[str, Any]] = []
        self._keepold_switch_reuse_idx_i64_dev: Optional[torch.Tensor] = None
        self._keepold_old_slots_i64_by_layer: dict[int, torch.Tensor] = {}
        self._keepold_old_page_fast_ready = False
        self._keepold_old_page_fast_reason = "inactive"
        self._keepold_old_page_prepare_ms = 0.0
        self._keepold_gpu_index_idx_i64_dev: Optional[torch.Tensor] = None
        self._keepold_gpu_index_fast_ready = False
        self._keepold_gpu_index_fast_reason = "inactive"
        self._keepold_gpu_index_prepare_ms = 0.0
        self._keepold_chunk_page_shared_switch_reuse_idx_i64_dev: Optional[
            torch.Tensor] = None
        self._keepold_chunk_page_shared_cpu_reuse_idx_i64_dev: Optional[
            torch.Tensor] = None
        self._keepold_chunk_page_shared_old_slots_i64_dev: Optional[
            torch.Tensor] = None
        self._keepold_chunk_page_shared_index_ready = False
        self._keepold_chunk_page_shared_index_reason = "inactive"
        self._keepold_chunk_page_shared_index_prepare_count = 0
        self._keepold_chunk_page_exec_granularity = (
            _CHUNK_PAGE_EXEC_GRANULARITY_TOKEN)
        self._keepold_index_mode = _KEEPOLD_INDEX_MODE_SLOT_MAPPING
        self._keepold_switch_scope = _KEEPOLD_SWITCH_SCOPE_ALL_NON_IMPORTANT
        self._keepold_chunk_page_source_id: Optional[str] = None
        self._keepold_chunk_page_entry_registered = False
        self._keepold_chunk_page_post_register_infos: list[dict[str, int]] = []
        self._keepold_chunk_page_deferred_record: Optional[dict[str, Any]] = None
        self._keepold_chunk_page_deferred_committed = False
        self._keepold_chunk_page_dryrun_supported = False
        self._keepold_chunk_page_dryrun_reason = "inactive"
        self._keepold_chunk_page_plan: Optional[dict[str, Any]] = None
        self._keepold_chunk_page_old_slots_cpu = torch.empty(0, dtype=torch.long)
        self._keepold_chunk_page_dst_idx_cpu = torch.empty(0, dtype=torch.long)
        self._keepold_chunk_page_old_positions_cpu = torch.empty(
            0, dtype=torch.long)
        self._keepold_chunk_page_new_positions_cpu = torch.empty(
            0, dtype=torch.long)
        self._keepold_chunk_page_gpu_span_block_ids_cpu = torch.empty(
            0, dtype=torch.long)
        self._keepold_chunk_page_gpu_span_page_offsets_cpu = torch.empty(
            0, dtype=torch.long)
        self._keepold_chunk_page_gpu_span_dst_starts_cpu = torch.empty(
            0, dtype=torch.long)
        self._keepold_chunk_page_gpu_span_lengths_cpu = torch.empty(
            0, dtype=torch.long)
        self._keepold_chunk_page_gpu_span_old_abs_starts_cpu = torch.empty(
            0, dtype=torch.long)
        self._keepold_chunk_page_gpu_span_new_abs_starts_cpu = torch.empty(
            0, dtype=torch.long)
        self._keepold_chunk_page_cpu_span_src_starts_cpu = torch.empty(
            0, dtype=torch.long)
        self._keepold_chunk_page_cpu_span_dst_starts_cpu = torch.empty(
            0, dtype=torch.long)
        self._keepold_chunk_page_cpu_span_lengths_cpu = torch.empty(
            0, dtype=torch.long)
        self._keepold_chunk_page_span_cpu_copy_src_starts_cpu = torch.empty(
            0, dtype=torch.long)
        self._keepold_chunk_page_span_cpu_copy_dst_starts_cpu = torch.empty(
            0, dtype=torch.long)
        self._keepold_chunk_page_span_cpu_copy_lengths_cpu = torch.empty(
            0, dtype=torch.long)
        self._keepold_span_reuse_old_pos_for_rope_cpu = torch.empty(
            0, dtype=torch.long)
        self._keepold_chunk_page_gpu_span_block_ids_dev: Optional[torch.Tensor] = None
        self._keepold_chunk_page_gpu_span_page_offsets_dev: Optional[
            torch.Tensor] = None
        self._keepold_chunk_page_gpu_span_dst_starts_dev: Optional[
            torch.Tensor] = None
        self._keepold_chunk_page_gpu_span_lengths_dev: Optional[torch.Tensor] = None
        self._keepold_chunk_page_gpu_span_old_abs_starts_dev: Optional[
            torch.Tensor] = None
        self._keepold_chunk_page_gpu_span_new_abs_starts_dev: Optional[
            torch.Tensor] = None
        self._keepold_chunk_page_cpu_span_src_starts_dev: Optional[
            torch.Tensor] = None
        self._keepold_chunk_page_cpu_span_dst_starts_dev: Optional[
            torch.Tensor] = None
        self._keepold_chunk_page_cpu_span_lengths_dev: Optional[
            torch.Tensor] = None
        self._keepold_chunk_page_span_cpu_copy_src_starts_dev: Optional[
            torch.Tensor] = None
        self._keepold_chunk_page_span_cpu_copy_dst_starts_dev: Optional[
            torch.Tensor] = None
        self._keepold_chunk_page_span_cpu_copy_lengths_dev: Optional[
            torch.Tensor] = None
        self._keepold_span_reuse_old_pos_for_rope_dev: Optional[torch.Tensor] = None
        self._keepold_chunk_page_span_coverage_checked = False
        self._keepold_chunk_page_span_coverage_reason = "inactive"
        self._keepold_chunk_page_span_full_reuse_tokens = 0
        self._keepold_chunk_page_span_prefix_uncovered_tokens = 0
        self._keepold_chunk_page_span_exec_ready = False
        self._keepold_chunk_page_span_exec_reason = "inactive"
        self._keepold_chunk_page_span_plan_build_ms = 0.0
        self._keepold_chunk_page_gpu_span_count = 0
        self._keepold_chunk_page_cpu_span_count = 0
        self._keepold_chunk_page_span_index_h2d_submit_ms = 0.0
        self._keepold_chunk_page_span_index_h2d_device_ms = float("nan")
        self._keepold_chunk_page_lookup_status = "inactive"
        self._keepold_chunk_page_lookup_reason = "inactive"
        self._keepold_chunk_page_target_reuse_tokens = 0
        self._keepold_chunk_page_gathered_target_tokens = 0
        self._keepold_chunk_page_cpu_fallback_tokens = 0
        self._keepold_chunk_page_skipped_out_of_range = 0
        self._keepold_chunk_page_skipped_missing_dst = 0
        self._keepold_chunk_page_skipped_duplicate = 0
        self._keepold_chunk_page_noncontiguous_span_count = 0
        self._keepold_chunk_page_old_slots_build_ms = 0.0
        self._keepold_chunk_page_exec_success = False
        self._keepold_chunk_page_exec_fallback_reason = "inactive"
        self._keepold_chunk_page_exec_tokens = 0
        self._keepold_chunk_page_exec_gpu_tokens = 0
        self._keepold_chunk_page_exec_cpu_tokens = 0
        self._keepold_chunk_page_lookup_hash_ms = 0.0
        self._keepold_chunk_page_lookup_index_ms = 0.0
        self._keepold_chunk_page_lookup_filter_ms = 0.0
        self._keepold_chunk_page_lookup_candidate_scan_ms = 0.0
        self._keepold_chunk_page_lookup_block_validity_check_ms = 0.0
        self._keepold_chunk_page_lookup_invalid_filter_ms = 0.0
        self._keepold_chunk_page_lookup_candidates_scanned = 0
        self._keepold_chunk_page_lookup_candidates_valid = 0
        self._keepold_chunk_page_lookup_candidates_invalid = 0
        self._keepold_chunk_page_metadata_only_candidate_count = 0
        self._keepold_chunk_page_metadata_only_filtered_count = 0
        self._keepold_chunk_page_metadata_only_false_hit_count = 0
        self._keepold_chunk_page_runtime_prelookup_source_id = ""
        self._keepold_chunk_page_runtime_prelookup_source_is_metadata_only = 0
        self._keepold_chunk_page_runtime_prelookup_source_dtype = ""
        self._keepold_chunk_page_runtime_prelookup_source_kv_layout = ""
        self._keepold_chunk_page_runtime_prelookup_source_block_size = 0
        self._keepold_chunk_page_lookup_materialize_ms = 0.0
        self._keepold_chunk_page_lookup_total_ms = 0.0
        self._keepold_chunk_page_lookup_cache_hit = 0
        self._keepold_chunk_page_lookup_cache_reason = "inactive"
        self._keepold_chunk_page_lookup_cache_saved_ms = 0.0
        self._keepold_chunk_page_lookup_cache_signature: Optional[str] = None
        self._keepold_chunk_page_lookup_cache_plan: Optional[Any] = None
        self._keepold_chunk_page_lookup_cache_total_ms = 0.0
        self._keepold_chunk_page_lookup_calls_in_runtime = 0
        self._keepold_timing_v2_index_tensor_h2d_submit_ms = 0.0
        self._keepold_timing_v2_index_tensor_h2d_device_ms = float("nan")
        self._keepold_timing_v2_chunk_page_exec_partition_ms = 0.0
        self._keepold_chunk_page_partition_build_ms = 0.0
        self._keepold_chunk_page_mixed_plan_materialize_ms = 0.0
        self._keepold_chunk_page_cpu_fallback_rows_build_ms = 0.0
        self._keepold_chunk_page_old_page_plan_build_ms = 0.0
        self._keepold_layer_metrics_mode = _normalize_keepold_layer_metrics_mode(
            getattr(self.config, "keepold_layer_metrics_mode",
                    _KEEPOLD_LAYER_METRICS_MODE_FULL))
        self._keepold_emit_layer_metrics_full = (
            self._keepold_layer_metrics_mode == _KEEPOLD_LAYER_METRICS_MODE_FULL)
        self._keepold_layer_metrics_agg_sum: dict[str, float] = {}
        self._keepold_layer_metrics_agg_cnt: dict[str, int] = {}
        self._keepold_layer_metrics_agg_latest: dict[str, Any] = {}
        self._keepold_layer_metrics_agg_layers = 0
        self._keepold_cached_bridge: Optional[_KeepOldPageBridge] = None
        self._keepold_cached_bridge_layers: Optional[
            dict[int, _KeepOldLayerSnapshot]] = None
        self._keepold_paged_layout_meta_by_layer: dict[
            int, Optional[tuple[int, int]]] = {}
        self._kv_eq_probe_enabled = False
        self._kv_eq_probe_max_tokens = 0
        self._kv_eq_probe_positions_cpu = torch.empty(0, dtype=torch.long)
        self._kv_eq_probe_layer_ids: Optional[set[int]] = None
        self._kv_eq_probe_tag = ""
        self._kv_eq_probe_records: dict[int, dict[str, Any]] = {}

        if self.config.metrics_path is not None:
            self.config.metrics_path.parent.mkdir(parents=True, exist_ok=True)

    def reconfigure(self, config: SelectiveRecomputeConfig) -> None:
        self.config = config
        self._source_loaded = False
        self._src_layer_kv = {}
        self._src_layer_kv_dynamic = {}
        self._dynamic_src_ready = False
        self._reuse_positions_cpu = torch.empty(0, dtype=torch.long)
        self._reuse_old_positions_cpu = torch.empty(0, dtype=torch.long)
        self._base_reuse_positions_cpu = torch.empty(0, dtype=torch.long)
        self._base_reuse_old_positions_cpu = torch.empty(0, dtype=torch.long)
        self._reuse_pos_to_old_cpu = torch.empty(0, dtype=torch.long)
        self._recompute_positions_cpu = torch.empty(0, dtype=torch.long)
        self._active = False
        self._forward_prompt_len = 0
        self._full_positions = None
        self._recompute_positions_dev = None
        self._reuse_positions_dev = None
        self._reuse_old_positions_dev = None
        self._base_reuse_positions_dev = None
        self._base_reuse_old_positions_dev = None
        self._lmcache_cops_module = None
        self._reuse_k_staging = None
        self._reuse_v_staging = None
        self._keepold_bridge_key = None
        self._keepold_phase = _KEEPOLD_PHASE_OFF
        self._keepold_switch_prompt_positions_cpu = torch.empty(
            0, dtype=torch.long)
        self._keepold_switch_prompt_positions_dev = None
        self._keepold_switch_reuse_idx_dev = None
        self._keepold_switch_reuse_idx_cpu = torch.empty(0, dtype=torch.long)
        self._keepold_cpu_reuse_idx_dev = None
        self._keepold_cpu_reuse_idx_cpu = torch.empty(0, dtype=torch.long)
        self._keepold_cpu_subset_layers = {}
        self._keepold_cpu_subset_total_prepare_ms = 0.0
        self._keepold_cpu_subset_total_prepare_envelope_ms = 0.0
        self._keepold_cpu_subset_total_pin_ms = 0.0
        self._keepold_cpu_subset_layer_loop_ms = 0.0
        self._keepold_cpu_subset_index_select_ms = 0.0
        self._keepold_cpu_subset_contiguous_ms = 0.0
        self._keepold_cpu_subset_store_ms = 0.0
        self._keepold_cpu_subset_num_layers = 0
        self._keepold_cpu_subset_tokens = 0
        self._keepold_cpu_subset_ready = False
        self._keepold_cpu_fallback_mode_effective = (
            _KEEPOLD_CPU_FALLBACK_MODE_SUBSET)
        self._keepold_cpu_fallback_dense_enabled = False
        self._keepold_cpu_fallback_dense_reason = "inactive"
        self._keepold_cpu_fallback_tokens = 0
        self._keepold_cpu_fallback_ratio = 0.0
        self._keepold_cpu_fallback_subset_skipped = False
        self._keepold_cpu_fallback_span_direct_ready = False
        self._keepold_cpu_fallback_span_direct_reason = "inactive"
        self._keepold_cpu_fallback_span_direct_span_count = 0
        self._keepold_cpu_fallback_span_direct_tokens = 0
        self._keepold_cpu_fallback_span_direct_build_ms = 0.0
        self._keepold_cpu_fallback_span_direct_src_starts_cpu = torch.empty(
            0, dtype=torch.long)
        self._keepold_cpu_fallback_span_direct_dst_starts_cpu = torch.empty(
            0, dtype=torch.long)
        self._keepold_cpu_fallback_span_direct_lengths_cpu = torch.empty(
            0, dtype=torch.long)
        self._keepold_overlap_cpu_gpu_eligible = False
        self._keepold_overlap_cpu_gpu_reason = "disabled"
        self._reuse_cpu_k_staging = None
        self._reuse_cpu_v_staging = None
        self._layer_kv_pipeline_starts = {}
        self._keepold_diag_layer_breakdown_enabled = bool(
            getattr(self.config, "keepold_diagnostic_layer_breakdown", False))
        self._keepold_diag_layer_components = {}
        self._keepold_diag_layer_pipeline_events = []
        self._keepold_switch_reuse_idx_i64_dev = None
        self._keepold_old_slots_i64_by_layer = {}
        self._keepold_old_page_fast_ready = False
        self._keepold_old_page_fast_reason = "inactive"
        self._keepold_old_page_prepare_ms = 0.0
        self._keepold_gpu_index_idx_i64_dev = None
        self._keepold_gpu_index_fast_ready = False
        self._keepold_gpu_index_fast_reason = "inactive"
        self._keepold_gpu_index_prepare_ms = 0.0
        self._keepold_chunk_page_shared_switch_reuse_idx_i64_dev = None
        self._keepold_chunk_page_shared_cpu_reuse_idx_i64_dev = None
        self._keepold_chunk_page_shared_old_slots_i64_dev = None
        self._keepold_chunk_page_shared_index_ready = False
        self._keepold_chunk_page_shared_index_reason = "inactive"
        self._keepold_chunk_page_shared_index_prepare_count = 0
        self._keepold_chunk_page_exec_granularity = (
            _CHUNK_PAGE_EXEC_GRANULARITY_TOKEN)
        self._keepold_index_mode = _KEEPOLD_INDEX_MODE_SLOT_MAPPING
        self._keepold_switch_scope = _KEEPOLD_SWITCH_SCOPE_ALL_NON_IMPORTANT
        self._keepold_chunk_page_source_id = None
        self._keepold_chunk_page_entry_registered = False
        self._keepold_chunk_page_post_register_infos = []
        self._keepold_chunk_page_deferred_record = None
        self._keepold_chunk_page_deferred_committed = False
        self._keepold_chunk_page_dryrun_supported = False
        self._keepold_chunk_page_dryrun_reason = "inactive"
        self._keepold_chunk_page_plan = None
        self._keepold_chunk_page_old_slots_cpu = torch.empty(
            0, dtype=torch.long)
        self._keepold_chunk_page_dst_idx_cpu = torch.empty(0, dtype=torch.long)
        self._keepold_chunk_page_old_positions_cpu = torch.empty(
            0, dtype=torch.long)
        self._keepold_chunk_page_new_positions_cpu = torch.empty(
            0, dtype=torch.long)
        self._keepold_chunk_page_gpu_span_block_ids_cpu = torch.empty(
            0, dtype=torch.long)
        self._keepold_chunk_page_gpu_span_page_offsets_cpu = torch.empty(
            0, dtype=torch.long)
        self._keepold_chunk_page_gpu_span_dst_starts_cpu = torch.empty(
            0, dtype=torch.long)
        self._keepold_chunk_page_gpu_span_lengths_cpu = torch.empty(
            0, dtype=torch.long)
        self._keepold_chunk_page_gpu_span_old_abs_starts_cpu = torch.empty(
            0, dtype=torch.long)
        self._keepold_chunk_page_gpu_span_new_abs_starts_cpu = torch.empty(
            0, dtype=torch.long)
        self._keepold_chunk_page_cpu_span_src_starts_cpu = torch.empty(
            0, dtype=torch.long)
        self._keepold_chunk_page_cpu_span_dst_starts_cpu = torch.empty(
            0, dtype=torch.long)
        self._keepold_chunk_page_cpu_span_lengths_cpu = torch.empty(
            0, dtype=torch.long)
        self._keepold_chunk_page_span_cpu_copy_src_starts_cpu = torch.empty(
            0, dtype=torch.long)
        self._keepold_chunk_page_span_cpu_copy_dst_starts_cpu = torch.empty(
            0, dtype=torch.long)
        self._keepold_chunk_page_span_cpu_copy_lengths_cpu = torch.empty(
            0, dtype=torch.long)
        self._keepold_span_reuse_old_pos_for_rope_cpu = torch.empty(
            0, dtype=torch.long)
        self._keepold_chunk_page_gpu_span_block_ids_dev = None
        self._keepold_chunk_page_gpu_span_page_offsets_dev = None
        self._keepold_chunk_page_gpu_span_dst_starts_dev = None
        self._keepold_chunk_page_gpu_span_lengths_dev = None
        self._keepold_chunk_page_gpu_span_old_abs_starts_dev = None
        self._keepold_chunk_page_gpu_span_new_abs_starts_dev = None
        self._keepold_chunk_page_cpu_span_src_starts_dev = None
        self._keepold_chunk_page_cpu_span_dst_starts_dev = None
        self._keepold_chunk_page_cpu_span_lengths_dev = None
        self._keepold_chunk_page_span_cpu_copy_src_starts_dev = None
        self._keepold_chunk_page_span_cpu_copy_dst_starts_dev = None
        self._keepold_chunk_page_span_cpu_copy_lengths_dev = None
        self._keepold_span_reuse_old_pos_for_rope_dev = None
        self._keepold_chunk_page_span_coverage_checked = False
        self._keepold_chunk_page_span_coverage_reason = "inactive"
        self._keepold_chunk_page_span_full_reuse_tokens = 0
        self._keepold_chunk_page_span_prefix_uncovered_tokens = 0
        self._keepold_chunk_page_span_exec_ready = False
        self._keepold_chunk_page_span_exec_reason = "inactive"
        self._keepold_chunk_page_span_plan_build_ms = 0.0
        self._keepold_chunk_page_gpu_span_count = 0
        self._keepold_chunk_page_cpu_span_count = 0
        self._keepold_chunk_page_span_index_h2d_submit_ms = 0.0
        self._keepold_chunk_page_span_index_h2d_device_ms = float("nan")
        self._keepold_chunk_page_lookup_status = "inactive"
        self._keepold_chunk_page_lookup_reason = "inactive"
        self._keepold_chunk_page_target_reuse_tokens = 0
        self._keepold_chunk_page_gathered_target_tokens = 0
        self._keepold_chunk_page_cpu_fallback_tokens = 0
        self._keepold_chunk_page_skipped_out_of_range = 0
        self._keepold_chunk_page_skipped_missing_dst = 0
        self._keepold_chunk_page_skipped_duplicate = 0
        self._keepold_chunk_page_noncontiguous_span_count = 0
        self._keepold_chunk_page_old_slots_build_ms = 0.0
        self._keepold_chunk_page_exec_success = False
        self._keepold_chunk_page_exec_fallback_reason = "inactive"
        self._keepold_chunk_page_exec_tokens = 0
        self._keepold_chunk_page_exec_gpu_tokens = 0
        self._keepold_chunk_page_exec_cpu_tokens = 0
        self._keepold_chunk_page_lookup_hash_ms = 0.0
        self._keepold_chunk_page_lookup_index_ms = 0.0
        self._keepold_chunk_page_lookup_filter_ms = 0.0
        self._keepold_chunk_page_lookup_candidate_scan_ms = 0.0
        self._keepold_chunk_page_lookup_block_validity_check_ms = 0.0
        self._keepold_chunk_page_lookup_invalid_filter_ms = 0.0
        self._keepold_chunk_page_lookup_candidates_scanned = 0
        self._keepold_chunk_page_lookup_candidates_valid = 0
        self._keepold_chunk_page_lookup_candidates_invalid = 0
        self._keepold_chunk_page_metadata_only_candidate_count = 0
        self._keepold_chunk_page_metadata_only_filtered_count = 0
        self._keepold_chunk_page_metadata_only_false_hit_count = 0
        self._keepold_chunk_page_runtime_prelookup_source_id = ""
        self._keepold_chunk_page_runtime_prelookup_source_is_metadata_only = 0
        self._keepold_chunk_page_runtime_prelookup_source_dtype = ""
        self._keepold_chunk_page_runtime_prelookup_source_kv_layout = ""
        self._keepold_chunk_page_runtime_prelookup_source_block_size = 0
        self._keepold_chunk_page_lookup_materialize_ms = 0.0
        self._keepold_chunk_page_lookup_total_ms = 0.0
        self._keepold_chunk_page_lookup_cache_hit = 0
        self._keepold_chunk_page_lookup_cache_reason = "inactive"
        self._keepold_chunk_page_lookup_cache_saved_ms = 0.0
        self._keepold_chunk_page_lookup_cache_signature = None
        self._keepold_chunk_page_lookup_cache_plan = None
        self._keepold_chunk_page_lookup_cache_total_ms = 0.0
        self._keepold_chunk_page_lookup_calls_in_runtime = 0
        self._keepold_timing_v2_index_tensor_h2d_submit_ms = 0.0
        self._keepold_timing_v2_index_tensor_h2d_device_ms = float("nan")
        self._keepold_timing_v2_chunk_page_exec_partition_ms = 0.0
        self._keepold_chunk_page_partition_build_ms = 0.0
        self._keepold_chunk_page_mixed_plan_materialize_ms = 0.0
        self._keepold_chunk_page_cpu_fallback_rows_build_ms = 0.0
        self._keepold_chunk_page_old_page_plan_build_ms = 0.0
        self._keepold_layer_metrics_mode = _normalize_keepold_layer_metrics_mode(
            getattr(self.config, "keepold_layer_metrics_mode",
                    _KEEPOLD_LAYER_METRICS_MODE_FULL))
        self._keepold_emit_layer_metrics_full = (
            self._keepold_layer_metrics_mode == _KEEPOLD_LAYER_METRICS_MODE_FULL)
        self._keepold_layer_metrics_agg_sum = {}
        self._keepold_layer_metrics_agg_cnt = {}
        self._keepold_layer_metrics_agg_latest = {}
        self._keepold_layer_metrics_agg_layers = 0
        self._keepold_cached_bridge = None
        self._keepold_cached_bridge_layers = None
        self._keepold_paged_layout_meta_by_layer = {}
        self._kv_eq_probe_enabled = False
        self._kv_eq_probe_max_tokens = 0
        self._kv_eq_probe_positions_cpu = torch.empty(0, dtype=torch.long)
        self._kv_eq_probe_layer_ids = None
        self._kv_eq_probe_tag = ""
        self._kv_eq_probe_records = {}
        if self.config.metrics_path is not None:
            self.config.metrics_path.parent.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _reshape_to_kv_heads(
        tensor: torch.Tensor,
        num_kv_heads: int,
        head_size: int,
    ) -> torch.Tensor:
        return reshape_to_kv_heads(tensor, num_kv_heads, head_size)

    def _append_metrics(self, metrics: dict[str, Any]) -> None:
        append_metrics_jsonl(self.config.metrics_path, metrics)

    def append_metric_event(self, metrics: dict[str, Any]) -> None:
        self._append_metrics(metrics)

    def configure_kv_equivalence_probe(
        self,
        *,
        enabled: bool,
        probe_positions: Sequence[int],
        layer_ids: Optional[Sequence[int]] = None,
        max_tokens: int = 0,
        tag: str = "",
    ) -> None:
        self._kv_eq_probe_enabled = bool(enabled)
        self._kv_eq_probe_max_tokens = int(max(0, int(max_tokens)))
        self._kv_eq_probe_tag = str(tag)
        self._kv_eq_probe_records = {}
        if not self._kv_eq_probe_enabled:
            self._kv_eq_probe_positions_cpu = torch.empty(0, dtype=torch.long)
            self._kv_eq_probe_layer_ids = None
            return
        pos = torch.tensor([int(x) for x in probe_positions], dtype=torch.long)
        if int(pos.numel()) > 0:
            pos = torch.unique(pos, sorted=True).contiguous()
            if self._kv_eq_probe_max_tokens > 0 and int(
                    pos.numel()) > self._kv_eq_probe_max_tokens:
                n = int(self._kv_eq_probe_max_tokens)
                idx = torch.linspace(
                    0,
                    int(pos.numel()) - 1,
                    steps=n,
                    dtype=torch.float32,
                ).round().to(dtype=torch.long)
                pos = pos.index_select(0, idx).contiguous()
        self._kv_eq_probe_positions_cpu = pos
        if layer_ids is None:
            self._kv_eq_probe_layer_ids = None
        else:
            self._kv_eq_probe_layer_ids = {int(x) for x in layer_ids}

    def pop_kv_equivalence_probe_records(self) -> dict[int, dict[str, Any]]:
        out = self._kv_eq_probe_records
        self._kv_eq_probe_records = {}
        return out

    def _kv_eq_probe_layer_enabled(self, layer_id: int) -> bool:
        if not self._kv_eq_probe_enabled:
            return False
        layer_ids = self._kv_eq_probe_layer_ids
        if layer_ids is None:
            return True
        return int(layer_id) in layer_ids

    def should_emit_layer_metrics_full(self) -> bool:
        return bool(self._keepold_emit_layer_metrics_full)

    def _reset_layer_metrics_aggregate(self) -> None:
        self._keepold_layer_metrics_agg_sum = {}
        self._keepold_layer_metrics_agg_cnt = {}
        self._keepold_layer_metrics_agg_latest = {}
        self._keepold_layer_metrics_agg_layers = 0

    def _record_layer_metrics_aggregate_value(self, key: str, value: Any) -> None:
        try:
            v = float(value)
        except (TypeError, ValueError):
            return
        if math.isnan(v) or math.isinf(v):
            return
        self._keepold_layer_metrics_agg_sum[key] = (
            float(self._keepold_layer_metrics_agg_sum.get(key, 0.0)) + v)
        self._keepold_layer_metrics_agg_cnt[key] = (
            int(self._keepold_layer_metrics_agg_cnt.get(key, 0)) + 1)

    def _record_layer_metrics_aggregate_latest(self, key: str, value: Any) -> None:
        if value is None:
            return
        self._keepold_layer_metrics_agg_latest[key] = value

    def record_compose_layer_metrics_aggregate(
        self,
        *,
        reuse_tokens_h2d: int,
        timing_v2_kv_h2d_k_submit_ms: float,
        timing_v2_kv_h2d_v_submit_ms: float,
        timing_v2_kv_h2d_device_ms: float,
        timing_v2_gpu_index_scatter_submit_ms: float,
        timing_v2_gpu_index_scatter_device_ms: float,
        timing_v2_old_page_gather_submit_ms: float,
        timing_v2_old_page_gather_device_ms: float,
        timing_v2_reuse_rope_submit_ms: float,
        timing_v2_reuse_rope_device_ms: float,
        timing_v2_reuse_rope_wait_ms: float,
        t_gpu_index_copy_sync_overhead_ms: float,
        span_gpu_gather_submit_ms: float,
        span_gpu_gather_device_ms: float,
        span_cpu_copy_submit_ms: float,
        span_cpu_copy_device_ms: float,
        span_cpu_copy_span_count: int,
        span_cpu_copy_tokens: int,
        span_no_dense_copy: int,
        cpu_fallback_span_direct_used: float,
        cpu_fallback_span_direct_span_count: float,
        cpu_fallback_span_direct_tokens: float,
        cpu_fallback_span_direct_copy_submit_ms: float,
        cpu_fallback_dense_enabled: bool,
        cpu_fallback_tokens: int,
        cpu_fallback_ratio: float,
        cpu_fallback_subset_skipped: bool,
        chunk_page_exec_granularity: str,
        cpu_fallback_mode: str,
        cpu_fallback_dense_reason: str,
        token_zero_submit_ms: float,
        token_zero_skipped: int,
        token_zero_skip_reason: str,
        token_full_coverage_ready: int,
        token_full_coverage_reason: str,
        token_full_coverage_gpu_rows: int,
        token_full_coverage_cpu_rows: int,
        token_full_coverage_total_rows: int,
        token_full_coverage_overlap_rows: int,
        token_full_coverage_missing_rows: int,
        compose_fixed_total_ms: float,
        compose_layout_cache_hit: int,
        compose_bridge_cache_hit: int,
        compose_metrics_minimal_used: int,
        compose_return_pack_ms: float,
    ) -> None:
        if self.should_emit_layer_metrics_full():
            return
        self._keepold_layer_metrics_agg_layers += 1
        self._record_layer_metrics_aggregate_value("reuse_tokens_h2d",
                                                   reuse_tokens_h2d)
        self._record_layer_metrics_aggregate_value(
            "timing_v2_kv_h2d_k_submit_ms", timing_v2_kv_h2d_k_submit_ms)
        self._record_layer_metrics_aggregate_value(
            "timing_v2_kv_h2d_v_submit_ms", timing_v2_kv_h2d_v_submit_ms)
        self._record_layer_metrics_aggregate_value("timing_v2_kv_h2d_device_ms",
                                                   timing_v2_kv_h2d_device_ms)
        self._record_layer_metrics_aggregate_value(
            "timing_v2_gpu_index_scatter_submit_ms",
            timing_v2_gpu_index_scatter_submit_ms)
        self._record_layer_metrics_aggregate_value(
            "timing_v2_gpu_index_scatter_device_ms",
            timing_v2_gpu_index_scatter_device_ms)
        self._record_layer_metrics_aggregate_value(
            "timing_v2_old_page_gather_submit_ms",
            timing_v2_old_page_gather_submit_ms)
        self._record_layer_metrics_aggregate_value(
            "timing_v2_old_page_gather_device_ms",
            timing_v2_old_page_gather_device_ms)
        self._record_layer_metrics_aggregate_value(
            "timing_v2_reuse_rope_submit_ms", timing_v2_reuse_rope_submit_ms)
        self._record_layer_metrics_aggregate_value(
            "timing_v2_reuse_rope_device_ms", timing_v2_reuse_rope_device_ms)
        self._record_layer_metrics_aggregate_value(
            "timing_v2_reuse_rope_wait_ms", timing_v2_reuse_rope_wait_ms)
        self._record_layer_metrics_aggregate_value(
            "t_gpu_index_copy_sync_overhead_ms",
            t_gpu_index_copy_sync_overhead_ms)
        self._record_layer_metrics_aggregate_value("span_gpu_gather_submit_ms",
                                                   span_gpu_gather_submit_ms)
        self._record_layer_metrics_aggregate_value("span_gpu_gather_device_ms",
                                                   span_gpu_gather_device_ms)
        self._record_layer_metrics_aggregate_value("span_cpu_copy_submit_ms",
                                                   span_cpu_copy_submit_ms)
        self._record_layer_metrics_aggregate_value("span_cpu_copy_device_ms",
                                                   span_cpu_copy_device_ms)
        self._record_layer_metrics_aggregate_value("span_cpu_copy_span_count",
                                                   span_cpu_copy_span_count)
        self._record_layer_metrics_aggregate_value("span_cpu_copy_tokens",
                                                   span_cpu_copy_tokens)
        self._record_layer_metrics_aggregate_value("span_no_dense_copy",
                                                   span_no_dense_copy)
        self._record_layer_metrics_aggregate_value(
            "cpu_fallback_span_direct_used",
            cpu_fallback_span_direct_used)
        self._record_layer_metrics_aggregate_value(
            "cpu_fallback_span_direct_span_count",
            cpu_fallback_span_direct_span_count)
        self._record_layer_metrics_aggregate_value(
            "cpu_fallback_span_direct_tokens",
            cpu_fallback_span_direct_tokens)
        self._record_layer_metrics_aggregate_value(
            "cpu_fallback_span_direct_copy_submit_ms",
            cpu_fallback_span_direct_copy_submit_ms)
        self._record_layer_metrics_aggregate_value(
            "cpu_fallback_dense_enabled",
            1 if cpu_fallback_dense_enabled else 0)
        self._record_layer_metrics_aggregate_value("cpu_fallback_tokens",
                                                   cpu_fallback_tokens)
        self._record_layer_metrics_aggregate_value("cpu_fallback_ratio",
                                                   cpu_fallback_ratio)
        self._record_layer_metrics_aggregate_value(
            "cpu_fallback_subset_skipped",
            1 if cpu_fallback_subset_skipped else 0)
        self._record_layer_metrics_aggregate_value("token_zero_submit_ms",
                                                   token_zero_submit_ms)
        self._record_layer_metrics_aggregate_value("token_zero_skipped",
                                                   token_zero_skipped)
        self._record_layer_metrics_aggregate_value(
            "token_full_coverage_ready", token_full_coverage_ready)
        self._record_layer_metrics_aggregate_value(
            "token_full_coverage_gpu_rows", token_full_coverage_gpu_rows)
        self._record_layer_metrics_aggregate_value(
            "token_full_coverage_cpu_rows", token_full_coverage_cpu_rows)
        self._record_layer_metrics_aggregate_value(
            "token_full_coverage_total_rows", token_full_coverage_total_rows)
        self._record_layer_metrics_aggregate_value(
            "token_full_coverage_overlap_rows",
            token_full_coverage_overlap_rows)
        self._record_layer_metrics_aggregate_value(
            "token_full_coverage_missing_rows",
            token_full_coverage_missing_rows)
        self._record_layer_metrics_aggregate_value("compose_fixed_total_ms",
                                                   compose_fixed_total_ms)
        self._record_layer_metrics_aggregate_value("compose_layout_cache_hit",
                                                   compose_layout_cache_hit)
        self._record_layer_metrics_aggregate_value("compose_bridge_cache_hit",
                                                   compose_bridge_cache_hit)
        self._record_layer_metrics_aggregate_value(
            "compose_metrics_minimal_used",
            compose_metrics_minimal_used)
        self._record_layer_metrics_aggregate_value("compose_return_pack_ms",
                                                   compose_return_pack_ms)
        self._record_layer_metrics_aggregate_latest(
            "chunk_page_exec_granularity", str(chunk_page_exec_granularity))
        self._record_layer_metrics_aggregate_latest("cpu_fallback_mode",
                                                    str(cpu_fallback_mode))
        self._record_layer_metrics_aggregate_latest(
            "cpu_fallback_dense_reason", str(cpu_fallback_dense_reason))
        self._record_layer_metrics_aggregate_latest(
            "token_zero_skip_reason", str(token_zero_skip_reason))
        self._record_layer_metrics_aggregate_latest(
            "token_full_coverage_reason", str(token_full_coverage_reason))

    def _emit_layer_metrics_aggregate_event(self) -> None:
        if self.should_emit_layer_metrics_full():
            return
        if self._keepold_layer_metrics_agg_layers <= 0:
            return
        payload: dict[str, Any] = {
            "event": "selective_layer_metrics_aggregate",
            "keepold_phase": self._keepold_phase,
            "layer_count": int(self._keepold_layer_metrics_agg_layers),
            "keepold_layer_metrics_mode": self._keepold_layer_metrics_mode,
            "ts_unix": time.time(),
        }
        for key, total in self._keepold_layer_metrics_agg_sum.items():
            cnt = int(self._keepold_layer_metrics_agg_cnt.get(key, 0))
            if cnt <= 0:
                continue
            payload[key] = float(total) / float(cnt)
        payload.update(self._keepold_layer_metrics_agg_latest)
        self._append_metrics(payload)

    def get_keepold_layer_snapshot_cached(
        self,
        *,
        layer_id: int,
    ) -> Optional[_KeepOldLayerSnapshot]:
        bridge_layers = self._keepold_cached_bridge_layers
        if bridge_layers is not None:
            return bridge_layers.get(int(layer_id))
        bridge = self._get_keepold_bridge(create=False)
        self._keepold_cached_bridge = bridge
        self._keepold_cached_bridge_layers = (
            bridge.layers if bridge is not None else None)
        if self._keepold_cached_bridge_layers is None:
            return None
        return self._keepold_cached_bridge_layers.get(int(layer_id))

    def infer_paged_layout_meta_cached(
        self,
        *,
        layer_id: int,
        cache: torch.Tensor,
        num_kv_heads: int,
        head_size: int,
    ) -> Optional[tuple[int, int]]:
        layer_id_int = int(layer_id)
        if layer_id_int in self._keepold_paged_layout_meta_by_layer:
            return self._keepold_paged_layout_meta_by_layer[layer_id_int]
        layout_meta = self._infer_paged_cache_layout_code_page_size(
            cache=cache,
            num_kv_heads=num_kv_heads,
            head_size=head_size,
        )
        self._keepold_paged_layout_meta_by_layer[layer_id_int] = layout_meta
        return layout_meta

    def _diag_request_phase(self) -> str:
        if self._keepold_phase == _KEEPOLD_PHASE_RECORD:
            return "first"
        if self._keepold_phase == _KEEPOLD_PHASE_REUSE:
            return "second"
        return str(self._keepold_phase)

    def _diag_backend_label(self) -> str:
        if self._keepold_index_mode == _KEEPOLD_INDEX_MODE_CHUNK_PAGE:
            return "chunk_page"
        return "cpu"

    def _diag_exec_granularity_label(self) -> str:
        if not self._chunk_page_exec_enabled():
            return "cpu"
        if self._keepold_index_mode != _KEEPOLD_INDEX_MODE_CHUNK_PAGE:
            return "cpu"
        if (self._keepold_chunk_page_exec_granularity
                == _CHUNK_PAGE_EXEC_GRANULARITY_SPAN
                and self._keepold_chunk_page_span_exec_ready):
            return "span"
        return "token"

    def _diag_rank_meta(self) -> dict[str, int]:
        rank = int(os.environ.get("RANK", "0"))
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        try:
            if (torch.distributed.is_available()
                    and torch.distributed.is_initialized()):
                rank = int(torch.distributed.get_rank())
                world_size = int(torch.distributed.get_world_size())
                local_rank = int(os.environ.get("LOCAL_RANK", local_rank))
        except Exception:
            pass
        tp_size_cfg = int(
            max(1, int(getattr(self.config, "keepold_tensor_parallel_size", 1))))
        return {
            "rank": int(rank),
            "local_rank": int(local_rank),
            "world_size": int(world_size),
            "tensor_parallel_size": int(tp_size_cfg),
        }

    def _nvtx_diag_enabled(self) -> bool:
        return bool(getattr(self.config, "keepold_nvtx_diagnostic", False))

    def _nvtx_layer_id_enabled(self, layer_id: int) -> bool:
        total_layers = max(len(self._src_layer_kv), len(self._src_layer_kv_dynamic))
        if total_layers <= 0:
            return True
        lid = int(layer_id)
        return lid < 3 or lid >= max(0, total_layers - 3)

    def maybe_nvtx_range_push(
        self,
        name: str,
        *,
        layer_id: Optional[int] = None,
        include_layer_id: bool = False,
    ) -> bool:
        if not self._nvtx_diag_enabled():
            return False
        nvtx_obj = getattr(torch.cuda, "nvtx", None)
        if nvtx_obj is None:
            return False
        label = str(name)
        if include_layer_id and layer_id is not None and self._nvtx_layer_id_enabled(layer_id):
            label = f"{label}_L{int(layer_id)}"
        try:
            nvtx_obj.range_push(label)
            return True
        except Exception:
            return False

    def maybe_nvtx_range_pop(self, pushed: bool) -> None:
        if not pushed or not self._nvtx_diag_enabled():
            return
        nvtx_obj = getattr(torch.cuda, "nvtx", None)
        if nvtx_obj is None:
            return
        try:
            nvtx_obj.range_pop()
        except Exception:
            return

    def record_layer_compose_diag(
        self,
        *,
        layer_id: int,
        layer_name: str,
        fields: dict[str, Any],
    ) -> None:
        if not self._keepold_diag_layer_breakdown_enabled:
            return
        payload = dict(fields)
        payload.update({
            "layer_id": int(layer_id),
            "layer_name": str(layer_name),
            "request_phase": self._diag_request_phase(),
            "backend": self._diag_backend_label(),
            "exec_granularity": self._diag_exec_granularity_label(),
            "keepold_phase": str(self._keepold_phase),
            **self._diag_rank_meta(),
        })
        self._keepold_diag_layer_components[int(layer_id)] = payload

    def pop_layer_compose_diag(self, *, layer_id: int) -> Optional[dict[str, Any]]:
        return self._keepold_diag_layer_components.pop(int(layer_id), None)

    def _prepare_keepold_overlap_compose_eligibility(self) -> None:
        self._keepold_overlap_cpu_gpu_eligible = False
        self._keepold_overlap_cpu_gpu_reason = "disabled"
        if not bool(getattr(self.config, "keepold_overlap_cpu_gpu_compose", False)):
            return
        if self._keepold_phase != _KEEPOLD_PHASE_REUSE:
            self._keepold_overlap_cpu_gpu_reason = "phase_not_reuse"
            return
        if self._keepold_index_mode != _KEEPOLD_INDEX_MODE_CHUNK_PAGE:
            self._keepold_overlap_cpu_gpu_reason = "index_mode_not_chunk_page"
            return
        if self._keepold_chunk_page_exec_granularity != _CHUNK_PAGE_EXEC_GRANULARITY_TOKEN:
            self._keepold_overlap_cpu_gpu_reason = "exec_granularity_not_token"
            return
        if self._keepold_switch_reuse_idx_cpu is None:
            self._keepold_overlap_cpu_gpu_reason = "missing_switch_idx"
            return
        if self._keepold_cpu_reuse_idx_cpu is None:
            self._keepold_overlap_cpu_gpu_reason = "missing_cpu_idx"
            return
        switch_idx_cpu = self._keepold_switch_reuse_idx_cpu
        cpu_idx_cpu = self._keepold_cpu_reuse_idx_cpu
        if int(switch_idx_cpu.numel()) <= 0:
            self._keepold_overlap_cpu_gpu_reason = "empty_switch_idx"
            return
        if int(cpu_idx_cpu.numel()) <= 0:
            self._keepold_overlap_cpu_gpu_reason = "empty_cpu_idx"
            return
        total_reuse = int(self._reuse_positions_cpu.numel())
        if total_reuse <= 0:
            self._keepold_overlap_cpu_gpu_reason = "empty_reuse_positions"
            return
        max_cpu = int(cpu_idx_cpu.max().item()) if int(cpu_idx_cpu.numel()) > 0 else -1
        max_switch = (int(switch_idx_cpu.max().item())
                      if int(switch_idx_cpu.numel()) > 0 else -1)
        if max(max_cpu, max_switch) >= total_reuse:
            self._keepold_overlap_cpu_gpu_reason = "idx_out_of_range"
            return
        if int(cpu_idx_cpu.min().item()) < 0 or int(switch_idx_cpu.min().item()) < 0:
            self._keepold_overlap_cpu_gpu_reason = "idx_negative"
            return
        occupied = torch.zeros(total_reuse, dtype=torch.bool)
        occupied.index_fill_(0, cpu_idx_cpu, True)
        overlap_count = int(occupied.index_select(0, switch_idx_cpu).sum().item())
        if overlap_count > 0:
            self._keepold_overlap_cpu_gpu_reason = "cpu_gpu_rows_overlap"
            return
        self._keepold_overlap_cpu_gpu_eligible = True
        self._keepold_overlap_cpu_gpu_reason = "ok"

    def buffer_layer_pipeline_diag_event(self, event: dict[str, Any]) -> None:
        if not self._keepold_diag_layer_breakdown_enabled:
            return
        self._keepold_diag_layer_pipeline_events.append(dict(event))

    def pop_layer_pipeline_diag_events(self) -> list[dict[str, Any]]:
        if not self._keepold_diag_layer_breakdown_enabled:
            return []
        out = self._keepold_diag_layer_pipeline_events
        self._keepold_diag_layer_pipeline_events = []
        return out

    def _reset_chunk_page_dryrun_state(self) -> None:
        self._keepold_chunk_page_plan = None
        self._keepold_chunk_page_old_slots_cpu = torch.empty(0,
                                                             dtype=torch.long)
        self._keepold_chunk_page_dst_idx_cpu = torch.empty(0, dtype=torch.long)
        self._keepold_chunk_page_old_positions_cpu = torch.empty(
            0, dtype=torch.long)
        self._keepold_chunk_page_new_positions_cpu = torch.empty(
            0, dtype=torch.long)
        self._keepold_chunk_page_gpu_span_block_ids_cpu = torch.empty(
            0, dtype=torch.long)
        self._keepold_chunk_page_gpu_span_page_offsets_cpu = torch.empty(
            0, dtype=torch.long)
        self._keepold_chunk_page_gpu_span_dst_starts_cpu = torch.empty(
            0, dtype=torch.long)
        self._keepold_chunk_page_gpu_span_lengths_cpu = torch.empty(
            0, dtype=torch.long)
        self._keepold_chunk_page_gpu_span_old_abs_starts_cpu = torch.empty(
            0, dtype=torch.long)
        self._keepold_chunk_page_gpu_span_new_abs_starts_cpu = torch.empty(
            0, dtype=torch.long)
        self._keepold_chunk_page_cpu_span_src_starts_cpu = torch.empty(
            0, dtype=torch.long)
        self._keepold_chunk_page_cpu_span_dst_starts_cpu = torch.empty(
            0, dtype=torch.long)
        self._keepold_chunk_page_cpu_span_lengths_cpu = torch.empty(
            0, dtype=torch.long)
        self._keepold_chunk_page_span_cpu_copy_src_starts_cpu = torch.empty(
            0, dtype=torch.long)
        self._keepold_chunk_page_span_cpu_copy_dst_starts_cpu = torch.empty(
            0, dtype=torch.long)
        self._keepold_chunk_page_span_cpu_copy_lengths_cpu = torch.empty(
            0, dtype=torch.long)
        self._keepold_span_reuse_old_pos_for_rope_cpu = torch.empty(
            0, dtype=torch.long)
        self._keepold_chunk_page_span_exec_ready = False
        self._keepold_chunk_page_span_exec_reason = "inactive"
        self._keepold_chunk_page_span_plan_build_ms = 0.0
        self._keepold_chunk_page_gpu_span_count = 0
        self._keepold_chunk_page_cpu_span_count = 0
        self._keepold_chunk_page_span_coverage_checked = False
        self._keepold_chunk_page_span_coverage_reason = "inactive"
        self._keepold_chunk_page_span_full_reuse_tokens = 0
        self._keepold_chunk_page_span_prefix_uncovered_tokens = 0
        self._keepold_chunk_page_span_index_h2d_submit_ms = 0.0
        self._keepold_chunk_page_span_index_h2d_device_ms = float("nan")
        self._keepold_chunk_page_lookup_status = "inactive"
        self._keepold_chunk_page_lookup_reason = "inactive"
        self._keepold_chunk_page_target_reuse_tokens = 0
        self._keepold_chunk_page_gathered_target_tokens = 0
        self._keepold_chunk_page_cpu_fallback_tokens = 0
        self._keepold_chunk_page_skipped_out_of_range = 0
        self._keepold_chunk_page_skipped_missing_dst = 0
        self._keepold_chunk_page_skipped_duplicate = 0
        self._keepold_chunk_page_noncontiguous_span_count = 0
        self._keepold_chunk_page_old_slots_build_ms = 0.0
        self._keepold_chunk_page_exec_success = False
        self._keepold_chunk_page_exec_fallback_reason = "inactive"
        self._keepold_chunk_page_exec_tokens = 0
        self._keepold_chunk_page_exec_gpu_tokens = 0
        self._keepold_chunk_page_exec_cpu_tokens = 0
        self._keepold_chunk_page_lookup_hash_ms = 0.0
        self._keepold_chunk_page_lookup_index_ms = 0.0
        self._keepold_chunk_page_lookup_filter_ms = 0.0
        self._keepold_chunk_page_lookup_candidate_scan_ms = 0.0
        self._keepold_chunk_page_lookup_block_validity_check_ms = 0.0
        self._keepold_chunk_page_lookup_invalid_filter_ms = 0.0
        self._keepold_chunk_page_lookup_candidates_scanned = 0
        self._keepold_chunk_page_lookup_candidates_valid = 0
        self._keepold_chunk_page_lookup_candidates_invalid = 0
        self._keepold_chunk_page_metadata_only_candidate_count = 0
        self._keepold_chunk_page_metadata_only_filtered_count = 0
        self._keepold_chunk_page_metadata_only_false_hit_count = 0
        self._keepold_chunk_page_runtime_prelookup_source_id = ""
        self._keepold_chunk_page_runtime_prelookup_source_is_metadata_only = 0
        self._keepold_chunk_page_runtime_prelookup_source_dtype = ""
        self._keepold_chunk_page_runtime_prelookup_source_kv_layout = ""
        self._keepold_chunk_page_runtime_prelookup_source_block_size = 0
        self._keepold_chunk_page_lookup_materialize_ms = 0.0
        self._keepold_chunk_page_lookup_total_ms = 0.0
        self._keepold_chunk_page_lookup_cache_hit = 0
        self._keepold_chunk_page_lookup_cache_reason = "inactive"
        self._keepold_chunk_page_lookup_cache_saved_ms = 0.0

    def _chunk_page_metrics_defaults(self) -> dict[str, Any]:
        return {
            "record_ms": 0.0,
            "lookup_ms": 0.0,
            "old_slots_build_ms": 0.0,
            "old_slots_total_ms": 0.0,
            "old_slots_map_build_ms": 0.0,
            "old_slots_span_iter_ms": 0.0,
            "old_slots_tensor_materialize_ms": 0.0,
            "old_slots_position_build_ms": 0.0,
            "old_slots_check_ms": 0.0,
            "old_slots_fastpath_used": 0,
            "old_slots_fastpath_reason": "inactive",
            "old_slots_span_count": 0,
            "old_slots_token_count": 0,
            "indexed_chunks": 0,
            "indexed_page_spans": 0,
            "indexed_full_page_spans": 0,
            "indexed_partial_page_spans": 0,
            "indexed_full_page_tokens": 0,
            "indexed_partial_tokens": 0,
            "hit_chunks": 0,
            "miss_chunks": 0,
            "candidate_count": 0,
            "same_source_hit_count": 0,
            "global_cross_source_hit_count": 0,
            "gpu_gather_full_tokens": 0,
            "gpu_gather_partial_tokens": 0,
            "target_reuse_tokens": 0,
            "gathered_target_tokens": 0,
            "cpu_fallback_tokens": 0,
            "skipped_out_of_range": 0,
            "skipped_missing_dst": 0,
            "skipped_duplicate": 0,
            "noncontiguous_span_count": 0,
            "lru_evictions": 0,
            "lru_capacity_evictions": 0,
            "lru_bucket_evictions": 0,
            "capacity_full_events": 0,
            "chunk_page_exec_tokens": 0,
            "chunk_page_exec_gpu_tokens": 0,
            "chunk_page_exec_cpu_tokens": 0,
            "chunk_page_exec_old_slots_build_ms": 0.0,
            "chunk_page_span_plan_build_ms": 0.0,
            "span_plan_build_ms": 0.0,
            "span_plan_total_ms": 0.0,
            "span_plan_make_pos_map_ms": 0.0,
            "span_plan_iter_gpu_spans_ms": 0.0,
            "span_plan_build_cpu_spans_ms": 0.0,
            "span_plan_merge_gpu_spans_ms": 0.0,
            "span_plan_tensor_materialize_ms": 0.0,
            "span_plan_debug_check_ms": 0.0,
            "span_plan_other_ms": 0.0,
            "span_plan_gpu_span_count": 0,
            "span_plan_cpu_span_count": 0,
            "span_plan_gpu_span_count_before_merge": 0,
            "span_plan_gpu_span_count_after_merge": 0,
            "span_plan_cpu_span_count_before_merge": 0,
            "span_plan_cpu_span_count_after_merge": 0,
            "span_plan_gpu_tokens": 0,
            "span_plan_cpu_tokens": 0,
            "span_plan_target_reuse_tokens": 0,
            "span_plan_full_reuse_tokens": 0,
            "span_plan_prefix_uncovered_tokens": 0,
            "span_plan_requires_token_fallback": 0,
            "span_plan_fallback_reason": "inactive",
            "chunk_page_exec_granularity": _CHUNK_PAGE_EXEC_GRANULARITY_TOKEN,
            "span_exec_ready": 0,
            "span_exec_reason": "inactive",
            "gpu_span_count": 0,
            "cpu_span_count": 0,
            "span_index_h2d_submit_ms": 0.0,
            "span_index_h2d_device_ms": float("nan"),
            "chunk_page_exec_fallback_reason": "inactive",
            "chunk_page_exec_success": 0,
            "chunk_page_cpu_fallback_tokens": 0,
            "gpu_index_prepare_total_ms": 0.0,
            "gpu_index_prepare_old_slots_ms": 0.0,
            "gpu_index_prepare_dst_idx_ms": 0.0,
            "gpu_index_prepare_positions_ms": 0.0,
            "gpu_index_prepare_h2d_ms": 0.0,
            "gpu_index_prepare_cache_hit": 0,
            "gpu_index_prepare_cache_reason": "inactive",
            "timing_v2_chunk_page_record_ms": 0.0,
            "timing_v2_chunk_page_lookup_ms": 0.0,
            "timing_v2_chunk_page_old_slots_build_ms": 0.0,
            "timing_v2_chunk_page_exec_partition_ms": 0.0,
            "chunk_page_partition_build_ms": 0.0,
            "chunk_page_mixed_plan_materialize_ms": 0.0,
            "chunk_page_cpu_fallback_rows_build_ms": 0.0,
            "chunk_page_old_page_plan_build_ms": 0.0,
            "chunk_page_lookup_hash_ms": 0.0,
            "chunk_page_lookup_index_ms": 0.0,
            "chunk_page_lookup_filter_ms": 0.0,
            "chunk_page_lookup_candidate_scan_ms": 0.0,
            "chunk_page_lookup_block_validity_check_ms": 0.0,
            "chunk_page_lookup_invalid_filter_ms": 0.0,
            "chunk_page_lookup_candidates_scanned": 0,
            "chunk_page_lookup_candidates_valid": 0,
            "chunk_page_lookup_candidates_invalid": 0,
            "chunk_page_metadata_only_candidate_count": 0,
            "chunk_page_metadata_only_filtered_count": 0,
            "chunk_page_metadata_only_false_hit_count": 0,
            "runtime_prelookup_source_id": "",
            "runtime_prelookup_source_is_metadata_only": 0,
            "runtime_prelookup_source_dtype": "",
            "runtime_prelookup_source_kv_layout": "",
            "runtime_prelookup_source_block_size": 0,
            "chunk_page_lookup_materialize_ms": 0.0,
            "chunk_page_lookup_total_ms": 0.0,
            "chunk_page_lookup_cache_hit": 0,
            "chunk_page_lookup_cache_reason": "inactive",
            "chunk_page_lookup_cache_saved_ms": 0.0,
            "chunk_page_lookup_calls_in_runtime": 0,
            "chunk_page_lookup_cache_mode": "diagnostic_only_single_call",
            "chunk_page_global_lookup_enabled": 0,
            "chunk_page_no_replace_existing_hash": 0,
            "chunk_page_block_invalidation_enabled": 0,
            "chunk_page_index_entries": 0,
            "chunk_page_index_chunk_hashes": 0,
            "chunk_page_current_entry_count": 0,
            "chunk_page_current_chunk_hash_count": 0,
            "chunk_page_max_chunks_raw_config": 0,
            "chunk_page_max_entries_per_hash_config": 0,
            "chunk_page_index_sources": 0,
            "entries_per_hash_max": 0,
            "entries_by_block_size": 0,
            "blocks_by_entry_size": 0,
            "chunk_page_register_attempts": 0,
            "chunk_page_register_success": 0,
            "chunk_page_register_skipped_duplicate_hash": 0,
            "chunk_page_register_skipped_bucket_full": 0,
            "chunk_page_lru_unit": _KEEPOLD_LRU_UNIT_ENTRY,
            "chunk_page_lru_evict_batch_chunks": 0,
            "chunk_page_lru_low_watermark_chunks": -1,
            "chunk_page_lru_chunk_hash_capacity_evictions": 0,
            "chunk_page_lru_evicted_chunk_hashes": 0,
            "chunk_page_lru_chunk_hash_evicted_entries_sum": 0,
            "chunk_page_lru_chunk_hash_evicted_entries_max_per_hash": 0,
            "chunk_page_block_invalidations": 0,
            "chunk_page_block_invalidated_entries": 0,
            "chunk_page_block_invalidated_blocks": 0,
            "chunk_page_metadata_only_entries_current": 0,
            "chunk_page_real_entries_current": 0,
            "post_register_ms": 0.0,
            "post_register_candidate_chunks": 0,
            "post_register_success": 0,
            "post_register_skipped_existing_hash": 0,
            "post_register_skipped_bucket_full": 0,
            "update_entry_on_hit": 1,
            "register_hit_chunks_after_sample": 0,
        }

    def _chunk_page_index_stats_payload(
        self,
        stats: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        stats = stats or {}
        return {
            "chunk_page_global_lookup_enabled":
            1 if bool(self.config.keepold_chunk_page_global_lookup) else 0,
            "chunk_page_no_replace_existing_hash":
            1 if bool(self.config.keepold_chunk_page_no_replace_existing_hash)
            else 0,
            "chunk_page_block_invalidation_enabled":
            1 if bool(self.config.keepold_chunk_page_block_invalidation) else 0,
            "chunk_page_index_entries": int(stats.get("entries", 0)),
            "chunk_page_index_chunk_hashes": int(
                stats.get("chunk_hash_count", 0)),
            "chunk_page_current_entry_count": int(
                stats.get("current_entry_count", stats.get("entries", 0))),
            "chunk_page_current_chunk_hash_count": int(
                stats.get("current_chunk_hash_count",
                          stats.get("chunk_hash_count", 0))),
            "chunk_page_max_chunks_raw_config": int(
                stats.get("max_chunks_raw_config", 0)),
            "chunk_page_max_entries_per_hash_config": int(
                stats.get("max_entries_per_hash_config", 0)),
            "chunk_page_index_sources": int(stats.get("sources", 0)),
            "entries_per_hash_max": int(stats.get("entries_per_hash_max", 0)),
            "entries_by_block_size": int(stats.get("entries_by_block_size", 0)),
            "blocks_by_entry_size": int(stats.get("blocks_by_entry_size", 0)),
            "lru_evictions": int(stats.get("lru_evictions", 0)),
            "lru_capacity_evictions": int(
                stats.get("lru_capacity_evictions", 0)),
            "lru_bucket_evictions": int(
                stats.get("lru_bucket_evictions", 0)),
            "chunk_page_lru_chunk_hash_capacity_evictions": int(
                stats.get("lru_chunk_hash_capacity_evictions", 0)),
            "chunk_page_lru_evicted_chunk_hashes": int(
                stats.get("lru_evicted_chunk_hashes", 0)),
            "chunk_page_lru_chunk_hash_evicted_entries_sum": int(
                stats.get("lru_chunk_hash_evicted_entries_sum", 0)),
            "chunk_page_lru_chunk_hash_evicted_entries_max_per_hash": int(
                stats.get("lru_chunk_hash_evicted_entries_max_per_hash", 0)),
            "capacity_full_events": int(stats.get("capacity_full_events", 0)),
            "chunk_page_register_attempts": int(
                stats.get("register_attempts", 0)),
            "chunk_page_register_success": int(stats.get("register_success", 0)),
            "chunk_page_register_skipped_duplicate_hash": int(
                stats.get("register_skipped_duplicate_hash", 0)),
            "chunk_page_register_skipped_bucket_full": int(
                stats.get("register_skipped_bucket_full", 0)),
            "chunk_page_block_invalidations": int(
                stats.get("block_invalidations", 0)),
            "chunk_page_block_invalidated_entries": int(
                stats.get("block_invalidated_entries", 0)),
            "chunk_page_block_invalidated_blocks": int(
                stats.get("block_invalidated_blocks", 0)),
            "chunk_page_metadata_only_entries_current": int(
                stats.get("metadata_only_entries_current", 0)),
            "chunk_page_real_entries_current": int(
                stats.get("real_entries_current", 0)),
            "chunk_page_lru_unit": str(
                stats.get("lru_unit", _KEEPOLD_LRU_UNIT_ENTRY)),
            "chunk_page_lru_evict_batch_chunks": int(
                stats.get("lru_evict_batch_chunks", 0)),
            "chunk_page_lru_low_watermark_chunks": int(
                stats.get("lru_low_watermark_chunks", -1)),
            "update_entry_on_hit":
            1 if bool(self.config.keepold_chunk_page_update_entry_on_hit) else 0,
            "register_hit_chunks_after_sample":
            1 if bool(
                self.config.keepold_chunk_page_register_hit_chunks_after_sample)
            else 0,
        }

    def _timing_mode(self) -> str:
        mode = str(getattr(self.config, "keepold_timing_mode",
                           _KEEPOLD_TIMING_MODE_LIGHT)).strip().lower()
        if mode not in (_KEEPOLD_TIMING_MODE_LIGHT,
                        _KEEPOLD_TIMING_MODE_CUDA_EVENT):
            return _KEEPOLD_TIMING_MODE_LIGHT
        return mode

    def _timing_cuda_event_enabled(self) -> bool:
        return self._timing_mode() == _KEEPOLD_TIMING_MODE_CUDA_EVENT

    def _accumulate_index_tensor_h2d_timing(
        self,
        *,
        submit_ms: float,
        device_ms: float,
    ) -> None:
        self._keepold_timing_v2_index_tensor_h2d_submit_ms += float(submit_ms)
        if math.isnan(float(device_ms)):
            return
        if math.isnan(self._keepold_timing_v2_index_tensor_h2d_device_ms):
            self._keepold_timing_v2_index_tensor_h2d_device_ms = float(device_ms)
        else:
            self._keepold_timing_v2_index_tensor_h2d_device_ms += float(device_ms)

    def _chunk_page_emit_metrics(
        self,
        *,
        phase: str,
        status: str,
        reason: str,
        fields: Optional[dict[str, Any]] = None,
    ) -> None:
        phase_norm = str(phase)
        chunk_page_exec_enabled = bool(
            self.config.keepold_chunk_page_execution_enabled
            and self._keepold_index_mode == _KEEPOLD_INDEX_MODE_CHUNK_PAGE
            and phase_norm == _KEEPOLD_PHASE_REUSE)
        payload = self._chunk_page_metrics_defaults()
        payload.update(self._chunk_page_index_stats_payload())
        if fields:
            payload.update(fields)
        payload["chunk_page_lookup_calls_in_runtime"] = int(
            self._keepold_chunk_page_lookup_calls_in_runtime)
        if "chunk_page_lookup_cache_mode" not in payload or not str(
                payload["chunk_page_lookup_cache_mode"]).strip():
            payload["chunk_page_lookup_cache_mode"] = (
                "runtime_reuse"
                if int(self._keepold_chunk_page_lookup_calls_in_runtime) > 1
                else "diagnostic_only_single_call")
        payload.update({
            "event": "selective_chunk_page_dry_run",
            "phase": phase_norm,
            "status": str(status),
            "reason": str(reason),
            "keepold_index_mode": self._keepold_index_mode,
            "keepold_switch_scope": self._keepold_switch_scope,
            "chunk_page_exec_granularity":
            self._keepold_chunk_page_exec_granularity,
            "keepold_bridge_key": self._keepold_bridge_key,
            "clear_each_sample": bool(
                self.config.keepold_chunk_page_clear_each_sample),
            "chunk_page_global_lookup_enabled":
            1 if bool(self.config.keepold_chunk_page_global_lookup) else 0,
            "chunk_page_no_replace_existing_hash":
            1 if bool(self.config.keepold_chunk_page_no_replace_existing_hash)
            else 0,
            "chunk_page_block_invalidation_enabled":
            1 if bool(self.config.keepold_chunk_page_block_invalidation) else 0,
            "update_entry_on_hit":
            1 if bool(self.config.keepold_chunk_page_update_entry_on_hit) else 0,
            "register_hit_chunks_after_sample":
            1 if bool(
                self.config.keepold_chunk_page_register_hit_chunks_after_sample)
            else 0,
            "chunk_page_dry_run_only": 0 if chunk_page_exec_enabled else 1,
            "chunk_page_execution_enabled": 1 if chunk_page_exec_enabled else 0,
            "rank_local_only": True,
            "ts_unix": time.time(),
        })
        self._append_metrics(payload)

    def _chunk_page_dryrun_allowed(self) -> tuple[bool, str]:
        if self._keepold_index_mode != _KEEPOLD_INDEX_MODE_CHUNK_PAGE:
            return False, "index_mode_not_chunk_page"
        if self._keepold_switch_scope != _KEEPOLD_SWITCH_SCOPE_ALL_NON_IMPORTANT:
            return False, "unsupported_switch_scope"
        if not self._keepold_chunk_page_source_id:
            return False, "missing_source_id"
        if not self.config.keepold_prompt_token_ids:
            return False, "missing_prompt_token_ids"
        if not self.config.keepold_chunk_infos:
            return False, "missing_chunk_infos"
        return True, "ok"

    def _chunk_page_exec_enabled(self) -> bool:
        return bool(
            self.config.keepold_chunk_page_execution_enabled
            and self._keepold_index_mode == _KEEPOLD_INDEX_MODE_CHUNK_PAGE
            and self._keepold_phase == _KEEPOLD_PHASE_REUSE)

    def _apply_chunk_page_exec_partition(self, *, device: torch.device) -> None:
        if not self._chunk_page_exec_enabled():
            return
        total_reuse = int(self._reuse_positions_cpu.numel())
        if total_reuse <= 0:
            return
        t_part0 = time.perf_counter()
        self._keepold_chunk_page_partition_build_ms = 0.0
        self._keepold_chunk_page_mixed_plan_materialize_ms = 0.0
        self._keepold_chunk_page_cpu_fallback_rows_build_ms = 0.0
        self._keepold_chunk_page_old_page_plan_build_ms = 0.0

        all_cpu_idx = torch.arange(total_reuse, dtype=torch.long)
        timing_cuda_event = bool(self._timing_cuda_event_enabled()
                                 and device.type == "cuda")

        def _fallback_all_cpu(reason: str) -> None:
            idx_start_event: Optional[torch.cuda.Event] = None
            idx_end_event: Optional[torch.cuda.Event] = None
            if timing_cuda_event:
                idx_start_event = torch.cuda.Event(enable_timing=True)
                idx_end_event = torch.cuda.Event(enable_timing=True)
                idx_start_event.record(torch.cuda.current_stream(device))
            t_idx0 = time.perf_counter()
            t_part1 = time.perf_counter()
            self._keepold_timing_v2_chunk_page_exec_partition_ms = float(
                (t_part1 - t_part0) * 1000.0)
            self._keepold_chunk_page_old_page_plan_build_ms = 0.0
            self._keepold_chunk_page_cpu_fallback_rows_build_ms = 0.0
            self._keepold_switch_reuse_idx_cpu = torch.empty(0, dtype=torch.long)
            self._keepold_switch_prompt_positions_cpu = torch.empty(
                0, dtype=torch.long)
            self._keepold_cpu_reuse_idx_cpu = all_cpu_idx
            self._keepold_switch_reuse_idx_dev = torch.empty(
                0, dtype=torch.long, device=device)
            self._keepold_switch_prompt_positions_dev = torch.empty(
                0, dtype=torch.long, device=device)
            self._keepold_cpu_reuse_idx_dev = all_cpu_idx.to(
                device=device, non_blocking=False)
            t_idx1 = time.perf_counter()
            idx_device_ms = float("nan")
            if idx_end_event is not None:
                idx_end_event.record(torch.cuda.current_stream(device))
                torch.cuda.synchronize(device)
                idx_device_ms = float(idx_start_event.elapsed_time(idx_end_event))
            self._accumulate_index_tensor_h2d_timing(
                submit_ms=(t_idx1 - t_idx0) * 1000.0,
                device_ms=idx_device_ms,
            )
            self._keepold_chunk_page_mixed_plan_materialize_ms = float(
                (t_idx1 - t_idx0) * 1000.0)
            self._keepold_chunk_page_partition_build_ms = float(
                self._keepold_timing_v2_chunk_page_exec_partition_ms)
            self._keepold_chunk_page_exec_success = False
            self._keepold_chunk_page_exec_fallback_reason = str(reason)
            self._keepold_chunk_page_exec_gpu_tokens = 0
            self._keepold_chunk_page_exec_cpu_tokens = int(total_reuse)
            self._append_metrics({
                "event": "selective_chunk_page_exec_partition",
                "keepold_phase": self._keepold_phase,
                "status": "fallback_all_cpu",
                "reason": str(reason),
                "chunk_page_execution_enabled": 1,
                "chunk_page_exec_success": 0,
                "chunk_page_exec_tokens": int(
                    self._keepold_chunk_page_exec_tokens),
                "chunk_page_exec_gpu_tokens": 0,
                "chunk_page_exec_cpu_tokens": int(total_reuse),
                "chunk_page_cpu_fallback_tokens": int(
                    self._keepold_chunk_page_cpu_fallback_tokens),
                "timing_v2_chunk_page_exec_partition_ms": float(
                    self._keepold_timing_v2_chunk_page_exec_partition_ms),
                "chunk_page_partition_build_ms": float(
                    self._keepold_chunk_page_partition_build_ms),
                "chunk_page_mixed_plan_materialize_ms": float(
                    self._keepold_chunk_page_mixed_plan_materialize_ms),
                "chunk_page_cpu_fallback_rows_build_ms": float(
                    self._keepold_chunk_page_cpu_fallback_rows_build_ms),
                "chunk_page_old_page_plan_build_ms": float(
                    self._keepold_chunk_page_old_page_plan_build_ms),
                "ts_unix": time.time(),
            })

        if self._keepold_chunk_page_lookup_status != "ok":
            _fallback_all_cpu(
                f"lookup_{self._keepold_chunk_page_lookup_status}:"
                f"{self._keepold_chunk_page_lookup_reason}")
            return

        if (self._keepold_chunk_page_exec_granularity
                == _CHUNK_PAGE_EXEC_GRANULARITY_SPAN
                and self._keepold_chunk_page_span_exec_ready):
            gpu_dst = self._keepold_chunk_page_gpu_span_dst_starts_cpu
            gpu_len = self._keepold_chunk_page_gpu_span_lengths_cpu
            cpu_dst = self._keepold_chunk_page_cpu_span_dst_starts_cpu
            cpu_len = self._keepold_chunk_page_cpu_span_lengths_cpu
            if int(gpu_dst.numel()) != int(gpu_len.numel()):
                _fallback_all_cpu("span_gpu_dst_len_mismatch")
                return
            if int(cpu_dst.numel()) != int(cpu_len.numel()):
                _fallback_all_cpu("span_cpu_dst_len_mismatch")
                return
            cover_gpu = torch.zeros(total_reuse, dtype=torch.bool)
            cover_cpu = torch.zeros(total_reuse, dtype=torch.bool)
            for i in range(int(gpu_len.numel())):
                st = int(gpu_dst[i].item())
                ln = int(gpu_len[i].item())
                if ln <= 0:
                    continue
                ed = st + ln
                if st < 0 or ed > total_reuse:
                    _fallback_all_cpu("span_gpu_dst_out_of_range")
                    return
                cover_gpu[st:ed] = True
            for i in range(int(cpu_len.numel())):
                st = int(cpu_dst[i].item())
                ln = int(cpu_len[i].item())
                if ln <= 0:
                    continue
                ed = st + ln
                if st < 0 or ed > total_reuse:
                    _fallback_all_cpu("span_cpu_dst_out_of_range")
                    return
                cover_cpu[st:ed] = True
            if bool((cover_gpu & cover_cpu).any().item()):
                _fallback_all_cpu("span_gpu_cpu_overlap")
                return
            self._keepold_switch_reuse_idx_cpu = torch.empty(0, dtype=torch.long)
            self._keepold_switch_prompt_positions_cpu = torch.empty(
                0, dtype=torch.long)
            self._keepold_cpu_reuse_idx_cpu = torch.empty(0, dtype=torch.long)
            self._keepold_switch_reuse_idx_dev = torch.empty(
                0, dtype=torch.long, device=device)
            self._keepold_switch_prompt_positions_dev = torch.empty(
                0, dtype=torch.long, device=device)
            self._keepold_cpu_reuse_idx_dev = torch.empty(
                0, dtype=torch.long, device=device)
            t_part1 = time.perf_counter()
            self._keepold_timing_v2_chunk_page_exec_partition_ms = float(
                (t_part1 - t_part0) * 1000.0)
            self._keepold_chunk_page_old_page_plan_build_ms = float(
                (t_part1 - t_part0) * 1000.0)
            self._keepold_chunk_page_cpu_fallback_rows_build_ms = 0.0
            self._keepold_chunk_page_mixed_plan_materialize_ms = 0.0
            self._keepold_chunk_page_partition_build_ms = float(
                self._keepold_timing_v2_chunk_page_exec_partition_ms)
            self._append_metrics({
                "event": "selective_chunk_page_exec_partition",
                "keepold_phase": self._keepold_phase,
                "status": "ok_span",
                "reason": self._keepold_chunk_page_exec_fallback_reason,
                "chunk_page_execution_enabled": 1,
                "chunk_page_exec_granularity":
                self._keepold_chunk_page_exec_granularity,
                "chunk_page_exec_success": 1 if self._keepold_chunk_page_exec_success else 0,
                "chunk_page_exec_tokens": int(
                    self._keepold_chunk_page_exec_tokens),
                "chunk_page_exec_gpu_tokens": int(
                    self._keepold_chunk_page_exec_gpu_tokens),
                "chunk_page_exec_cpu_tokens": int(
                    self._keepold_chunk_page_exec_cpu_tokens),
                "gpu_span_count": int(self._keepold_chunk_page_gpu_span_count),
                "cpu_span_count": int(self._keepold_chunk_page_cpu_span_count),
                "chunk_page_cpu_fallback_tokens": int(
                    self._keepold_chunk_page_cpu_fallback_tokens),
                "timing_v2_chunk_page_exec_partition_ms": float(
                    self._keepold_timing_v2_chunk_page_exec_partition_ms),
                "chunk_page_partition_build_ms": float(
                    self._keepold_chunk_page_partition_build_ms),
                "chunk_page_mixed_plan_materialize_ms": float(
                    self._keepold_chunk_page_mixed_plan_materialize_ms),
                "chunk_page_cpu_fallback_rows_build_ms": float(
                    self._keepold_chunk_page_cpu_fallback_rows_build_ms),
                "chunk_page_old_page_plan_build_ms": float(
                    self._keepold_chunk_page_old_page_plan_build_ms),
                "ts_unix": time.time(),
            })
            return

        t_old_plan0 = time.perf_counter()
        dst_idx_cpu = self._keepold_chunk_page_dst_idx_cpu.to(
            dtype=torch.long, device="cpu").contiguous()
        old_slots_cpu = self._keepold_chunk_page_old_slots_cpu.to(
            dtype=torch.long, device="cpu").contiguous()
        old_pos_cpu = self._keepold_chunk_page_old_positions_cpu.to(
            dtype=torch.long, device="cpu").contiguous()
        new_pos_cpu = self._keepold_chunk_page_new_positions_cpu.to(
            dtype=torch.long, device="cpu").contiguous()

        if int(old_slots_cpu.numel()) != int(dst_idx_cpu.numel()):
            _fallback_all_cpu("old_slots_dst_len_mismatch")
            return
        if int(old_pos_cpu.numel()) != int(dst_idx_cpu.numel()):
            _fallback_all_cpu("missing_old_positions")
            return
        if int(new_pos_cpu.numel()) != int(dst_idx_cpu.numel()):
            _fallback_all_cpu("missing_new_positions")
            return
        if int(dst_idx_cpu.numel()) > 0:
            min_idx = int(dst_idx_cpu.min().item())
            max_idx = int(dst_idx_cpu.max().item())
            if min_idx < 0 or max_idx >= total_reuse:
                _fallback_all_cpu("dst_idx_out_of_range")
                return
            if int(old_slots_cpu.min().item()) < 0:
                _fallback_all_cpu("old_slots_negative")
                return
            if int(torch.unique(dst_idx_cpu).numel()) != int(dst_idx_cpu.numel()):
                _fallback_all_cpu("dst_idx_duplicate")
                return
            expected_new = self._reuse_positions_cpu.index_select(0, dst_idx_cpu)
            if not torch.equal(expected_new, new_pos_cpu):
                _fallback_all_cpu("new_positions_mismatch")
                return
        t_old_plan1 = time.perf_counter()
        self._keepold_chunk_page_old_page_plan_build_ms = float(
            (t_old_plan1 - t_old_plan0) * 1000.0)

        t_cpu_rows0 = time.perf_counter()
        keep_mask = torch.ones(total_reuse, dtype=torch.bool)
        if int(dst_idx_cpu.numel()) > 0:
            keep_mask[dst_idx_cpu] = False
        cpu_reuse_idx_cpu = all_cpu_idx[keep_mask].contiguous()
        t_cpu_rows1 = time.perf_counter()
        self._keepold_chunk_page_cpu_fallback_rows_build_ms = float(
            (t_cpu_rows1 - t_cpu_rows0) * 1000.0)

        t_mix0 = time.perf_counter()
        self._keepold_switch_reuse_idx_cpu = dst_idx_cpu
        self._keepold_switch_prompt_positions_cpu = old_pos_cpu
        self._keepold_cpu_reuse_idx_cpu = cpu_reuse_idx_cpu
        idx_start_event: Optional[torch.cuda.Event] = None
        idx_end_event: Optional[torch.cuda.Event] = None
        if timing_cuda_event:
            idx_start_event = torch.cuda.Event(enable_timing=True)
            idx_end_event = torch.cuda.Event(enable_timing=True)
            idx_start_event.record(torch.cuda.current_stream(device))
        t_idx0 = time.perf_counter()
        self._keepold_switch_reuse_idx_dev = dst_idx_cpu.to(
            device=device, non_blocking=False)
        self._keepold_switch_prompt_positions_dev = old_pos_cpu.to(
            device=device, non_blocking=False)
        self._keepold_cpu_reuse_idx_dev = cpu_reuse_idx_cpu.to(
            device=device, non_blocking=False)
        t_idx1 = time.perf_counter()
        idx_device_ms = float("nan")
        if idx_end_event is not None:
            idx_end_event.record(torch.cuda.current_stream(device))
            torch.cuda.synchronize(device)
            idx_device_ms = float(idx_start_event.elapsed_time(idx_end_event))
        self._accumulate_index_tensor_h2d_timing(
            submit_ms=(t_idx1 - t_idx0) * 1000.0,
            device_ms=idx_device_ms,
        )
        t_mix1 = time.perf_counter()
        self._keepold_chunk_page_mixed_plan_materialize_ms = float(
            (t_mix1 - t_mix0) * 1000.0)

        self._keepold_chunk_page_exec_success = True
        self._keepold_chunk_page_exec_fallback_reason = (
            "partial_cpu_fallback"
            if self._keepold_chunk_page_cpu_fallback_tokens > 0 else "ok")
        self._keepold_chunk_page_exec_gpu_tokens = int(dst_idx_cpu.numel())
        self._keepold_chunk_page_exec_cpu_tokens = int(cpu_reuse_idx_cpu.numel())
        t_part1 = time.perf_counter()
        self._keepold_timing_v2_chunk_page_exec_partition_ms = float(
            (t_part1 - t_part0) * 1000.0)
        self._keepold_chunk_page_partition_build_ms = float(
            self._keepold_chunk_page_old_page_plan_build_ms
            + self._keepold_chunk_page_cpu_fallback_rows_build_ms
            + self._keepold_chunk_page_mixed_plan_materialize_ms)
        self._append_metrics({
            "event": "selective_chunk_page_exec_partition",
            "keepold_phase": self._keepold_phase,
            "status": "ok",
            "reason": self._keepold_chunk_page_exec_fallback_reason,
            "chunk_page_execution_enabled": 1,
            "chunk_page_exec_success": 1,
            "chunk_page_exec_tokens": int(self._keepold_chunk_page_exec_tokens),
            "chunk_page_exec_gpu_tokens": int(dst_idx_cpu.numel()),
            "chunk_page_exec_cpu_tokens": int(cpu_reuse_idx_cpu.numel()),
            "chunk_page_cpu_fallback_tokens": int(
                self._keepold_chunk_page_cpu_fallback_tokens),
            "timing_v2_chunk_page_exec_partition_ms": float(
                self._keepold_timing_v2_chunk_page_exec_partition_ms),
            "chunk_page_partition_build_ms": float(
                self._keepold_chunk_page_partition_build_ms),
            "chunk_page_mixed_plan_materialize_ms": float(
                self._keepold_chunk_page_mixed_plan_materialize_ms),
            "chunk_page_cpu_fallback_rows_build_ms": float(
                self._keepold_chunk_page_cpu_fallback_rows_build_ms),
            "chunk_page_old_page_plan_build_ms": float(
                self._keepold_chunk_page_old_page_plan_build_ms),
            "ts_unix": time.time(),
        })

    def _prepare_chunk_page_lookup_dry_run(self) -> None:
        self._keepold_chunk_page_lookup_calls_in_runtime += 1
        self._reset_chunk_page_dryrun_state()
        allowed, reason = self._chunk_page_dryrun_allowed()
        if not allowed:
            self._keepold_chunk_page_lookup_status = "skipped"
            self._keepold_chunk_page_lookup_reason = str(reason)
            self._keepold_chunk_page_exec_fallback_reason = str(reason)
            self._chunk_page_emit_metrics(
                phase=self._keepold_phase,
                status="skipped",
                reason=reason,
            )
            return
        if self._keepold_phase != _KEEPOLD_PHASE_REUSE:
            self._keepold_chunk_page_lookup_status = "skipped"
            self._keepold_chunk_page_lookup_reason = "phase_not_reuse"
            return

        assert self._keepold_chunk_page_source_id is not None
        source_id = self._keepold_chunk_page_source_id
        global_lookup = bool(self.config.keepold_chunk_page_global_lookup)
        index = get_chunk_page_index(
            max_chunks=self.config.keepold_chunk_page_max_chunks,
            max_entries_per_hash=self.config.keepold_chunk_page_max_entries_per_hash,
            global_lookup_enabled=global_lookup,
            no_replace_existing_hash=bool(
                self.config.keepold_chunk_page_no_replace_existing_hash),
            block_invalidation_enabled=bool(
                self.config.keepold_chunk_page_block_invalidation),
            lru_unit=str(self.config.keepold_chunk_page_lru_unit),
            lru_evict_batch_chunks=int(
                self.config.keepold_chunk_page_lru_evict_batch_chunks),
            lru_low_watermark_chunks=int(
                self.config.keepold_chunk_page_lru_low_watermark_chunks),
        )
        source = index.sources.get(source_id)
        if (source is None or not source.valid
                or not bool(getattr(source, "valid_for_lookup_source", True))):
            if not global_lookup:
                self._keepold_chunk_page_lookup_status = "miss"
                self._keepold_chunk_page_lookup_reason = "missing_source"
                self._keepold_chunk_page_exec_fallback_reason = "missing_source"
                self._chunk_page_emit_metrics(
                    phase=self._keepold_phase,
                    status="miss",
                    reason="missing_source",
                )
                return
            source = None
            for candidate_source in index.sources.values():
                if (candidate_source is not None and candidate_source.valid
                        and bool(getattr(candidate_source,
                                         "valid_for_lookup_source", True))
                        and not bool(getattr(candidate_source,
                                             "is_metadata_only_source", False))):
                    source = candidate_source
                    break
            if source is None:
                for candidate_source in index.sources.values():
                    if (candidate_source is not None and candidate_source.valid
                            and bool(getattr(candidate_source,
                                             "valid_for_lookup_source", True))):
                        source = candidate_source
                        break
        block_size = int(source.block_size) if source is not None else 1
        dtype = str(source.dtype) if source is not None and source.dtype else None
        kv_layout = (str(source.kv_layout)
                     if source is not None and source.kv_layout else None)
        runtime_prelookup_source_id = (
            str(getattr(source, "source_id", "")) if source is not None else "")
        runtime_prelookup_source_is_metadata_only = int(
            bool(getattr(source, "is_metadata_only_source", False))
            if source is not None else False)
        runtime_prelookup_source_dtype = str(dtype or "")
        runtime_prelookup_source_kv_layout = str(kv_layout or "")
        runtime_prelookup_source_block_size = int(block_size)
        self._keepold_chunk_page_runtime_prelookup_source_id = str(
            runtime_prelookup_source_id)
        self._keepold_chunk_page_runtime_prelookup_source_is_metadata_only = int(
            runtime_prelookup_source_is_metadata_only)
        self._keepold_chunk_page_runtime_prelookup_source_dtype = str(
            runtime_prelookup_source_dtype)
        self._keepold_chunk_page_runtime_prelookup_source_kv_layout = str(
            runtime_prelookup_source_kv_layout)
        self._keepold_chunk_page_runtime_prelookup_source_block_size = int(
            runtime_prelookup_source_block_size)
        prompt_token_ids = self.config.keepold_prompt_token_ids or []
        chunk_infos = self.config.keepold_chunk_infos or []
        lookup_calls_in_runtime = int(self._keepold_chunk_page_lookup_calls_in_runtime)
        lookup_cache_mode = (
            "runtime_reuse"
            if lookup_calls_in_runtime > 1 else "diagnostic_only_single_call")
        if lookup_calls_in_runtime > 1:
            chunk_sig_items = []
            for info in chunk_infos:
                chunk_sig_items.append((
                    int(info.get("orig_idx", -1)),
                    int(info.get("start", -1)),
                    int(info.get("end", -1)),
                ))
            lookup_signature = json.dumps(
                {
                    "source_id": str(source_id),
                    "prompt_len": int(self.config.prompt_len),
                    "block_size": int(block_size),
                    "hash_algo": str(self.config.keepold_chunk_page_hash_algo),
                    "global_lookup": bool(
                        self.config.keepold_chunk_page_global_lookup),
                    "model_id": str(self.config.keepold_model_id or ""),
                    "dtype": str(dtype or ""),
                    "kv_layout": str(kv_layout or ""),
                    "prompt_token_ids": [int(x) for x in prompt_token_ids],
                    "chunk_infos": chunk_sig_items,
                },
                sort_keys=True,
                separators=(",", ":"),
            )
        else:
            # Single-call lookup is the normal case in GS.
            # Avoid materializing a huge prompt-token JSON signature that will
            # never be reused by runtime cache.
            lookup_signature = (
                "single_call:"
                f"{str(source_id)}|"
                f"prompt_len={int(self.config.prompt_len)}|"
                f"block_size={int(block_size)}|"
                f"chunk_count={int(len(chunk_infos))}|"
                f"hash_algo={str(self.config.keepold_chunk_page_hash_algo)}|"
                f"global_lookup={int(bool(self.config.keepold_chunk_page_global_lookup))}|"
                f"model_id={str(self.config.keepold_model_id or '')}|"
                f"dtype={str(dtype or '')}|"
                f"kv_layout={str(kv_layout or '')}"
            )

        lookup_hash_ms = 0.0
        lookup_index_ms = 0.0
        lookup_filter_ms = 0.0
        lookup_candidate_scan_ms = 0.0
        lookup_block_validity_check_ms = 0.0
        lookup_invalid_filter_ms = 0.0
        lookup_candidates_scanned = 0
        lookup_candidates_valid = 0
        lookup_candidates_invalid = 0
        metadata_only_candidate_count = 0
        metadata_only_filtered_count = 0
        metadata_only_false_hit_count = 0
        lookup_materialize_ms = 0.0
        lookup_total_ms = 0.0
        lookup_cache_hit = 0
        lookup_cache_reason = "diagnostic_only_single_call"
        lookup_cache_saved_ms = 0.0

        lookup_nvtx_pushed = self.maybe_nvtx_range_push("keepold_lookup")
        t_lookup0 = time.perf_counter()
        t_hash0 = time.perf_counter()
        _ = hash(lookup_signature)
        t_hash1 = time.perf_counter()
        lookup_hash_ms = float((t_hash1 - t_hash0) * 1000.0)

        t_index0 = time.perf_counter()
        _ = index.stats()
        t_index1 = time.perf_counter()
        lookup_index_ms = float((t_index1 - t_index0) * 1000.0)

        t_filter0 = time.perf_counter()
        if source is None:
            plan = build_all_miss_page_reuse_plan(
                prompt_token_ids=prompt_token_ids,
                chunk_infos=chunk_infos,
            )
        else:
            plan = build_page_reuse_plan(
                index=index,
                source_id=str(source.source_id),
                global_lookup=global_lookup,
                prompt_token_ids=prompt_token_ids,
                chunk_infos=chunk_infos,
                block_size=block_size,
                hash_algo=self.config.keepold_chunk_page_hash_algo,
                model_id=self.config.keepold_model_id,
                dtype=dtype,
                kv_layout=kv_layout,
                touch_on_hit=bool(
                    self.config.keepold_chunk_page_update_entry_on_hit),
            )
        t_filter1 = time.perf_counter()
        lookup_filter_ms = float((t_filter1 - t_filter0) * 1000.0)
        lookup_profile = getattr(plan, "lookup_profile", {}) or {}
        lookup_hash_ms += float(lookup_profile.get("lookup_hash_ms", 0.0))
        lookup_candidate_scan_ms = float(
            lookup_profile.get("lookup_candidate_scan_ms", 0.0))
        lookup_block_validity_check_ms = float(
            lookup_profile.get("lookup_block_validity_check_ms", 0.0))
        lookup_invalid_filter_ms = float(
            lookup_profile.get("lookup_invalid_filter_ms", 0.0))
        lookup_candidates_scanned = int(
            lookup_profile.get("lookup_candidates_scanned", 0.0))
        lookup_candidates_valid = int(
            lookup_profile.get("lookup_candidates_valid", 0.0))
        lookup_candidates_invalid = int(
            lookup_profile.get("lookup_candidates_invalid", 0.0))
        metadata_only_candidate_count = int(
            lookup_profile.get("metadata_only_candidate_count", 0.0))
        metadata_only_filtered_count = int(
            lookup_profile.get("metadata_only_filtered_count", 0.0))
        metadata_only_false_hit_count = int(
            lookup_profile.get("metadata_only_false_hit_count", 0.0))
        profile_materialize_ms = float(
            lookup_profile.get("lookup_materialize_ms", 0.0))
        if profile_materialize_ms > 0.0:
            lookup_materialize_ms = profile_materialize_ms

        t_materialize0 = time.perf_counter()
        _ = int(plan.indexed_chunks)
        _ = int(plan.hit_chunks)
        _ = int(plan.miss_chunks)
        _ = int(plan.candidate_count)
        t_materialize1 = time.perf_counter()
        lookup_materialize_overhead_ms = float((t_materialize1 - t_materialize0)
                                               * 1000.0)
        lookup_materialize_ms += float(lookup_materialize_overhead_ms)
        t_lookup1 = time.perf_counter()
        lookup_total_ms = float((t_lookup1 - t_lookup0) * 1000.0)
        self._keepold_chunk_page_lookup_cache_signature = lookup_signature
        self._keepold_chunk_page_lookup_cache_plan = plan
        self._keepold_chunk_page_lookup_cache_total_ms = float(lookup_total_ms)

        assert plan is not None
        lookup_ms = float(lookup_total_ms)
        self._keepold_chunk_page_lookup_hash_ms = float(lookup_hash_ms)
        self._keepold_chunk_page_lookup_index_ms = float(lookup_index_ms)
        self._keepold_chunk_page_lookup_filter_ms = float(lookup_filter_ms)
        self._keepold_chunk_page_lookup_candidate_scan_ms = float(
            lookup_candidate_scan_ms)
        self._keepold_chunk_page_lookup_block_validity_check_ms = float(
            lookup_block_validity_check_ms)
        self._keepold_chunk_page_lookup_invalid_filter_ms = float(
            lookup_invalid_filter_ms)
        self._keepold_chunk_page_lookup_candidates_scanned = int(
            lookup_candidates_scanned)
        self._keepold_chunk_page_lookup_candidates_valid = int(
            lookup_candidates_valid)
        self._keepold_chunk_page_lookup_candidates_invalid = int(
            lookup_candidates_invalid)
        self._keepold_chunk_page_metadata_only_candidate_count = int(
            metadata_only_candidate_count)
        self._keepold_chunk_page_metadata_only_filtered_count = int(
            metadata_only_filtered_count)
        self._keepold_chunk_page_metadata_only_false_hit_count = int(
            metadata_only_false_hit_count)
        self._keepold_chunk_page_lookup_materialize_ms = float(
            lookup_materialize_ms)
        self._keepold_chunk_page_lookup_total_ms = float(lookup_total_ms)
        self._keepold_chunk_page_lookup_cache_hit = int(lookup_cache_hit)
        self._keepold_chunk_page_lookup_cache_reason = str(lookup_cache_reason)
        self._keepold_chunk_page_lookup_cache_saved_ms = float(
            lookup_cache_saved_ms)
        if bool(self.config.keepold_chunk_page_register_hit_chunks_after_sample):
            self._keepold_chunk_page_post_register_infos = [
                dict(info) for info in chunk_infos
            ]
        else:
            miss_keys = set(plan.miss_chunk_keys)
            self._keepold_chunk_page_post_register_infos = [
                dict(info) for info in chunk_infos
                if (int(info.get("orig_idx", -1)), int(info.get("start", -1)),
                    int(info.get("end", -1))) in miss_keys
            ]
        self.maybe_nvtx_range_pop(lookup_nvtx_pushed)

        need_positions = bool(self.config.keepold_chunk_page_debug_checks
                              or self.config.keepold_chunk_page_execution_enabled)
        span_mode_requested = bool(
            self._keepold_chunk_page_exec_granularity
            == _CHUNK_PAGE_EXEC_GRANULARITY_SPAN
            and self._chunk_page_exec_enabled())
        span_token_fallback_reason = ""
        if span_mode_requested:
            span_build = build_span_reuse_plan(
                plan=plan,
                reuse_positions_cpu=self._reuse_positions_cpu,
                prompt_len=self.config.prompt_len,
                block_size=block_size,
                debug_check_duplicates=bool(
                    self.config.keepold_chunk_page_debug_checks),
            )
            self._keepold_chunk_page_span_plan_build_ms = float(
                span_build["span_plan_build_ms"])
            self._keepold_chunk_page_gpu_span_block_ids_cpu = (
                span_build["gpu_span_block_ids"]).contiguous()
            self._keepold_chunk_page_gpu_span_page_offsets_cpu = (
                span_build["gpu_span_page_offsets"]).contiguous()
            self._keepold_chunk_page_gpu_span_dst_starts_cpu = (
                span_build["gpu_span_dst_starts"]).contiguous()
            self._keepold_chunk_page_gpu_span_lengths_cpu = (
                span_build["gpu_span_lengths"]).contiguous()
            self._keepold_chunk_page_gpu_span_old_abs_starts_cpu = (
                span_build["gpu_span_old_abs_starts"]).contiguous()
            self._keepold_chunk_page_gpu_span_new_abs_starts_cpu = (
                span_build["gpu_span_new_abs_starts"]).contiguous()
            self._keepold_chunk_page_cpu_span_src_starts_cpu = (
                span_build["cpu_span_src_starts"]).contiguous()
            self._keepold_chunk_page_cpu_span_dst_starts_cpu = (
                span_build["cpu_span_dst_starts"]).contiguous()
            self._keepold_chunk_page_cpu_span_lengths_cpu = (
                span_build["cpu_span_lengths"]).contiguous()

            # Ensure span execution covers the *entire* reuse buffer, not only
            # chunk target rows.  build_span_reuse_plan() may only describe
            # chunk spans, while reuse buffer also contains prefix/template rows
            # such as prefix_capture_tokens.  Those uncovered rows must be copied
            # from CPU via CPU spans, otherwise compose_full_kv will fail with
            # "span exec incomplete coverage".
            total_reuse_int = int(self._reuse_positions_cpu.numel())
            self._keepold_chunk_page_span_full_reuse_tokens = int(total_reuse_int)
            self._keepold_chunk_page_span_prefix_uncovered_tokens = 0
            self._keepold_chunk_page_span_coverage_checked = False
            self._keepold_chunk_page_span_coverage_reason = "inactive"
            gpu_dst_tmp = self._keepold_chunk_page_gpu_span_dst_starts_cpu.to(
                device="cpu", dtype=torch.long).contiguous()
            gpu_len_tmp = self._keepold_chunk_page_gpu_span_lengths_cpu.to(
                device="cpu", dtype=torch.long).contiguous()
            cpu_src_tmp = self._keepold_chunk_page_cpu_span_src_starts_cpu.to(
                device="cpu", dtype=torch.long).contiguous()
            cpu_dst_tmp = self._keepold_chunk_page_cpu_span_dst_starts_cpu.to(
                device="cpu", dtype=torch.long).contiguous()
            cpu_len_tmp = self._keepold_chunk_page_cpu_span_lengths_cpu.to(
                device="cpu", dtype=torch.long).contiguous()

            cover_tmp = torch.zeros(total_reuse_int, dtype=torch.bool)
            _bad_span_cover = False

            for _i in range(int(gpu_len_tmp.numel())):
                _st = int(gpu_dst_tmp[_i].item())
                _ln = int(gpu_len_tmp[_i].item())
                if _ln <= 0:
                    continue
                _ed = _st + _ln
                if _st < 0 or _ed > total_reuse_int:
                    span_build["requires_token_fallback"] = True
                    span_build["fallback_reason"] = "span_gpu_dst_out_of_full_reuse_range"
                    _bad_span_cover = True
                    break
                if bool(cover_tmp[_st:_ed].any().item()):
                    span_build["requires_token_fallback"] = True
                    span_build["fallback_reason"] = "span_gpu_dst_overlap"
                    _bad_span_cover = True
                    break
                cover_tmp[_st:_ed] = True

            if not _bad_span_cover:
                if (int(cpu_src_tmp.numel()) != int(cpu_dst_tmp.numel())
                        or int(cpu_src_tmp.numel()) != int(cpu_len_tmp.numel())):
                    span_build["requires_token_fallback"] = True
                    span_build["fallback_reason"] = "span_cpu_metadata_len_mismatch"
                    _bad_span_cover = True

            _src_starts = []
            _dst_starts = []
            _lengths = []

            if not _bad_span_cover:
                for _i in range(int(cpu_len_tmp.numel())):
                    _src = int(cpu_src_tmp[_i].item())
                    _dst = int(cpu_dst_tmp[_i].item())
                    _ln = int(cpu_len_tmp[_i].item())
                    if _ln <= 0:
                        continue
                    _ed = _dst + _ln
                    if _dst < 0 or _ed > total_reuse_int:
                        span_build["requires_token_fallback"] = True
                        span_build["fallback_reason"] = "span_cpu_dst_out_of_full_reuse_range"
                        _bad_span_cover = True
                        break
                    if bool(cover_tmp[_dst:_ed].any().item()):
                        span_build["requires_token_fallback"] = True
                        span_build["fallback_reason"] = "span_cpu_gpu_or_cpu_overlap"
                        _bad_span_cover = True
                        break
                    cover_tmp[_dst:_ed] = True
                    _src_starts.append(_src)
                    _dst_starts.append(_dst)
                    _lengths.append(_ln)

            # Add CPU spans for every still-uncovered reuse row.  This captures
            # prefix/template rows and any chunk fallback rows not emitted by the
            # first-version span builder.  Current source KV is in reuse order,
            # so src_start == dst_start.
            if not _bad_span_cover and not bool(span_build["requires_token_fallback"]):
                self._keepold_chunk_page_span_prefix_uncovered_tokens = int(
                    (~cover_tmp).sum().item())
                _j = 0
                while _j < total_reuse_int:
                    if bool(cover_tmp[_j].item()):
                        _j += 1
                        continue
                    _st = _j
                    _j += 1
                    while _j < total_reuse_int and not bool(cover_tmp[_j].item()):
                        _j += 1
                    _ln = _j - _st
                    _src_starts.append(_st)
                    _dst_starts.append(_st)
                    _lengths.append(_ln)
                    cover_tmp[_st:_j] = True

                self._keepold_chunk_page_cpu_span_src_starts_cpu = torch.tensor(
                    _src_starts, dtype=torch.long)
                self._keepold_chunk_page_cpu_span_dst_starts_cpu = torch.tensor(
                    _dst_starts, dtype=torch.long)
                self._keepold_chunk_page_cpu_span_lengths_cpu = torch.tensor(
                    _lengths, dtype=torch.long)
                self._keepold_chunk_page_span_cpu_copy_src_starts_cpu = (
                    self._keepold_chunk_page_cpu_span_src_starts_cpu.contiguous())
                self._keepold_chunk_page_span_cpu_copy_dst_starts_cpu = (
                    self._keepold_chunk_page_cpu_span_dst_starts_cpu.contiguous())
                self._keepold_chunk_page_span_cpu_copy_lengths_cpu = (
                    self._keepold_chunk_page_cpu_span_lengths_cpu.contiguous())
                span_build["cpu_span_count"] = int(len(_lengths))
                span_build["cpu_span_tokens"] = int(sum(_lengths))
                self._keepold_chunk_page_span_coverage_checked = True
                self._keepold_chunk_page_span_coverage_reason = "ok"
            else:
                self._keepold_chunk_page_span_cpu_copy_src_starts_cpu = torch.empty(
                    0, dtype=torch.long)
                self._keepold_chunk_page_span_cpu_copy_dst_starts_cpu = torch.empty(
                    0, dtype=torch.long)
                self._keepold_chunk_page_span_cpu_copy_lengths_cpu = torch.empty(
                    0, dtype=torch.long)
                if _bad_span_cover:
                    self._keepold_chunk_page_span_coverage_reason = str(
                        span_build.get("fallback_reason",
                                       "span_coverage_check_failed"))
                elif bool(span_build["requires_token_fallback"]):
                    self._keepold_chunk_page_span_coverage_reason = str(
                        span_build.get("fallback_reason",
                                       "span_requires_token_fallback"))
                else:
                    self._keepold_chunk_page_span_coverage_reason = (
                        "span_coverage_not_checked")

            self._keepold_chunk_page_gpu_span_count = int(
                span_build["gpu_span_count"])
            self._keepold_chunk_page_cpu_span_count = int(
                span_build["cpu_span_count"])
            self._keepold_chunk_page_lookup_status = "ok"
            self._keepold_chunk_page_lookup_reason = "ok"
            self._keepold_chunk_page_target_reuse_tokens = int(
                span_build["target_reuse_tokens"])
            self._keepold_chunk_page_gathered_target_tokens = int(
                span_build["gpu_span_tokens"])
            self._keepold_chunk_page_cpu_fallback_tokens = int(
                span_build["cpu_span_tokens"])
            self._keepold_chunk_page_skipped_out_of_range = int(
                span_build["skipped_out_of_range"])
            self._keepold_chunk_page_skipped_missing_dst = int(
                span_build["skipped_missing_dst"])
            self._keepold_chunk_page_skipped_duplicate = int(
                span_build["skipped_duplicate"])
            self._keepold_chunk_page_noncontiguous_span_count = int(
                span_build["noncontiguous_span_count"])
            self._keepold_chunk_page_old_slots_build_ms = 0.0
            self._keepold_chunk_page_exec_tokens = int(
                span_build["target_reuse_tokens"])
            self._keepold_chunk_page_exec_gpu_tokens = int(
                span_build["gpu_span_tokens"])
            self._keepold_chunk_page_exec_cpu_tokens = int(
                span_build["cpu_span_tokens"])
            self._keepold_chunk_page_span_exec_ready = bool(
                not span_build["requires_token_fallback"])
            self._keepold_chunk_page_span_exec_reason = (
                "ok" if self._keepold_chunk_page_span_exec_ready else
                str(span_build["fallback_reason"]))
            # Prepare request-level rope old-position vector for span path once.
            span_rope_old_pos = self._reuse_old_positions_cpu.clone().contiguous()
            if self._keepold_chunk_page_span_exec_ready:
                for i in range(
                        int(self._keepold_chunk_page_gpu_span_lengths_cpu.numel())):
                    seg_len = int(
                        self._keepold_chunk_page_gpu_span_lengths_cpu[i].item())
                    if seg_len <= 0:
                        continue
                    dst_start = int(
                        self._keepold_chunk_page_gpu_span_dst_starts_cpu[i].item())
                    dst_end = dst_start + seg_len
                    old_abs_start = int(
                        self._keepold_chunk_page_gpu_span_old_abs_starts_cpu[i].item())
                    span_rope_old_pos[dst_start:dst_end] = (
                        old_abs_start + torch.arange(seg_len, dtype=torch.long))
            self._keepold_span_reuse_old_pos_for_rope_cpu = span_rope_old_pos
            if (self._keepold_chunk_page_span_exec_ready
                    and self.config.keepold_chunk_page_debug_checks):
                token_dbg = build_old_slots_and_dst_idx(
                    plan=plan,
                    reuse_positions_cpu=self._reuse_positions_cpu,
                    prompt_len=self.config.prompt_len,
                    block_size=block_size,
                    debug_build_positions=False,
                    debug_check_duplicates=True,
                    mode="auto",
                    debug_compare_fast_slow=bool(
                        self.config.keepold_chunk_page_debug_checks),
                )
                dbg_old_slots = token_dbg["old_slots"]
                dbg_dst_idx = token_dbg["dst_idx"]
                assert isinstance(dbg_old_slots, torch.Tensor)
                assert isinstance(dbg_dst_idx, torch.Tensor)

                span_old_slots_parts: list[torch.Tensor] = []
                span_dst_idx_parts: list[torch.Tensor] = []
                for i in range(int(self._keepold_chunk_page_gpu_span_lengths_cpu.numel())):
                    seg_len = int(self._keepold_chunk_page_gpu_span_lengths_cpu[i].item())
                    if seg_len <= 0:
                        continue
                    block_id = int(self._keepold_chunk_page_gpu_span_block_ids_cpu[i].item())
                    page_off = int(
                        self._keepold_chunk_page_gpu_span_page_offsets_cpu[i].item())
                    dst_start = int(self._keepold_chunk_page_gpu_span_dst_starts_cpu[i].item())
                    offs = torch.arange(seg_len, dtype=torch.long)
                    span_old_slots_parts.append(
                        (block_id * int(block_size) + page_off + offs).contiguous())
                    span_dst_idx_parts.append((dst_start + offs).contiguous())
                if span_old_slots_parts:
                    span_old_slots_dbg = torch.cat(span_old_slots_parts, dim=0).contiguous()
                    span_dst_idx_dbg = torch.cat(span_dst_idx_parts, dim=0).contiguous()
                else:
                    span_old_slots_dbg = torch.empty(0, dtype=torch.long)
                    span_dst_idx_dbg = torch.empty(0, dtype=torch.long)

                debug_compare_ok = True
                if int(span_dst_idx_dbg.numel()) != int(dbg_dst_idx.numel()):
                    debug_compare_ok = False
                else:
                    if int(span_dst_idx_dbg.numel()) > 0:
                        span_order = torch.argsort(span_dst_idx_dbg)
                        token_order = torch.argsort(dbg_dst_idx)
                        span_dst_sorted = span_dst_idx_dbg.index_select(0, span_order)
                        token_dst_sorted = dbg_dst_idx.index_select(0, token_order)
                        span_old_sorted = span_old_slots_dbg.index_select(
                            0, span_order)
                        token_old_sorted = dbg_old_slots.index_select(
                            0, token_order)
                        if (not torch.equal(span_dst_sorted, token_dst_sorted)
                                or not torch.equal(span_old_sorted, token_old_sorted)):
                            debug_compare_ok = False
                    if int(token_dbg["cpu_fallback_tokens"]) != int(
                            span_build["cpu_span_tokens"]):
                        debug_compare_ok = False

                if not debug_compare_ok:
                    self._keepold_chunk_page_span_exec_ready = False
                    self._keepold_chunk_page_span_exec_reason = (
                        "span_debug_compare_failed")
                    self._keepold_chunk_page_span_coverage_reason = (
                        "span_debug_compare_failed")
            has_invalid = (
                int(span_build["skipped_out_of_range"]) > 0
                or int(span_build["skipped_missing_dst"]) > 0
                or int(span_build["skipped_duplicate"]) > 0
                or bool(span_build["requires_token_fallback"])
            )
            self._keepold_chunk_page_exec_success = not has_invalid
            self._keepold_chunk_page_exec_fallback_reason = (
                "ok" if int(span_build["cpu_span_tokens"]) == 0
                else "partial_cpu_fallback")
            self._keepold_chunk_page_old_slots_cpu = torch.empty(
                0, dtype=torch.long)
            self._keepold_chunk_page_dst_idx_cpu = torch.empty(
                0, dtype=torch.long)
            # Build compact old/new position tensors for rope rewrite only.
            gpu_tokens = int(self._keepold_chunk_page_exec_gpu_tokens)
            if gpu_tokens > 0:
                old_pos = torch.empty(gpu_tokens, dtype=torch.long)
                new_pos = torch.empty(gpu_tokens, dtype=torch.long)
                w = 0
                for i in range(int(self._keepold_chunk_page_gpu_span_lengths_cpu.numel())):
                    seg_len = int(self._keepold_chunk_page_gpu_span_lengths_cpu[i].item())
                    if seg_len <= 0:
                        continue
                    old_st = int(self._keepold_chunk_page_gpu_span_old_abs_starts_cpu[i].item())
                    new_st = int(self._keepold_chunk_page_gpu_span_new_abs_starts_cpu[i].item())
                    sl = slice(w, w + seg_len)
                    offs = torch.arange(seg_len, dtype=torch.long)
                    old_pos[sl] = old_st + offs
                    new_pos[sl] = new_st + offs
                    w += seg_len
                self._keepold_chunk_page_old_positions_cpu = old_pos[:w].contiguous()
                self._keepold_chunk_page_new_positions_cpu = new_pos[:w].contiguous()
            else:
                self._keepold_chunk_page_old_positions_cpu = torch.empty(
                    0, dtype=torch.long)
                self._keepold_chunk_page_new_positions_cpu = torch.empty(
                    0, dtype=torch.long)
            self._keepold_chunk_page_plan = {
                "gather_spans": plan.gather_spans,
                "cpu_fallback_spans": plan.cpu_fallback_spans,
            }
            if not self._keepold_chunk_page_span_exec_ready:
                span_token_fallback_reason = str(
                    self._keepold_chunk_page_span_exec_reason)
            else:
                stats = index.stats()
                self._chunk_page_emit_metrics(
                    phase=self._keepold_phase,
                    status="ok",
                    reason="exec_ready",
                    fields={
                        "lookup_ms": float(lookup_ms),
                        "chunk_page_lookup_hash_ms": float(
                            self._keepold_chunk_page_lookup_hash_ms),
                        "chunk_page_lookup_index_ms": float(
                            self._keepold_chunk_page_lookup_index_ms),
                        "chunk_page_lookup_filter_ms": float(
                            self._keepold_chunk_page_lookup_filter_ms),
                        "chunk_page_lookup_candidate_scan_ms": float(
                            self._keepold_chunk_page_lookup_candidate_scan_ms),
                        "chunk_page_lookup_block_validity_check_ms": float(
                            self._keepold_chunk_page_lookup_block_validity_check_ms),
                        "chunk_page_lookup_invalid_filter_ms": float(
                            self._keepold_chunk_page_lookup_invalid_filter_ms),
                        "chunk_page_lookup_candidates_scanned": int(
                            self._keepold_chunk_page_lookup_candidates_scanned),
                        "chunk_page_lookup_candidates_valid": int(
                            self._keepold_chunk_page_lookup_candidates_valid),
                        "chunk_page_lookup_candidates_invalid": int(
                            self._keepold_chunk_page_lookup_candidates_invalid),
                        "chunk_page_metadata_only_candidate_count": int(
                            self._keepold_chunk_page_metadata_only_candidate_count),
                        "chunk_page_metadata_only_filtered_count": int(
                            self._keepold_chunk_page_metadata_only_filtered_count),
                        "chunk_page_metadata_only_false_hit_count": int(
                            self._keepold_chunk_page_metadata_only_false_hit_count),
                        "runtime_prelookup_source_id": str(
                            self._keepold_chunk_page_runtime_prelookup_source_id),
                        "runtime_prelookup_source_is_metadata_only": int(
                            self._keepold_chunk_page_runtime_prelookup_source_is_metadata_only),
                        "runtime_prelookup_source_dtype": str(
                            self._keepold_chunk_page_runtime_prelookup_source_dtype),
                        "runtime_prelookup_source_kv_layout": str(
                            self._keepold_chunk_page_runtime_prelookup_source_kv_layout),
                        "runtime_prelookup_source_block_size": int(
                            self._keepold_chunk_page_runtime_prelookup_source_block_size),
                        "chunk_page_lookup_materialize_ms": float(
                            self._keepold_chunk_page_lookup_materialize_ms),
                        "chunk_page_lookup_total_ms": float(
                            self._keepold_chunk_page_lookup_total_ms),
                        "chunk_page_lookup_cache_hit": int(
                            self._keepold_chunk_page_lookup_cache_hit),
                        "chunk_page_lookup_cache_reason": str(
                            self._keepold_chunk_page_lookup_cache_reason),
                        "chunk_page_lookup_cache_saved_ms": float(
                            self._keepold_chunk_page_lookup_cache_saved_ms),
                        "chunk_page_lookup_calls_in_runtime": int(
                            lookup_calls_in_runtime),
                        "chunk_page_lookup_cache_mode": str(
                            lookup_cache_mode),
                        "old_slots_build_ms": 0.0,
                        "indexed_chunks": int(plan.indexed_chunks),
                        "hit_chunks": int(plan.hit_chunks),
                        "miss_chunks": int(plan.miss_chunks),
                        "candidate_count": int(plan.candidate_count),
                        "same_source_hit_count": int(
                            plan.same_source_hit_count),
                        "global_cross_source_hit_count": int(
                            plan.global_cross_source_hit_count),
                        "gpu_gather_full_tokens": int(plan.gpu_gather_full_tokens),
                        "gpu_gather_partial_tokens": int(plan.gpu_gather_partial_tokens),
                        "target_reuse_tokens": int(
                            span_build["target_reuse_tokens"]),
                        "gathered_target_tokens": int(span_build["gpu_span_tokens"]),
                        "cpu_fallback_tokens": int(span_build["cpu_span_tokens"]),
                        "skipped_out_of_range": int(
                            span_build["skipped_out_of_range"]),
                        "skipped_missing_dst": int(
                            span_build["skipped_missing_dst"]),
                        "skipped_duplicate": int(
                            span_build["skipped_duplicate"]),
                        "noncontiguous_span_count": int(
                            span_build["noncontiguous_span_count"]),
                        "lru_evictions": int(stats["lru_evictions"]),
                        **self._chunk_page_index_stats_payload(stats),
                        "chunk_page_exec_tokens": int(
                            span_build["target_reuse_tokens"]),
                        "chunk_page_exec_gpu_tokens": int(
                            span_build["gpu_span_tokens"]),
                        "chunk_page_exec_cpu_tokens": int(
                            span_build["cpu_span_tokens"]),
                        "chunk_page_exec_old_slots_build_ms": 0.0,
                        "chunk_page_span_plan_build_ms": float(
                            span_build["span_plan_build_ms"]),
                        "span_plan_build_ms": float(
                            span_build["span_plan_build_ms"]),
                        "span_plan_total_ms": float(
                            span_build["span_plan_total_ms"]),
                        "span_plan_make_pos_map_ms": float(
                            span_build["span_plan_make_pos_map_ms"]),
                        "span_plan_iter_gpu_spans_ms": float(
                            span_build["span_plan_iter_gpu_spans_ms"]),
                        "span_plan_build_cpu_spans_ms": float(
                            span_build["span_plan_build_cpu_spans_ms"]),
                        "span_plan_merge_gpu_spans_ms": float(
                            span_build["span_plan_merge_gpu_spans_ms"]),
                        "span_plan_tensor_materialize_ms": float(
                            span_build["span_plan_tensor_materialize_ms"]),
                        "span_plan_debug_check_ms": float(
                            span_build["span_plan_debug_check_ms"]),
                        "span_plan_other_ms": float(
                            span_build["span_plan_other_ms"]),
                        "span_plan_gpu_span_count": int(
                            span_build["gpu_span_count"]),
                        "span_plan_cpu_span_count": int(
                            span_build["cpu_span_count"]),
                        "span_plan_gpu_span_count_before_merge": int(
                            span_build["span_plan_gpu_span_count_before_merge"]),
                        "span_plan_gpu_span_count_after_merge": int(
                            span_build["span_plan_gpu_span_count_after_merge"]),
                        "span_plan_cpu_span_count_before_merge": int(
                            span_build["span_plan_cpu_span_count_before_merge"]),
                        "span_plan_cpu_span_count_after_merge": int(
                            span_build["span_plan_cpu_span_count_after_merge"]),
                        "span_plan_gpu_tokens": int(
                            span_build["gpu_span_tokens"]),
                        "span_plan_cpu_tokens": int(
                            span_build["cpu_span_tokens"]),
                        "span_plan_target_reuse_tokens": int(
                            span_build["target_reuse_tokens"]),
                        "span_plan_full_reuse_tokens": int(
                            self._keepold_chunk_page_span_full_reuse_tokens),
                        "span_plan_prefix_uncovered_tokens": int(
                            self._keepold_chunk_page_span_prefix_uncovered_tokens),
                        "span_plan_requires_token_fallback":
                        1 if span_build["requires_token_fallback"] else 0,
                        "span_plan_fallback_reason": str(
                            span_build["fallback_reason"]),
                        "chunk_page_exec_granularity":
                        _CHUNK_PAGE_EXEC_GRANULARITY_SPAN,
                        "span_exec_ready": 1,
                        "span_exec_reason":
                        self._keepold_chunk_page_span_exec_reason,
                        "span_coverage_checked":
                        1 if self._keepold_chunk_page_span_coverage_checked else 0,
                        "span_coverage_reason":
                        self._keepold_chunk_page_span_coverage_reason,
                        "gpu_span_count": int(span_build["gpu_span_count"]),
                        "cpu_span_count": int(span_build["cpu_span_count"]),
                        "span_index_h2d_submit_ms": float(
                            self._keepold_chunk_page_span_index_h2d_submit_ms),
                        "span_index_h2d_device_ms": float(
                            self._keepold_chunk_page_span_index_h2d_device_ms),
                        "chunk_page_exec_fallback_reason":
                        self._keepold_chunk_page_exec_fallback_reason,
                        "chunk_page_exec_success":
                        1 if self._keepold_chunk_page_exec_success else 0,
                        "chunk_page_cpu_fallback_tokens": int(
                            span_build["cpu_span_tokens"]),
                        "timing_v2_chunk_page_lookup_ms": float(lookup_ms),
                        "timing_v2_chunk_page_old_slots_build_ms": 0.0,
                        "timing_v2_chunk_page_exec_partition_ms": float(
                            self._keepold_timing_v2_chunk_page_exec_partition_ms),
                        "chunk_page_partition_build_ms": float(
                            self._keepold_chunk_page_partition_build_ms),
                        "chunk_page_mixed_plan_materialize_ms": float(
                            self._keepold_chunk_page_mixed_plan_materialize_ms),
                        "chunk_page_cpu_fallback_rows_build_ms": float(
                            self._keepold_chunk_page_cpu_fallback_rows_build_ms),
                        "chunk_page_old_page_plan_build_ms": float(
                            self._keepold_chunk_page_old_page_plan_build_ms),
                    },
                )
                return

        old_slots_nvtx_pushed = self.maybe_nvtx_range_push("keepold_old_slots_build")
        try:
            build = build_old_slots_and_dst_idx(
                plan=plan,
                reuse_positions_cpu=self._reuse_positions_cpu,
                prompt_len=self.config.prompt_len,
                block_size=block_size,
                debug_build_positions=need_positions,
                debug_check_duplicates=bool(
                    self.config.keepold_chunk_page_debug_checks),
                mode="auto",
                debug_compare_fast_slow=bool(
                    self.config.keepold_chunk_page_debug_checks),
            )
        finally:
            self.maybe_nvtx_range_pop(old_slots_nvtx_pushed)
        old_slots = build["old_slots"]
        dst_idx = build["dst_idx"]
        old_positions = build["old_positions"]
        new_positions = build["new_positions"]
        assert isinstance(old_slots, torch.Tensor)
        assert isinstance(dst_idx, torch.Tensor)
        assert isinstance(old_positions, torch.Tensor)
        assert isinstance(new_positions, torch.Tensor)

        self._keepold_chunk_page_old_slots_cpu = old_slots.contiguous()
        self._keepold_chunk_page_dst_idx_cpu = dst_idx.contiguous()
        self._keepold_chunk_page_old_positions_cpu = old_positions.contiguous()
        self._keepold_chunk_page_new_positions_cpu = new_positions.contiguous()
        self._keepold_chunk_page_plan = {
            "gather_spans": plan.gather_spans,
            "cpu_fallback_spans": plan.cpu_fallback_spans,
        }
        self._keepold_chunk_page_lookup_status = "ok"
        self._keepold_chunk_page_lookup_reason = "ok"
        self._keepold_chunk_page_target_reuse_tokens = int(
            build["target_reuse_tokens"])
        self._keepold_chunk_page_gathered_target_tokens = int(
            build["gathered_target_tokens"])
        self._keepold_chunk_page_cpu_fallback_tokens = int(
            build["cpu_fallback_tokens"])
        self._keepold_chunk_page_skipped_out_of_range = int(
            build["skipped_out_of_range"])
        self._keepold_chunk_page_skipped_missing_dst = int(
            build["skipped_missing_dst"])
        self._keepold_chunk_page_skipped_duplicate = int(
            build["skipped_duplicate"])
        self._keepold_chunk_page_noncontiguous_span_count = int(
            build["noncontiguous_span_count"])
        self._keepold_chunk_page_old_slots_build_ms = float(
            build["old_slots_build_ms"])
        self._keepold_chunk_page_span_plan_build_ms = 0.0
        self._keepold_chunk_page_span_exec_ready = False
        self._keepold_chunk_page_span_exec_reason = (
            "inactive" if not span_token_fallback_reason else
            f"fallback_to_token:{span_token_fallback_reason}")
        self._keepold_chunk_page_exec_tokens = int(build["target_reuse_tokens"])
        self._keepold_chunk_page_exec_gpu_tokens = int(
            build["gathered_target_tokens"])
        self._keepold_chunk_page_exec_cpu_tokens = int(
            build["cpu_fallback_tokens"])

        has_missing = (plan.miss_chunks > 0
                       or int(build["cpu_fallback_tokens"]) > 0
                       or int(build["skipped_out_of_range"]) > 0
                       or int(build["skipped_missing_dst"]) > 0
                       or int(build["skipped_duplicate"]) > 0)
        self._keepold_chunk_page_exec_success = not has_missing
        self._keepold_chunk_page_exec_fallback_reason = (
            "ok" if not has_missing else "partial_cpu_fallback")

        if (self.config.keepold_chunk_page_debug_checks
                and int(dst_idx.numel()) > 0
                and int(old_positions.numel()) == int(dst_idx.numel())
                and int(new_positions.numel()) == int(dst_idx.numel())):
            dry_old = self._reuse_old_positions_cpu.clone()
            dry_new = self._reuse_positions_cpu.clone()
            dry_old.index_copy_(0, dst_idx, old_positions)
            dry_new.index_copy_(0, dst_idx, new_positions)
            # Keep these for stage-2 wiring validation.
            self._keepold_chunk_page_old_positions_cpu = dry_old
            self._keepold_chunk_page_new_positions_cpu = dry_new
        elif (self.config.keepold_chunk_page_execution_enabled
              and int(dst_idx.numel()) > 0
              and int(old_positions.numel()) == int(dst_idx.numel())
              and int(new_positions.numel()) == int(dst_idx.numel())):
            # Stage-2 execution wiring consumes subset old/new positions.
            self._keepold_chunk_page_old_positions_cpu = old_positions.contiguous()
            self._keepold_chunk_page_new_positions_cpu = new_positions.contiguous()
        else:
            self._keepold_chunk_page_old_positions_cpu = (
                self._reuse_old_positions_cpu)
            self._keepold_chunk_page_new_positions_cpu = (
                self._reuse_positions_cpu)

        stats = index.stats()
        self._chunk_page_emit_metrics(
            phase=self._keepold_phase,
            status="ok",
            reason=("exec_ready" if self._chunk_page_exec_enabled() else
                    "dry_run_only"),
            fields={
                "lookup_ms": float(lookup_ms),
                "chunk_page_lookup_hash_ms": float(
                    self._keepold_chunk_page_lookup_hash_ms),
                "chunk_page_lookup_index_ms": float(
                    self._keepold_chunk_page_lookup_index_ms),
                "chunk_page_lookup_filter_ms": float(
                    self._keepold_chunk_page_lookup_filter_ms),
                "chunk_page_lookup_candidate_scan_ms": float(
                    self._keepold_chunk_page_lookup_candidate_scan_ms),
                "chunk_page_lookup_block_validity_check_ms": float(
                    self._keepold_chunk_page_lookup_block_validity_check_ms),
                "chunk_page_lookup_invalid_filter_ms": float(
                    self._keepold_chunk_page_lookup_invalid_filter_ms),
                "chunk_page_lookup_candidates_scanned": int(
                    self._keepold_chunk_page_lookup_candidates_scanned),
                "chunk_page_lookup_candidates_valid": int(
                    self._keepold_chunk_page_lookup_candidates_valid),
                "chunk_page_lookup_candidates_invalid": int(
                    self._keepold_chunk_page_lookup_candidates_invalid),
                "chunk_page_metadata_only_candidate_count": int(
                    self._keepold_chunk_page_metadata_only_candidate_count),
                "chunk_page_metadata_only_filtered_count": int(
                    self._keepold_chunk_page_metadata_only_filtered_count),
                "chunk_page_metadata_only_false_hit_count": int(
                    self._keepold_chunk_page_metadata_only_false_hit_count),
                "runtime_prelookup_source_id": str(
                    self._keepold_chunk_page_runtime_prelookup_source_id),
                "runtime_prelookup_source_is_metadata_only": int(
                    self._keepold_chunk_page_runtime_prelookup_source_is_metadata_only),
                "runtime_prelookup_source_dtype": str(
                    self._keepold_chunk_page_runtime_prelookup_source_dtype),
                "runtime_prelookup_source_kv_layout": str(
                    self._keepold_chunk_page_runtime_prelookup_source_kv_layout),
                "runtime_prelookup_source_block_size": int(
                    self._keepold_chunk_page_runtime_prelookup_source_block_size),
                "chunk_page_lookup_materialize_ms": float(
                    self._keepold_chunk_page_lookup_materialize_ms),
                "chunk_page_lookup_total_ms": float(
                    self._keepold_chunk_page_lookup_total_ms),
                "chunk_page_lookup_cache_hit": int(
                    self._keepold_chunk_page_lookup_cache_hit),
                "chunk_page_lookup_cache_reason": str(
                    self._keepold_chunk_page_lookup_cache_reason),
                "chunk_page_lookup_cache_saved_ms": float(
                    self._keepold_chunk_page_lookup_cache_saved_ms),
                "chunk_page_lookup_calls_in_runtime": int(
                    lookup_calls_in_runtime),
                "chunk_page_lookup_cache_mode": str(lookup_cache_mode),
                "old_slots_build_ms": float(build["old_slots_build_ms"]),
                "old_slots_total_ms": float(build["old_slots_total_ms"]),
                "old_slots_map_build_ms": float(
                    build["old_slots_map_build_ms"]),
                "old_slots_span_iter_ms": float(
                    build["old_slots_span_iter_ms"]),
                "old_slots_tensor_materialize_ms": float(
                    build["old_slots_tensor_materialize_ms"]),
                "old_slots_position_build_ms": float(
                    build["old_slots_position_build_ms"]),
                "old_slots_check_ms": float(build["old_slots_check_ms"]),
                "old_slots_fastpath_used": int(
                    build["old_slots_fastpath_used"]),
                "old_slots_fastpath_reason": str(
                    build["old_slots_fastpath_reason"]),
                "old_slots_span_count": int(build["old_slots_span_count"]),
                "old_slots_token_count": int(build["old_slots_token_count"]),
                "indexed_chunks": int(plan.indexed_chunks),
                "hit_chunks": int(plan.hit_chunks),
                "miss_chunks": int(plan.miss_chunks),
                "candidate_count": int(plan.candidate_count),
                "same_source_hit_count": int(plan.same_source_hit_count),
                "global_cross_source_hit_count": int(
                    plan.global_cross_source_hit_count),
                "gpu_gather_full_tokens": int(plan.gpu_gather_full_tokens),
                "gpu_gather_partial_tokens": int(plan.gpu_gather_partial_tokens),
                "target_reuse_tokens": int(build["target_reuse_tokens"]),
                "gathered_target_tokens": int(build["gathered_target_tokens"]),
                "cpu_fallback_tokens": int(build["cpu_fallback_tokens"]),
                "skipped_out_of_range": int(build["skipped_out_of_range"]),
                "skipped_missing_dst": int(build["skipped_missing_dst"]),
                "skipped_duplicate": int(build["skipped_duplicate"]),
                "noncontiguous_span_count": int(
                    build["noncontiguous_span_count"]),
                "lru_evictions": int(stats["lru_evictions"]),
                **self._chunk_page_index_stats_payload(stats),
                "chunk_page_exec_tokens": int(build["target_reuse_tokens"]),
                "chunk_page_exec_gpu_tokens": int(
                    build["gathered_target_tokens"]),
                "chunk_page_exec_cpu_tokens": int(build["cpu_fallback_tokens"]),
                "chunk_page_exec_old_slots_build_ms": float(
                    build["old_slots_build_ms"]),
                "chunk_page_span_plan_build_ms": float(
                    self._keepold_chunk_page_span_plan_build_ms),
                "chunk_page_exec_granularity":
                self._keepold_chunk_page_exec_granularity,
                "span_exec_ready":
                1 if self._keepold_chunk_page_span_exec_ready else 0,
                "span_exec_reason":
                self._keepold_chunk_page_span_exec_reason,
                "gpu_span_count": int(self._keepold_chunk_page_gpu_span_count),
                "cpu_span_count": int(self._keepold_chunk_page_cpu_span_count),
                "span_index_h2d_submit_ms": float(
                    self._keepold_chunk_page_span_index_h2d_submit_ms),
                "span_index_h2d_device_ms": float(
                    self._keepold_chunk_page_span_index_h2d_device_ms),
                "chunk_page_exec_fallback_reason":
                self._keepold_chunk_page_exec_fallback_reason,
                "chunk_page_exec_success":
                1 if self._keepold_chunk_page_exec_success else 0,
                "chunk_page_cpu_fallback_tokens": int(
                    build["cpu_fallback_tokens"]),
                "timing_v2_chunk_page_lookup_ms": float(lookup_ms),
                "timing_v2_chunk_page_old_slots_build_ms": float(
                    build["old_slots_build_ms"]),
                "timing_v2_chunk_page_exec_partition_ms": float(
                    self._keepold_timing_v2_chunk_page_exec_partition_ms),
                "chunk_page_partition_build_ms": float(
                    self._keepold_chunk_page_partition_build_ms),
                "chunk_page_mixed_plan_materialize_ms": float(
                    self._keepold_chunk_page_mixed_plan_materialize_ms),
                "chunk_page_cpu_fallback_rows_build_ms": float(
                    self._keepold_chunk_page_cpu_fallback_rows_build_ms),
                "chunk_page_old_page_plan_build_ms": float(
                    self._keepold_chunk_page_old_page_plan_build_ms),
            },
        )

    def _get_keepold_bridge(self,
                            *,
                            create: bool = False
                            ) -> Optional[_KeepOldPageBridge]:
        key = self._keepold_bridge_key
        if not key:
            return None
        bridge = _KEEPOLD_PAGE_BRIDGES.get(key)
        if bridge is not None:
            return bridge
        if not create:
            return None
        bridge = _KeepOldPageBridge(
            prompt_len=int(self.config.prompt_len),
            switch_prompt_positions_dev=None,
            layers={},
            ts_unix=time.time(),
        )
        _KEEPOLD_PAGE_BRIDGES[key] = bridge
        return bridge

    @staticmethod
    def _extract_tokens_from_paged_cache(
        cache: torch.Tensor,
        slots: torch.Tensor,
        num_kv_heads: int,
        head_size: int,
    ) -> torch.Tensor:
        return extract_tokens_from_paged_cache(
            cache,
            slots,
            num_kv_heads,
            head_size,
        )

    @staticmethod
    def _fused_extract_scatter_k_from_paged_cache(
        *,
        key_cache: torch.Tensor,
        old_slots: torch.Tensor,
        dst_reuse_idx: torch.Tensor,
        reuse_k: torch.Tensor,
        num_kv_heads: int,
        head_size: int,
        chunk_size: int,
        mode: str,
    ) -> dict[str, Any]:
        return fused_extract_scatter_k_from_paged_cache(
            key_cache=key_cache,
            old_slots=old_slots,
            dst_reuse_idx=dst_reuse_idx,
            reuse_k=reuse_k,
            num_kv_heads=num_kv_heads,
            head_size=head_size,
            chunk_size=chunk_size,
            mode=mode,
        )

    @staticmethod
    def _fused_extract_scatter_kv_from_paged_cache(
        *,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        old_slots: torch.Tensor,
        dst_reuse_idx: torch.Tensor,
        reuse_k: torch.Tensor,
        reuse_v: torch.Tensor,
        num_kv_heads: int,
        head_size: int,
        mode: str,
    ) -> dict[str, Any]:
        return fused_extract_scatter_kv_from_paged_cache(
            key_cache=key_cache,
            value_cache=value_cache,
            old_slots=old_slots,
            dst_reuse_idx=dst_reuse_idx,
            reuse_k=reuse_k,
            reuse_v=reuse_v,
            num_kv_heads=num_kv_heads,
            head_size=head_size,
            mode=mode,
        )

    @staticmethod
    def _fused_extract_scatter_kv_from_paged_cache_fast(
        *,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        old_slots_i64: torch.Tensor,
        dst_idx_i64: torch.Tensor,
        reuse_k: torch.Tensor,
        reuse_v: torch.Tensor,
        layout_code: int,
        page_size: int,
    ) -> dict[str, Any]:
        return fused_extract_scatter_kv_from_paged_cache_cuda_op_fast(
            key_cache=key_cache,
            value_cache=value_cache,
            old_slots_i64=old_slots_i64,
            dst_idx_i64=dst_idx_i64,
            reuse_k=reuse_k,
            reuse_v=reuse_v,
            layout_code=layout_code,
            page_size=page_size,
        )

    @staticmethod
    def _fused_extract_scatter_kv_from_paged_cache_spans_fast(
        *,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        span_block_ids_i64: torch.Tensor,
        span_page_offsets_i64: torch.Tensor,
        span_dst_starts_i64: torch.Tensor,
        span_lengths_i64: torch.Tensor,
        reuse_k: torch.Tensor,
        reuse_v: torch.Tensor,
        layout_code: int,
        page_size: int,
    ) -> dict[str, Any]:
        return fused_extract_scatter_kv_from_paged_cache_spans_cuda_op_fast(
            key_cache=key_cache,
            value_cache=value_cache,
            span_block_ids_i64=span_block_ids_i64,
            span_page_offsets_i64=span_page_offsets_i64,
            span_dst_starts_i64=span_dst_starts_i64,
            span_lengths_i64=span_lengths_i64,
            reuse_k=reuse_k,
            reuse_v=reuse_v,
            layout_code=layout_code,
            page_size=page_size,
        )

    @staticmethod
    def _infer_paged_cache_layout_code_page_size(
        *,
        cache: torch.Tensor,
        num_kv_heads: int,
        head_size: int,
    ) -> Optional[tuple[int, int]]:
        if cache.dim() != 4:
            return None
        if (int(cache.shape[2]) == int(num_kv_heads)
                and int(cache.shape[3]) == int(head_size)):
            return 0, int(cache.shape[1])
        if (int(cache.shape[1]) == int(num_kv_heads)
                and int(cache.shape[3]) == int(head_size)):
            return 1, int(cache.shape[2])
        return None

    def _reset_keepold_old_page_fastpath(
        self,
        *,
        reset_timing_v2_index: bool = True,
    ) -> None:
        self._keepold_switch_reuse_idx_i64_dev = None
        self._keepold_old_slots_i64_by_layer = {}
        self._keepold_old_page_fast_ready = False
        self._keepold_old_page_fast_reason = "inactive"
        self._keepold_old_page_prepare_ms = 0.0
        self._keepold_chunk_page_shared_switch_reuse_idx_i64_dev = None
        self._keepold_chunk_page_shared_cpu_reuse_idx_i64_dev = None
        self._keepold_chunk_page_shared_old_slots_i64_dev = None
        self._keepold_chunk_page_gpu_span_block_ids_dev = None
        self._keepold_chunk_page_gpu_span_page_offsets_dev = None
        self._keepold_chunk_page_gpu_span_dst_starts_dev = None
        self._keepold_chunk_page_gpu_span_lengths_dev = None
        self._keepold_chunk_page_gpu_span_old_abs_starts_dev = None
        self._keepold_chunk_page_gpu_span_new_abs_starts_dev = None
        self._keepold_chunk_page_cpu_span_src_starts_dev = None
        self._keepold_chunk_page_cpu_span_dst_starts_dev = None
        self._keepold_chunk_page_cpu_span_lengths_dev = None
        self._keepold_chunk_page_span_cpu_copy_src_starts_dev = None
        self._keepold_chunk_page_span_cpu_copy_dst_starts_dev = None
        self._keepold_chunk_page_span_cpu_copy_lengths_dev = None
        self._keepold_span_reuse_old_pos_for_rope_dev = None
        self._keepold_chunk_page_shared_index_ready = False
        self._keepold_chunk_page_shared_index_reason = "inactive"
        self._keepold_chunk_page_shared_index_prepare_count = 0
        self._keepold_chunk_page_span_index_h2d_submit_ms = 0.0
        self._keepold_chunk_page_span_index_h2d_device_ms = float("nan")
        if reset_timing_v2_index:
            self._keepold_timing_v2_index_tensor_h2d_submit_ms = 0.0
            self._keepold_timing_v2_index_tensor_h2d_device_ms = float("nan")
        self._keepold_gpu_index_idx_i64_dev = None
        self._keepold_gpu_index_fast_ready = False
        self._keepold_gpu_index_fast_reason = "inactive"
        self._keepold_gpu_index_prepare_ms = 0.0

    def _prepare_keepold_chunk_page_shared_index_tensors(
        self,
        *,
        device: torch.device,
    ) -> None:
        prep_total_ms = 0.0
        prep_old_slots_ms = 0.0
        prep_dst_idx_ms = 0.0
        prep_positions_ms = 0.0
        prep_h2d_ms = 0.0
        prep_cache_hit = 0
        prep_cache_reason = "unknown"
        prep_t0 = time.perf_counter()
        if not self._chunk_page_exec_enabled():
            self._keepold_chunk_page_shared_index_reason = (
                "chunk_page_exec_disabled")
            return
        if self._keepold_chunk_page_shared_index_ready:
            prep_cache_hit = 1
            prep_cache_reason = "already_ready"
            prep_total_ms = float((time.perf_counter() - prep_t0) * 1000.0)
            self._append_metrics({
                "event": "selective_keepold_chunk_page_index_prepare",
                "keepold_phase": self._keepold_phase,
                "prepare_count": int(
                    self._keepold_chunk_page_shared_index_prepare_count),
                "prepared_once": int(
                    self._keepold_chunk_page_shared_index_prepare_count == 1),
                "fast_ready": bool(self._keepold_chunk_page_shared_index_ready),
                "fast_reason": self._keepold_chunk_page_shared_index_reason,
                "gpu_index_prepare_total_ms": float(prep_total_ms),
                "gpu_index_prepare_old_slots_ms": float(prep_old_slots_ms),
                "gpu_index_prepare_dst_idx_ms": float(prep_dst_idx_ms),
                "gpu_index_prepare_positions_ms": float(prep_positions_ms),
                "gpu_index_prepare_h2d_ms": float(prep_h2d_ms),
                "gpu_index_prepare_cache_hit": int(prep_cache_hit),
                "gpu_index_prepare_cache_reason": str(prep_cache_reason),
                "ts_unix": time.time(),
            })
            return

        self._keepold_chunk_page_shared_index_prepare_count += 1
        self._keepold_chunk_page_shared_index_reason = "unknown"

        if (self._keepold_chunk_page_exec_granularity
                == _CHUNK_PAGE_EXEC_GRANULARITY_SPAN
                and self._keepold_chunk_page_span_exec_ready):
            prep_cache_hit = 0
            prep_cache_reason = "build_span"
            def _ensure_i64_cpu(t: torch.Tensor) -> torch.Tensor:
                return t.to(device="cpu", dtype=torch.long).contiguous()

            def _ensure_i64_dev(t: torch.Tensor) -> torch.Tensor:
                out = t
                if out.device != device:
                    out = out.to(device=device, non_blocking=False)
                if out.dtype != torch.long:
                    out = out.to(dtype=torch.long)
                if not out.is_contiguous():
                    out = out.contiguous()
                return out

            gpu_block = _ensure_i64_cpu(self._keepold_chunk_page_gpu_span_block_ids_cpu)
            gpu_off = _ensure_i64_cpu(self._keepold_chunk_page_gpu_span_page_offsets_cpu)
            gpu_dst = _ensure_i64_cpu(self._keepold_chunk_page_gpu_span_dst_starts_cpu)
            gpu_len = _ensure_i64_cpu(self._keepold_chunk_page_gpu_span_lengths_cpu)
            gpu_old = _ensure_i64_cpu(
                self._keepold_chunk_page_gpu_span_old_abs_starts_cpu)
            gpu_new = _ensure_i64_cpu(
                self._keepold_chunk_page_gpu_span_new_abs_starts_cpu)
            prep_old_slots_ms += 0.0
            prep_dst_idx_ms += float(0.0)
            prep_positions_ms += float(0.0)
            cpu_src = _ensure_i64_cpu(
                self._keepold_chunk_page_span_cpu_copy_src_starts_cpu)
            cpu_dst = _ensure_i64_cpu(
                self._keepold_chunk_page_span_cpu_copy_dst_starts_cpu)
            cpu_len = _ensure_i64_cpu(
                self._keepold_chunk_page_span_cpu_copy_lengths_cpu)
            rope_old_pos_cpu = _ensure_i64_cpu(
                self._keepold_span_reuse_old_pos_for_rope_cpu)
            n_gpu = int(gpu_len.numel())
            n_cpu = int(cpu_len.numel())
            if (int(gpu_block.numel()) != n_gpu
                    or int(gpu_off.numel()) != n_gpu
                    or int(gpu_dst.numel()) != n_gpu
                    or int(gpu_old.numel()) != n_gpu
                    or int(gpu_new.numel()) != n_gpu):
                self._keepold_chunk_page_shared_index_reason = (
                    "span_gpu_metadata_len_mismatch")
                self._keepold_chunk_page_shared_index_ready = False
                return
            if int(cpu_src.numel()) != n_cpu or int(cpu_dst.numel()) != n_cpu:
                self._keepold_chunk_page_shared_index_reason = (
                    "span_cpu_metadata_len_mismatch")
                self._keepold_chunk_page_shared_index_ready = False
                return

            timing_cuda_event = bool(self._timing_cuda_event_enabled()
                                     and device.type == "cuda")
            idx_start_event: Optional[torch.cuda.Event] = None
            idx_end_event: Optional[torch.cuda.Event] = None
            if timing_cuda_event:
                idx_start_event = torch.cuda.Event(enable_timing=True)
                idx_end_event = torch.cuda.Event(enable_timing=True)
                idx_start_event.record(torch.cuda.current_stream(device))
            t_idx_submit0 = time.perf_counter()
            self._keepold_chunk_page_gpu_span_block_ids_dev = _ensure_i64_dev(
                gpu_block)
            self._keepold_chunk_page_gpu_span_page_offsets_dev = _ensure_i64_dev(
                gpu_off)
            self._keepold_chunk_page_gpu_span_dst_starts_dev = _ensure_i64_dev(
                gpu_dst)
            self._keepold_chunk_page_gpu_span_lengths_dev = _ensure_i64_dev(
                gpu_len)
            self._keepold_chunk_page_gpu_span_old_abs_starts_dev = _ensure_i64_dev(
                gpu_old)
            self._keepold_chunk_page_gpu_span_new_abs_starts_dev = _ensure_i64_dev(
                gpu_new)
            self._keepold_chunk_page_cpu_span_src_starts_dev = _ensure_i64_dev(
                cpu_src)
            self._keepold_chunk_page_cpu_span_dst_starts_dev = _ensure_i64_dev(
                cpu_dst)
            self._keepold_chunk_page_cpu_span_lengths_dev = _ensure_i64_dev(
                cpu_len)
            self._keepold_chunk_page_span_cpu_copy_src_starts_dev = (
                self._keepold_chunk_page_cpu_span_src_starts_dev)
            self._keepold_chunk_page_span_cpu_copy_dst_starts_dev = (
                self._keepold_chunk_page_cpu_span_dst_starts_dev)
            self._keepold_chunk_page_span_cpu_copy_lengths_dev = (
                self._keepold_chunk_page_cpu_span_lengths_dev)
            self._keepold_span_reuse_old_pos_for_rope_dev = _ensure_i64_dev(
                rope_old_pos_cpu)
            t_idx_submit1 = time.perf_counter()
            prep_h2d_ms = float((t_idx_submit1 - t_idx_submit0) * 1000.0)
            idx_device_ms = float("nan")
            if idx_end_event is not None:
                idx_end_event.record(torch.cuda.current_stream(device))
                torch.cuda.synchronize(device)
                idx_device_ms = float(idx_start_event.elapsed_time(idx_end_event))
            submit_ms = float((t_idx_submit1 - t_idx_submit0) * 1000.0)
            self._keepold_chunk_page_span_index_h2d_submit_ms = submit_ms
            self._keepold_chunk_page_span_index_h2d_device_ms = float(idx_device_ms)
            self._accumulate_index_tensor_h2d_timing(
                submit_ms=submit_ms,
                device_ms=idx_device_ms,
            )
            self._keepold_chunk_page_shared_switch_reuse_idx_i64_dev = None
            self._keepold_chunk_page_shared_cpu_reuse_idx_i64_dev = None
            self._keepold_chunk_page_shared_old_slots_i64_dev = None
            self._keepold_chunk_page_shared_index_reason = "ok_span"
            self._keepold_chunk_page_shared_index_ready = True
            self._append_metrics({
                "event": "selective_keepold_chunk_page_index_prepare",
                "keepold_phase": self._keepold_phase,
                "prepare_count": int(
                    self._keepold_chunk_page_shared_index_prepare_count),
                "prepared_once": int(
                    self._keepold_chunk_page_shared_index_prepare_count == 1),
                "fast_ready": True,
                "fast_reason": "ok_span",
                "chunk_page_exec_granularity":
                self._keepold_chunk_page_exec_granularity,
                "gpu_span_count": int(n_gpu),
                "cpu_span_count": int(n_cpu),
                "gpu_span_tokens": int(gpu_len.sum().item()) if n_gpu > 0 else 0,
                "cpu_span_tokens": int(cpu_len.sum().item()) if n_cpu > 0 else 0,
                "span_coverage_checked":
                1 if self._keepold_chunk_page_span_coverage_checked else 0,
                "span_coverage_reason":
                self._keepold_chunk_page_span_coverage_reason,
                "span_plan_full_reuse_tokens":
                int(self._keepold_chunk_page_span_full_reuse_tokens),
                "span_plan_prefix_uncovered_tokens":
                int(self._keepold_chunk_page_span_prefix_uncovered_tokens),
                "span_index_h2d_submit_ms": float(
                    self._keepold_chunk_page_span_index_h2d_submit_ms),
                "span_index_h2d_device_ms": float(
                    self._keepold_chunk_page_span_index_h2d_device_ms),
                "timing_v2_index_tensor_h2d_submit_ms": float(
                    self._keepold_timing_v2_index_tensor_h2d_submit_ms),
                "timing_v2_index_tensor_h2d_device_ms": float(
                    self._keepold_timing_v2_index_tensor_h2d_device_ms),
                "gpu_index_prepare_total_ms": float(
                    (time.perf_counter() - prep_t0) * 1000.0),
                "gpu_index_prepare_old_slots_ms": float(prep_old_slots_ms),
                "gpu_index_prepare_dst_idx_ms": float(prep_dst_idx_ms),
                "gpu_index_prepare_positions_ms": float(prep_positions_ms),
                "gpu_index_prepare_h2d_ms": float(prep_h2d_ms),
                "gpu_index_prepare_cache_hit": int(prep_cache_hit),
                "gpu_index_prepare_cache_reason": str(prep_cache_reason),
                "ts_unix": time.time(),
            })
            return

        switch_src = self._keepold_switch_reuse_idx_dev
        if switch_src is None:
            prep_cache_reason = "missing_switch_reuse_idx_dev"
            self._keepold_chunk_page_shared_index_reason = (
                "missing_switch_reuse_idx_dev")
            return
        cpu_idx_src = self._keepold_cpu_reuse_idx_dev
        if cpu_idx_src is None:
            prep_cache_reason = "missing_cpu_reuse_idx_dev"
            self._keepold_chunk_page_shared_index_reason = (
                "missing_cpu_reuse_idx_dev")
            return

        def _ensure_device_i64_contig(t: torch.Tensor) -> torch.Tensor:
            out = t
            if out.device != device:
                out = out.to(device=device, non_blocking=False)
            if out.dtype != torch.long:
                out = out.to(dtype=torch.long)
            if not out.is_contiguous():
                out = out.contiguous()
            return out

        timing_cuda_event = bool(self._timing_cuda_event_enabled()
                                 and device.type == "cuda")
        idx_start_event: Optional[torch.cuda.Event] = None
        idx_end_event: Optional[torch.cuda.Event] = None
        if timing_cuda_event:
            idx_start_event = torch.cuda.Event(enable_timing=True)
            idx_end_event = torch.cuda.Event(enable_timing=True)
            idx_start_event.record(torch.cuda.current_stream(device))

        t_idx_submit0 = time.perf_counter()
        t_dst0 = time.perf_counter()
        switch_i64 = _ensure_device_i64_contig(switch_src)
        t_dst1 = time.perf_counter()
        prep_dst_idx_ms = float((t_dst1 - t_dst0) * 1000.0)
        t_pos0 = time.perf_counter()
        cpu_idx_i64 = _ensure_device_i64_contig(cpu_idx_src)
        t_pos1 = time.perf_counter()
        prep_positions_ms = float((t_pos1 - t_pos0) * 1000.0)
        t_old0 = time.perf_counter()
        old_slots_cpu = self._keepold_chunk_page_old_slots_cpu.to(
            device="cpu", dtype=torch.long).contiguous()
        t_old1 = time.perf_counter()
        prep_old_slots_ms = float((t_old1 - t_old0) * 1000.0)
        old_slots_len = int(old_slots_cpu.numel())
        switch_len = int(switch_i64.numel())
        old_slots_i64: Optional[torch.Tensor]
        if old_slots_len != switch_len:
            old_slots_i64 = None
        elif old_slots_len > 0:
            t_oldh2d0 = time.perf_counter()
            old_slots_i64 = old_slots_cpu.to(
                device=device, dtype=torch.long,
                non_blocking=False).contiguous()
            t_oldh2d1 = time.perf_counter()
            prep_h2d_ms += float((t_oldh2d1 - t_oldh2d0) * 1000.0)
        else:
            old_slots_i64 = torch.empty(0, dtype=torch.long, device=device)
        t_idx_submit1 = time.perf_counter()
        prep_h2d_ms += float((t_idx_submit1 - t_idx_submit0) * 1000.0)

        idx_device_ms = float("nan")
        if idx_end_event is not None:
            idx_end_event.record(torch.cuda.current_stream(device))
            torch.cuda.synchronize(device)
            idx_device_ms = float(idx_start_event.elapsed_time(idx_end_event))
        self._accumulate_index_tensor_h2d_timing(
            submit_ms=(t_idx_submit1 - t_idx_submit0) * 1000.0,
            device_ms=idx_device_ms,
        )

        self._keepold_chunk_page_shared_switch_reuse_idx_i64_dev = switch_i64
        self._keepold_chunk_page_shared_cpu_reuse_idx_i64_dev = cpu_idx_i64
        self._keepold_chunk_page_shared_old_slots_i64_dev = old_slots_i64
        if old_slots_i64 is None:
            prep_cache_hit = 0
            prep_cache_reason = "chunk_page_old_slots_dst_len_mismatch"
            self._keepold_chunk_page_shared_index_reason = (
                "chunk_page_old_slots_dst_len_mismatch")
            self._keepold_chunk_page_shared_index_ready = False
        else:
            prep_cache_hit = 0
            prep_cache_reason = "ok"
            self._keepold_chunk_page_shared_index_reason = "ok"
            self._keepold_chunk_page_shared_index_ready = True

        old_slots_tokens = int(old_slots_len)
        prep_total_ms = float((time.perf_counter() - prep_t0) * 1000.0)
        self._append_metrics({
            "event": "selective_keepold_chunk_page_index_prepare",
            "keepold_phase": self._keepold_phase,
            "prepare_count": int(
                self._keepold_chunk_page_shared_index_prepare_count),
            "prepared_once": int(
                self._keepold_chunk_page_shared_index_prepare_count == 1),
            "fast_ready": bool(self._keepold_chunk_page_shared_index_ready),
            "fast_reason": self._keepold_chunk_page_shared_index_reason,
            "switch_tokens": int(switch_len),
            "cpu_reuse_tokens": int(cpu_idx_i64.numel()),
            "old_slots_tokens": int(old_slots_tokens),
            "timing_v2_index_tensor_h2d_submit_ms": float(
                self._keepold_timing_v2_index_tensor_h2d_submit_ms),
            "timing_v2_index_tensor_h2d_device_ms": float(
                self._keepold_timing_v2_index_tensor_h2d_device_ms),
            "gpu_index_prepare_total_ms": float(prep_total_ms),
            "gpu_index_prepare_old_slots_ms": float(prep_old_slots_ms),
            "gpu_index_prepare_dst_idx_ms": float(prep_dst_idx_ms),
            "gpu_index_prepare_positions_ms": float(prep_positions_ms),
            "gpu_index_prepare_h2d_ms": float(prep_h2d_ms),
            "gpu_index_prepare_cache_hit": int(prep_cache_hit),
            "gpu_index_prepare_cache_reason": str(prep_cache_reason),
            "ts_unix": time.time(),
        })

    def _prepare_keepold_old_page_fastpath(self, *, device: torch.device) -> None:
        self._reset_keepold_old_page_fastpath(reset_timing_v2_index=False)
        if self._keepold_phase != _KEEPOLD_PHASE_REUSE:
            self._keepold_old_page_fast_reason = "phase_not_reuse"
            return
        if self.config.keepold_fused_k_mode != _KEEPOLD_FUSED_K_CUDA_OP:
            self._keepold_old_page_fast_reason = "fused_mode_not_cuda_op"
            return
        if self.config.dynamic_update_backend != _DYNAMIC_UPDATE_BACKEND_CPU_REBUILD:
            self._keepold_old_page_fast_reason = "backend_not_cpu_rebuild"
            return
        if self._chunk_page_exec_enabled():
            self._prepare_keepold_chunk_page_shared_index_tensors(device=device)
            if (self._keepold_chunk_page_exec_granularity
                    == _CHUNK_PAGE_EXEC_GRANULARITY_SPAN
                    and self._keepold_chunk_page_span_exec_ready):
                if not self._keepold_chunk_page_shared_index_ready:
                    self._keepold_old_page_fast_reason = (
                        self._keepold_chunk_page_shared_index_reason
                        or "chunk_page_span_index_prepare_failed")
                    return
                span_lengths = self._keepold_chunk_page_gpu_span_lengths_dev
                span_count = int(0 if span_lengths is None else span_lengths.numel())
                span_tokens = int(
                    0 if span_lengths is None or span_count == 0 else
                    span_lengths.sum().item())
                self._keepold_switch_reuse_idx_i64_dev = None
                self._keepold_gpu_index_idx_i64_dev = None
                self._keepold_old_slots_i64_by_layer = {}
                self._keepold_old_page_fast_ready = bool(span_count > 0 and span_tokens > 0)
                self._keepold_old_page_fast_reason = (
                    "chunk_page_span_exec_ok"
                    if self._keepold_old_page_fast_ready else
                    "chunk_page_exec_no_gpu_tokens")
                self._keepold_old_page_prepare_ms = 0.0
                self._append_metrics({
                    "event": "selective_keepold_old_page_prepare",
                    "keepold_phase": self._keepold_phase,
                    "chunk_page_exec_granularity":
                    self._keepold_chunk_page_exec_granularity,
                    "prepared_layers": int(len(self._src_layer_kv)),
                    "expected_layers": int(len(self._src_layer_kv)),
                    "switch_tokens": int(span_tokens),
                    "chunk_page_exec_enabled": 1,
                    "chunk_page_exec_gpu_tokens": int(span_tokens),
                    "chunk_page_exec_cpu_tokens":
                    int(self._keepold_chunk_page_exec_cpu_tokens),
                    "gpu_span_count": int(span_count),
                    "cpu_span_count": int(self._keepold_chunk_page_cpu_span_count),
                    "span_index_h2d_submit_ms": float(
                        self._keepold_chunk_page_span_index_h2d_submit_ms),
                    "span_index_h2d_device_ms": float(
                        self._keepold_chunk_page_span_index_h2d_device_ms),
                    "timing_v2_index_tensor_h2d_submit_ms": float(
                        self._keepold_timing_v2_index_tensor_h2d_submit_ms),
                    "timing_v2_index_tensor_h2d_device_ms": float(
                        self._keepold_timing_v2_index_tensor_h2d_device_ms),
                    "fast_ready": bool(self._keepold_old_page_fast_ready),
                    "fast_reason": self._keepold_old_page_fast_reason,
                    "ts_unix": time.time(),
                })
                return
            switch_reuse_idx_i64 = (
                self._keepold_chunk_page_shared_switch_reuse_idx_i64_dev)
            old_slots_i64 = self._keepold_chunk_page_shared_old_slots_i64_dev
            if switch_reuse_idx_i64 is None:
                self._keepold_old_page_fast_reason = (
                    self._keepold_chunk_page_shared_index_reason
                    or "missing_switch_reuse_idx_i64")
                return
            t0 = time.perf_counter()
            old_slots_len = int(0 if old_slots_i64 is None else old_slots_i64.numel())
            if not self._keepold_chunk_page_shared_index_ready:
                self._keepold_old_page_fast_reason = (
                    self._keepold_chunk_page_shared_index_reason
                    or "chunk_page_shared_index_prepare_failed")
                self._keepold_switch_reuse_idx_i64_dev = switch_reuse_idx_i64
                return
            self._keepold_switch_reuse_idx_i64_dev = switch_reuse_idx_i64
            self._keepold_gpu_index_idx_i64_dev = (
                self._keepold_chunk_page_shared_cpu_reuse_idx_i64_dev)
            if old_slots_len == 0:
                t1 = time.perf_counter()
                self._keepold_old_page_prepare_ms = (t1 - t0) * 1000.0
                self._keepold_old_page_fast_ready = False
                self._keepold_old_page_fast_reason = "chunk_page_exec_no_gpu_tokens"
                self._append_metrics({
                    "event": "selective_keepold_old_page_prepare",
                    "keepold_phase": self._keepold_phase,
                    "prepared_layers": 0,
                    "expected_layers": int(len(self._src_layer_kv)),
                    "switch_tokens": int(switch_reuse_idx_i64.numel()),
                    "chunk_page_exec_enabled": 1,
                    "chunk_page_exec_gpu_tokens": int(switch_reuse_idx_i64.numel()),
                    "chunk_page_exec_cpu_tokens":
                    int(self._keepold_cpu_reuse_idx_cpu.numel()),
                    "t_keepold_old_page_prepare_ms": float(
                        self._keepold_old_page_prepare_ms),
                    "timing_v2_index_tensor_h2d_submit_ms": float(
                        self._keepold_timing_v2_index_tensor_h2d_submit_ms),
                    "timing_v2_index_tensor_h2d_device_ms": float(
                        self._keepold_timing_v2_index_tensor_h2d_device_ms),
                    "chunk_page_shared_index_prepare_count": int(
                        self._keepold_chunk_page_shared_index_prepare_count),
                    "chunk_page_shared_index_prepared_once": int(
                        self._keepold_chunk_page_shared_index_prepare_count == 1),
                    "chunk_page_shared_index_ready": bool(
                        self._keepold_chunk_page_shared_index_ready),
                    "chunk_page_shared_index_reason":
                    self._keepold_chunk_page_shared_index_reason,
                    "chunk_page_shared_index_all_layers_share_old_slots": True,
                    "chunk_page_shared_index_unique_old_slots_ptr_count": 0,
                    "fast_ready": False,
                    "fast_reason": self._keepold_old_page_fast_reason,
                    "ts_unix": time.time(),
                })
                return
            assert old_slots_i64 is not None
            if int(old_slots_i64.min().item()) < 0:
                raise ValueError("chunk_page old_slots contains negative value")
            prepared = 0
            for lid in self._src_layer_kv.keys():
                self._keepold_old_slots_i64_by_layer[int(lid)] = old_slots_i64
                prepared += 1
            unique_old_slots_ptr_count = len(
                {
                    int(t.data_ptr())
                    for t in self._keepold_old_slots_i64_by_layer.values()
                    if int(t.numel()) > 0
                })
            t1 = time.perf_counter()
            self._keepold_old_page_prepare_ms = (t1 - t0) * 1000.0
            self._keepold_old_page_fast_ready = (prepared > 0)
            self._keepold_old_page_fast_reason = (
                "chunk_page_exec_ok"
                if self._keepold_old_page_fast_ready else "no_layer_snapshots")
            self._append_metrics({
                "event": "selective_keepold_old_page_prepare",
                "keepold_phase": self._keepold_phase,
                "prepared_layers": int(prepared),
                "expected_layers": int(len(self._src_layer_kv)),
                "switch_tokens": int(switch_reuse_idx_i64.numel()),
                "chunk_page_exec_enabled": 1,
                "chunk_page_exec_gpu_tokens": int(switch_reuse_idx_i64.numel()),
                "chunk_page_exec_cpu_tokens":
                int(self._keepold_cpu_reuse_idx_cpu.numel()),
                "t_keepold_old_page_prepare_ms": float(
                    self._keepold_old_page_prepare_ms),
                "timing_v2_index_tensor_h2d_submit_ms": float(
                    self._keepold_timing_v2_index_tensor_h2d_submit_ms),
                "timing_v2_index_tensor_h2d_device_ms": float(
                    self._keepold_timing_v2_index_tensor_h2d_device_ms),
                "chunk_page_shared_index_prepare_count": int(
                    self._keepold_chunk_page_shared_index_prepare_count),
                "chunk_page_shared_index_prepared_once": int(
                    self._keepold_chunk_page_shared_index_prepare_count == 1),
                "chunk_page_shared_index_ready": bool(
                    self._keepold_chunk_page_shared_index_ready),
                "chunk_page_shared_index_reason":
                self._keepold_chunk_page_shared_index_reason,
                "chunk_page_shared_index_all_layers_share_old_slots": int(
                    unique_old_slots_ptr_count <= 1),
                "chunk_page_shared_index_unique_old_slots_ptr_count": int(
                    unique_old_slots_ptr_count),
                "fast_ready": bool(self._keepold_old_page_fast_ready),
                "fast_reason": self._keepold_old_page_fast_reason,
                "ts_unix": time.time(),
            })
            return
        if self._keepold_switch_reuse_idx_dev is None:
            self._keepold_old_page_fast_reason = "missing_switch_reuse_idx_dev"
            return
        if self._keepold_switch_prompt_positions_dev is None:
            self._keepold_old_page_fast_reason = (
                "missing_switch_prompt_positions_dev")
            return
        if int(self._keepold_switch_reuse_idx_dev.numel()) == 0:
            self._keepold_old_page_fast_reason = "empty_switch_reuse_idx"
            return
        bridge = self._get_keepold_bridge(create=False)
        if bridge is None:
            self._keepold_old_page_fast_reason = "missing_keepold_bridge"
            return

        t0 = time.perf_counter()
        switch_prompt_pos = self._keepold_switch_prompt_positions_dev
        timing_cuda_event = (self._timing_cuda_event_enabled()
                             and device.type == "cuda")
        idx_start_event: Optional[torch.cuda.Event] = None
        idx_end_event: Optional[torch.cuda.Event] = None
        if timing_cuda_event:
            idx_start_event = torch.cuda.Event(enable_timing=True)
            idx_end_event = torch.cuda.Event(enable_timing=True)
            idx_start_event.record(torch.cuda.current_stream(device))
        t_idx_submit0 = time.perf_counter()
        switch_reuse_idx_i64 = self._keepold_switch_reuse_idx_dev.to(
            device=device, dtype=torch.long, non_blocking=False).contiguous()
        t_idx_submit1 = time.perf_counter()
        if idx_end_event is not None:
            idx_end_event.record(torch.cuda.current_stream(device))
            torch.cuda.synchronize(device)
            self._keepold_timing_v2_index_tensor_h2d_device_ms = float(
                idx_start_event.elapsed_time(idx_end_event))
        else:
            self._keepold_timing_v2_index_tensor_h2d_device_ms = float("nan")
        self._keepold_timing_v2_index_tensor_h2d_submit_ms = float(
            (t_idx_submit1 - t_idx_submit0) * 1000.0)
        max_switch_prompt = int(switch_prompt_pos.max().item())
        if int(switch_reuse_idx_i64.max().item()) >= int(
                self._reuse_positions_cpu.numel()):
            raise ValueError(
                "keepold switch reuse index out of range: "
                f"max_switch_reuse={int(switch_reuse_idx_i64.max().item())}, "
                f"reuse_tokens={int(self._reuse_positions_cpu.numel())}")
        layer_count = 0
        prepared = 0
        for lid in self._src_layer_kv.keys():
            layer_count += 1
            snapshot = bridge.layers.get(int(lid))
            if snapshot is None:
                continue
            slot_mapping = snapshot.slot_mapping
            if slot_mapping.device != device:
                slot_mapping = slot_mapping.to(device=device, non_blocking=False)
            if slot_mapping.dtype != torch.long:
                slot_mapping = slot_mapping.to(dtype=torch.long)
            if not slot_mapping.is_contiguous():
                slot_mapping = slot_mapping.contiguous()
            if int(slot_mapping.numel()) <= max_switch_prompt:
                raise ValueError(
                    "keepold switch prompt index out of range for "
                    f"layer={int(lid)}: max_switch={max_switch_prompt} "
                    f"slot_count={int(slot_mapping.numel())}")
            old_slots_i64 = slot_mapping.index_select(
                0, switch_prompt_pos).contiguous()
            if int(old_slots_i64.min().item()) < 0:
                raise ValueError(
                    "keepold old_slots contains negative value: "
                    f"layer={int(lid)}")
            self._keepold_old_slots_i64_by_layer[int(lid)] = old_slots_i64
            prepared += 1
        t1 = time.perf_counter()
        self._keepold_switch_reuse_idx_i64_dev = switch_reuse_idx_i64
        self._keepold_old_page_prepare_ms = (t1 - t0) * 1000.0
        self._keepold_old_page_fast_ready = (prepared > 0)
        if self._keepold_old_page_fast_ready:
            self._keepold_old_page_fast_reason = "ok"
        else:
            self._keepold_old_page_fast_reason = "no_layer_snapshots"
        self._append_metrics({
            "event": "selective_keepold_old_page_prepare",
            "keepold_phase": self._keepold_phase,
            "prepared_layers": int(prepared),
            "expected_layers": int(layer_count),
            "switch_tokens": int(switch_prompt_pos.numel()),
            "t_keepold_old_page_prepare_ms": float(
                self._keepold_old_page_prepare_ms),
            "timing_v2_index_tensor_h2d_submit_ms": float(
                self._keepold_timing_v2_index_tensor_h2d_submit_ms),
            "timing_v2_index_tensor_h2d_device_ms": float(
                self._keepold_timing_v2_index_tensor_h2d_device_ms),
            "fast_ready": bool(self._keepold_old_page_fast_ready),
            "fast_reason": self._keepold_old_page_fast_reason,
            "ts_unix": time.time(),
        })

    def _prepare_keepold_gpu_index_fastpath(self, *, device: torch.device) -> None:
        self._keepold_gpu_index_idx_i64_dev = None
        self._keepold_gpu_index_fast_ready = False
        self._keepold_gpu_index_fast_reason = "inactive"
        self._keepold_gpu_index_prepare_ms = 0.0
        if self._keepold_phase != _KEEPOLD_PHASE_REUSE:
            self._keepold_gpu_index_fast_reason = "phase_not_reuse"
            return
        if self.config.keepold_fused_k_mode != _KEEPOLD_FUSED_K_CUDA_OP:
            self._keepold_gpu_index_fast_reason = "fused_mode_not_cuda_op"
            return
        if self._chunk_page_exec_enabled():
            if (self._keepold_chunk_page_exec_granularity
                    == _CHUNK_PAGE_EXEC_GRANULARITY_SPAN
                    and self._keepold_chunk_page_span_exec_ready):
                self._prepare_keepold_chunk_page_shared_index_tensors(device=device)
                self._keepold_gpu_index_idx_i64_dev = None
                self._keepold_gpu_index_prepare_ms = 0.0
                self._keepold_gpu_index_fast_ready = bool(
                    self._keepold_chunk_page_shared_index_ready)
                self._keepold_gpu_index_fast_reason = (
                    "span_exec_not_needed"
                    if self._keepold_chunk_page_shared_index_ready else
                    self._keepold_chunk_page_shared_index_reason)
                self._append_metrics({
                    "event": "selective_keepold_gpu_index_prepare",
                    "keepold_phase": self._keepold_phase,
                    "chunk_page_exec_granularity":
                    self._keepold_chunk_page_exec_granularity,
                    "cpu_reuse_tokens": int(self._keepold_chunk_page_exec_cpu_tokens),
                    "gpu_span_count": int(self._keepold_chunk_page_gpu_span_count),
                    "cpu_span_count": int(self._keepold_chunk_page_cpu_span_count),
                    "t_keepold_gpu_index_prepare_ms": 0.0,
                    "timing_v2_index_tensor_h2d_submit_ms": float(
                        self._keepold_timing_v2_index_tensor_h2d_submit_ms),
                    "timing_v2_index_tensor_h2d_device_ms": float(
                        self._keepold_timing_v2_index_tensor_h2d_device_ms),
                    "fast_ready": bool(self._keepold_gpu_index_fast_ready),
                    "fast_reason": self._keepold_gpu_index_fast_reason,
                    "ts_unix": time.time(),
                })
                return
            t0 = time.perf_counter()
            self._prepare_keepold_chunk_page_shared_index_tensors(device=device)
            idx_i64 = self._keepold_chunk_page_shared_cpu_reuse_idx_i64_dev
            if idx_i64 is None:
                self._keepold_gpu_index_fast_reason = (
                    self._keepold_chunk_page_shared_index_reason
                    or "missing_shared_cpu_reuse_idx_i64")
                return
            if int(idx_i64.numel()) > 0:
                max_idx = int(idx_i64.max().item())
                if max_idx >= int(self._reuse_positions_cpu.numel()):
                    raise ValueError(
                        "keepold gpu_index idx out of range: "
                        f"max_idx={max_idx}, reuse_tokens={int(self._reuse_positions_cpu.numel())}"
                    )
            t1 = time.perf_counter()
            self._keepold_gpu_index_idx_i64_dev = idx_i64
            self._keepold_gpu_index_prepare_ms = (t1 - t0) * 1000.0
            self._keepold_gpu_index_fast_ready = True
            self._keepold_gpu_index_fast_reason = "chunk_page_exec_shared_ok"
            self._append_metrics({
                "event": "selective_keepold_gpu_index_prepare",
                "keepold_phase": self._keepold_phase,
                "cpu_reuse_tokens": int(idx_i64.numel()),
                "t_keepold_gpu_index_prepare_ms": float(
                    self._keepold_gpu_index_prepare_ms),
                "timing_v2_index_tensor_h2d_submit_ms": float(
                    self._keepold_timing_v2_index_tensor_h2d_submit_ms),
                "timing_v2_index_tensor_h2d_device_ms": float(
                    self._keepold_timing_v2_index_tensor_h2d_device_ms),
                "chunk_page_shared_index_prepare_count": int(
                    self._keepold_chunk_page_shared_index_prepare_count),
                "chunk_page_shared_index_prepared_once": int(
                    self._keepold_chunk_page_shared_index_prepare_count == 1),
                "chunk_page_shared_index_ready": bool(
                    self._keepold_chunk_page_shared_index_ready),
                "chunk_page_shared_index_reason":
                self._keepold_chunk_page_shared_index_reason,
                "fast_ready": bool(self._keepold_gpu_index_fast_ready),
                "fast_reason": self._keepold_gpu_index_fast_reason,
                "ts_unix": time.time(),
            })
            return
        if self._keepold_cpu_reuse_idx_dev is None:
            self._keepold_gpu_index_fast_reason = "missing_cpu_reuse_idx_dev"
            return
        t0 = time.perf_counter()
        timing_cuda_event = (self._timing_cuda_event_enabled()
                             and device.type == "cuda")
        idx_start_event: Optional[torch.cuda.Event] = None
        idx_end_event: Optional[torch.cuda.Event] = None
        if timing_cuda_event:
            idx_start_event = torch.cuda.Event(enable_timing=True)
            idx_end_event = torch.cuda.Event(enable_timing=True)
            idx_start_event.record(torch.cuda.current_stream(device))
        t_idx_submit0 = time.perf_counter()
        idx_i64 = self._keepold_cpu_reuse_idx_dev.to(
            device=device, dtype=torch.long, non_blocking=False).contiguous()
        t_idx_submit1 = time.perf_counter()
        device_ms = float("nan")
        if idx_end_event is not None:
            idx_end_event.record(torch.cuda.current_stream(device))
            torch.cuda.synchronize(device)
            device_ms = float(idx_start_event.elapsed_time(idx_end_event))
        if int(idx_i64.numel()) > 0:
            max_idx = int(idx_i64.max().item())
            if max_idx >= int(self._reuse_positions_cpu.numel()):
                raise ValueError(
                    "keepold gpu_index idx out of range: "
                    f"max_idx={max_idx}, reuse_tokens={int(self._reuse_positions_cpu.numel())}")
        t1 = time.perf_counter()
        self._accumulate_index_tensor_h2d_timing(
            submit_ms=(t_idx_submit1 - t_idx_submit0) * 1000.0,
            device_ms=device_ms,
        )
        self._keepold_gpu_index_idx_i64_dev = idx_i64
        self._keepold_gpu_index_prepare_ms = (t1 - t0) * 1000.0
        self._keepold_gpu_index_fast_ready = True
        self._keepold_gpu_index_fast_reason = "ok"
        self._append_metrics({
            "event": "selective_keepold_gpu_index_prepare",
            "keepold_phase": self._keepold_phase,
            "cpu_reuse_tokens": int(idx_i64.numel()),
            "t_keepold_gpu_index_prepare_ms": float(
                self._keepold_gpu_index_prepare_ms),
            "timing_v2_index_tensor_h2d_submit_ms": float(
                self._keepold_timing_v2_index_tensor_h2d_submit_ms),
            "timing_v2_index_tensor_h2d_device_ms": float(
                self._keepold_timing_v2_index_tensor_h2d_device_ms),
            "fast_ready": bool(self._keepold_gpu_index_fast_ready),
            "fast_reason": self._keepold_gpu_index_fast_reason,
            "ts_unix": time.time(),
        })

    @staticmethod
    def _fused_scatter_rows_to_reuse_k(
        *,
        src_rows: torch.Tensor,
        dst_reuse_idx: torch.Tensor,
        reuse_k: torch.Tensor,
        mode: str,
    ) -> dict[str, Any]:
        return fused_scatter_rows_to_reuse_k(
            src_rows=src_rows,
            dst_reuse_idx=dst_reuse_idx,
            reuse_k=reuse_k,
            mode=mode,
        )

    @staticmethod
    def _fused_scatter_rows_to_reuse_kv(
        *,
        src_k_rows: torch.Tensor,
        src_v_rows: torch.Tensor,
        dst_reuse_idx: torch.Tensor,
        reuse_k: torch.Tensor,
        reuse_v: torch.Tensor,
        mode: str,
    ) -> dict[str, Any]:
        return fused_scatter_rows_to_reuse_kv(
            src_k_rows=src_k_rows,
            src_v_rows=src_v_rows,
            dst_reuse_idx=dst_reuse_idx,
            reuse_k=reuse_k,
            reuse_v=reuse_v,
            mode=mode,
        )

    @staticmethod
    def _fused_scatter_rows_to_reuse_kv_fast(
        *,
        src_k_rows: torch.Tensor,
        src_v_rows: torch.Tensor,
        dst_idx_i64: torch.Tensor,
        reuse_k: torch.Tensor,
        reuse_v: torch.Tensor,
    ) -> dict[str, Any]:
        return fused_scatter_rows_to_reuse_kv_cuda_op_fast(
            src_k_rows=src_k_rows,
            src_v_rows=src_v_rows,
            dst_idx_i64=dst_idx_i64,
            reuse_k=reuse_k,
            reuse_v=reuse_v,
        )

    def _build_keepold_partition_cpu(self) -> None:
        self._keepold_switch_reuse_idx_cpu = torch.empty(0, dtype=torch.long)
        self._keepold_cpu_reuse_idx_cpu = torch.empty(0, dtype=torch.long)
        if self._keepold_phase != _KEEPOLD_PHASE_REUSE:
            return
        switch_pos_cpu = self._keepold_switch_prompt_positions_cpu
        if switch_pos_cpu.numel() == 0:
            return
        total_reuse = int(self._reuse_positions_cpu.numel())
        reuse_pos_to_idx = torch.full(
            (self.config.prompt_len, ),
            -1,
            dtype=torch.long,
        )
        if total_reuse > 0:
            reuse_pos_to_idx[self._reuse_positions_cpu] = torch.arange(
                total_reuse, dtype=torch.long)
        reuse_idx = reuse_pos_to_idx.index_select(0, switch_pos_cpu)
        valid_mask = reuse_idx >= 0
        switch_valid = switch_pos_cpu[valid_mask].contiguous()
        switch_reuse_idx = reuse_idx[valid_mask].contiguous()
        self._keepold_switch_prompt_positions_cpu = switch_valid
        self._keepold_switch_reuse_idx_cpu = switch_reuse_idx
        if switch_reuse_idx.numel() == 0:
            return
        keep_mask = torch.ones(total_reuse, dtype=torch.bool)
        keep_mask[switch_reuse_idx] = False
        self._keepold_cpu_reuse_idx_cpu = torch.arange(
            total_reuse, dtype=torch.long)[keep_mask].contiguous()

    def _reset_keepold_cpu_fallback_span_direct(self) -> None:
        self._keepold_cpu_fallback_span_direct_ready = False
        self._keepold_cpu_fallback_span_direct_reason = "inactive"
        self._keepold_cpu_fallback_span_direct_span_count = 0
        self._keepold_cpu_fallback_span_direct_tokens = 0
        self._keepold_cpu_fallback_span_direct_build_ms = 0.0
        self._keepold_cpu_fallback_span_direct_src_starts_cpu = torch.empty(
            0, dtype=torch.long)
        self._keepold_cpu_fallback_span_direct_dst_starts_cpu = torch.empty(
            0, dtype=torch.long)
        self._keepold_cpu_fallback_span_direct_lengths_cpu = torch.empty(
            0, dtype=torch.long)

    def _cpu_fallback_subset_mode_code(self) -> float:
        mode = str(self._keepold_cpu_fallback_mode_effective).strip().lower()
        if mode == _KEEPOLD_CPU_FALLBACK_MODE_SUBSET:
            return 0.0
        if mode == _KEEPOLD_CPU_FALLBACK_MODE_DENSE:
            return 1.0
        if mode == _KEEPOLD_CPU_FALLBACK_MODE_AUTO:
            return 2.0
        if mode == _KEEPOLD_CPU_FALLBACK_MODE_SUBSET_SPAN_DIRECT:
            return 3.0
        if mode == "span":
            return 4.0
        return -1.0

    def _prepare_keepold_cpu_fallback_span_direct(self) -> None:
        t0 = time.perf_counter()
        self._reset_keepold_cpu_fallback_span_direct()
        if self._keepold_phase != _KEEPOLD_PHASE_REUSE:
            self._keepold_cpu_fallback_span_direct_reason = "phase_not_reuse"
            return
        if self.config.dynamic_update_backend != _DYNAMIC_UPDATE_BACKEND_CPU_REBUILD:
            self._keepold_cpu_fallback_span_direct_reason = "backend_not_cpu_rebuild"
            return
        if not self._chunk_page_exec_enabled():
            self._keepold_cpu_fallback_span_direct_reason = "chunk_page_exec_disabled"
            return
        if (self._keepold_chunk_page_exec_granularity
                != _CHUNK_PAGE_EXEC_GRANULARITY_TOKEN):
            self._keepold_cpu_fallback_span_direct_reason = (
                "chunk_page_exec_not_token")
            return
        cpu_idx = self._keepold_cpu_reuse_idx_cpu
        if cpu_idx is None or int(cpu_idx.numel()) == 0:
            self._keepold_cpu_fallback_span_direct_reason = "no_cpu_fallback_tokens"
            return
        cpu_idx = cpu_idx.to(device="cpu", dtype=torch.long).contiguous()
        total_reuse = int(self._reuse_positions_cpu.numel())
        if total_reuse <= 0:
            self._keepold_cpu_fallback_span_direct_reason = "empty_reuse_positions"
            return
        min_idx = int(cpu_idx.min().item())
        max_idx = int(cpu_idx.max().item())
        if min_idx < 0 or max_idx >= total_reuse:
            self._keepold_cpu_fallback_span_direct_reason = "cpu_idx_out_of_range"
            return
        if int(torch.unique(cpu_idx).numel()) != int(cpu_idx.numel()):
            self._keepold_cpu_fallback_span_direct_reason = "cpu_idx_duplicate"
            return
        if int(cpu_idx.numel()) > 1:
            step = cpu_idx[1:] - cpu_idx[:-1]
            if bool((step < 1).any().item()):
                self._keepold_cpu_fallback_span_direct_reason = "cpu_idx_not_sorted"
                return

        switch_idx = self._keepold_switch_reuse_idx_cpu
        if switch_idx is not None and int(switch_idx.numel()) > 0:
            switch_idx = switch_idx.to(device="cpu",
                                       dtype=torch.long).contiguous()
            if int(switch_idx.min().item()) < 0 or int(
                    switch_idx.max().item()) >= total_reuse:
                self._keepold_cpu_fallback_span_direct_reason = (
                    "switch_idx_out_of_range")
                return
            occ = torch.zeros(total_reuse, dtype=torch.bool)
            occ.index_fill_(0, switch_idx, True)
            if bool(occ.index_select(0, cpu_idx).any().item()):
                self._keepold_cpu_fallback_span_direct_reason = "cpu_gpu_rows_overlap"
                return

        # Span-direct requires full reuse-order source rows in each layer.
        for lid, (src_k_raw, src_v_raw) in self._src_layer_kv.items():
            src_tokens_k = int(src_k_raw.shape[0])
            src_tokens_v = int(src_v_raw.shape[0])
            if src_tokens_k != src_tokens_v:
                self._keepold_cpu_fallback_span_direct_reason = (
                    f"kv_token_mismatch_layer_{int(lid)}")
                return
            if src_tokens_k != total_reuse:
                self._keepold_cpu_fallback_span_direct_reason = (
                    f"source_not_full_reuse_rows_layer_{int(lid)}")
                return

        n = int(cpu_idx.numel())
        if n <= 0:
            self._keepold_cpu_fallback_span_direct_reason = "no_cpu_fallback_tokens"
            return
        if n == 1:
            src_starts = cpu_idx.clone()
            dst_starts = cpu_idx.clone()
            lengths = torch.ones(1, dtype=torch.long)
        else:
            split_points = torch.nonzero(
                (cpu_idx[1:] - cpu_idx[:-1]) != 1,
                as_tuple=False,
            ).flatten() + 1
            starts_idx = torch.cat([
                torch.tensor([0], dtype=torch.long), split_points
            ]).contiguous()
            ends_idx = torch.cat([
                split_points,
                torch.tensor([n], dtype=torch.long),
            ]).contiguous()
            src_starts = cpu_idx.index_select(0, starts_idx).contiguous()
            dst_starts = src_starts.clone().contiguous()
            lengths = (ends_idx - starts_idx).to(dtype=torch.long).contiguous()

        covered = int(lengths.sum().item()) if int(lengths.numel()) > 0 else 0
        if covered != n:
            self._keepold_cpu_fallback_span_direct_reason = (
                "span_cover_mismatch")
            return

        if self.config.keepold_chunk_page_debug_checks and int(lengths.numel()) > 0:
            rebuilt_parts: list[torch.Tensor] = []
            for i in range(int(lengths.numel())):
                seg_len = int(lengths[i].item())
                seg_start = int(src_starts[i].item())
                rebuilt_parts.append(
                    (seg_start + torch.arange(seg_len, dtype=torch.long)))
            rebuilt = torch.cat(rebuilt_parts, dim=0).contiguous()
            if not torch.equal(rebuilt, cpu_idx):
                self._keepold_cpu_fallback_span_direct_reason = (
                    "debug_rebuild_mismatch")
                return

        self._keepold_cpu_fallback_span_direct_src_starts_cpu = src_starts
        self._keepold_cpu_fallback_span_direct_dst_starts_cpu = dst_starts
        self._keepold_cpu_fallback_span_direct_lengths_cpu = lengths
        self._keepold_cpu_fallback_span_direct_span_count = int(lengths.numel())
        self._keepold_cpu_fallback_span_direct_tokens = int(covered)
        self._keepold_cpu_fallback_span_direct_build_ms = float(
            (time.perf_counter() - t0) * 1000.0)
        self._keepold_cpu_fallback_span_direct_reason = "ok"
        self._keepold_cpu_fallback_span_direct_ready = True

    def _prepare_keepold_cpu_subsets(self) -> None:
        self._keepold_cpu_subset_layers = {}
        self._keepold_cpu_subset_total_prepare_ms = 0.0
        self._keepold_cpu_subset_total_prepare_envelope_ms = 0.0
        self._keepold_cpu_subset_total_pin_ms = 0.0
        self._keepold_cpu_subset_layer_loop_ms = 0.0
        self._keepold_cpu_subset_index_select_ms = 0.0
        self._keepold_cpu_subset_contiguous_ms = 0.0
        self._keepold_cpu_subset_store_ms = 0.0
        self._keepold_cpu_subset_num_layers = 0
        self._keepold_cpu_subset_tokens = 0
        self._keepold_cpu_subset_ready = False
        self._keepold_cpu_fallback_mode_effective = (
            _KEEPOLD_CPU_FALLBACK_MODE_SUBSET)
        self._keepold_cpu_fallback_dense_enabled = False
        self._keepold_cpu_fallback_dense_reason = "inactive"
        self._keepold_cpu_fallback_tokens = 0
        self._keepold_cpu_fallback_ratio = 0.0
        self._keepold_cpu_fallback_subset_skipped = False
        self._reset_keepold_cpu_fallback_span_direct()

        def _emit_prepare_event(*, layers_prepared: int,
                                switch_tokens: int) -> None:
            self._append_metrics({
                "event": "selective_keepold_cpu_subset_prepare",
                "layers_prepared": int(layers_prepared),
                "cpu_reuse_tokens": int(self._keepold_cpu_fallback_tokens),
                "switch_reuse_tokens": int(switch_tokens),
                "t_cpu_subset_prepare_total_ms": float(
                    self._keepold_cpu_subset_total_prepare_ms),
                "t_pin_memory_total_ms": float(self._keepold_cpu_subset_total_pin_ms),
                "t_cpu_subset_prepare_mean_ms": float(
                    self._keepold_cpu_subset_total_prepare_ms
                    / max(1, int(layers_prepared))),
                "t_pin_memory_mean_ms": float(
                    self._keepold_cpu_subset_total_pin_ms
                    / max(1, int(layers_prepared))),
                "timing_v2_cpu_subset_select_ms": float(
                    self._keepold_cpu_subset_total_prepare_ms),
                "timing_v2_cpu_subset_pin_ms": float(
                    self._keepold_cpu_subset_total_pin_ms),
                "cpu_fallback_mode":
                self._keepold_cpu_fallback_mode_effective,
                "cpu_fallback_dense_enabled":
                1 if self._keepold_cpu_fallback_dense_enabled else 0,
                "cpu_fallback_dense_reason":
                self._keepold_cpu_fallback_dense_reason,
                "cpu_fallback_tokens": int(self._keepold_cpu_fallback_tokens),
                "cpu_fallback_ratio": float(self._keepold_cpu_fallback_ratio),
                "cpu_fallback_subset_skipped":
                1 if self._keepold_cpu_fallback_subset_skipped else 0,
                "cpu_fallback_auto_threshold":
                float(_KEEPOLD_CPU_FALLBACK_AUTO_THRESHOLD),
                "cpu_fallback_subset_total_prepare_ms": float(
                    self._keepold_cpu_subset_total_prepare_envelope_ms),
                "cpu_fallback_subset_layer_loop_ms": float(
                    self._keepold_cpu_subset_layer_loop_ms),
                "cpu_fallback_subset_index_select_ms": float(
                    self._keepold_cpu_subset_index_select_ms),
                "cpu_fallback_subset_contiguous_ms": float(
                    self._keepold_cpu_subset_contiguous_ms),
                "cpu_fallback_subset_pin_ms": float(
                    self._keepold_cpu_subset_total_pin_ms),
                "cpu_fallback_subset_store_ms": float(
                    self._keepold_cpu_subset_store_ms),
                "cpu_fallback_subset_num_layers": int(
                    self._keepold_cpu_subset_num_layers),
                "cpu_fallback_subset_tokens": int(
                    self._keepold_cpu_subset_tokens),
                "cpu_fallback_subset_mode":
                self._keepold_cpu_fallback_mode_effective,
                "cpu_fallback_subset_mode_code": float(
                    self._cpu_fallback_subset_mode_code()),
                "cpu_fallback_span_direct_ready":
                1 if self._keepold_cpu_fallback_span_direct_ready else 0,
                "cpu_fallback_span_direct_reason":
                self._keepold_cpu_fallback_span_direct_reason,
                "cpu_fallback_span_direct_span_count": int(
                    self._keepold_cpu_fallback_span_direct_span_count),
                "cpu_fallback_span_direct_tokens": int(
                    self._keepold_cpu_fallback_span_direct_tokens),
                "cpu_fallback_span_direct_build_ms": float(
                    self._keepold_cpu_fallback_span_direct_build_ms),
                "ts_unix": time.time(),
            })

        if (self._keepold_phase != _KEEPOLD_PHASE_REUSE
                or self.config.dynamic_update_backend
                != _DYNAMIC_UPDATE_BACKEND_CPU_REBUILD):
            return

        # Span-level chunk-page execution has its own CPU fallback span copy.
        # It must not materialize token-level CPU subsets here.
        #
        # Important: this function can be called before build_span_reuse_plan()
        # has set span_exec_ready=True.  Therefore do NOT depend on
        # _keepold_chunk_page_span_exec_ready here.  If the user requested
        # span granularity for chunk_page execution in reuse phase, skip legacy
        # token-level subset preparation unconditionally.
        if (self._keepold_chunk_page_exec_granularity
                == _CHUNK_PAGE_EXEC_GRANULARITY_SPAN
                and self._chunk_page_exec_enabled()):
            total_reuse = int(self._reuse_positions_cpu.numel())
            cpu_tokens = int(self._keepold_chunk_page_exec_cpu_tokens)
            gpu_tokens = int(self._keepold_chunk_page_exec_gpu_tokens)
            denom = max(1, total_reuse)
            self._keepold_cpu_fallback_mode_effective = "span"
            self._keepold_cpu_fallback_dense_enabled = False
            self._keepold_cpu_fallback_dense_reason = "span_exec_no_subset_early"
            self._keepold_cpu_fallback_tokens = cpu_tokens
            self._keepold_cpu_fallback_ratio = float(cpu_tokens) / float(denom)
            self._keepold_cpu_fallback_subset_skipped = True
            self._keepold_cpu_subset_layers = {}
            self._keepold_cpu_subset_ready = True
            self._keepold_cpu_subset_total_prepare_ms = 0.0
            self._keepold_cpu_subset_total_prepare_envelope_ms = 0.0
            self._keepold_cpu_subset_total_pin_ms = 0.0
            _emit_prepare_event(
                layers_prepared=0,
                switch_tokens=int(gpu_tokens),
            )
            return

        if self._keepold_switch_reuse_idx_cpu.numel() == 0:
            return

        switch_tokens = int(self._keepold_switch_reuse_idx_cpu.numel())
        cpu_reuse_idx = self._keepold_cpu_reuse_idx_cpu
        total_reuse = int(self._reuse_positions_cpu.numel())
        cpu_fallback_tokens = int(cpu_reuse_idx.numel())
        fallback_ratio = (
            float(cpu_fallback_tokens) / float(total_reuse)
            if total_reuse > 0 else 0.0)
        self._keepold_cpu_fallback_tokens = int(cpu_fallback_tokens)
        self._keepold_cpu_fallback_ratio = float(fallback_ratio)

        if (self._chunk_page_exec_enabled()
                and self._keepold_chunk_page_exec_granularity
                == _CHUNK_PAGE_EXEC_GRANULARITY_SPAN
                and self._keepold_chunk_page_span_exec_ready):
            self._keepold_cpu_fallback_mode_effective = (
                _KEEPOLD_CPU_FALLBACK_MODE_SUBSET)
            self._keepold_cpu_fallback_dense_enabled = False
            self._keepold_cpu_fallback_dense_reason = "span_exec_direct_copy"
            self._keepold_cpu_fallback_subset_skipped = True
            self._keepold_cpu_subset_ready = False
            _emit_prepare_event(layers_prepared=0, switch_tokens=switch_tokens)
            return

        mode_cfg = _normalize_keepold_cpu_fallback_mode(
            getattr(self.config, "keepold_cpu_fallback_mode",
                    _KEEPOLD_CPU_FALLBACK_MODE_SUBSET))
        dense_enabled = False
        dense_reason = "subset_mode"
        if mode_cfg == _KEEPOLD_CPU_FALLBACK_MODE_DENSE:
            dense_enabled = True
            dense_reason = "forced_dense"
        elif mode_cfg == _KEEPOLD_CPU_FALLBACK_MODE_AUTO:
            if (cpu_fallback_tokens > 0
                    and fallback_ratio >= _KEEPOLD_CPU_FALLBACK_AUTO_THRESHOLD):
                dense_enabled = True
                dense_reason = (
                    f"auto_ratio_ge_{_KEEPOLD_CPU_FALLBACK_AUTO_THRESHOLD:.2f}")
            else:
                dense_reason = (
                    f"auto_ratio_lt_{_KEEPOLD_CPU_FALLBACK_AUTO_THRESHOLD:.2f}")
        self._keepold_cpu_fallback_mode_effective = mode_cfg
        self._keepold_cpu_fallback_dense_enabled = bool(dense_enabled)
        self._keepold_cpu_fallback_dense_reason = str(dense_reason)

        if cpu_reuse_idx.numel() == 0:
            self._keepold_cpu_fallback_subset_skipped = True
            self._keepold_cpu_subset_ready = True
            self._keepold_cpu_fallback_dense_reason = "no_cpu_fallback_tokens"
            _emit_prepare_event(layers_prepared=0, switch_tokens=switch_tokens)
            return
        if dense_enabled:
            self._keepold_cpu_fallback_subset_skipped = True
            self._keepold_cpu_subset_ready = False
            _emit_prepare_event(layers_prepared=0, switch_tokens=switch_tokens)
            return
        self._prepare_keepold_cpu_fallback_span_direct()
        if self._keepold_cpu_fallback_span_direct_ready:
            self._keepold_cpu_fallback_mode_effective = (
                _KEEPOLD_CPU_FALLBACK_MODE_SUBSET_SPAN_DIRECT)
            self._keepold_cpu_fallback_subset_skipped = True
            self._keepold_cpu_subset_ready = False
            _emit_prepare_event(layers_prepared=0, switch_tokens=switch_tokens)
            return
        self._keepold_cpu_fallback_subset_skipped = False
        idx_old_subset = self._reuse_old_positions_cpu.index_select(
            0, cpu_reuse_idx).contiguous()
        self._keepold_cpu_subset_tokens = int(cpu_fallback_tokens)
        t_subset_total0 = time.perf_counter()

        for lid, (src_k_raw, src_v_raw) in self._src_layer_kv.items():
            t_layer0 = time.perf_counter()
            src_k = src_k_raw.detach()
            src_v = src_v_raw.detach()
            src_tokens = int(src_k.shape[0])
            if int(src_v.shape[0]) != src_tokens:
                raise RuntimeError(
                    "keepold CPU subset source K/V token mismatch: "
                    f"layer={lid}, k={src_tokens}, v={int(src_v.shape[0])}")
            if src_tokens == total_reuse:
                src_idx = cpu_reuse_idx
            else:
                if idx_old_subset.numel() > 0:
                    max_old = int(idx_old_subset.max().item())
                    if max_old >= src_tokens:
                        raise ValueError(
                            "keepold CPU subset index out of range: "
                            f"layer={lid}, max_old={max_old}, src_tokens={src_tokens}"
                        )
                src_idx = idx_old_subset

            t_sel0 = time.perf_counter()
            k_subset = src_k.index_select(0, src_idx)
            v_subset = src_v.index_select(0, src_idx)
            t_sel1 = time.perf_counter()
            t_select_ms = (t_sel1 - t_sel0) * 1000.0

            t_contig0 = time.perf_counter()
            k_subset = k_subset.contiguous()
            v_subset = v_subset.contiguous()
            t_contig1 = time.perf_counter()
            t_contiguous_ms = (t_contig1 - t_contig0) * 1000.0
            t_prepare_ms = t_select_ms + t_contiguous_ms

            t_pin0 = time.perf_counter()
            if torch.cuda.is_available():
                if not k_subset.is_pinned():
                    k_subset = k_subset.pin_memory()
                if not v_subset.is_pinned():
                    v_subset = v_subset.pin_memory()
            t_pin1 = time.perf_counter()
            t_pin_ms = (t_pin1 - t_pin0) * 1000.0

            self._keepold_cpu_subset_total_prepare_ms += t_prepare_ms
            self._keepold_cpu_subset_total_pin_ms += t_pin_ms
            self._keepold_cpu_subset_index_select_ms += t_select_ms
            self._keepold_cpu_subset_contiguous_ms += t_contiguous_ms
            t_store0 = time.perf_counter()
            self._keepold_cpu_subset_layers[int(lid)] = _KeepOldCpuSubsetLayer(
                k_cpu=k_subset,
                v_cpu=v_subset,
                t_subset_prepare_ms=t_prepare_ms,
                t_pin_memory_ms=t_pin_ms,
            )
            t_store1 = time.perf_counter()
            self._keepold_cpu_subset_store_ms += (t_store1 - t_store0) * 1000.0
            self._keepold_cpu_subset_layer_loop_ms += (
                time.perf_counter() - t_layer0) * 1000.0

        t_subset_total1 = time.perf_counter()
        self._keepold_cpu_subset_total_prepare_envelope_ms = float(
            (t_subset_total1 - t_subset_total0) * 1000.0)
        self._keepold_cpu_subset_num_layers = int(len(self._keepold_cpu_subset_layers))

        self._keepold_cpu_subset_ready = True
        _emit_prepare_event(
            layers_prepared=int(len(self._keepold_cpu_subset_layers)),
            switch_tokens=switch_tokens,
        )

    def _load_source_once(self) -> None:
        if self._source_loaded:
            return
        t_source0 = time.perf_counter()
        t_cpu_prepare_ms = 0.0
        t_cpu_contiguous_ms = 0.0
        t_pin_memory_ms = 0.0
        pin_attempted = bool(torch.cuda.is_available())
        pre_pinned_k_layers = 0
        pre_pinned_v_layers = 0
        newly_pinned_k_layers = 0
        newly_pinned_v_layers = 0
        cpu_moved_k_layers = 0
        cpu_moved_v_layers = 0
        contiguous_fixed_k_layers = 0
        contiguous_fixed_v_layers = 0
        if self.config.layer_kv_data is not None:
            layer_kv = self.config.layer_kv_data
            if not isinstance(layer_kv, dict):
                raise ValueError("layer_kv_data must be a dict")
        else:
            if self.config.pt_path is None:
                raise RuntimeError(
                    "Selective runtime requires pt_path or layer_kv_data.")
            if not self.config.pt_path.exists():
                raise FileNotFoundError(
                    f"Selective runtime pt_path not found: {self.config.pt_path}")

            obj = torch.load(self.config.pt_path, map_location="cpu")
            if not isinstance(obj, dict) or "layer_kv" not in obj:
                raise ValueError(
                    f"Invalid selective layerwise KV file: {self.config.pt_path}")
            layer_kv = obj["layer_kv"]
            if not isinstance(layer_kv, dict):
                raise ValueError("'layer_kv' must be a dict in selective KV file")

        layer_items: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        for lid_raw, entry in layer_kv.items():
            lid = int(lid_raw)
            if not isinstance(entry, dict) or "k" not in entry or "v" not in entry:
                raise ValueError(
                    f"Invalid layer entry in selective KV file: layer={lid}")
            src_k = entry["k"]
            src_v = entry["v"]
            if (not isinstance(src_k, torch.Tensor)
                    or not isinstance(src_v, torch.Tensor)):
                raise ValueError(
                    f"Layer {lid} has non-tensor K/V in selective KV file")

            if src_k.device.type == "cpu" and src_k.is_pinned():
                pre_pinned_k_layers += 1
            if src_v.device.type == "cpu" and src_v.is_pinned():
                pre_pinned_v_layers += 1

            t_prepare0 = time.perf_counter()
            if src_k.requires_grad:
                src_k = src_k.detach()
            if src_k.device.type != "cpu":
                src_k = src_k.to(device="cpu")
                cpu_moved_k_layers += 1
            if not src_k.is_contiguous():
                t_cont0 = time.perf_counter()
                src_k = src_k.contiguous()
                t_cont1 = time.perf_counter()
                t_cpu_contiguous_ms += (t_cont1 - t_cont0) * 1000.0
                contiguous_fixed_k_layers += 1

            if src_v.requires_grad:
                src_v = src_v.detach()
            if src_v.device.type != "cpu":
                src_v = src_v.to(device="cpu")
                cpu_moved_v_layers += 1
            if not src_v.is_contiguous():
                t_cont0 = time.perf_counter()
                src_v = src_v.contiguous()
                t_cont1 = time.perf_counter()
                t_cpu_contiguous_ms += (t_cont1 - t_cont0) * 1000.0
                contiguous_fixed_v_layers += 1
            t_prepare1 = time.perf_counter()
            t_cpu_prepare_ms += (t_prepare1 - t_prepare0) * 1000.0
            if pin_attempted:
                t_pin0 = time.perf_counter()
                if not src_k.is_pinned():
                    src_k = src_k.pin_memory()
                    newly_pinned_k_layers += 1
                if not src_v.is_pinned():
                    src_v = src_v.pin_memory()
                    newly_pinned_v_layers += 1
                t_pin1 = time.perf_counter()
                t_pin_memory_ms += (t_pin1 - t_pin0) * 1000.0
            if src_k.shape[0] != src_v.shape[0]:
                raise ValueError(f"Layer {lid} K/V token mismatch")
            layer_items[lid] = (src_k, src_v)

        if not layer_items:
            raise ValueError(
                f"No layer entries found in selective KV file: {self.config.pt_path}"
            )

        recompute_pos = torch.tensor(
            self.config.recompute_positions, dtype=torch.long).contiguous()
        recompute_pos = torch.unique(recompute_pos, sorted=True)
        if recompute_pos.numel() == 0:
            raise ValueError("recompute_positions is empty")
        if recompute_pos.min().item() < 0:
            raise ValueError("recompute_positions contains negative index")
        if recompute_pos.max().item() >= self.config.prompt_len:
            raise ValueError(
                "recompute_positions out of range for prompt_len "
                f"{self.config.prompt_len}")

        full_pos = torch.arange(self.config.prompt_len, dtype=torch.long)
        keep_mask = torch.ones(self.config.prompt_len, dtype=torch.bool)
        keep_mask[recompute_pos] = False
        reuse_pos = full_pos[keep_mask].contiguous()
        if self.config.reuse_old_positions is None:
            reuse_old_pos = torch.arange(
                int(reuse_pos.numel()), dtype=torch.long).contiguous()
        else:
            reuse_old_pos = torch.tensor(
                self.config.reuse_old_positions, dtype=torch.long).contiguous()
            if int(reuse_old_pos.numel()) != int(reuse_pos.numel()):
                raise ValueError(
                    "reuse_old_positions length mismatch: "
                    f"reuse_old_positions={int(reuse_old_pos.numel())} "
                    f"reuse_tokens={int(reuse_pos.numel())}")
            if reuse_old_pos.numel() > 0 and int(reuse_old_pos.min().item()) < 0:
                raise ValueError("reuse_old_positions contains negative index")

        # Validate layer token count against reuse token count.
        expected = int(reuse_pos.numel())
        for lid, (k_cpu, v_cpu) in layer_items.items():
            if int(k_cpu.shape[0]) != expected:
                raise ValueError(
                    "Selective KV token count mismatch: "
                    f"layer={lid} kv_tokens={int(k_cpu.shape[0])} "
                    f"reuse_tokens={expected}")
            if int(v_cpu.shape[0]) != expected:
                raise ValueError(
                    "Selective KV value token count mismatch: "
                    f"layer={lid} kv_tokens={int(v_cpu.shape[0])} "
                    f"reuse_tokens={expected}")

        self._src_layer_kv = layer_items
        self._src_layer_kv_dynamic = {}
        self._dynamic_src_ready = False
        self._recompute_positions_cpu = recompute_pos
        self._reuse_positions_cpu = reuse_pos
        self._reuse_old_positions_cpu = reuse_old_pos
        self._base_reuse_positions_cpu = reuse_pos.clone().contiguous()
        self._base_reuse_old_positions_cpu = reuse_old_pos.clone().contiguous()
        reuse_pos_to_old = torch.full(
            (self.config.prompt_len, ),
            -1,
            dtype=torch.long,
        )
        if reuse_pos.numel() > 0:
            reuse_pos_to_old[reuse_pos] = reuse_old_pos
        self._reuse_pos_to_old_cpu = reuse_pos_to_old.contiguous()

        keepold_phase = _normalize_keepold_phase(self.config.keepold_phase)
        keepold_bridge_key = (str(self.config.keepold_bridge_key).strip()
                              if self.config.keepold_bridge_key else None)
        if keepold_phase != _KEEPOLD_PHASE_OFF and not keepold_bridge_key:
            raise ValueError(
                "keepold_phase requires keepold_bridge_key to be set.")
        switch_pos_cpu = torch.empty(0, dtype=torch.long)
        if self.config.keepold_switch_prompt_positions is not None:
            switch_pos_cpu = torch.tensor(
                [int(x) for x in self.config.keepold_switch_prompt_positions],
                dtype=torch.long,
            ).contiguous()
            if switch_pos_cpu.numel() > 0:
                switch_pos_cpu = torch.unique(switch_pos_cpu,
                                              sorted=True).contiguous()
                if int(switch_pos_cpu.min().item()) < 0:
                    raise ValueError(
                        "keepold_switch_prompt_positions contains negative index"
                    )
                if int(switch_pos_cpu.max().item()) >= int(self.config.prompt_len):
                    raise ValueError(
                        "keepold_switch_prompt_positions out of range for "
                        f"prompt_len={self.config.prompt_len}")
        keepold_index_mode = str(self.config.keepold_index_mode).strip().lower()
        if keepold_index_mode not in (_KEEPOLD_INDEX_MODE_SLOT_MAPPING,
                                      _KEEPOLD_INDEX_MODE_CHUNK_PAGE):
            raise ValueError(
                "Invalid keepold_index_mode: "
                f"{self.config.keepold_index_mode}. Expected one of "
                f"{_KEEPOLD_INDEX_MODE_SLOT_MAPPING}, {_KEEPOLD_INDEX_MODE_CHUNK_PAGE}."
            )
        keepold_switch_scope = str(self.config.keepold_switch_scope).strip().lower()
        if keepold_switch_scope not in (_KEEPOLD_SWITCH_SCOPE_D_ONLY,
                                        _KEEPOLD_SWITCH_SCOPE_ALL_NON_IMPORTANT):
            raise ValueError(
                "Invalid keepold_switch_scope: "
                f"{self.config.keepold_switch_scope}. Expected one of "
                f"{_KEEPOLD_SWITCH_SCOPE_D_ONLY}, "
                f"{_KEEPOLD_SWITCH_SCOPE_ALL_NON_IMPORTANT}.")
        hash_algo = normalize_chunk_page_hash_algo(
            self.config.keepold_chunk_page_hash_algo)
        self.config.keepold_chunk_page_hash_algo = hash_algo
        if int(self.config.keepold_chunk_page_max_chunks) <= 0:
            raise ValueError("keepold_chunk_page_max_chunks must be > 0")
        if int(self.config.keepold_chunk_page_max_entries_per_hash) <= 0:
            raise ValueError("keepold_chunk_page_max_entries_per_hash must be > 0")
        self.config.keepold_chunk_page_lru_unit = _normalize_keepold_lru_unit(
            getattr(self.config, "keepold_chunk_page_lru_unit",
                    _KEEPOLD_LRU_UNIT_ENTRY))
        if int(getattr(self.config, "keepold_chunk_page_lru_evict_batch_chunks",
                       0)) < 0:
            raise ValueError(
                "keepold_chunk_page_lru_evict_batch_chunks must be >= 0")
        chunk_page_exec_granularity = _normalize_chunk_page_exec_granularity(
            getattr(self.config, "keepold_chunk_page_exec_granularity",
                    _CHUNK_PAGE_EXEC_GRANULARITY_TOKEN))
        if keepold_index_mode != _KEEPOLD_INDEX_MODE_CHUNK_PAGE:
            chunk_page_exec_granularity = _CHUNK_PAGE_EXEC_GRANULARITY_TOKEN

        self._keepold_bridge_key = keepold_bridge_key
        self._keepold_phase = keepold_phase
        self._keepold_index_mode = keepold_index_mode
        self._keepold_chunk_page_exec_granularity = chunk_page_exec_granularity
        self._keepold_switch_scope = keepold_switch_scope
        self._keepold_chunk_page_source_id = keepold_bridge_key
        self._keepold_chunk_page_entry_registered = False
        self._keepold_chunk_page_dryrun_supported = (
            keepold_index_mode == _KEEPOLD_INDEX_MODE_CHUNK_PAGE
            and keepold_switch_scope == _KEEPOLD_SWITCH_SCOPE_ALL_NON_IMPORTANT)
        self._keepold_chunk_page_dryrun_reason = (
            "ok" if self._keepold_chunk_page_dryrun_supported else
            ("unsupported_switch_scope" if keepold_index_mode
             == _KEEPOLD_INDEX_MODE_CHUNK_PAGE else "index_mode_slot_mapping"))
        self._reset_chunk_page_dryrun_state()
        self._keepold_switch_prompt_positions_cpu = switch_pos_cpu
        self._keepold_switch_prompt_positions_dev = None
        self._keepold_switch_reuse_idx_dev = None
        self._keepold_switch_reuse_idx_cpu = torch.empty(0, dtype=torch.long)
        self._keepold_cpu_reuse_idx_dev = None
        self._keepold_cpu_reuse_idx_cpu = torch.empty(0, dtype=torch.long)
        self._keepold_cpu_subset_layers = {}
        self._keepold_cpu_subset_total_prepare_ms = 0.0
        self._keepold_cpu_subset_total_prepare_envelope_ms = 0.0
        self._keepold_cpu_subset_total_pin_ms = 0.0
        self._keepold_cpu_subset_ready = False
        self._build_keepold_partition_cpu()
        self._prepare_keepold_cpu_subsets()
        self._source_loaded = True
        t_source1 = time.perf_counter()

        common_meta = {
            "num_layers": int(len(layer_items)),
            "prompt_len": int(self.config.prompt_len),
            "reuse_tokens": int(reuse_pos.numel()),
            "ts_unix": time.time(),
        }
        self._append_metrics({
            "event": "selective_source_prepare",
            "t_cpu_prepare_ms": t_cpu_prepare_ms,
            "timing_v2_cpu_source_contiguous_ms": float(t_cpu_contiguous_ms),
            "cpu_moved_k_layers": int(cpu_moved_k_layers),
            "cpu_moved_v_layers": int(cpu_moved_v_layers),
            "contiguous_fixed_k_layers": int(contiguous_fixed_k_layers),
            "contiguous_fixed_v_layers": int(contiguous_fixed_v_layers),
            **common_meta,
        })
        self._append_metrics({
            "event": "selective_source_pin",
            "t_pin_memory_ms": t_pin_memory_ms,
            "timing_v2_cpu_pin_ms": float(t_pin_memory_ms),
            "pin_attempted": pin_attempted,
            "pre_pinned_k_layers": int(pre_pinned_k_layers),
            "pre_pinned_v_layers": int(pre_pinned_v_layers),
            "newly_pinned_k_layers": int(newly_pinned_k_layers),
            "newly_pinned_v_layers": int(newly_pinned_v_layers),
            "cpu_moved_k_layers": int(cpu_moved_k_layers),
            "cpu_moved_v_layers": int(cpu_moved_v_layers),
            "contiguous_fixed_k_layers": int(contiguous_fixed_k_layers),
            "contiguous_fixed_v_layers": int(contiguous_fixed_v_layers),
            "pin_success_k_layers": int(pre_pinned_k_layers + newly_pinned_k_layers),
            "pin_success_v_layers": int(pre_pinned_v_layers + newly_pinned_v_layers),
            "is_pinned_all": bool(
                pin_attempted
                and (pre_pinned_k_layers + newly_pinned_k_layers) == len(layer_items)
                and (pre_pinned_v_layers + newly_pinned_v_layers) == len(layer_items)),
            **common_meta,
        })
        self._append_metrics({
            "event": "selective_source_load",
            "t_source_load_total_ms": (t_source1 - t_source0) * 1000.0,
            "timing_v2_cpu_source_load_ms": float((t_source1 - t_source0) *
                                                   1000.0),
            "timing_v2_source_load_envelope_ms": float((t_source1 - t_source0)
                                                        * 1000.0),
            **common_meta,
        })
        self._append_metrics({
            "event": "selective_keepold_config",
            "keepold_phase": self._keepold_phase,
            "keepold_bridge_key": self._keepold_bridge_key,
            "keepold_index_mode": self._keepold_index_mode,
            "keepold_switch_scope": self._keepold_switch_scope,
            "keepold_switch_prompt_tokens": int(
                self._keepold_switch_prompt_positions_cpu.numel()),
            "keepold_fused_k_mode": self.config.keepold_fused_k_mode,
            "keepold_fused_k_chunk_size": int(
                self.config.keepold_fused_k_chunk_size),
            "keepold_chunk_page_hash_algo":
            self.config.keepold_chunk_page_hash_algo,
            "keepold_chunk_page_max_chunks":
            int(self.config.keepold_chunk_page_max_chunks),
            "keepold_chunk_page_max_entries_per_hash":
            int(self.config.keepold_chunk_page_max_entries_per_hash),
            "keepold_chunk_page_lru_unit":
            str(self.config.keepold_chunk_page_lru_unit),
            "keepold_chunk_page_lru_evict_batch_chunks":
            int(self.config.keepold_chunk_page_lru_evict_batch_chunks),
            "keepold_chunk_page_lru_low_watermark_chunks":
            int(self.config.keepold_chunk_page_lru_low_watermark_chunks),
            "keepold_chunk_page_clear_each_sample":
            bool(self.config.keepold_chunk_page_clear_each_sample),
            "keepold_chunk_page_global_lookup":
            bool(self.config.keepold_chunk_page_global_lookup),
            "keepold_chunk_page_no_replace_existing_hash":
            bool(self.config.keepold_chunk_page_no_replace_existing_hash),
            "keepold_chunk_page_block_invalidation":
            bool(self.config.keepold_chunk_page_block_invalidation),
            "keepold_chunk_page_defer_register":
            bool(self.config.keepold_chunk_page_defer_register),
            "keepold_chunk_page_register_hit_chunks_after_sample":
            bool(self.config.keepold_chunk_page_register_hit_chunks_after_sample),
            "keepold_chunk_page_update_entry_on_hit":
            bool(self.config.keepold_chunk_page_update_entry_on_hit),
            "keepold_chunk_page_debug_checks":
            bool(self.config.keepold_chunk_page_debug_checks),
            "keepold_chunk_page_execution_enabled":
            bool(self.config.keepold_chunk_page_execution_enabled),
            "keepold_chunk_page_exec_granularity":
            self._keepold_chunk_page_exec_granularity,
            "keepold_timing_mode": self._timing_mode(),
            "keepold_layer_metrics_mode": self._keepold_layer_metrics_mode,
            "keepold_overlap_cpu_gpu_compose": bool(
                getattr(self.config, "keepold_overlap_cpu_gpu_compose", False)),
            "keepold_cpu_fallback_mode": _normalize_keepold_cpu_fallback_mode(
                getattr(self.config, "keepold_cpu_fallback_mode",
                        _KEEPOLD_CPU_FALLBACK_MODE_SUBSET)),
            **common_meta,
        })

        logger.info(
            "Selective runtime source loaded: layers=%d prompt_len=%d "
            "reuse_tokens=%d recompute_tokens=%d reuse_rope_impl=%s "
            "dynamic_update_backend=%s",
            len(self._src_layer_kv),
            self.config.prompt_len,
            int(self._reuse_positions_cpu.numel()),
            int(self._recompute_positions_cpu.numel()),
            self.config.reuse_rope_impl,
            self.config.dynamic_update_backend,
        )

    def begin_forward(self, positions: torch.Tensor) -> bool:
        t_begin_forward0 = time.perf_counter()
        if not self.config.enabled:
            self._active = False
            return False

        t_bf_load0 = time.perf_counter()
        self._load_source_once()
        t_bf_load1 = time.perf_counter()
        t_begin_forward_load_source_ms = float((t_bf_load1 - t_bf_load0) * 1000.0)
        self._dynamic_src_ready = False
        if self.config.prefill_only and positions.numel() <= 1:
            self._active = False
            return False
        if int(positions.numel()) != self.config.prompt_len:
            self._active = False
            return False

        device = positions.device
        t_bf_pos0 = time.perf_counter()
        self._full_positions = positions.detach().contiguous()
        self._recompute_positions_dev = self._recompute_positions_cpu.to(
            device=device, non_blocking=False)
        self._reuse_positions_dev = self._reuse_positions_cpu.to(
            device=device, non_blocking=False)
        self._reuse_old_positions_dev = self._reuse_old_positions_cpu.to(
            device=device, non_blocking=False)
        self._base_reuse_positions_dev = self._base_reuse_positions_cpu.to(
            device=device, non_blocking=False)
        self._base_reuse_old_positions_dev = self._base_reuse_old_positions_cpu.to(
            device=device, non_blocking=False)
        self._keepold_switch_prompt_positions_dev = None
        self._keepold_switch_reuse_idx_dev = None
        self._keepold_cpu_reuse_idx_dev = None
        self._layer_kv_pipeline_starts = {}
        self._keepold_diag_layer_components = {}
        self._reset_layer_metrics_aggregate()
        self._keepold_cached_bridge = None
        self._keepold_cached_bridge_layers = None
        self._keepold_paged_layout_meta_by_layer = {}
        self._reset_keepold_old_page_fastpath()
        t_bf_pos1 = time.perf_counter()
        t_begin_forward_position_h2d_ms = float((t_bf_pos1 - t_bf_pos0) * 1000.0)

        defer_chunk_page_register = bool(
            self._keepold_phase == _KEEPOLD_PHASE_REUSE
            and self.config.keepold_chunk_page_defer_register
            and self._keepold_index_mode == _KEEPOLD_INDEX_MODE_CHUNK_PAGE)

        if self._keepold_phase == _KEEPOLD_PHASE_RECORD:
            bridge = self._get_keepold_bridge(create=True)
            if bridge is not None:
                if int(bridge.prompt_len) != int(self.config.prompt_len):
                    raise RuntimeError(
                        "keepold bridge prompt_len mismatch: "
                        f"bridge={bridge.prompt_len}, "
                        f"current={self.config.prompt_len}")
                if self._keepold_switch_prompt_positions_cpu.numel() > 0:
                    switch_dev = self._keepold_switch_prompt_positions_cpu.to(
                        device=device, non_blocking=False)
                    self._keepold_switch_prompt_positions_dev = switch_dev
                    bridge.switch_prompt_positions_dev = (
                        switch_dev.detach().clone().contiguous())
                elif bridge.switch_prompt_positions_dev is not None:
                    self._keepold_switch_prompt_positions_dev = (
                        bridge.switch_prompt_positions_dev.to(
                            device=device, non_blocking=False))
        elif self._keepold_phase == _KEEPOLD_PHASE_REUSE:
            switch_dev: Optional[torch.Tensor] = None
            if self._keepold_switch_prompt_positions_cpu.numel() > 0:
                switch_dev = self._keepold_switch_prompt_positions_cpu.to(
                    device=device, non_blocking=False)
            else:
                bridge = self._get_keepold_bridge(create=False)
                if bridge is not None and bridge.switch_prompt_positions_dev is not None:
                    switch_dev = bridge.switch_prompt_positions_dev.to(
                        device=device, non_blocking=False)
            if switch_dev is not None and switch_dev.numel() > 0:
                if self._keepold_switch_reuse_idx_cpu.numel() > 0:
                    self._keepold_switch_prompt_positions_dev = (
                        self._keepold_switch_prompt_positions_cpu.to(
                            device=device, non_blocking=False))
                    self._keepold_switch_reuse_idx_dev = (
                        self._keepold_switch_reuse_idx_cpu.to(
                            device=device, non_blocking=False))
                    self._keepold_cpu_reuse_idx_dev = (
                        self._keepold_cpu_reuse_idx_cpu.to(
                            device=device, non_blocking=False))
                    switch_reuse_tokens = int(
                        self._keepold_switch_reuse_idx_cpu.numel())
                    cpu_reuse_tokens = int(self._keepold_cpu_reuse_idx_cpu.numel())
                else:
                    reuse_pos_to_idx = torch.full(
                        (self.config.prompt_len, ),
                        -1,
                        dtype=torch.long,
                        device=device,
                    )
                    reuse_pos_to_idx[self._reuse_positions_dev] = torch.arange(
                        int(self._reuse_positions_dev.numel()),
                        dtype=torch.long,
                        device=device,
                    )
                    reuse_idx = reuse_pos_to_idx.index_select(0, switch_dev)
                    valid_mask = reuse_idx >= 0
                    switch_prompt_valid = switch_dev[valid_mask].contiguous()
                    switch_reuse_idx = reuse_idx[valid_mask].contiguous()
                    switch_reuse_tokens = int(switch_reuse_idx.numel())
                    if switch_reuse_idx.numel() > 0:
                        total_reuse = int(self._reuse_positions_dev.numel())
                        keep_mask = torch.ones(total_reuse,
                                               dtype=torch.bool,
                                               device=device)
                        keep_mask[switch_reuse_idx] = False
                        cpu_reuse_idx = torch.arange(
                            total_reuse, dtype=torch.long, device=device)[keep_mask]
                        self._keepold_switch_prompt_positions_dev = (
                            switch_prompt_valid)
                        self._keepold_switch_reuse_idx_dev = switch_reuse_idx
                        self._keepold_cpu_reuse_idx_dev = cpu_reuse_idx.contiguous()
                        self._keepold_cpu_reuse_idx_cpu = (
                            self._keepold_cpu_reuse_idx_dev.to(
                                device="cpu", non_blocking=False))
                    cpu_reuse_tokens = int(
                        self._keepold_cpu_reuse_idx_cpu.numel())
                self._append_metrics({
                    "event": "selective_keepold_partition",
                    "keepold_phase": self._keepold_phase,
                    "switch_prompt_tokens": int(switch_dev.numel()),
                    "switch_reuse_tokens": int(switch_reuse_tokens),
                    "cpu_reuse_tokens": int(cpu_reuse_tokens),
                    "cpu_subset_ready": bool(self._keepold_cpu_subset_ready),
                    "ts_unix": time.time(),
                })
        bridge_cached = self._get_keepold_bridge(create=False)
        self._keepold_cached_bridge = bridge_cached
        self._keepold_cached_bridge_layers = (
            bridge_cached.layers if bridge_cached is not None else None)
        t_begin_forward_chunk_page_lookup_ms = 0.0
        t_begin_forward_exec_partition_ms = 0.0
        t_begin_forward_cpu_subset_refresh_ms = 0.0
        cpu_subset_refreshed_after_partition = 0
        cpu_subset_refresh_reason = "inactive"
        cpu_subset_refresh_before_rows = int(
            self._keepold_cpu_reuse_idx_cpu.numel())
        cpu_subset_refresh_before_tokens = int(
            self._keepold_cpu_fallback_tokens)
        cpu_subset_refresh_before_span_direct_tokens = int(
            self._keepold_cpu_fallback_span_direct_tokens)
        if self._keepold_index_mode == _KEEPOLD_INDEX_MODE_CHUNK_PAGE:
            t_bf_lookup0 = time.perf_counter()
            self._prepare_chunk_page_lookup_dry_run()
            t_bf_lookup1 = time.perf_counter()
            t_begin_forward_chunk_page_lookup_ms = float(
                (t_bf_lookup1 - t_bf_lookup0) * 1000.0)
            t_bf_exec0 = time.perf_counter()
            self._apply_chunk_page_exec_partition(device=device)
            t_bf_exec1 = time.perf_counter()
            t_begin_forward_exec_partition_ms = float(
                (t_bf_exec1 - t_bf_exec0) * 1000.0)
            # CPU subset/spans are initially prepared in _load_source_once()
            # before chunk-page lookup/partition runs. In global chunk-page
            # mode, lookup can change CPU/GPU row partitioning significantly.
            # If partitioned CPU rows changed, refresh subset prep so compose
            # path does not re-materialize per-layer CPU subset dynamically.
            if (self._keepold_phase == _KEEPOLD_PHASE_REUSE
                    and self.config.dynamic_update_backend
                    == _DYNAMIC_UPDATE_BACKEND_CPU_REBUILD):
                expected_cpu_rows = int(self._keepold_chunk_page_exec_cpu_tokens)
                current_cpu_rows = int(self._keepold_cpu_reuse_idx_cpu.numel())
                current_cpu_tokens = int(self._keepold_cpu_fallback_tokens)
                current_span_direct_tokens = int(
                    self._keepold_cpu_fallback_span_direct_tokens)
                span_direct_marked = bool(
                    self._keepold_cpu_fallback_subset_skipped
                    and self._keepold_cpu_fallback_span_direct_ready)
                need_refresh = (
                    current_cpu_rows != expected_cpu_rows
                    or current_cpu_tokens != expected_cpu_rows
                    or (span_direct_marked
                        and current_span_direct_tokens != expected_cpu_rows))
                if need_refresh:
                    t_bf_subset_refresh0 = time.perf_counter()
                    self._prepare_keepold_cpu_subsets()
                    t_bf_subset_refresh1 = time.perf_counter()
                    t_begin_forward_cpu_subset_refresh_ms = float(
                        (t_bf_subset_refresh1 - t_bf_subset_refresh0) * 1000.0)
                    cpu_subset_refreshed_after_partition = 1
                    cpu_subset_refresh_reason = "partition_mismatch_refresh"
                else:
                    cpu_subset_refresh_reason = "partition_aligned"
            else:
                cpu_subset_refresh_reason = "not_reuse_cpu_rebuild"
        self._forward_prompt_len = int(positions.numel())
        t_bf_old_page0 = time.perf_counter()
        self._prepare_keepold_old_page_fastpath(device=device)
        t_bf_old_page1 = time.perf_counter()
        t_begin_forward_old_page_prepare_ms = float(
            (t_bf_old_page1 - t_bf_old_page0) * 1000.0)
        t_bf_gpu_idx0 = time.perf_counter()
        self._prepare_keepold_gpu_index_fastpath(device=device)
        t_bf_gpu_idx1 = time.perf_counter()
        t_begin_forward_gpu_index_prepare_ms = float(
            (t_bf_gpu_idx1 - t_bf_gpu_idx0) * 1000.0)
        self._prepare_keepold_overlap_compose_eligibility()
        t_begin_forward1 = time.perf_counter()
        self._append_metrics({
            "event": "selective_begin_forward_envelope",
            "keepold_phase": self._keepold_phase,
            "keepold_index_mode": self._keepold_index_mode,
            "chunk_page_execution_enabled": bool(self._chunk_page_exec_enabled()),
            "t_begin_forward_total_ms": float(
                (t_begin_forward1 - t_begin_forward0) * 1000.0),
            "t_begin_forward_load_source_ms": float(
                t_begin_forward_load_source_ms),
            "t_begin_forward_position_h2d_ms": float(
                t_begin_forward_position_h2d_ms),
            "t_begin_forward_chunk_page_lookup_ms": float(
                t_begin_forward_chunk_page_lookup_ms),
            "t_begin_forward_exec_partition_ms": float(
                t_begin_forward_exec_partition_ms),
            "t_begin_forward_cpu_subset_refresh_ms": float(
                t_begin_forward_cpu_subset_refresh_ms),
            "cpu_subset_refreshed_after_partition": int(
                cpu_subset_refreshed_after_partition),
            "cpu_subset_refresh_reason": str(cpu_subset_refresh_reason),
            "cpu_subset_refresh_before_rows": int(
                cpu_subset_refresh_before_rows),
            "cpu_subset_refresh_before_tokens": int(
                cpu_subset_refresh_before_tokens),
            "cpu_subset_refresh_before_span_direct_tokens": int(
                cpu_subset_refresh_before_span_direct_tokens),
            "cpu_subset_refresh_after_rows": int(
                self._keepold_cpu_reuse_idx_cpu.numel()),
            "cpu_subset_refresh_after_tokens": int(
                self._keepold_cpu_fallback_tokens),
            "cpu_subset_refresh_after_span_direct_tokens": int(
                self._keepold_cpu_fallback_span_direct_tokens),
            "t_begin_forward_old_page_prepare_ms": float(
                t_begin_forward_old_page_prepare_ms),
            "t_begin_forward_gpu_index_prepare_ms": float(
                t_begin_forward_gpu_index_prepare_ms),
            "keepold_overlap_cpu_gpu_compose_enabled":
            1 if bool(getattr(self.config, "keepold_overlap_cpu_gpu_compose", False)) else 0,
            "keepold_overlap_cpu_gpu_compose_eligible":
            1 if bool(self._keepold_overlap_cpu_gpu_eligible) else 0,
            "keepold_overlap_cpu_gpu_compose_reason":
            str(self._keepold_overlap_cpu_gpu_reason),
            "ts_unix": time.time(),
        })
        self._active = True
        return True

    def end_forward(self) -> None:
        self._emit_layer_metrics_aggregate_event()
        self._active = False
        self._full_positions = None
        self._recompute_positions_dev = None
        self._reuse_positions_dev = None
        self._reuse_old_positions_dev = None
        self._base_reuse_positions_dev = None
        self._base_reuse_old_positions_dev = None
        self._keepold_switch_prompt_positions_dev = None
        self._keepold_switch_reuse_idx_dev = None
        self._keepold_cpu_reuse_idx_dev = None
        self._layer_kv_pipeline_starts = {}
        self._keepold_diag_layer_components = {}
        self._reset_layer_metrics_aggregate()
        self._keepold_cached_bridge = None
        self._keepold_cached_bridge_layers = None
        self._keepold_paged_layout_meta_by_layer = {}
        self._reset_keepold_old_page_fastpath()
        self._forward_prompt_len = 0

    def commit_deferred_chunk_page_entries(self) -> dict[str, Any]:
        """Register current request chunk-page entries after generate returns.

        Global stream uses this to keep request TTFT free of index registration
        work: lookup happens in begin_forward(), while registration is explicitly
        committed by the runner after the single natural request finishes.
        """
        t0 = time.perf_counter()
        if not bool(self.config.keepold_chunk_page_defer_register):
            return {"status": "skipped", "reason": "defer_register_disabled"}
        if self._keepold_chunk_page_deferred_committed:
            return {"status": "skipped", "reason": "already_committed"}
        record = self._keepold_chunk_page_deferred_record
        if not isinstance(record, dict):
            return {"status": "skipped", "reason": "missing_deferred_record"}

        candidate_infos = [dict(x) for x in self._keepold_chunk_page_post_register_infos]
        index = get_chunk_page_index(
            max_chunks=self.config.keepold_chunk_page_max_chunks,
            max_entries_per_hash=self.config.keepold_chunk_page_max_entries_per_hash,
            global_lookup_enabled=bool(self.config.keepold_chunk_page_global_lookup),
            no_replace_existing_hash=bool(
                self.config.keepold_chunk_page_no_replace_existing_hash),
            block_invalidation_enabled=bool(
                self.config.keepold_chunk_page_block_invalidation),
            lru_unit=str(self.config.keepold_chunk_page_lru_unit),
            lru_evict_batch_chunks=int(
                self.config.keepold_chunk_page_lru_evict_batch_chunks),
            lru_low_watermark_chunks=int(
                self.config.keepold_chunk_page_lru_low_watermark_chunks),
        )
        rec_stats: dict[str, Any] = {
            "register_attempts": 0,
            "register_success": 0,
            "register_skipped_duplicate_hash": 0,
            "register_skipped_bucket_full": 0,
        }
        if candidate_infos:
            entries, rec_stats = build_record_entries(
                index=index,
                source_id=str(record.get("source_id", "")),
                prompt_token_ids=(self.config.keepold_prompt_token_ids or []),
                chunk_infos=candidate_infos,
                logical_to_physical=list(record.get("logical_to_physical", [])),
                block_size=int(record.get("block_size", 1)),
                hash_algo=self.config.keepold_chunk_page_hash_algo,
                model_id=self.config.keepold_model_id,
                dtype=(str(record.get("dtype")) if record.get("dtype") else None),
                kv_layout=(str(record.get("kv_layout"))
                           if record.get("kv_layout") else None),
                no_replace_existing_hash=bool(
                    self.config.keepold_chunk_page_no_replace_existing_hash),
                max_entries_per_hash=int(
                    self.config.keepold_chunk_page_max_entries_per_hash),
            )
            register_success = 0
            lru_evicted_entries = 0
            lru_evicted_chunk_hashes = 0
            lru_evicted_entries_per_chunk_max = 0
            lru_trigger_reason = "none"
            for entry in entries:
                if index.register_entry(entry):
                    register_success += 1
                reg_lru = index.get_last_register_lru_stats()
                lru_evicted_entries += int(
                    reg_lru.get("lru_evicted_entries", 0))
                lru_evicted_chunk_hashes += int(
                    reg_lru.get("lru_evicted_chunk_hashes", 0))
                lru_evicted_entries_per_chunk_max = max(
                    int(lru_evicted_entries_per_chunk_max),
                    int(reg_lru.get("lru_evicted_entries_per_chunk_max", 0)),
                )
                reason = str(reg_lru.get("lru_trigger_reason", "none"))
                pri = {
                    "none": 0,
                    "bucket_capacity": 1,
                    "entry_capacity": 2,
                    "chunk_hash_capacity": 3,
                }
                if pri.get(reason, 0) >= pri.get(lru_trigger_reason, 0):
                    lru_trigger_reason = reason
            rec_stats["register_success"] = int(register_success)
            rec_stats["lru_evicted_entries"] = int(lru_evicted_entries)
            rec_stats["lru_evicted_chunk_hashes"] = int(
                lru_evicted_chunk_hashes)
            rec_stats["lru_evicted_entries_per_chunk_max"] = int(
                lru_evicted_entries_per_chunk_max)
            rec_stats["lru_trigger_reason"] = str(lru_trigger_reason)
            register_rejected = max(0, int(len(entries)) - int(register_success))
            if register_rejected > 0:
                rec_stats["register_skipped_duplicate_hash"] = int(
                    rec_stats.get("register_skipped_duplicate_hash", 0)
                ) + int(register_rejected)
                rec_stats["register_skipped_bucket_full"] = int(
                    rec_stats.get("register_skipped_bucket_full", 0)
                ) + int(register_rejected)

        t1 = time.perf_counter()
        index_stats = index.stats()
        post_register_ms = float((t1 - t0) * 1000.0)
        out: dict[str, Any] = {
            "status": "ok",
            "reason": "committed",
            "post_register_ms": post_register_ms,
            "post_register_candidate_chunks": int(len(candidate_infos)),
            "post_register_success": int(rec_stats.get("register_success", 0)),
            "post_register_skipped_existing_hash": int(
                rec_stats.get("register_skipped_duplicate_hash", 0)),
            "post_register_skipped_bucket_full": int(
                rec_stats.get("register_skipped_bucket_full", 0)),
            "chunk_page_register_attempts": int(
                rec_stats.get("register_attempts", 0)),
            "chunk_page_register_success": int(
                rec_stats.get("register_success", 0)),
            "chunk_page_register_skipped_duplicate_hash": int(
                rec_stats.get("register_skipped_duplicate_hash", 0)),
            "chunk_page_register_skipped_bucket_full": int(
                rec_stats.get("register_skipped_bucket_full", 0)),
            "lru_evicted_entries": int(
                rec_stats.get("lru_evicted_entries", 0)),
            "lru_evicted_chunk_hashes": int(
                rec_stats.get("lru_evicted_chunk_hashes", 0)),
            "lru_evicted_entries_per_chunk_max": int(
                rec_stats.get("lru_evicted_entries_per_chunk_max", 0)),
            "lru_trigger_reason": str(
                rec_stats.get("lru_trigger_reason", "none")),
            **self._chunk_page_index_stats_payload(index_stats),
        }
        self._append_metrics({
            "event": "selective_chunk_page_post_register",
            "keepold_phase": self._keepold_phase,
            "keepold_bridge_key": self._keepold_bridge_key,
            **out,
            "ts_unix": time.time(),
        })
        self._keepold_chunk_page_deferred_committed = True
        return out

    def is_active(self) -> bool:
        return self._active

    def get_recompute_positions(self, device: torch.device) -> torch.Tensor:
        if not self._active or self._recompute_positions_dev is None:
            raise RuntimeError("Selective runtime is not active for this forward")
        if self._recompute_positions_dev.device != device:
            self._recompute_positions_dev = self._recompute_positions_cpu.to(
                device=device, non_blocking=False)
        return self._recompute_positions_dev

    def get_full_positions(self, device: torch.device) -> torch.Tensor:
        if not self._active or self._full_positions is None:
            raise RuntimeError("Selective runtime is not active for this forward")
        if self._full_positions.device != device:
            self._full_positions = self._full_positions.to(device=device,
                                                           non_blocking=False)
        return self._full_positions

    def get_reuse_positions(self, device: torch.device) -> torch.Tensor:
        if not self._active or self._reuse_positions_dev is None:
            raise RuntimeError("Selective runtime is not active for this forward")
        if self._reuse_positions_dev.device != device:
            self._reuse_positions_dev = self._reuse_positions_cpu.to(
                device=device, non_blocking=False)
        return self._reuse_positions_dev

    def get_reuse_old_positions(self, device: torch.device) -> torch.Tensor:
        if not self._active or self._reuse_old_positions_dev is None:
            raise RuntimeError("Selective runtime is not active for this forward")
        if self._reuse_old_positions_dev.device != device:
            self._reuse_old_positions_dev = self._reuse_old_positions_cpu.to(
                device=device, non_blocking=False)
        return self._reuse_old_positions_dev

    def get_base_reuse_positions(self, device: torch.device) -> torch.Tensor:
        if not self._active or self._base_reuse_positions_dev is None:
            raise RuntimeError("Selective runtime is not active for this forward")
        if self._base_reuse_positions_dev.device != device:
            self._base_reuse_positions_dev = self._base_reuse_positions_cpu.to(
                device=device, non_blocking=False)
        return self._base_reuse_positions_dev

    def get_base_reuse_old_positions(self, device: torch.device) -> torch.Tensor:
        if not self._active or self._base_reuse_old_positions_dev is None:
            raise RuntimeError("Selective runtime is not active for this forward")
        if self._base_reuse_old_positions_dev.device != device:
            self._base_reuse_old_positions_dev = self._base_reuse_old_positions_cpu.to(
                device=device, non_blocking=False)
        return self._base_reuse_old_positions_dev

    def _rebuild_dynamic_source_layer_kv(self) -> None:
        if not self._source_loaded:
            return
        if not self._src_layer_kv:
            self._src_layer_kv_dynamic = {}
            self._dynamic_src_ready = False
            return

        pin_attempted = bool(torch.cuda.is_available())
        idx_cpu = self._reuse_old_positions_cpu
        dynamic_items: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        t_dynamic_gather_ms = 0.0
        t_dynamic_pin_ms = 0.0
        direct_reuse_layers = 0

        for lid, (src_k, src_v) in self._src_layer_kv.items():
            if int(src_k.shape[0]) == int(idx_cpu.numel()):
                dynamic_items[lid] = (src_k, src_v)
                direct_reuse_layers += 1
                continue

            t_g0 = time.perf_counter()
            subset_tokens = int(idx_cpu.numel())
            src_k_dyn = torch.empty(
                (subset_tokens, *tuple(src_k.shape[1:])),
                dtype=src_k.dtype,
                device="cpu",
                pin_memory=pin_attempted,
            )
            src_v_dyn = torch.empty(
                (subset_tokens, *tuple(src_v.shape[1:])),
                dtype=src_v.dtype,
                device="cpu",
                pin_memory=pin_attempted,
            )
            torch.index_select(src_k, 0, idx_cpu, out=src_k_dyn)
            torch.index_select(src_v, 0, idx_cpu, out=src_v_dyn)
            t_g1 = time.perf_counter()
            t_dynamic_gather_ms += (t_g1 - t_g0) * 1000.0

            dynamic_items[lid] = (src_k_dyn, src_v_dyn)

        self._src_layer_kv_dynamic = dynamic_items
        self._dynamic_src_ready = True
        self._append_metrics({
            "event": "selective_dynamic_source_rebuild",
            "num_layers": int(len(dynamic_items)),
            "direct_reuse_layers": int(direct_reuse_layers),
            "rebuilt_layers": int(len(dynamic_items) - direct_reuse_layers),
            "reuse_tokens": int(self._reuse_positions_cpu.numel()),
            "recompute_tokens": int(self._recompute_positions_cpu.numel()),
            "t_dynamic_gather_ms": t_dynamic_gather_ms,
            "t_dynamic_pin_ms": t_dynamic_pin_ms,
            "pin_attempted": pin_attempted,
            "ts_unix": time.time(),
        })

    def update_recompute_positions(
        self,
        recompute_positions: Sequence[int] | torch.Tensor,
    ) -> None:
        """Dynamically update recompute/reuse split for the current forward."""
        if not self._active:
            raise RuntimeError(
                "update_recompute_positions called when runtime is inactive")

        if isinstance(recompute_positions, torch.Tensor):
            recompute_pos = recompute_positions.detach().to(
                device="cpu", dtype=torch.long).contiguous()
        else:
            recompute_pos = torch.tensor(
                [int(x) for x in recompute_positions],
                dtype=torch.long,
            ).contiguous()

        recompute_pos = torch.unique(recompute_pos, sorted=True)
        if recompute_pos.numel() == 0:
            raise ValueError("Dynamic recompute_positions is empty")
        if int(recompute_pos.min().item()) < 0:
            raise ValueError(
                "Dynamic recompute_positions contains negative index")
        if int(recompute_pos.max().item()) >= self.config.prompt_len:
            raise ValueError(
                "Dynamic recompute_positions out of range for prompt_len "
                f"{self.config.prompt_len}")

        full_pos = torch.arange(self.config.prompt_len, dtype=torch.long)
        keep_mask = torch.ones(self.config.prompt_len, dtype=torch.bool)
        keep_mask[recompute_pos] = False
        reuse_pos = full_pos[keep_mask].contiguous()
        if reuse_pos.numel() > 0:
            if self._reuse_pos_to_old_cpu.numel() != self.config.prompt_len:
                raise RuntimeError(
                    "reuse_pos_to_old mapping is not initialized correctly")
            reuse_old_pos = self._reuse_pos_to_old_cpu.index_select(
                0, reuse_pos).contiguous()
            if int(reuse_old_pos.min().item()) < 0:
                raise ValueError(
                    "Dynamic recompute_positions selected non-reusable token "
                    "into reuse set.")
        else:
            reuse_old_pos = torch.empty(0, dtype=torch.long)

        self._recompute_positions_cpu = recompute_pos
        self._reuse_positions_cpu = reuse_pos
        self._reuse_old_positions_cpu = reuse_old_pos

        if self._full_positions is not None:
            device = self._full_positions.device
            self._recompute_positions_dev = self._recompute_positions_cpu.to(
                device=device, non_blocking=False)
            self._reuse_positions_dev = self._reuse_positions_cpu.to(
                device=device, non_blocking=False)
            self._reuse_old_positions_dev = self._reuse_old_positions_cpu.to(
                device=device, non_blocking=False)

        if self.config.dynamic_update_backend == _DYNAMIC_UPDATE_BACKEND_CPU_REBUILD:
            # Rebuild dynamic CPU source once after layer-1 updates, so layer2+
            # can avoid per-layer CPU gather/pin fallback in compose_full_kv.
            self._rebuild_dynamic_source_layer_kv()
        elif (self.config.dynamic_update_backend
              == _DYNAMIC_UPDATE_BACKEND_GPU_POSITIONS_ONLY):
            # Blend-only fast path: keep source tensors in their base order and
            # only update position sets. compose_full_kv will resolve dynamic
            # reuse via device-side position indexing.
            self._src_layer_kv_dynamic = {}
            self._dynamic_src_ready = False
            self._append_metrics({
                "event": "selective_dynamic_source_rebuild_skipped",
                "reason": "gpu_positions_only_backend",
                "prompt_len": int(self.config.prompt_len),
                "recompute_tokens": int(recompute_pos.numel()),
                "reuse_tokens": int(reuse_pos.numel()),
                "ts_unix": time.time(),
            })
        else:
            raise ValueError(
                "Invalid selective dynamic_update_backend: "
                f"{self.config.dynamic_update_backend}. Expected one of "
                f"{_DYNAMIC_UPDATE_BACKEND_CPU_REBUILD}, "
                f"{_DYNAMIC_UPDATE_BACKEND_GPU_POSITIONS_ONLY}.")

        self._append_metrics({
            "event": "selective_recompute_update",
            "prompt_len": int(self.config.prompt_len),
            "recompute_tokens": int(recompute_pos.numel()),
            "reuse_tokens": int(reuse_pos.numel()),
            "dynamic_update_backend": self.config.dynamic_update_backend,
            "ts_unix": time.time(),
        })

    def _get_lmcache_cops(self):
        from keychunk.runtime.rope import load
        return load()

    def compose_full_kv(
        self,
        *,
        layer_id: int,
        new_k: torch.Tensor,
        new_v: torch.Tensor,
        layer_name: str,
        reuse_rope_cos_sin_cache: Optional[torch.Tensor] = None,
        reuse_rope_is_neox_style: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        probe_enabled = (
            bool(self._kv_eq_probe_enabled)
            and self._kv_eq_probe_layer_enabled(int(layer_id))
            and int(self._kv_eq_probe_positions_cpu.numel()) > 0
        )
        if probe_enabled:
            probe_positions = self._kv_eq_probe_positions_cpu.to(
                device=new_k.device,
                dtype=torch.long,
                non_blocking=False,
            ).contiguous()
            full_k, full_v, reuse_probe_k, reuse_probe_v = compose_full_kv_impl(
                self,
                layer_id=layer_id,
                new_k=new_k,
                new_v=new_v,
                layer_name=layer_name,
                reuse_rope_cos_sin_cache=reuse_rope_cos_sin_cache,
                reuse_rope_is_neox_style=reuse_rope_is_neox_style,
                probe_positions=probe_positions,
            )
            if reuse_probe_k is not None and reuse_probe_v is not None:
                self._kv_eq_probe_records[int(layer_id)] = {
                    "layer_id": int(layer_id),
                    "layer_name": str(layer_name),
                    "probe_positions": self._kv_eq_probe_positions_cpu.clone(
                    ).contiguous(),
                    "k": reuse_probe_k.detach().to(
                        device="cpu", dtype=torch.float32).contiguous(),
                    "v": reuse_probe_v.detach().to(
                        device="cpu", dtype=torch.float32).contiguous(),
                    "k_dtype": str(reuse_probe_k.dtype),
                    "v_dtype": str(reuse_probe_v.dtype),
                    "tag": str(self._kv_eq_probe_tag),
                }
        else:
            full_k, full_v, _, _ = compose_full_kv_impl(
                self,
                layer_id=layer_id,
                new_k=new_k,
                new_v=new_v,
                layer_name=layer_name,
                reuse_rope_cos_sin_cache=reuse_rope_cos_sin_cache,
                reuse_rope_is_neox_style=reuse_rope_is_neox_style,
            )
        return full_k, full_v

    def compose_full_kv_with_reuse_probe(
        self,
        *,
        layer_id: int,
        new_k: torch.Tensor,
        new_v: torch.Tensor,
        layer_name: str,
        probe_positions: torch.Tensor,
        reuse_rope_cos_sin_cache: Optional[torch.Tensor] = None,
        reuse_rope_is_neox_style: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        full_k, full_v, reuse_probe_k, reuse_probe_v = compose_full_kv_impl(
            self,
            layer_id=layer_id,
            new_k=new_k,
            new_v=new_v,
            layer_name=layer_name,
            reuse_rope_cos_sin_cache=reuse_rope_cos_sin_cache,
            reuse_rope_is_neox_style=reuse_rope_is_neox_style,
            probe_positions=probe_positions,
        )
        if reuse_probe_k is None or reuse_probe_v is None:
            raise RuntimeError("Failed to collect reuse probe KV")
        return full_k, full_v, reuse_probe_k, reuse_probe_v

    def mark_layer_kv_pipeline_start(self, *, layer_id: int) -> None:
        self._layer_kv_pipeline_starts[int(layer_id)] = time.perf_counter()

    def write_layer_kv_to_paged_cache(
        self,
        *,
        attn_layer: Any,
        layer_name: str,
        full_k: torch.Tensor,
        full_v: torch.Tensor,
        layer_id: int,
    ) -> None:
        if not self._active:
            raise RuntimeError(
                "write_layer_kv_to_paged_cache called when runtime is inactive")
        from vllm.forward_context import get_forward_context

        forward_context = get_forward_context()
        attn_metadata = forward_context.attn_metadata
        if isinstance(attn_metadata, dict):
            attn_metadata = attn_metadata[layer_name]

        if (hasattr(attn_metadata, "query_start_loc")
                and getattr(attn_metadata, "query_start_loc") is not None):
            # First version only supports one request per forward.
            if int(attn_metadata.query_start_loc.numel()) != 2:
                raise RuntimeError(
                    "Selective runtime currently supports single-request prefill only."
                )

        num_actual_tokens = int(attn_metadata.num_actual_tokens)
        if self.config.prefill_only and num_actual_tokens <= 1:
            return
        if num_actual_tokens <= 0:
            return

        kv_cache = attn_layer.kv_cache[forward_context.virtual_engine]
        key_cache, value_cache = kv_cache.unbind(0)
        slot_mapping = attn_metadata.slot_mapping[:num_actual_tokens]
        defer_chunk_page_register = bool(
            self._keepold_phase == _KEEPOLD_PHASE_REUSE
            and self.config.keepold_chunk_page_defer_register
            and self._keepold_index_mode == _KEEPOLD_INDEX_MODE_CHUNK_PAGE)

        if self._keepold_phase == _KEEPOLD_PHASE_RECORD:
            bridge = self._get_keepold_bridge(create=True)
            if bridge is not None:
                bridge.layers[int(layer_id)] = _KeepOldLayerSnapshot(
                    key_cache=key_cache,
                    value_cache=value_cache,
                    slot_mapping=slot_mapping.detach().clone().contiguous(),
                    num_actual_tokens=int(num_actual_tokens),
                )
                bridge.ts_unix = time.time()
                self._append_metrics({
                    "event": "selective_old_page_record",
                    "layer_name": layer_name,
                    "layer_id": int(layer_id),
                    "keepold_bridge_key": self._keepold_bridge_key,
                    "num_actual_tokens": int(num_actual_tokens),
                    "slot_tokens_recorded": int(slot_mapping.numel()),
                    "ts_unix": time.time(),
                })

        if int(full_k.shape[0]) < num_actual_tokens:
            raise ValueError(
                f"full_k length {int(full_k.shape[0])} is shorter than "
                f"num_actual_tokens {num_actual_tokens}")
        if int(full_v.shape[0]) < num_actual_tokens:
            raise ValueError(
                f"full_v length {int(full_v.shape[0])} is shorter than "
                f"num_actual_tokens {num_actual_tokens}")

        timing_cuda_event = (self._timing_cuda_event_enabled()
                             and full_k.device.type == "cuda")
        page_start_event: Optional[torch.cuda.Event] = None
        page_end_event: Optional[torch.cuda.Event] = None
        if timing_cuda_event:
            page_start_event = torch.cuda.Event(enable_timing=True)
            page_end_event = torch.cuda.Event(enable_timing=True)
            page_start_event.record(torch.cuda.current_stream(full_k.device))
        t0 = time.perf_counter()
        ops.reshape_and_cache_flash(
            full_k[:num_actual_tokens],
            full_v[:num_actual_tokens],
            key_cache,
            value_cache,
            slot_mapping,
            attn_layer.kv_cache_dtype,
            attn_layer._k_scale,
            attn_layer._v_scale,
        )
        t_submit = time.perf_counter()
        if page_end_event is not None:
            page_end_event.record(torch.cuda.current_stream(full_k.device))
        t_wait0 = t_submit
        if full_k.device.type == "cuda":
            torch.cuda.synchronize(full_k.device)
        t_wait1 = time.perf_counter()
        page_device_ms = float("nan")
        if page_start_event is not None and page_end_event is not None:
            page_device_ms = float(page_start_event.elapsed_time(page_end_event))
        t1 = t_wait1
        page_write_submit_ms = float((t_submit - t0) * 1000.0)
        page_write_wait_ms = float((t_wait1 - t_wait0) * 1000.0)
        self._append_metrics({
            "event": "selective_layer_cache_write",
            "layer_name": layer_name,
            "layer_id": layer_id,
            "num_actual_tokens": num_actual_tokens,
            "t_page_write_ms": (t1 - t0) * 1000.0,
            "timing_v2_page_write_submit_ms": float(page_write_submit_ms),
            "timing_v2_page_write_device_ms": float(page_device_ms),
            "timing_v2_page_write_wait_ms": float(page_write_wait_ms),
            "ts_unix": time.time(),
        })

        if ((self._keepold_phase == _KEEPOLD_PHASE_RECORD
             or defer_chunk_page_register)
                and self._keepold_index_mode == _KEEPOLD_INDEX_MODE_CHUNK_PAGE):
            allowed, reason = self._chunk_page_dryrun_allowed()
            if not allowed:
                if int(layer_id) == 0:
                    self._chunk_page_emit_metrics(
                        phase=self._keepold_phase,
                        status="skipped",
                        reason=reason,
                    )
            else:
                t_rec0 = time.perf_counter()
                layout_meta = self._infer_paged_cache_layout_code_page_size(
                    cache=key_cache,
                    num_kv_heads=int(full_k.shape[1]),
                    head_size=int(full_k.shape[2]),
                )
                if layout_meta is None:
                    if int(layer_id) == 0:
                        self._chunk_page_emit_metrics(
                            phase=self._keepold_phase,
                            status="skipped",
                            reason="layout_infer_failed",
                        )
                else:
                    layout_code, page_size = layout_meta
                    source_id = str(self._keepold_chunk_page_source_id or "")
                    index = get_chunk_page_index(
                        max_chunks=self.config.keepold_chunk_page_max_chunks,
                        max_entries_per_hash=self.
                        config.keepold_chunk_page_max_entries_per_hash,
                        global_lookup_enabled=bool(
                            self.config.keepold_chunk_page_global_lookup),
                        no_replace_existing_hash=bool(
                            self.config.
                            keepold_chunk_page_no_replace_existing_hash),
                        block_invalidation_enabled=bool(
                            self.config.keepold_chunk_page_block_invalidation),
                        lru_unit=str(self.config.keepold_chunk_page_lru_unit),
                        lru_evict_batch_chunks=int(
                            self.config.
                            keepold_chunk_page_lru_evict_batch_chunks),
                        lru_low_watermark_chunks=int(
                            self.config.
                            keepold_chunk_page_lru_low_watermark_chunks),
                    )
                    kv_layout = self.config.keepold_kv_layout
                    if kv_layout is None:
                        kv_layout = f"layout_{int(layout_code)}"
                    src = index.register_source(
                        source_id=source_id,
                        block_size=int(page_size),
                        layout_code=int(layout_code),
                        model_id=self.config.keepold_model_id,
                        dtype=str(key_cache.dtype),
                        kv_layout=kv_layout,
                    )
                    index.update_source_layer_cache(
                        source_id=source_id,
                        layer_id=int(layer_id),
                        key_cache=key_cache,
                        value_cache=value_cache,
                    )

                    emit_stats: dict[str, Any] = {}
                    rec_stats: dict[str, Any] = {
                        "register_attempts": 0,
                        "register_success": 0,
                        "register_skipped_duplicate_hash": 0,
                        "register_skipped_bucket_full": 0,
                    }
                    should_build_record_entries = (
                        self._keepold_phase == _KEEPOLD_PHASE_RECORD
                        and not self._keepold_chunk_page_entry_registered)
                    should_capture_deferred_record = (
                        defer_chunk_page_register
                        and self._keepold_chunk_page_deferred_record is None)
                    if should_build_record_entries or should_capture_deferred_record:
                        slot_cpu = slot_mapping.detach().to(
                            device="cpu", dtype=torch.long).contiguous()
                        prompt_len = int(self.config.prompt_len)
                        block_size = int(page_size)
                        num_logical_blocks = (
                            prompt_len + block_size - 1) // block_size
                        logical_to_physical: list[int] = []
                        for logical_block in range(num_logical_blocks):
                            pos = logical_block * block_size
                            if pos >= int(slot_cpu.numel()):
                                break
                            slot = int(slot_cpu[pos].item())
                            if slot < 0:
                                raise ValueError(
                                    "chunk_page record got negative slot: "
                                    f"layer={int(layer_id)}, pos={pos}, slot={slot}"
                                )
                            logical_to_physical.append(slot // block_size)
                        if logical_to_physical and should_capture_deferred_record:
                            self._keepold_chunk_page_deferred_record = {
                                "source_id": str(source_id),
                                "logical_to_physical": list(logical_to_physical),
                                "block_size": int(block_size),
                                "dtype": str(src.dtype) if src.dtype else "",
                                "kv_layout": str(src.kv_layout) if src.kv_layout else "",
                            }
                            t_rec1 = time.perf_counter()
                            index_stats = index.stats()
                            self._append_metrics({
                                "event": "selective_chunk_page_deferred_register_ready",
                                "keepold_phase": self._keepold_phase,
                                "source_id": str(source_id),
                                "post_register_candidate_chunks": int(
                                    len(self._keepold_chunk_page_post_register_infos)),
                                "record_ms": float((t_rec1 - t_rec0) * 1000.0),
                                "timing_v2_chunk_page_record_ms": 0.0,
                                **self._chunk_page_index_stats_payload(index_stats),
                                "ts_unix": time.time(),
                            })
                            should_build_record_entries = False

                        if logical_to_physical and should_build_record_entries:
                            entries, rec_stats = build_record_entries(
                                index=index,
                                source_id=source_id,
                                prompt_token_ids=(
                                    self.config.keepold_prompt_token_ids or []),
                                chunk_infos=(self.config.keepold_chunk_infos
                                             or []),
                                logical_to_physical=logical_to_physical,
                                block_size=block_size,
                                hash_algo=self.config.
                                keepold_chunk_page_hash_algo,
                                model_id=self.config.keepold_model_id,
                                dtype=src.dtype,
                                kv_layout=src.kv_layout,
                                no_replace_existing_hash=bool(
                                    self.config.
                                    keepold_chunk_page_no_replace_existing_hash),
                                max_entries_per_hash=int(
                                    self.config.
                                    keepold_chunk_page_max_entries_per_hash),
                            )
                            register_success = 0
                            for entry in entries:
                                if index.register_entry(entry):
                                    register_success += 1
                            rec_stats["register_success"] = int(
                                register_success)
                            register_rejected = max(
                                0, int(len(entries)) - int(register_success))
                            if register_rejected > 0:
                                rec_stats[
                                    "register_skipped_duplicate_hash"] = int(
                                        rec_stats.get(
                                            "register_skipped_duplicate_hash",
                                            0)) + int(register_rejected)
                                rec_stats[
                                    "register_skipped_bucket_full"] = int(
                                        rec_stats.get(
                                            "register_skipped_bucket_full",
                                            0)) + int(register_rejected)
                            emit_stats.update(rec_stats)
                            self._keepold_chunk_page_entry_registered = True
                        elif should_build_record_entries:
                            self._chunk_page_emit_metrics(
                                phase=self._keepold_phase,
                                status="skipped",
                                reason="empty_logical_to_physical",
                            )

                        if should_build_record_entries:
                            t_rec1 = time.perf_counter()
                            index_stats = index.stats()
                            index_payload = self._chunk_page_index_stats_payload(
                                index_stats)
                            emit_stats.update({
                                "record_ms": float((t_rec1 - t_rec0) * 1000.0),
                                "timing_v2_chunk_page_record_ms":
                                float((t_rec1 - t_rec0) * 1000.0),
                                "lru_evictions": int(index_stats["lru_evictions"]),
                                **index_payload,
                                "chunk_page_register_attempts": int(
                                    rec_stats.get("register_attempts", 0)),
                                "chunk_page_register_success": int(
                                    rec_stats.get("register_success", 0)),
                                "chunk_page_register_skipped_duplicate_hash": int(
                                    rec_stats.get(
                                        "register_skipped_duplicate_hash", 0)),
                                "chunk_page_register_skipped_bucket_full": int(
                                    rec_stats.get(
                                        "register_skipped_bucket_full", 0)),
                            })
                            self._chunk_page_emit_metrics(
                                phase=self._keepold_phase,
                                status="recorded",
                                reason="dry_run_only",
                                fields=emit_stats,
                            )

        pipeline_t0 = self._layer_kv_pipeline_starts.pop(int(layer_id), None)
        if pipeline_t0 is not None:
            pipeline_ms = float((t1 - pipeline_t0) * 1000.0)
            self._append_metrics({
                "event": "selective_layer_kv_pipeline_total",
                "layer_name": layer_name,
                "layer_id": int(layer_id),
                "num_actual_tokens": int(num_actual_tokens),
                "keepold_phase": self._keepold_phase,
                "t_layer_kv_pipeline_total_ms": float(pipeline_ms),
                "timing_v2_layer_pipeline_envelope_ms":
                float(pipeline_ms),
                "ts_unix": time.time(),
            })
            if self._keepold_diag_layer_breakdown_enabled:
                compose_diag = self.pop_layer_compose_diag(layer_id=int(layer_id))
                if compose_diag is None:
                    compose_diag = {
                        "layer_id": int(layer_id),
                        "layer_name": str(layer_name),
                        "request_phase": self._diag_request_phase(),
                        "backend": self._diag_backend_label(),
                        "exec_granularity": self._diag_exec_granularity_label(),
                        **self._diag_rank_meta(),
                    }

                def _f(key: str, default: float = 0.0) -> float:
                    try:
                        v = float(compose_diag.get(key, default))
                    except (TypeError, ValueError):
                        v = float(default)
                    if math.isnan(v) or math.isinf(v):
                        return float(default)
                    return float(v)

                component_sum_ms = (
                    _f("diag_layer_source_prepare_ms")
                    + _f("diag_layer_cpu_subset_select_ms")
                    + _f("diag_layer_cpu_subset_pin_ms")
                    + _f("diag_layer_cpu_kv_h2d_submit_ms")
                    + _f("diag_layer_gpu_index_scatter_submit_ms")
                    + _f("diag_layer_old_page_gather_submit_ms")
                    + _f("diag_layer_rope_reposition_submit_ms")
                    + float(page_write_submit_ms)
                )
                compose_residual_ms = _f("diag_layer_compose_residual_ms")
                compose_wall_known_sum_ms = _f(
                    "diag_layer_compose_wall_known_sum_ms")
                compose_wall_residual_ms = _f(
                    "diag_layer_compose_wall_residual_ms")
                component_sum_with_compose_residual_ms = float(
                    component_sum_ms + compose_residual_ms)
                component_sum_with_compose_wall_ms = float(
                    compose_wall_known_sum_ms
                    + compose_wall_residual_ms
                    + float(page_write_submit_ms)
                    + float(page_write_wait_ms))
                diag_payload = {
                    "event": "diag_layer_pipeline_breakdown",
                    "sample_scope": "request_local",
                    "layer_name": str(layer_name),
                    "layer_id": int(layer_id),
                    "request_phase": str(compose_diag.get("request_phase", "")),
                    "backend": str(compose_diag.get("backend", "cpu")),
                    "exec_granularity": str(
                        compose_diag.get("exec_granularity", "cpu")),
                    "keepold_phase": str(compose_diag.get("keepold_phase", "")),
                    "rank": int(compose_diag.get("rank", 0)),
                    "local_rank": int(compose_diag.get("local_rank", 0)),
                    "world_size": int(compose_diag.get("world_size", 1)),
                    "tensor_parallel_size": int(
                        compose_diag.get("tensor_parallel_size", 1)),
                    "diag_layer_pipeline_envelope_ms": float(pipeline_ms),
                    "diag_layer_source_prepare_ms": _f(
                        "diag_layer_source_prepare_ms"),
                    "diag_layer_cpu_subset_select_ms": _f(
                        "diag_layer_cpu_subset_select_ms"),
                    "diag_layer_cpu_subset_pin_ms": _f(
                        "diag_layer_cpu_subset_pin_ms"),
                    "diag_layer_cpu_kv_h2d_submit_ms": _f(
                        "diag_layer_cpu_kv_h2d_submit_ms"),
                    "diag_layer_cpu_kv_h2d_device_ms": _f(
                        "diag_layer_cpu_kv_h2d_device_ms"),
                    "diag_layer_cpu_kv_h2d_wait_ms": _f(
                        "diag_layer_cpu_kv_h2d_wait_ms"),
                    "diag_layer_gpu_index_scatter_submit_ms": _f(
                        "diag_layer_gpu_index_scatter_submit_ms"),
                    "diag_layer_gpu_index_scatter_device_ms": _f(
                        "diag_layer_gpu_index_scatter_device_ms"),
                    "diag_layer_gpu_index_scatter_wait_ms": _f(
                        "diag_layer_gpu_index_scatter_wait_ms"),
                    "diag_layer_reuse_buffer_zero_submit_ms": _f(
                        "diag_layer_reuse_buffer_zero_submit_ms"),
                    "diag_layer_reuse_buffer_zero_device_ms": _f(
                        "diag_layer_reuse_buffer_zero_device_ms"),
                    "diag_layer_reuse_buffer_zero_wait_ms": _f(
                        "diag_layer_reuse_buffer_zero_wait_ms"),
                    "diag_layer_reuse_staging_shape_check_ms": _f(
                        "diag_layer_reuse_staging_shape_check_ms"),
                    "diag_layer_cpu_staging_shape_check_ms": _f(
                        "diag_layer_cpu_staging_shape_check_ms"),
                    "diag_layer_staging_alloc_ms": _f(
                        "diag_layer_staging_alloc_ms"),
                    "diag_layer_src_reshape_ms": _f(
                        "diag_layer_src_reshape_ms"),
                    "diag_layer_position_index_select_ms": _f(
                        "diag_layer_position_index_select_ms"),
                    "diag_layer_get_positions_ms": _f(
                        "diag_layer_get_positions_ms"),
                    "diag_layer_cpu_subset_fetch_ms": _f(
                        "diag_layer_cpu_subset_fetch_ms"),
                    "diag_layer_cpu_subset_reshape_ms": _f(
                        "diag_layer_cpu_subset_reshape_ms"),
                    "diag_layer_cpu_subset_to_staging_submit_ms": _f(
                        "diag_layer_cpu_subset_to_staging_submit_ms"),
                    "diag_layer_staging_to_reuse_scatter_submit_ms": _f(
                        "diag_layer_staging_to_reuse_scatter_submit_ms"),
                    "diag_layer_old_page_gather_submit_ms": _f(
                        "diag_layer_old_page_gather_submit_ms"),
                    "diag_layer_old_page_gather_device_ms": _f(
                        "diag_layer_old_page_gather_device_ms"),
                    "diag_layer_old_page_gather_wait_ms": _f(
                        "diag_layer_old_page_gather_wait_ms"),
                    "diag_layer_old_page_precheck_ms": _f(
                        "diag_layer_old_page_precheck_ms"),
                    "diag_layer_old_page_call_wrapper_ms": _f(
                        "diag_layer_old_page_call_wrapper_ms"),
                    "diag_layer_old_page_fused_call_ms": _f(
                        "diag_layer_old_page_fused_call_ms"),
                    "diag_layer_old_page_fused_total_ms": _f(
                        "diag_layer_old_page_fused_total_ms"),
                    "diag_layer_rope_reposition_submit_ms": _f(
                        "diag_layer_rope_reposition_submit_ms"),
                    "diag_layer_rope_reposition_device_ms": _f(
                        "diag_layer_rope_reposition_device_ms"),
                    "diag_layer_rope_wait_ms": _f("diag_layer_rope_wait_ms"),
                    "diag_layer_rope_prepare_ms": _f(
                        "diag_layer_rope_prepare_ms"),
                    "diag_layer_rope_total_ms": _f("diag_layer_rope_total_ms"),
                    "diag_layer_compose_python_overhead_ms": _f(
                        "diag_layer_compose_python_overhead_ms"),
                    "diag_layer_compose_total_ms": _f(
                        "diag_layer_compose_total_ms"),
                    "diag_layer_compose_known_sum_ms": _f(
                        "diag_layer_compose_known_sum_ms"),
                    "diag_layer_compose_residual_ms": float(
                        compose_residual_ms),
                    "diag_layer_compose_wall_control_ms": _f(
                        "diag_layer_compose_wall_control_ms"),
                    "diag_layer_compose_wall_zero_ms": _f(
                        "diag_layer_compose_wall_zero_ms"),
                    "diag_layer_compose_wall_cpu_source_ms": _f(
                        "diag_layer_compose_wall_cpu_source_ms"),
                    "diag_layer_compose_wall_cpu_h2d_ms": _f(
                        "diag_layer_compose_wall_cpu_h2d_ms"),
                    "diag_layer_compose_wall_cpu_scatter_ms": _f(
                        "diag_layer_compose_wall_cpu_scatter_ms"),
                    "diag_layer_compose_wall_gpu_source_ms": _f(
                        "diag_layer_compose_wall_gpu_source_ms"),
                    "diag_layer_compose_wall_rope_ms": _f(
                        "diag_layer_compose_wall_rope_ms"),
                    "diag_layer_compose_wall_return_pack_ms": _f(
                        "diag_layer_compose_wall_return_pack_ms"),
                    "diag_layer_compose_wall_known_sum_ms": _f(
                        "diag_layer_compose_wall_known_sum_ms"),
                    "diag_layer_compose_wall_residual_ms": _f(
                        "diag_layer_compose_wall_residual_ms"),
                    "diag_layer_token_branch_enter_ms": _f(
                        "diag_layer_token_branch_enter_ms"),
                    "diag_layer_bridge_lookup_ms": _f(
                        "diag_layer_bridge_lookup_ms"),
                    "diag_layer_switch_index_prepare_ms": _f(
                        "diag_layer_switch_index_prepare_ms"),
                    "diag_layer_cpu_reuse_idx_prepare_ms": _f(
                        "diag_layer_cpu_reuse_idx_prepare_ms"),
                    "diag_layer_dense_check_ms": _f(
                        "diag_layer_dense_check_ms"),
                    "diag_layer_token_coverage_check_ms": _f(
                        "diag_layer_token_coverage_check_ms"),
                    "diag_layer_old_page_layout_infer_ms": _f(
                        "diag_layer_old_page_layout_infer_ms"),
                    "diag_layer_fused_kv_precheck_ms": _f(
                        "diag_layer_fused_kv_precheck_ms"),
                    "diag_layer_overlap_enabled": _f(
                        "diag_layer_overlap_enabled"),
                    "diag_layer_overlap_used": _f(
                        "diag_layer_overlap_used"),
                    "diag_layer_overlap_reason": str(
                        compose_diag.get("diag_layer_overlap_reason", "")),
                    "diag_layer_cpu_stream_submit_ms": _f(
                        "diag_layer_cpu_stream_submit_ms"),
                    "diag_layer_old_page_stream_submit_ms": _f(
                        "diag_layer_old_page_stream_submit_ms"),
                    "diag_layer_overlap_wait_submit_ms": _f(
                        "diag_layer_overlap_wait_submit_ms"),
                    "diag_layer_overlap_saved_estimated_ms": _f(
                        "diag_layer_overlap_saved_estimated_ms"),
                    "diag_layer_metrics_append_ms": _f(
                        "diag_layer_metrics_append_ms"),
                    "diag_layer_return_pack_ms": _f(
                        "diag_layer_return_pack_ms"),
                    "diag_layer_rope_impl": str(
                        compose_diag.get("diag_layer_rope_impl", "")),
                    "diag_layer_rope_tokens": int(
                        compose_diag.get("diag_layer_rope_tokens", 0)),
                    "diag_layer_page_cache_write_submit_ms":
                    float(page_write_submit_ms),
                    "diag_layer_page_cache_write_device_ms":
                    float(page_device_ms),
                    "diag_layer_page_cache_write_wait_ms":
                    float(page_write_wait_ms),
                    "diag_layer_page_cache_write_tokens": int(num_actual_tokens),
                    "diag_layer_page_cache_write_mode": "reshape_and_cache_flash",
                    "diag_layer_component_sum_ms": float(component_sum_ms),
                    "diag_layer_component_sum_with_compose_residual_ms": float(
                        component_sum_with_compose_residual_ms),
                    "diag_layer_component_sum_with_compose_wall_ms": float(
                        component_sum_with_compose_wall_ms),
                    "diag_layer_unaccounted_ms": float(
                        pipeline_ms - component_sum_ms),
                    "diag_layer_unaccounted_after_compose_ms": float(
                        pipeline_ms - component_sum_with_compose_residual_ms),
                    "diag_layer_unaccounted_after_compose_wall_ms": float(
                        pipeline_ms - component_sum_with_compose_wall_ms),
                    "diag_old_page_tokens": int(
                        compose_diag.get("diag_old_page_tokens", 0)),
                    "diag_cpu_source_tokens": int(
                        compose_diag.get("diag_cpu_source_tokens", 0)),
                    "diag_gpu_target_tokens": int(
                        compose_diag.get("diag_gpu_target_tokens", 0)),
                    "diag_old_page_fused_kv_used": int(
                        compose_diag.get("diag_old_page_fused_kv_used", 0)),
                    "diag_old_page_fused_kv_reason": str(
                        compose_diag.get("diag_old_page_fused_kv_reason", "")),
                    "diag_old_page_fused_span_count": int(
                        compose_diag.get("diag_old_page_fused_span_count", 0)),
                    "diag_old_page_fused_token_count": int(
                        compose_diag.get("diag_old_page_fused_token_count", 0)),
                    "diag_old_page_fused_call_ms": _f(
                        "diag_old_page_fused_call_ms"),
                    "diag_old_page_fused_total_ms": _f(
                        "diag_old_page_fused_total_ms"),
                    "diag_old_page_submit_ms": _f(
                        "diag_layer_old_page_gather_submit_ms"),
                    "diag_old_page_device_ms": _f(
                        "diag_layer_old_page_gather_device_ms"),
                    "diag_token_old_slots_build_ms": _f(
                        "diag_token_old_slots_build_ms"),
                    "diag_token_old_slots_count": int(
                        compose_diag.get("diag_token_old_slots_count", 0)),
                    "diag_token_dst_idx_count": int(
                        compose_diag.get("diag_token_dst_idx_count", 0)),
                    "diag_token_zero_submit_ms": _f(
                        "diag_token_zero_submit_ms"),
                    "diag_token_zero_skipped": int(
                        compose_diag.get("diag_token_zero_skipped", 0)),
                    "diag_token_zero_skip_reason": str(
                        compose_diag.get("diag_token_zero_skip_reason", "")),
                    "diag_token_full_coverage_ready": int(
                        compose_diag.get("diag_token_full_coverage_ready", 0)),
                    "diag_token_full_coverage_reason": str(
                        compose_diag.get("diag_token_full_coverage_reason", "")),
                    "diag_token_full_coverage_gpu_rows": int(
                        compose_diag.get("diag_token_full_coverage_gpu_rows", 0)),
                    "diag_token_full_coverage_cpu_rows": int(
                        compose_diag.get("diag_token_full_coverage_cpu_rows", 0)),
                    "diag_token_full_coverage_total_rows": int(
                        compose_diag.get("diag_token_full_coverage_total_rows", 0)),
                    "diag_token_full_coverage_overlap_rows": int(
                        compose_diag.get("diag_token_full_coverage_overlap_rows", 0)),
                    "diag_token_full_coverage_missing_rows": int(
                        compose_diag.get("diag_token_full_coverage_missing_rows", 0)),
                    "diag_layer_compose_fixed_total_ms": _f(
                        "diag_layer_compose_fixed_total_ms"),
                    "diag_layer_compose_layout_cache_hit": int(
                        compose_diag.get("diag_layer_compose_layout_cache_hit",
                                         0)),
                    "diag_layer_compose_bridge_cache_hit": int(
                        compose_diag.get("diag_layer_compose_bridge_cache_hit",
                                         0)),
                    "diag_layer_compose_metrics_minimal_used": int(
                        compose_diag.get(
                            "diag_layer_compose_metrics_minimal_used", 0)),
                    "diag_span_plan_build_ms": _f("diag_span_plan_build_ms"),
                    "diag_span_gpu_span_count": int(
                        compose_diag.get("diag_span_gpu_span_count", 0)),
                    "diag_span_cpu_span_count": int(
                        compose_diag.get("diag_span_cpu_span_count", 0)),
                    "diag_span_cpu_copy_tokens": int(
                        compose_diag.get("diag_span_cpu_copy_tokens", 0)),
                    "diag_span_no_dense_copy": int(
                        compose_diag.get("diag_span_no_dense_copy", 0)),
                    "diag_span_python_fallback_used": int(
                        compose_diag.get("diag_span_python_fallback_used", 0)),
                    "diag_full_cpu_copy_tokens": int(
                        compose_diag.get("diag_full_cpu_copy_tokens", 0)),
                    "diag_full_cpu_copy_submit_ms": _f(
                        "diag_full_cpu_copy_submit_ms"),
                    "diag_full_cpu_copy_device_ms": _f(
                        "diag_full_cpu_copy_device_ms"),
                    "ts_unix": time.time(),
                }
                self.buffer_layer_pipeline_diag_event(diag_payload)

    def expand_hidden_states(self, active_hidden_states: torch.Tensor) -> torch.Tensor:
        if not self._active:
            raise RuntimeError("expand_hidden_states called when runtime inactive")
        recompute_pos = self.get_recompute_positions(active_hidden_states.device)
        full_hidden = torch.zeros(
            (self._forward_prompt_len, int(active_hidden_states.shape[1])),
            dtype=active_hidden_states.dtype,
            device=active_hidden_states.device,
        )
        full_hidden.index_copy_(0, recompute_pos, active_hidden_states)
        return full_hidden


def get_selective_recompute_runtime() -> Optional[SelectiveRecomputeRuntime]:
    global _SELECTIVE_RUNTIME_SINGLETON, _SELECTIVE_RUNTIME_INITED
    if _SELECTIVE_RUNTIME_INITED:
        return _SELECTIVE_RUNTIME_SINGLETON
    _SELECTIVE_RUNTIME_INITED = True
    return _SELECTIVE_RUNTIME_SINGLETON


def configure_selective_recompute_runtime(
    *,
    pt_path: Optional[str] = None,
    layer_kv_data: Optional[dict[int, dict[str, torch.Tensor]]] = None,
    prompt_len: int,
    recompute_positions: Sequence[int],
    reuse_old_positions: Optional[Sequence[int]] = None,
    reuse_rope_impl: str = _ROPE_IMPL_NONE,
    lmcache_root: Optional[str] = None,
    prefill_only: bool = True,
    metrics_path: Optional[str] = None,
    dynamic_update_backend: str = _DYNAMIC_UPDATE_BACKEND_CPU_REBUILD,
    keepold_bridge_key: Optional[str] = None,
    keepold_phase: str = _KEEPOLD_PHASE_OFF,
    keepold_switch_prompt_positions: Optional[Sequence[int]] = None,
    keepold_fused_k_mode: str = _KEEPOLD_FUSED_K_OFF,
    keepold_fused_k_chunk_size: int = 0,
    keepold_index_mode: str = _KEEPOLD_INDEX_MODE_SLOT_MAPPING,
    keepold_switch_scope: str = _KEEPOLD_SWITCH_SCOPE_ALL_NON_IMPORTANT,
    keepold_chunk_infos: Optional[Sequence[dict[str, int]]] = None,
    keepold_prompt_token_ids: Optional[Sequence[int]] = None,
    keepold_chunk_page_hash_algo: str = "sha256",
    keepold_chunk_page_max_chunks: int = 128,
    keepold_chunk_page_max_entries_per_hash: int = 8,
    keepold_chunk_page_lru_unit: str = _KEEPOLD_LRU_UNIT_ENTRY,
    keepold_chunk_page_lru_evict_batch_chunks: int = 0,
    keepold_chunk_page_lru_low_watermark_chunks: int = -1,
    keepold_chunk_page_clear_each_sample: bool = True,
    keepold_chunk_page_global_lookup: bool = False,
    keepold_chunk_page_no_replace_existing_hash: bool = False,
    keepold_chunk_page_block_invalidation: bool = False,
    keepold_chunk_page_defer_register: bool = False,
    keepold_chunk_page_register_hit_chunks_after_sample: bool = False,
    keepold_chunk_page_update_entry_on_hit: bool = True,
    keepold_model_id: Optional[str] = None,
    keepold_kv_layout: Optional[str] = None,
    keepold_chunk_page_debug_checks: bool = False,
    keepold_chunk_page_execution_enabled: bool = False,
    keepold_chunk_page_exec_granularity: str = _CHUNK_PAGE_EXEC_GRANULARITY_TOKEN,
    keepold_timing_mode: str = _KEEPOLD_TIMING_MODE_LIGHT,
    keepold_layer_metrics_mode: str = _KEEPOLD_LAYER_METRICS_MODE_FULL,
    keepold_overlap_cpu_gpu_compose: bool = False,
    keepold_token_skip_zero_if_full_coverage: bool = False,
    keepold_cpu_fallback_mode: str = _KEEPOLD_CPU_FALLBACK_MODE_SUBSET,
    keepold_diagnostic_layer_breakdown: bool = False,
    keepold_rank_timing_diagnostic: bool = False,
    keepold_nvtx_diagnostic: bool = False,
    keepold_tensor_parallel_size: int = 1,
    eager_load_source: bool = False,
) -> None:
    global _SELECTIVE_RUNTIME_SINGLETON, _SELECTIVE_RUNTIME_INITED

    pt: Optional[Path] = None
    if pt_path is not None:
        pt = Path(pt_path)
        if not pt.exists():
            raise FileNotFoundError(
                f"Selective recompute pt_path not found: {pt}")
    if pt is None and layer_kv_data is None:
        raise ValueError("Either pt_path or layer_kv_data must be provided.")
    if layer_kv_data is not None:
        if not isinstance(layer_kv_data, dict):
            raise ValueError("layer_kv_data must be a dict.")
        if not layer_kv_data:
            raise ValueError("layer_kv_data is empty.")
    if prompt_len <= 0:
        raise ValueError("prompt_len must be > 0")
    if not recompute_positions:
        raise ValueError("recompute_positions must be non-empty")
    reuse_rope_impl = _normalize_rope_impl(reuse_rope_impl)
    if reuse_rope_impl not in (_ROPE_IMPL_NONE, _ROPE_IMPL_LMCACHE_COPS_DIRECT):
        raise ValueError(
            "Invalid selective reuse_rope_impl: "
            f"{reuse_rope_impl}. Expected one of "
            f"{_ROPE_IMPL_NONE}, {_ROPE_IMPL_LMCACHE_COPS_DIRECT}.")
    dynamic_update_backend = dynamic_update_backend.strip().lower()
    if dynamic_update_backend not in (
            _DYNAMIC_UPDATE_BACKEND_CPU_REBUILD,
            _DYNAMIC_UPDATE_BACKEND_GPU_POSITIONS_ONLY):
        raise ValueError(
            "Invalid selective dynamic_update_backend: "
            f"{dynamic_update_backend}. Expected one of "
            f"{_DYNAMIC_UPDATE_BACKEND_CPU_REBUILD}, "
            f"{_DYNAMIC_UPDATE_BACKEND_GPU_POSITIONS_ONLY}.")
    keepold_phase = _normalize_keepold_phase(keepold_phase)
    keepold_fused_k_mode = _normalize_keepold_fused_k_mode(
        keepold_fused_k_mode)
    if int(keepold_fused_k_chunk_size) < 0:
        raise ValueError("keepold_fused_k_chunk_size must be >= 0")
    keepold_index_mode = str(keepold_index_mode).strip().lower()
    if keepold_index_mode not in (_KEEPOLD_INDEX_MODE_SLOT_MAPPING,
                                  _KEEPOLD_INDEX_MODE_CHUNK_PAGE):
        raise ValueError(
            "Invalid keepold_index_mode: "
            f"{keepold_index_mode}. Expected one of "
            f"{_KEEPOLD_INDEX_MODE_SLOT_MAPPING}, {_KEEPOLD_INDEX_MODE_CHUNK_PAGE}.")
    keepold_switch_scope = str(keepold_switch_scope).strip().lower()
    if keepold_switch_scope not in (_KEEPOLD_SWITCH_SCOPE_D_ONLY,
                                    _KEEPOLD_SWITCH_SCOPE_ALL_NON_IMPORTANT):
        raise ValueError(
            "Invalid keepold_switch_scope: "
            f"{keepold_switch_scope}. Expected one of "
            f"{_KEEPOLD_SWITCH_SCOPE_D_ONLY}, "
            f"{_KEEPOLD_SWITCH_SCOPE_ALL_NON_IMPORTANT}.")
    keepold_chunk_page_hash_algo = normalize_chunk_page_hash_algo(
        keepold_chunk_page_hash_algo)
    if int(keepold_chunk_page_max_chunks) <= 0:
        raise ValueError("keepold_chunk_page_max_chunks must be > 0")
    if int(keepold_chunk_page_max_entries_per_hash) <= 0:
        raise ValueError("keepold_chunk_page_max_entries_per_hash must be > 0")
    keepold_chunk_page_lru_unit = _normalize_keepold_lru_unit(
        keepold_chunk_page_lru_unit)
    if int(keepold_chunk_page_lru_evict_batch_chunks) < 0:
        raise ValueError(
            "keepold_chunk_page_lru_evict_batch_chunks must be >= 0")
    keepold_timing_mode = str(keepold_timing_mode).strip().lower()
    if keepold_timing_mode not in (_KEEPOLD_TIMING_MODE_LIGHT,
                                   _KEEPOLD_TIMING_MODE_CUDA_EVENT):
        raise ValueError(
            "Invalid keepold_timing_mode: "
            f"{keepold_timing_mode}. Expected one of "
            f"{_KEEPOLD_TIMING_MODE_LIGHT}, {_KEEPOLD_TIMING_MODE_CUDA_EVENT}.")
    keepold_layer_metrics_mode = _normalize_keepold_layer_metrics_mode(
        keepold_layer_metrics_mode)
    keepold_chunk_page_exec_granularity = _normalize_chunk_page_exec_granularity(
        keepold_chunk_page_exec_granularity)
    if keepold_index_mode != _KEEPOLD_INDEX_MODE_CHUNK_PAGE:
        keepold_chunk_page_exec_granularity = _CHUNK_PAGE_EXEC_GRANULARITY_TOKEN
    keepold_cpu_fallback_mode = _normalize_keepold_cpu_fallback_mode(
        keepold_cpu_fallback_mode)
    if int(keepold_tensor_parallel_size) <= 0:
        raise ValueError("keepold_tensor_parallel_size must be > 0")
    lmcache_root_path = Path(lmcache_root) if lmcache_root else None
    if lmcache_root_path is not None and not lmcache_root_path.exists():
        raise FileNotFoundError(
            f"lmcache_root does not exist: {lmcache_root_path}")

    cfg = SelectiveRecomputeConfig(
        enabled=True,
        pt_path=pt,
        layer_kv_data=layer_kv_data,
        prompt_len=int(prompt_len),
        recompute_positions=[int(x) for x in recompute_positions],
        reuse_old_positions=([int(x)
                              for x in reuse_old_positions]
                             if reuse_old_positions is not None else None),
        reuse_rope_impl=reuse_rope_impl,
        lmcache_root=lmcache_root_path,
        prefill_only=prefill_only,
        metrics_path=Path(metrics_path) if metrics_path else None,
        dynamic_update_backend=dynamic_update_backend,
        keepold_bridge_key=(str(keepold_bridge_key).strip()
                            if keepold_bridge_key else None),
        keepold_phase=keepold_phase,
        keepold_switch_prompt_positions=(
            [int(x) for x in keepold_switch_prompt_positions]
            if keepold_switch_prompt_positions is not None else None),
        keepold_fused_k_mode=keepold_fused_k_mode,
        keepold_fused_k_chunk_size=int(keepold_fused_k_chunk_size),
        keepold_index_mode=keepold_index_mode,
        keepold_switch_scope=keepold_switch_scope,
        keepold_chunk_infos=(
            [{
                "orig_idx": int(x.get("orig_idx", -1)),
                "start": int(x["start"]),
                "end": int(x["end"]),
            } for x in keepold_chunk_infos]
            if keepold_chunk_infos is not None else None),
        keepold_prompt_token_ids=(
            [int(x) for x in keepold_prompt_token_ids]
            if keepold_prompt_token_ids is not None else None),
        keepold_chunk_page_hash_algo=keepold_chunk_page_hash_algo,
        keepold_chunk_page_max_chunks=int(keepold_chunk_page_max_chunks),
        keepold_chunk_page_max_entries_per_hash=int(
            keepold_chunk_page_max_entries_per_hash),
        keepold_chunk_page_lru_unit=str(keepold_chunk_page_lru_unit),
        keepold_chunk_page_lru_evict_batch_chunks=int(
            keepold_chunk_page_lru_evict_batch_chunks),
        keepold_chunk_page_lru_low_watermark_chunks=int(
            keepold_chunk_page_lru_low_watermark_chunks),
        keepold_chunk_page_clear_each_sample=bool(
            keepold_chunk_page_clear_each_sample),
        keepold_chunk_page_global_lookup=bool(
            keepold_chunk_page_global_lookup),
        keepold_chunk_page_no_replace_existing_hash=bool(
            keepold_chunk_page_no_replace_existing_hash),
        keepold_chunk_page_block_invalidation=bool(
            keepold_chunk_page_block_invalidation),
        keepold_chunk_page_defer_register=bool(
            keepold_chunk_page_defer_register),
        keepold_chunk_page_register_hit_chunks_after_sample=bool(
            keepold_chunk_page_register_hit_chunks_after_sample),
        keepold_chunk_page_update_entry_on_hit=bool(
            keepold_chunk_page_update_entry_on_hit),
        keepold_model_id=(str(keepold_model_id).strip()
                          if keepold_model_id else None),
        keepold_kv_layout=(str(keepold_kv_layout).strip()
                           if keepold_kv_layout else None),
        keepold_chunk_page_debug_checks=bool(keepold_chunk_page_debug_checks),
        keepold_chunk_page_execution_enabled=bool(
            keepold_chunk_page_execution_enabled),
        keepold_chunk_page_exec_granularity=keepold_chunk_page_exec_granularity,
        keepold_timing_mode=keepold_timing_mode,
        keepold_layer_metrics_mode=keepold_layer_metrics_mode,
        keepold_overlap_cpu_gpu_compose=bool(keepold_overlap_cpu_gpu_compose),
        keepold_token_skip_zero_if_full_coverage=bool(
            keepold_token_skip_zero_if_full_coverage),
        keepold_cpu_fallback_mode=keepold_cpu_fallback_mode,
        keepold_diagnostic_layer_breakdown=bool(
            keepold_diagnostic_layer_breakdown),
        keepold_rank_timing_diagnostic=bool(keepold_rank_timing_diagnostic),
        keepold_nvtx_diagnostic=bool(keepold_nvtx_diagnostic),
        keepold_tensor_parallel_size=int(keepold_tensor_parallel_size),
    )

    if _SELECTIVE_RUNTIME_SINGLETON is None:
        _SELECTIVE_RUNTIME_SINGLETON = SelectiveRecomputeRuntime(cfg)
    else:
        _SELECTIVE_RUNTIME_SINGLETON.reconfigure(cfg)
    _SELECTIVE_RUNTIME_INITED = True
    if eager_load_source and _SELECTIVE_RUNTIME_SINGLETON is not None:
        _SELECTIVE_RUNTIME_SINGLETON._load_source_once()
    if (keepold_phase == _KEEPOLD_PHASE_REUSE
            and keepold_fused_k_mode == "cuda_op"):
        ok, reason = prewarm_keepold_cuda_op()
        if not ok:
            logger.warning("keepold cuda_op prewarm failed: %s", reason)


def clear_selective_recompute_runtime() -> None:
    global _SELECTIVE_RUNTIME_SINGLETON, _SELECTIVE_RUNTIME_INITED
    _SELECTIVE_RUNTIME_SINGLETON = None
    _SELECTIVE_RUNTIME_INITED = True


def commit_deferred_chunk_page_entries() -> dict[str, Any]:
    runtime = get_selective_recompute_runtime()
    if runtime is None:
        return {"status": "skipped", "reason": "missing_runtime"}
    return runtime.commit_deferred_chunk_page_entries()


def clear_keepold_chunk_page_index(source_id: Optional[str] = None) -> None:
    clear_chunk_page_index(source_id=source_id)
