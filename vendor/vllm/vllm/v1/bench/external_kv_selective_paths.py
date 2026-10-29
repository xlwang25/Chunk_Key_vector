# SPDX-License-Identifier: Apache-2.0
"""Selective compose path logic (KeyChunk)."""

from __future__ import annotations

from typing import Optional, TYPE_CHECKING
import time

import torch

from .external_kv_common import (
    _DYNAMIC_UPDATE_BACKEND_CPU_REBUILD,
    _DYNAMIC_UPDATE_BACKEND_GPU_POSITIONS_ONLY,
    _KEEPOLD_FUSED_K_CUDA_OP,
    _KEEPOLD_PHASE_REUSE,
    _KEEPOLD_FUSED_K_OFF,
    _KEEPOLD_FUSED_K_TRITON,
    _ROPE_IMPL_NONE,
    _ROPE_IMPL_LMCACHE_COPS_DIRECT,
)

if TYPE_CHECKING:
    from .external_kv_selective import SelectiveRecomputeRuntime

_KEEPOLD_TRITON_GPU_INDEX_MIN_TOKENS = 256
_CHUNK_PAGE_EXEC_GRANULARITY_SPAN = "span"


def compose_full_kv(
    runtime: "SelectiveRecomputeRuntime",
    *,
    layer_id: int,
    new_k: torch.Tensor,
    new_v: torch.Tensor,
    layer_name: str,
    reuse_rope_cos_sin_cache: Optional[torch.Tensor] = None,
    reuse_rope_is_neox_style: bool = True,
    probe_positions: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    self = runtime
    cfg = self.config
    dynamic_update_backend = cfg.dynamic_update_backend
    reuse_rope_impl = cfg.reuse_rope_impl
    device_is_cuda = (new_k.device.type == "cuda")
    emit_layer_metrics_full = bool(self.should_emit_layer_metrics_full())
    keepold_phase = self._keepold_phase
    keepold_index_mode = self._keepold_index_mode
    if not self._active:
        raise RuntimeError("compose_full_kv called when runtime is inactive")
    if self._reuse_positions_dev is None:
        raise RuntimeError("reuse positions are not prepared on device")

    src_pair = None
    using_dynamic_source = False
    if self._dynamic_src_ready:
        src_pair = self._src_layer_kv_dynamic.get(layer_id)
        using_dynamic_source = src_pair is not None
    if src_pair is None:
        src_pair = self._src_layer_kv.get(layer_id)
        using_dynamic_source = False
    if src_pair is None:
        raise KeyError(
            f"Layer {layer_id} is missing in selective KV source for {layer_name}"
        )

    # Layer-level KV pipeline envelope timing starts here and ends after
    # this layer page write completes (recorded in write_layer_kv_to_paged_cache).
    self.mark_layer_kv_pipeline_start(layer_id=layer_id)
    layer_pipeline_nvtx_pushed = self.maybe_nvtx_range_push(
        "keepold_layer_pipeline",
        layer_id=layer_id,
        include_layer_id=True,
    )
    diag_layer_enabled = bool(
        getattr(self, "_keepold_diag_layer_breakdown_enabled", False))
    diag_compose_t0 = time.perf_counter() if diag_layer_enabled else 0.0
    diag_layer_metrics_append_ms = 0.0
    _append_metrics = self._append_metrics

    src_k_raw, src_v_raw = src_pair
    num_kv_heads = int(new_k.shape[1])
    head_size = int(new_k.shape[2])
    t_src_reshape0 = time.perf_counter() if diag_layer_enabled else 0.0
    src_k = self._reshape_to_kv_heads(src_k_raw, num_kv_heads, head_size)
    src_v = self._reshape_to_kv_heads(src_v_raw, int(new_v.shape[1]),
                                      int(new_v.shape[2]))
    diag_layer_src_reshape_ms = float(
        (time.perf_counter() - t_src_reshape0) * 1000.0
    ) if diag_layer_enabled else 0.0

    t_get_pos0 = time.perf_counter() if diag_layer_enabled else 0.0
    reuse_pos = self.get_reuse_positions(new_k.device)
    full_positions = self.get_full_positions(new_k.device)
    reuse_old_pos = self.get_reuse_old_positions(new_k.device)
    recompute_pos = self.get_recompute_positions(new_k.device)
    diag_layer_get_positions_ms = float(
        (time.perf_counter() - t_get_pos0) * 1000.0
    ) if diag_layer_enabled else 0.0
    t_pos_idx0 = time.perf_counter() if diag_layer_enabled else 0.0
    reuse_new_pos = full_positions.index_select(0, reuse_pos)
    diag_layer_position_index_select_ms = float(
        (time.perf_counter() - t_pos_idx0) * 1000.0
    ) if diag_layer_enabled else 0.0
    src_k_eff: torch.Tensor
    src_v_eff: torch.Tensor
    full_reuse_pos_for_index_copy = reuse_pos
    reuse_old_pos_for_rope = reuse_old_pos
    reuse_new_pos_for_rope = reuse_new_pos
    h2d_source_mode = "current_reuse_subset"
    keepold_subset_cpu_source = (
        keepold_phase == _KEEPOLD_PHASE_REUSE
        and dynamic_update_backend == _DYNAMIC_UPDATE_BACKEND_CPU_REBUILD
        and self._keepold_switch_reuse_idx_dev is not None
        and int(self._keepold_switch_reuse_idx_dev.numel()) > 0
        and int(self._keepold_cpu_reuse_idx_cpu.numel()) < int(
            self._reuse_old_positions_cpu.numel()))

    if dynamic_update_backend == _DYNAMIC_UPDATE_BACKEND_GPU_POSITIONS_ONLY:
        base_reuse_pos = self.get_base_reuse_positions(new_k.device)
        base_reuse_old_pos = self.get_base_reuse_old_positions(new_k.device)
        if int(src_k.shape[0]) == int(base_reuse_pos.numel()):
            # Blend-only fast path: keep source in base order and update
            # only recompute/reuse positions. This avoids CPU rebuild.
            src_k_eff = src_k
            src_v_eff = src_v
            full_reuse_pos_for_index_copy = base_reuse_pos
            reuse_old_pos_for_rope = base_reuse_old_pos
            t_pos_idx0 = time.perf_counter() if diag_layer_enabled else 0.0
            reuse_new_pos_for_rope = full_positions.index_select(
                0, base_reuse_pos)
            if diag_layer_enabled:
                diag_layer_position_index_select_ms += float(
                    (time.perf_counter() - t_pos_idx0) * 1000.0)
            h2d_source_mode = "base_reuse_full"
        else:
            _append_metrics({
                "event": "selective_dynamic_source_fallback",
                "layer_name": layer_name,
                "layer_id": layer_id,
                "using_dynamic_source": bool(using_dynamic_source),
                "src_tokens": int(src_k.shape[0]),
                "reuse_tokens": int(reuse_pos.numel()),
                "dynamic_update_backend": dynamic_update_backend,
                "reason": "base_reuse_len_mismatch",
                "ts_unix": time.time(),
            })
            # Fall back to the legacy CPU subset path below.
            if int(src_k.shape[0]) == int(reuse_pos.numel()):
                src_k_eff = src_k
                src_v_eff = src_v
            else:
                if int(reuse_old_pos.numel()) != int(reuse_pos.numel()):
                    raise ValueError(
                        f"Layer {layer_id} old-position token mismatch: "
                        f"reuse_old_pos={int(reuse_old_pos.numel())}, "
                        f"reuse_pos={int(reuse_pos.numel())}"
                    )
                if int(reuse_old_pos.numel()) > 0:
                    max_old = int(reuse_old_pos.max().item())
                    if max_old >= int(src_k.shape[0]):
                        raise ValueError(
                            f"Layer {layer_id} old-position index out of range: "
                            f"max_old={max_old}, src_k={int(src_k.shape[0])}"
                        )
                idx_cpu = self._reuse_old_positions_cpu
                if keepold_subset_cpu_source:
                    idx_cpu = idx_cpu.index_select(
                        0, self._keepold_cpu_reuse_idx_cpu).contiguous()
                src_k_eff = src_k.index_select(0, idx_cpu).contiguous()
                src_v_eff = src_v.index_select(0, idx_cpu).contiguous()
                if device_is_cuda:
                    if not src_k_eff.is_pinned():
                        src_k_eff = src_k_eff.pin_memory()
                    if not src_v_eff.is_pinned():
                        src_v_eff = src_v_eff.pin_memory()
    else:
        # Legacy/default path for keychunk and non-blend callers.
        if int(src_k.shape[0]) == int(reuse_pos.numel()):
            src_k_eff = src_k
            src_v_eff = src_v
        else:
            _append_metrics({
                "event": "selective_dynamic_source_fallback",
                "layer_name": layer_name,
                "layer_id": layer_id,
                "using_dynamic_source": bool(using_dynamic_source),
                "src_tokens": int(src_k.shape[0]),
                "reuse_tokens": int(reuse_pos.numel()),
                "dynamic_update_backend": dynamic_update_backend,
                "reason": "legacy_cpu_subset_path",
                "ts_unix": time.time(),
            })
            if int(reuse_old_pos.numel()) != int(reuse_pos.numel()):
                raise ValueError(
                    f"Layer {layer_id} old-position token mismatch: "
                    f"reuse_old_pos={int(reuse_old_pos.numel())}, "
                    f"reuse_pos={int(reuse_pos.numel())}"
                )
            if int(reuse_old_pos.numel()) > 0:
                max_old = int(reuse_old_pos.max().item())
                if max_old >= int(src_k.shape[0]):
                    raise ValueError(
                        f"Layer {layer_id} old-position index out of range: "
                        f"max_old={max_old}, src_k={int(src_k.shape[0])}"
                    )
            # src_k/src_v are CPU tensors in source order. Slice them by
            # the current reuse_old mapping.
            idx_cpu = self._reuse_old_positions_cpu
            if keepold_subset_cpu_source:
                idx_cpu = idx_cpu.index_select(
                    0, self._keepold_cpu_reuse_idx_cpu).contiguous()
            src_k_eff = src_k.index_select(0, idx_cpu).contiguous()
            src_v_eff = src_v.index_select(0, idx_cpu).contiguous()
            # Keep dynamic subset pageable->pinned behavior consistent with the
            # main path, so H2D can stay non-blocking.
            if device_is_cuda:
                if not src_k_eff.is_pinned():
                    src_k_eff = src_k_eff.pin_memory()
                if not src_v_eff.is_pinned():
                    src_v_eff = src_v_eff.pin_memory()

    if int(new_k.shape[0]) != int(recompute_pos.numel()):
        raise ValueError(
            f"Layer {layer_id} recompute token mismatch: "
            f"new_k={int(new_k.shape[0])}, recompute_pos={int(recompute_pos.numel())}"
        )

    diag_layer_reuse_buffer_zero_submit_ms = 0.0
    diag_layer_reuse_buffer_zero_device_ms = float("nan")
    diag_layer_reuse_staging_shape_check_ms = 0.0
    diag_layer_cpu_staging_shape_check_ms = 0.0
    diag_layer_staging_alloc_ms = 0.0
    diag_layer_cpu_subset_fetch_ms = 0.0
    diag_layer_cpu_subset_reshape_ms = 0.0
    diag_layer_cpu_subset_to_staging_submit_ms = 0.0
    diag_layer_staging_to_reuse_scatter_submit_ms = 0.0
    diag_layer_cpu_kv_h2d_wait_ms = 0.0
    diag_layer_gpu_index_scatter_wait_ms = 0.0
    diag_layer_old_page_gather_wait_ms = 0.0
    diag_layer_old_page_precheck_ms = 0.0
    diag_layer_old_page_call_wrapper_ms = 0.0
    diag_layer_reuse_buffer_zero_wait_ms = 0.0
    diag_layer_rope_prepare_ms = 0.0
    diag_layer_rope_total_ms = 0.0
    diag_layer_compose_python_overhead_ms = 0.0
    diag_layer_compose_total_ms = 0.0
    diag_layer_compose_known_sum_ms = 0.0
    diag_layer_compose_residual_ms = 0.0
    diag_layer_compose_wall_control_ms = 0.0
    diag_layer_compose_wall_zero_ms = 0.0
    diag_layer_compose_wall_cpu_source_ms = 0.0
    diag_layer_compose_wall_cpu_h2d_ms = 0.0
    diag_layer_compose_wall_cpu_scatter_ms = 0.0
    diag_layer_compose_wall_gpu_source_ms = 0.0
    diag_layer_compose_wall_rope_ms = 0.0
    diag_layer_compose_wall_return_pack_ms = 0.0
    diag_layer_compose_wall_known_sum_ms = 0.0
    diag_layer_compose_wall_residual_ms = 0.0
    diag_layer_token_branch_enter_ms = 0.0
    diag_layer_bridge_lookup_ms = 0.0
    diag_layer_switch_index_prepare_ms = 0.0
    diag_layer_cpu_reuse_idx_prepare_ms = 0.0
    diag_layer_token_coverage_check_ms = 0.0
    diag_layer_dense_check_ms = 0.0
    diag_layer_old_page_layout_infer_ms = 0.0
    diag_layer_fused_kv_precheck_ms = 0.0
    diag_layer_return_pack_ms = 0.0
    diag_layer_overlap_enabled = 0.0
    diag_layer_overlap_used = 0.0
    diag_layer_overlap_reason = "inactive"
    diag_layer_cpu_stream_submit_ms = 0.0
    diag_layer_old_page_stream_submit_ms = 0.0
    diag_layer_overlap_wait_submit_ms = 0.0
    diag_layer_overlap_saved_estimated_ms = 0.0
    diag_layer_cpu_fallback_span_direct_used = 0.0
    diag_layer_cpu_fallback_span_count = 0.0
    diag_layer_cpu_fallback_span_tokens = 0.0
    diag_layer_cpu_fallback_span_copy_submit_ms = 0.0
    diag_compose_fields: Optional[dict[str, object]] = None
    token_zero_submit_ms = 0.0
    token_zero_skipped = 0
    token_zero_skip_reason = "inactive"
    token_full_coverage_ready = 0
    token_full_coverage_reason = "inactive"
    token_full_coverage_gpu_rows = 0
    token_full_coverage_cpu_rows = 0
    token_full_coverage_total_rows = 0
    token_full_coverage_overlap_rows = 0
    token_full_coverage_missing_rows = 0
    compose_fixed_total_ms = 0.0
    compose_layout_cache_hit = 0
    compose_bridge_cache_hit = 0
    compose_metrics_minimal_used = 0
    compose_return_pack_ms = 0.0

    target_tokens = int(src_k_eff.shape[0])
    if keepold_subset_cpu_source:
        target_tokens = int(full_reuse_pos_for_index_copy.numel())
    target_k_shape = (target_tokens, num_kv_heads, head_size)
    target_v_shape = (target_tokens, int(new_v.shape[1]),
                      int(new_v.shape[2]))
    t_reuse_shape0 = time.perf_counter() if diag_layer_enabled else 0.0
    need_reuse_k_alloc = (
        self._reuse_k_staging is None
        or self._reuse_k_staging.shape != target_k_shape
        or self._reuse_k_staging.dtype != new_k.dtype
        or self._reuse_k_staging.device != new_k.device)
    need_reuse_v_alloc = (
        self._reuse_v_staging is None
        or self._reuse_v_staging.shape != target_v_shape
        or self._reuse_v_staging.dtype != new_v.dtype
        or self._reuse_v_staging.device != new_v.device)
    if diag_layer_enabled:
        diag_layer_reuse_staging_shape_check_ms = float(
            (time.perf_counter() - t_reuse_shape0) * 1000.0)
    if need_reuse_k_alloc:
        t_alloc0 = time.perf_counter() if diag_layer_enabled else 0.0
        self._reuse_k_staging = torch.empty(
            target_k_shape,
            dtype=new_k.dtype,
            device=new_k.device,
        )
        if diag_layer_enabled:
            diag_layer_staging_alloc_ms += float(
                (time.perf_counter() - t_alloc0) * 1000.0)
    if need_reuse_v_alloc:
        t_alloc0 = time.perf_counter() if diag_layer_enabled else 0.0
        self._reuse_v_staging = torch.empty(
            target_v_shape,
            dtype=new_v.dtype,
            device=new_v.device,
        )
        if diag_layer_enabled:
            diag_layer_staging_alloc_ms += float(
                (time.perf_counter() - t_alloc0) * 1000.0)

    reuse_k = self._reuse_k_staging
    reuse_v = self._reuse_v_staging
    keepold_mixed_source = False
    keepold_switch_tokens = 0
    keepold_old_page_tokens = 0
    t_old_page_gather_ms = 0.0
    t_old_slot_select_ms = 0.0
    t_old_page_extract_k_ms = 0.0
    t_old_page_extract_v_ms = 0.0
    t_old_page_dtype_cast_ms = 0.0
    t_old_page_scatter_back_ms = 0.0
    t_old_page_sync_overhead_ms = 0.0
    t_old_page_sync_pre_ms = 0.0
    t_old_page_sync_post_ms = 0.0
    old_page_fused_k_used = False
    old_page_fused_k_mode = cfg.keepold_fused_k_mode
    old_page_fused_k_fallback_reason = "disabled"
    old_page_fused_k_chunks = 0
    t_old_page_fused_k_total_ms = 0.0
    t_old_page_fused_k_gather_ms = 0.0
    t_old_page_fused_k_cast_ms = 0.0
    t_old_page_fused_k_scatter_ms = 0.0
    old_page_fused_kv_used = False
    old_page_fused_kv_fallback_reason = "disabled"
    t_old_page_fused_kv_total_ms = 0.0
    t_old_page_fused_kv_call_ms = 0.0
    span_gpu_fused_used = 0
    span_gpu_fused_reason = "inactive"
    span_fused_span_count = 0
    span_fused_token_count = 0
    span_python_fallback_used = 0
    span_python_fallback_reason = "inactive"
    gpu_index_fused_k_used = False
    gpu_index_fused_k_fallback_reason = "disabled"
    t_gpu_index_fused_k_ms = 0.0
    gpu_index_fused_kv_used = False
    gpu_index_fused_kv_fallback_reason = "disabled"
    t_gpu_index_fused_kv_ms = 0.0
    t_gpu_index_fused_kv_call_ms = 0.0
    t_h2d_submit_ms = 0.0
    t_cpu_subset_prepare_ms = 0.0
    t_cpu_subset_pin_ms = 0.0
    t_pure_h2d_copy_ms = 0.0
    t_gpu_index_copy_ms = 0.0
    t_cpu_subset_to_staging_copy_k_ms = 0.0
    t_cpu_subset_to_staging_copy_v_ms = 0.0
    t_staging_to_reuse_index_copy_k_ms = 0.0
    t_staging_to_reuse_index_copy_v_ms = 0.0
    t_gpu_index_copy_sync_overhead_ms = 0.0
    t_gpu_index_copy_accounted_ms = 0.0
    t_gpu_index_copy_unaccounted_ms = 0.0
    used_prepared_cpu_subset = False
    timing_mode = str(getattr(self.config, "keepold_timing_mode",
                              "light")).strip().lower()
    timing_cuda_event = bool(timing_mode == "cuda_event")
    timing_v2_index_tensor_h2d_submit_ms = float(
        getattr(self, "_keepold_timing_v2_index_tensor_h2d_submit_ms", 0.0))
    timing_v2_index_tensor_h2d_device_ms = float(
        getattr(self, "_keepold_timing_v2_index_tensor_h2d_device_ms",
                float("nan")))
    timing_v2_kv_h2d_k_submit_ms = 0.0
    timing_v2_kv_h2d_v_submit_ms = 0.0
    timing_v2_kv_h2d_device_ms = float("nan")
    timing_v2_gpu_index_scatter_submit_ms = 0.0
    timing_v2_gpu_index_scatter_device_ms = float("nan")
    timing_v2_old_page_gather_submit_ms = 0.0
    timing_v2_old_page_gather_device_ms = float("nan")
    timing_v2_reuse_rope_submit_ms = 0.0
    timing_v2_reuse_rope_device_ms = float("nan")
    timing_v2_reuse_rope_wait_ms = 0.0
    chunk_page_exec_enabled = bool(
        cfg.keepold_chunk_page_execution_enabled
        and keepold_index_mode == "chunk_page"
        and keepold_phase == _KEEPOLD_PHASE_REUSE)
    chunk_page_exec_success = bool(
        getattr(self, "_keepold_chunk_page_exec_success", False))
    chunk_page_exec_fallback_reason = str(
        getattr(self, "_keepold_chunk_page_exec_fallback_reason", "inactive"))
    chunk_page_exec_tokens = int(
        getattr(self, "_keepold_chunk_page_exec_tokens", 0))
    chunk_page_exec_gpu_tokens = int(
        getattr(self, "_keepold_chunk_page_exec_gpu_tokens", 0))
    chunk_page_exec_cpu_tokens = int(
        getattr(self, "_keepold_chunk_page_exec_cpu_tokens", 0))
    chunk_page_exec_old_slots_build_ms = float(
        getattr(self, "_keepold_chunk_page_old_slots_build_ms", 0.0))
    chunk_page_cpu_fallback_tokens = int(
        getattr(self, "_keepold_chunk_page_cpu_fallback_tokens", 0))
    chunk_page_exec_granularity = str(
        getattr(self, "_keepold_chunk_page_exec_granularity", "token"))
    chunk_page_span_exec_ready = bool(
        getattr(self, "_keepold_chunk_page_span_exec_ready", False))
    token_skip_zero_if_full_coverage = bool(
        getattr(cfg, "keepold_token_skip_zero_if_full_coverage", False))
    cpu_fallback_mode = str(
        getattr(self, "_keepold_cpu_fallback_mode_effective", "subset"))
    cpu_fallback_dense_enabled = bool(
        getattr(self, "_keepold_cpu_fallback_dense_enabled", False))
    cpu_fallback_dense_reason = str(
        getattr(self, "_keepold_cpu_fallback_dense_reason", "inactive"))
    cpu_fallback_tokens = int(
        getattr(self, "_keepold_cpu_fallback_tokens",
                chunk_page_exec_cpu_tokens))
    cpu_fallback_ratio = float(
        getattr(self, "_keepold_cpu_fallback_ratio", 0.0))
    cpu_fallback_subset_skipped = bool(
        getattr(self, "_keepold_cpu_fallback_subset_skipped", False))
    cpu_fallback_span_direct_ready = bool(
        getattr(self, "_keepold_cpu_fallback_span_direct_ready", False))
    cpu_fallback_span_direct_reason = str(
        getattr(self, "_keepold_cpu_fallback_span_direct_reason", "inactive"))
    cpu_fallback_span_direct_span_count = int(
        getattr(self, "_keepold_cpu_fallback_span_direct_span_count", 0))
    cpu_fallback_span_direct_tokens = int(
        getattr(self, "_keepold_cpu_fallback_span_direct_tokens", 0))
    cpu_fallback_span_direct_src_starts_cpu = getattr(
        self, "_keepold_cpu_fallback_span_direct_src_starts_cpu",
        torch.empty(0, dtype=torch.long))
    cpu_fallback_span_direct_dst_starts_cpu = getattr(
        self, "_keepold_cpu_fallback_span_direct_dst_starts_cpu",
        torch.empty(0, dtype=torch.long))
    cpu_fallback_span_direct_lengths_cpu = getattr(
        self, "_keepold_cpu_fallback_span_direct_lengths_cpu",
        torch.empty(0, dtype=torch.long))
    cpu_fallback_dense_applied = False
    cpu_fallback_dense_reason_runtime = cpu_fallback_dense_reason
    overlap_config_enabled = bool(
        getattr(cfg, "keepold_overlap_cpu_gpu_compose", False))
    overlap_runtime_eligible = bool(
        getattr(self, "_keepold_overlap_cpu_gpu_eligible", False))
    overlap_runtime_reason = str(
        getattr(self, "_keepold_overlap_cpu_gpu_reason", "disabled"))
    if diag_layer_enabled:
        diag_layer_overlap_enabled = 1.0 if overlap_config_enabled else 0.0
        diag_layer_overlap_reason = (
            overlap_runtime_reason if overlap_config_enabled else "disabled")
    overlap_used = False
    old_page_work_stream: Optional[torch.cuda.Stream] = None
    h2d_tokens = int(src_k_eff.shape[0])
    src_is_pinned_k = bool(src_k_eff.is_pinned())
    src_is_pinned_v = bool(src_v_eff.is_pinned())
    non_blocking_k = (device_is_cuda
                      and src_k_eff.device.type == "cpu"
                      and src_k_eff.is_pinned())
    non_blocking_v = (device_is_cuda
                      and src_v_eff.device.type == "cpu"
                      and src_v_eff.is_pinned())
    h2d_start_event: Optional[torch.cuda.Event] = None
    h2d_end_event: Optional[torch.cuda.Event] = None
    span_gpu_gather_submit_ms = 0.0
    span_gpu_gather_device_ms = float("nan")
    span_cpu_copy_submit_ms = 0.0
    span_cpu_copy_device_ms = float("nan")
    span_cpu_copy_span_count = 0
    span_cpu_copy_tokens = 0
    span_no_dense_copy = 0
    span_mode_active = False

    span_exec_active = bool(
        keepold_phase == _KEEPOLD_PHASE_REUSE
        and dynamic_update_backend == _DYNAMIC_UPDATE_BACKEND_CPU_REBUILD
        and chunk_page_exec_enabled
        and chunk_page_exec_granularity == _CHUNK_PAGE_EXEC_GRANULARITY_SPAN
        and chunk_page_span_exec_ready)

    if span_exec_active:
        layer_snapshot = self.get_keepold_layer_snapshot_cached(
            layer_id=layer_id)
        if layer_snapshot is None:
            _append_metrics({
                "event": "selective_old_page_unavailable",
                "layer_name": layer_name,
                "layer_id": layer_id,
                "reason": "missing_layer_snapshot",
                "keepold_bridge_key": self._keepold_bridge_key,
                "ts_unix": time.time(),
            })
            raise RuntimeError(
                "keepold old-page layer snapshot is missing: "
                f"layer={layer_id}, bridge_key={self._keepold_bridge_key}"
            )

        keepold_mixed_source = True
        span_mode_active = True
        span_no_dense_copy = 1
        # Do not zero the whole reuse buffer on the performance path.
        # Coverage checks below guarantee every row is written exactly once
        # by either CPU span copy or GPU old-page span gather.  Zeroing is kept
        # only for debug checks to expose coverage bugs.
        if bool(getattr(self.config, "keepold_chunk_page_debug_checks", False)):
            zero_start_event: Optional[torch.cuda.Event] = None
            zero_end_event: Optional[torch.cuda.Event] = None
            if (diag_layer_enabled and timing_cuda_event
                    and new_k.device.type == "cuda"):
                zero_start_event = torch.cuda.Event(enable_timing=True)
                zero_end_event = torch.cuda.Event(enable_timing=True)
                zero_start_event.record(torch.cuda.current_stream(new_k.device))
            t_zero0 = time.perf_counter() if diag_layer_enabled else 0.0
            reuse_k.zero_()
            reuse_v.zero_()
            if diag_layer_enabled:
                diag_layer_reuse_buffer_zero_submit_ms = float(
                    (time.perf_counter() - t_zero0) * 1000.0)
            if zero_end_event is not None:
                zero_end_event.record(torch.cuda.current_stream(new_k.device))
                t_zero_wait0 = time.perf_counter()
                torch.cuda.synchronize(new_k.device)
                diag_layer_reuse_buffer_zero_wait_ms += float(
                    (time.perf_counter() - t_zero_wait0) * 1000.0)
                diag_layer_reuse_buffer_zero_device_ms = float(
                    zero_start_event.elapsed_time(zero_end_event))
        full_reuse_tokens = int(full_reuse_pos_for_index_copy.numel())
        if full_reuse_tokens <= 0:
            raise RuntimeError("span exec requires non-empty reuse buffer")
        if int(src_k_eff.shape[0]) < full_reuse_tokens:
            raise RuntimeError(
                "span exec source tokens smaller than reuse tokens: "
                f"layer={layer_id}, src={int(src_k_eff.shape[0])}, "
                f"reuse={full_reuse_tokens}")
        if int(src_v_eff.shape[0]) < full_reuse_tokens:
            raise RuntimeError(
                "span exec source value tokens smaller than reuse tokens: "
                f"layer={layer_id}, src={int(src_v_eff.shape[0])}, "
                f"reuse={full_reuse_tokens}")

        gpu_len_cpu = self._keepold_chunk_page_gpu_span_lengths_cpu
        if gpu_len_cpu.device.type != "cpu" or gpu_len_cpu.dtype != torch.long:
            raise RuntimeError(
                "span exec gpu length metadata is not prepared as cpu int64")
        if not gpu_len_cpu.is_contiguous():
            gpu_len_cpu = gpu_len_cpu.contiguous()
        keepold_switch_tokens = int(
            gpu_len_cpu.sum().item()) if int(gpu_len_cpu.numel()) > 0 else 0
        keepold_old_page_tokens = int(keepold_switch_tokens)

        cpu_src_cpu = self._keepold_chunk_page_span_cpu_copy_src_starts_cpu
        cpu_dst_cpu = self._keepold_chunk_page_span_cpu_copy_dst_starts_cpu
        cpu_len_cpu = self._keepold_chunk_page_span_cpu_copy_lengths_cpu
        if (cpu_src_cpu.device.type != "cpu" or cpu_src_cpu.dtype != torch.long
                or cpu_dst_cpu.device.type != "cpu"
                or cpu_dst_cpu.dtype != torch.long
                or cpu_len_cpu.device.type != "cpu"
                or cpu_len_cpu.dtype != torch.long):
            raise RuntimeError(
                "span exec cpu copy metadata is not prepared as cpu int64")
        if not cpu_src_cpu.is_contiguous():
            cpu_src_cpu = cpu_src_cpu.contiguous()
        if not cpu_dst_cpu.is_contiguous():
            cpu_dst_cpu = cpu_dst_cpu.contiguous()
        if not cpu_len_cpu.is_contiguous():
            cpu_len_cpu = cpu_len_cpu.contiguous()
        if (int(cpu_src_cpu.numel()) != int(cpu_dst_cpu.numel())
                or int(cpu_src_cpu.numel()) != int(cpu_len_cpu.numel())):
            raise RuntimeError("span exec cpu metadata len mismatch")
        span_cpu_copy_span_count = int(cpu_len_cpu.numel())
        span_cpu_copy_tokens = int(
            cpu_len_cpu.sum().item()) if span_cpu_copy_span_count > 0 else 0
        h2d_tokens = int(span_cpu_copy_tokens)
        h2d_source_mode = "mixed_cpu_span_plus_old_page_span"
        cpu_fallback_dense_applied = False
        cpu_fallback_dense_reason_runtime = "span_exec_no_dense_copy"
        cpu_fallback_subset_skipped = True

        if h2d_tokens > 0:
            src_is_pinned_k = bool(src_k_eff.is_pinned())
            src_is_pinned_v = bool(src_v_eff.is_pinned())
            non_blocking_k = (new_k.device.type == "cuda"
                              and src_k_eff.device.type == "cpu"
                              and src_k_eff.is_pinned())
            non_blocking_v = (new_v.device.type == "cuda"
                              and src_v_eff.device.type == "cpu"
                              and src_v_eff.is_pinned())
            if new_k.device.type == "cuda":
                h2d_start_event = torch.cuda.Event(enable_timing=True)
                h2d_end_event = torch.cuda.Event(enable_timing=True)
                h2d_start_event.record(torch.cuda.current_stream(new_k.device))
            kv_h2d_start_event: Optional[torch.cuda.Event] = None
            kv_h2d_end_event: Optional[torch.cuda.Event] = None
            if timing_cuda_event and new_k.device.type == "cuda":
                kv_h2d_start_event = torch.cuda.Event(enable_timing=True)
                kv_h2d_end_event = torch.cuda.Event(enable_timing=True)
                kv_h2d_start_event.record(torch.cuda.current_stream(new_k.device))
            t_copy0 = time.perf_counter()
            t_copy_k0 = t_copy0
            cpu_src_list = cpu_src_cpu.tolist()
            cpu_dst_list = cpu_dst_cpu.tolist()
            cpu_len_list = cpu_len_cpu.tolist()
            for src_start, dst_start, seg_len in zip(
                    cpu_src_list, cpu_dst_list, cpu_len_list):
                if seg_len <= 0:
                    continue
                src_end = src_start + seg_len
                dst_end = dst_start + seg_len
                if (src_start < 0 or src_end > int(src_k_eff.shape[0])
                        or dst_start < 0 or dst_end > full_reuse_tokens):
                    raise RuntimeError(
                        "span exec cpu copy range out of bounds: "
                        f"layer={layer_id}, src=[{src_start},{src_end}), "
                        f"dst=[{dst_start},{dst_end}), "
                        f"src_tokens={int(src_k_eff.shape[0])}, "
                        f"reuse_tokens={full_reuse_tokens}")
                reuse_k[dst_start:dst_end].copy_(
                    src_k_eff[src_start:src_end], non_blocking=non_blocking_k)
            t_copy_k1 = time.perf_counter()
            timing_v2_kv_h2d_k_submit_ms = float((t_copy_k1 - t_copy_k0) * 1000.0)
            t_copy_v0 = t_copy_k1
            for src_start, dst_start, seg_len in zip(
                    cpu_src_list, cpu_dst_list, cpu_len_list):
                if seg_len <= 0:
                    continue
                src_end = src_start + seg_len
                dst_end = dst_start + seg_len
                reuse_v[dst_start:dst_end].copy_(
                    src_v_eff[src_start:src_end], non_blocking=non_blocking_v)
            t_copy_v1 = time.perf_counter()
            timing_v2_kv_h2d_v_submit_ms = float((t_copy_v1 - t_copy_v0) * 1000.0)
            if kv_h2d_end_event is not None:
                kv_h2d_end_event.record(torch.cuda.current_stream(new_k.device))
                t_h2d_wait0 = time.perf_counter()
                torch.cuda.synchronize(new_k.device)
                diag_layer_cpu_kv_h2d_wait_ms += float(
                    (time.perf_counter() - t_h2d_wait0) * 1000.0)
                timing_v2_kv_h2d_device_ms = float(
                    kv_h2d_start_event.elapsed_time(kv_h2d_end_event))
            if h2d_end_event is not None:
                h2d_end_event.record(torch.cuda.current_stream(new_k.device))
            t_copy1 = t_copy_v1
            t_h2d_submit_ms = float((t_copy1 - t_copy0) * 1000.0)
            t_pure_h2d_copy_ms = float(t_h2d_submit_ms)
            span_cpu_copy_submit_ms = float(t_h2d_submit_ms)
            span_cpu_copy_device_ms = float(timing_v2_kv_h2d_device_ms)

        if (self._keepold_span_reuse_old_pos_for_rope_dev is not None
                and self._keepold_span_reuse_old_pos_for_rope_dev.device
                == new_k.device):
            reuse_old_pos_for_rope = (
                self._keepold_span_reuse_old_pos_for_rope_dev.contiguous())
        else:
            rope_old_pos_cpu = self._keepold_span_reuse_old_pos_for_rope_cpu
            if rope_old_pos_cpu.device.type != "cpu":
                rope_old_pos_cpu = rope_old_pos_cpu.to(device="cpu",
                                                       dtype=torch.long)
            else:
                rope_old_pos_cpu = rope_old_pos_cpu.to(dtype=torch.long)
            reuse_old_pos_for_rope = rope_old_pos_cpu.to(
                device=new_k.device, non_blocking=False).contiguous()
        if int(reuse_old_pos_for_rope.numel()) != int(full_reuse_tokens):
            raise RuntimeError(
                "span exec rope position vector size mismatch: "
                f"layer={layer_id}, rope_tokens={int(reuse_old_pos_for_rope.numel())}, "
                f"reuse_tokens={full_reuse_tokens}")
        if keepold_switch_tokens > 0:
            old_page_start_event: Optional[torch.cuda.Event] = None
            old_page_end_event: Optional[torch.cuda.Event] = None
            if timing_cuda_event and new_k.device.type == "cuda":
                old_page_start_event = torch.cuda.Event(enable_timing=True)
                old_page_end_event = torch.cuda.Event(enable_timing=True)
                old_page_start_event.record(torch.cuda.current_stream(new_k.device))
            old_page_nvtx_pushed = self.maybe_nvtx_range_push(
                "keepold_old_page_gather",
                layer_id=layer_id,
                include_layer_id=True,
            )
            t_pg0 = time.perf_counter()
            fused_span_used = False
            fused_span_reason = "not_attempted"
            layout_meta = self.infer_paged_layout_meta_cached(
                layer_id=layer_id,
                cache=layer_snapshot.key_cache,
                num_kv_heads=num_kv_heads,
                head_size=head_size,
            )
            span_dev_ready = bool(
                self._keepold_chunk_page_gpu_span_block_ids_dev is not None
                and self._keepold_chunk_page_gpu_span_page_offsets_dev is not None
                and self._keepold_chunk_page_gpu_span_dst_starts_dev is not None
                and self._keepold_chunk_page_gpu_span_lengths_dev is not None)
            span_fused_span_count = int(
                0 if self._keepold_chunk_page_gpu_span_lengths_dev is None else
                self._keepold_chunk_page_gpu_span_lengths_dev.numel())
            span_fused_token_count = int(
                0 if self._keepold_chunk_page_gpu_span_lengths_dev is None
                or span_fused_span_count == 0 else
                self._keepold_chunk_page_gpu_span_lengths_dev.sum().item())
            if old_page_fused_k_mode != _KEEPOLD_FUSED_K_CUDA_OP:
                fused_span_reason = "fused_mode_not_cuda_op"
            elif layout_meta is None:
                fused_span_reason = "layout_infer_failed"
            elif not self._keepold_old_page_fast_ready:
                fused_span_reason = (
                    f"old_page_fast_not_ready:{self._keepold_old_page_fast_reason}")
            elif not span_dev_ready:
                fused_span_reason = "missing_span_dev_tensors"
            else:
                layout_code, page_size = layout_meta
                t_span_call0 = time.perf_counter()
                fused_span_stats = (
                    self._fused_extract_scatter_kv_from_paged_cache_spans_fast(
                        key_cache=layer_snapshot.key_cache,
                        value_cache=layer_snapshot.value_cache,
                        span_block_ids_i64=(
                            self._keepold_chunk_page_gpu_span_block_ids_dev),
                        span_page_offsets_i64=(
                            self._keepold_chunk_page_gpu_span_page_offsets_dev),
                        span_dst_starts_i64=(
                            self._keepold_chunk_page_gpu_span_dst_starts_dev),
                        span_lengths_i64=(
                            self._keepold_chunk_page_gpu_span_lengths_dev),
                        reuse_k=reuse_k,
                        reuse_v=reuse_v,
                        layout_code=int(layout_code),
                        page_size=int(page_size),
                    ))
                t_span_call1 = time.perf_counter()
                t_old_page_fused_kv_call_ms = float(
                    (t_span_call1 - t_span_call0) * 1000.0)
                old_page_fused_kv_used = bool(
                    fused_span_stats.get("used", False))
                old_page_fused_kv_fallback_reason = str(
                    fused_span_stats.get("reason", "unknown"))
                t_old_page_fused_kv_total_ms = float(
                    fused_span_stats.get("t_total_ms", 0.0) or 0.0)
                span_fused_span_count = int(
                    fused_span_stats.get("span_count", span_fused_span_count) or 0)
                span_fused_token_count = int(
                    fused_span_stats.get("token_count", span_fused_token_count) or 0)
                if old_page_fused_kv_used:
                    fused_span_used = True
                    fused_span_reason = "ok"
                    old_page_fused_k_used = True
                    old_page_fused_k_fallback_reason = "kv_span_kernel"
                    old_page_fused_k_chunks = int(
                        fused_span_stats.get("chunk_count", 0) or 0)
                    t_old_page_fused_k_total_ms = t_old_page_fused_kv_total_ms
                    t_old_page_fused_k_gather_ms = t_old_page_fused_kv_total_ms
                    t_old_page_extract_k_ms += t_old_page_fused_kv_total_ms * 0.5
                    t_old_page_extract_v_ms += t_old_page_fused_kv_total_ms * 0.5
                else:
                    fused_span_reason = str(old_page_fused_kv_fallback_reason)

            span_gpu_fused_used = 1 if fused_span_used else 0
            span_gpu_fused_reason = str(fused_span_reason)
            if not fused_span_used:
                if old_page_fused_kv_fallback_reason == "disabled":
                    old_page_fused_kv_fallback_reason = str(fused_span_reason)
                span_python_fallback_reason = str(fused_span_reason)
                debug_allow_python_fallback = bool(
                    self.config.keepold_chunk_page_debug_checks)
                if not debug_allow_python_fallback:
                    layout_meta_str = (
                        "None" if layout_meta is None else
                        f"layout_code={int(layout_meta[0])},page_size={int(layout_meta[1])}")
                    raise RuntimeError(
                        "span fused CUDA op unavailable in performance path; "
                        f"fused_span_reason={fused_span_reason}; "
                        f"old_page_fused_kv_fallback_reason={old_page_fused_kv_fallback_reason}; "
                        f"layout_meta={layout_meta_str}; "
                        f"old_page_fast_ready={int(self._keepold_old_page_fast_ready)}; "
                        f"old_page_fast_reason={self._keepold_old_page_fast_reason}; "
                        f"span_dev_ready={int(span_dev_ready)}; "
                        f"span_block_dev={int(self._keepold_chunk_page_gpu_span_block_ids_dev is not None)}; "
                        f"span_off_dev={int(self._keepold_chunk_page_gpu_span_page_offsets_dev is not None)}; "
                        f"span_dst_dev={int(self._keepold_chunk_page_gpu_span_dst_starts_dev is not None)}; "
                        f"span_len_dev={int(self._keepold_chunk_page_gpu_span_lengths_dev is not None)}; "
                        f"span_count={int(span_fused_span_count)}; "
                        f"token_count={int(span_fused_token_count)}")
                span_python_fallback_used = 1
                layout_meta_fb = self.infer_paged_layout_meta_cached(
                    layer_id=layer_id,
                    cache=layer_snapshot.key_cache,
                    num_kv_heads=num_kv_heads,
                    head_size=head_size,
                )
                if layout_meta_fb is None:
                    raise RuntimeError(
                        "span exec layout infer failed for old-page gather")
                _, page_size = layout_meta_fb
                gpu_dst_cpu = self._keepold_chunk_page_gpu_span_dst_starts_cpu
                if gpu_dst_cpu.device.type != "cpu":
                    gpu_dst_cpu = gpu_dst_cpu.to(device="cpu", dtype=torch.long)
                else:
                    gpu_dst_cpu = gpu_dst_cpu.to(dtype=torch.long)
                gpu_dst_cpu = gpu_dst_cpu.contiguous()
                gpu_old_abs_cpu = self._keepold_chunk_page_gpu_span_old_abs_starts_cpu
                if gpu_old_abs_cpu.device.type != "cpu":
                    gpu_old_abs_cpu = gpu_old_abs_cpu.to(device="cpu",
                                                         dtype=torch.long)
                else:
                    gpu_old_abs_cpu = gpu_old_abs_cpu.to(dtype=torch.long)
                gpu_old_abs_cpu = gpu_old_abs_cpu.contiguous()
                for j in range(int(gpu_len_cpu.numel())):
                    seg_len = int(gpu_len_cpu[j].item())
                    if seg_len <= 0:
                        continue
                    dst_start = int(gpu_dst_cpu[j].item())
                    dst_end = dst_start + seg_len
                    block_id = int(self._keepold_chunk_page_gpu_span_block_ids_cpu[j].item())
                    page_off = int(
                        self._keepold_chunk_page_gpu_span_page_offsets_cpu[j].item())
                    old_slots = (
                        block_id * int(page_size)
                        + page_off
                        + torch.arange(seg_len, device=new_k.device,
                                       dtype=torch.long))
                    t_extract_k0 = time.perf_counter()
                    old_k = self._extract_tokens_from_paged_cache(
                        layer_snapshot.key_cache, old_slots, num_kv_heads, head_size)
                    t_extract_k1 = time.perf_counter()
                    t_old_page_extract_k_ms += (t_extract_k1 - t_extract_k0) * 1000.0
                    t_extract_v0 = t_extract_k1
                    old_v = self._extract_tokens_from_paged_cache(
                        layer_snapshot.value_cache,
                        old_slots,
                        int(new_v.shape[1]),
                        int(new_v.shape[2]),
                    )
                    t_extract_v1 = time.perf_counter()
                    t_old_page_extract_v_ms += (t_extract_v1 - t_extract_v0) * 1000.0
                    if old_k.dtype != reuse_k.dtype:
                        old_k = old_k.to(dtype=reuse_k.dtype)
                    if old_v.dtype != reuse_v.dtype:
                        old_v = old_v.to(dtype=reuse_v.dtype)
                    reuse_k[dst_start:dst_end].copy_(old_k)
                    reuse_v[dst_start:dst_end].copy_(old_v)
                if old_page_fused_kv_fallback_reason == "disabled":
                    old_page_fused_kv_fallback_reason = str(fused_span_reason)
            else:
                span_python_fallback_used = 0
                span_python_fallback_reason = "none"
            if old_page_end_event is not None:
                old_page_end_event.record(torch.cuda.current_stream(new_k.device))
                t_old_page_wait0 = time.perf_counter()
                torch.cuda.synchronize(new_k.device)
                diag_layer_old_page_gather_wait_ms += float(
                    (time.perf_counter() - t_old_page_wait0) * 1000.0)
                timing_v2_old_page_gather_device_ms = float(
                    old_page_start_event.elapsed_time(old_page_end_event))
            t_pg1 = time.perf_counter()
            t_old_page_gather_ms = float((t_pg1 - t_pg0) * 1000.0)
            self.maybe_nvtx_range_pop(old_page_nvtx_pushed)
            timing_v2_old_page_gather_submit_ms = float(t_old_page_gather_ms)
            span_gpu_gather_submit_ms = float(t_old_page_gather_ms)
            span_gpu_gather_device_ms = float(timing_v2_old_page_gather_device_ms)
            if emit_layer_metrics_full:
                _append_metrics({
                    "event": "selective_old_page_gather",
                    "layer_name": layer_name,
                    "layer_id": layer_id,
                    "keepold_bridge_key": self._keepold_bridge_key,
                    "chunk_page_exec_granularity": chunk_page_exec_granularity,
                    "span_mode_active": 1,
                    "switched_reuse_tokens": int(keepold_switch_tokens),
                    "cpu_reuse_tokens": int(h2d_tokens),
                    "old_page_tokens": int(keepold_old_page_tokens),
                    "span_gpu_fused_used": int(span_gpu_fused_used),
                    "span_gpu_fused_reason": str(span_gpu_fused_reason),
                    "old_page_fused_kv_used": 1 if old_page_fused_kv_used else 0,
                    "old_page_fused_kv_fallback_reason":
                    str(old_page_fused_kv_fallback_reason),
                    "t_old_page_fused_kv_total_ms": float(
                        t_old_page_fused_kv_total_ms),
                    "t_old_page_fused_kv_call_ms": float(
                        t_old_page_fused_kv_call_ms),
                    "span_fused_span_count": int(span_fused_span_count),
                    "span_fused_token_count": int(span_fused_token_count),
                    "span_gpu_gather_submit_ms": float(span_gpu_gather_submit_ms),
                    "span_gpu_gather_device_ms": float(span_gpu_gather_device_ms),
                    "span_python_fallback_used": int(span_python_fallback_used),
                    "span_python_fallback_reason": str(span_python_fallback_reason),
                    "timing_v2_old_page_gather_submit_ms": float(
                        timing_v2_old_page_gather_submit_ms),
                    "timing_v2_old_page_gather_device_ms": float(
                        timing_v2_old_page_gather_device_ms),
                    "chunk_page_execution_enabled":
                    1 if chunk_page_exec_enabled else 0,
                    "chunk_page_exec_success":
                    1 if chunk_page_exec_success else 0,
                    "chunk_page_exec_fallback_reason":
                    chunk_page_exec_fallback_reason,
                    "chunk_page_exec_tokens": int(chunk_page_exec_tokens),
                    "chunk_page_exec_gpu_tokens": int(chunk_page_exec_gpu_tokens),
                    "chunk_page_exec_cpu_tokens": int(chunk_page_exec_cpu_tokens),
                    "chunk_page_exec_old_slots_build_ms":
                    float(chunk_page_exec_old_slots_build_ms),
                    "chunk_page_cpu_fallback_tokens":
                    int(chunk_page_cpu_fallback_tokens),
                    "cpu_fallback_mode": cpu_fallback_mode,
                    "cpu_fallback_dense_enabled":
                    1 if cpu_fallback_dense_enabled else 0,
                    "cpu_fallback_dense_applied":
                    1 if cpu_fallback_dense_applied else 0,
                    "cpu_fallback_dense_reason":
                    cpu_fallback_dense_reason_runtime,
                    "cpu_fallback_tokens": int(cpu_fallback_tokens),
                    "cpu_fallback_ratio": float(cpu_fallback_ratio),
                    "cpu_fallback_subset_skipped":
                    1 if cpu_fallback_subset_skipped else 0,
                    "cpu_fallback_span_direct_ready":
                    1 if cpu_fallback_span_direct_ready else 0,
                    "cpu_fallback_span_direct_reason":
                    cpu_fallback_span_direct_reason,
                    "cpu_fallback_span_direct_span_count":
                    int(cpu_fallback_span_direct_span_count),
                    "cpu_fallback_span_direct_tokens":
                    int(cpu_fallback_span_direct_tokens),
                    "overlap_cpu_gpu_compose_enabled":
                    1 if overlap_config_enabled else 0,
                    "overlap_cpu_gpu_compose_used":
                    1 if overlap_used else 0,
                    "overlap_cpu_gpu_compose_reason":
                    str(diag_layer_overlap_reason),
                    "timing_v2_index_tensor_h2d_submit_ms":
                    float(timing_v2_index_tensor_h2d_submit_ms),
                    "timing_v2_index_tensor_h2d_device_ms":
                    float(timing_v2_index_tensor_h2d_device_ms),
                    "ts_unix": time.time(),
                })

        span_cpu_copy_submit_ms = float(t_h2d_submit_ms)
        span_cpu_copy_device_ms = float(timing_v2_kv_h2d_device_ms)

    elif (keepold_phase == _KEEPOLD_PHASE_REUSE
            and dynamic_update_backend == _DYNAMIC_UPDATE_BACKEND_CPU_REBUILD
            and self._keepold_switch_reuse_idx_dev is not None
            and self._keepold_switch_prompt_positions_dev is not None
            and int(self._keepold_switch_reuse_idx_dev.numel()) > 0):
        t_token_branch0 = time.perf_counter() if diag_layer_enabled else 0.0
        t_bridge_lookup0 = time.perf_counter() if diag_layer_enabled else 0.0
        layer_snapshot = self.get_keepold_layer_snapshot_cached(
            layer_id=layer_id)
        if diag_layer_enabled:
            diag_layer_bridge_lookup_ms += float(
                (time.perf_counter() - t_bridge_lookup0) * 1000.0)
        if layer_snapshot is None:
            _append_metrics({
                "event": "selective_old_page_unavailable",
                "layer_name": layer_name,
                "layer_id": layer_id,
                "reason": "missing_layer_snapshot",
                "keepold_bridge_key": self._keepold_bridge_key,
                "ts_unix": time.time(),
            })
            raise RuntimeError(
                "keepold old-page layer snapshot is missing: "
                f"layer={layer_id}, bridge_key={self._keepold_bridge_key}"
            )
        else:
            keepold_mixed_source = True
            t_switch_prepare0 = time.perf_counter() if diag_layer_enabled else 0.0
            switch_reuse_idx = self._keepold_switch_reuse_idx_dev
            switch_prompt_pos = self._keepold_switch_prompt_positions_dev
            keepold_switch_tokens = int(switch_reuse_idx.numel())
            if diag_layer_enabled:
                diag_layer_switch_index_prepare_ms += float(
                    (time.perf_counter() - t_switch_prepare0) * 1000.0)
            t_cpu_reuse_prepare0 = time.perf_counter() if diag_layer_enabled else 0.0
            cpu_reuse_idx_dev = self._keepold_cpu_reuse_idx_dev
            cpu_reuse_idx_cpu = self._keepold_cpu_reuse_idx_cpu
            if cpu_reuse_idx_dev is None:
                cpu_reuse_idx_dev = torch.empty(0,
                                                dtype=torch.long,
                                                device=new_k.device)
            if cpu_reuse_idx_cpu is None:
                cpu_reuse_idx_cpu = torch.empty(0, dtype=torch.long)
            if diag_layer_enabled:
                diag_layer_cpu_reuse_idx_prepare_ms += float(
                    (time.perf_counter() - t_cpu_reuse_prepare0) * 1000.0)
            cpu_reuse_idx_dev_for_copy = cpu_reuse_idx_dev
            cpu_reuse_idx_cpu_for_copy = cpu_reuse_idx_cpu
            full_reuse_tokens = int(full_reuse_pos_for_index_copy.numel())
            should_run_coverage_check = bool(
                token_skip_zero_if_full_coverage or diag_layer_enabled)
            if not should_run_coverage_check:
                token_full_coverage_ready = 0
                token_full_coverage_reason = "feature_disabled"
                token_full_coverage_total_rows = 0
                token_full_coverage_gpu_rows = 0
                token_full_coverage_cpu_rows = 0
                token_full_coverage_overlap_rows = 0
                token_full_coverage_missing_rows = 0
            else:
                t_token_coverage0 = time.perf_counter() if diag_layer_enabled else 0.0
                token_full_coverage_total_rows = int(full_reuse_tokens)
                token_full_coverage_gpu_rows = int(keepold_switch_tokens)
                token_full_coverage_cpu_rows = int(cpu_reuse_idx_dev_for_copy.numel())
                if token_full_coverage_total_rows <= 0:
                    token_full_coverage_ready = 0
                    token_full_coverage_reason = "empty_reuse_rows"
                else:
                    try:
                        coverage_marks = torch.zeros(
                            token_full_coverage_total_rows,
                            dtype=torch.int32,
                            device=new_k.device,
                        )
                        if int(keepold_switch_tokens) > 0:
                            coverage_marks.index_add_(
                                0,
                                switch_reuse_idx,
                                torch.ones_like(switch_reuse_idx,
                                                dtype=torch.int32),
                            )
                        if int(cpu_reuse_idx_dev_for_copy.numel()) > 0:
                            coverage_marks.index_add_(
                                0,
                                cpu_reuse_idx_dev_for_copy,
                                torch.ones_like(cpu_reuse_idx_dev_for_copy,
                                                dtype=torch.int32),
                            )
                        token_full_coverage_overlap_rows = int(
                            (coverage_marks > 1).sum().item())
                        token_full_coverage_missing_rows = int(
                            (coverage_marks == 0).sum().item())
                        if (token_full_coverage_overlap_rows == 0
                                and token_full_coverage_missing_rows == 0):
                            token_full_coverage_ready = 1
                            token_full_coverage_reason = "ok"
                        elif token_full_coverage_overlap_rows > 0:
                            token_full_coverage_ready = 0
                            token_full_coverage_reason = "overlap_rows"
                        else:
                            token_full_coverage_ready = 0
                            token_full_coverage_reason = "missing_rows"
                    except Exception:
                        token_full_coverage_ready = 0
                        token_full_coverage_reason = "coverage_check_error"
                if diag_layer_enabled:
                    diag_layer_token_coverage_check_ms += float(
                        (time.perf_counter() - t_token_coverage0) * 1000.0)
            cpu_fallback_dense_reason_runtime = cpu_fallback_dense_reason
            t_dense_check0 = time.perf_counter() if diag_layer_enabled else 0.0
            if cpu_fallback_dense_enabled and full_reuse_tokens > 0:
                cpu_reuse_idx_dev_for_copy = torch.arange(
                    full_reuse_tokens, dtype=torch.long, device=new_k.device)
                cpu_reuse_idx_cpu_for_copy = torch.arange(
                    full_reuse_tokens, dtype=torch.long)
                cpu_fallback_dense_applied = True
                cpu_fallback_subset_skipped = True
                token_full_coverage_gpu_rows = int(keepold_switch_tokens)
                token_full_coverage_cpu_rows = int(cpu_reuse_idx_dev_for_copy.numel())
            if diag_layer_enabled:
                diag_layer_dense_check_ms += float(
                    (time.perf_counter() - t_dense_check0) * 1000.0)
                diag_layer_token_branch_enter_ms += float(
                    (time.perf_counter() - t_token_branch0) * 1000.0)

            should_skip_zero = bool(
                token_skip_zero_if_full_coverage
                and token_full_coverage_ready == 1)
            if should_skip_zero:
                token_zero_skipped = 1
                token_zero_skip_reason = "full_coverage"
                token_zero_submit_ms = 0.0
            else:
                token_zero_skipped = 0
                if not token_skip_zero_if_full_coverage:
                    token_zero_skip_reason = "feature_disabled"
                else:
                    token_zero_skip_reason = token_full_coverage_reason
                zero_start_event: Optional[torch.cuda.Event] = None
                zero_end_event: Optional[torch.cuda.Event] = None
                if (diag_layer_enabled and timing_cuda_event
                        and new_k.device.type == "cuda"):
                    zero_start_event = torch.cuda.Event(enable_timing=True)
                    zero_end_event = torch.cuda.Event(enable_timing=True)
                    zero_start_event.record(torch.cuda.current_stream(new_k.device))
                t_zero0 = time.perf_counter()
                reuse_k.zero_()
                reuse_v.zero_()
                token_zero_submit_ms = float((time.perf_counter() - t_zero0) *
                                             1000.0)
                if diag_layer_enabled:
                    diag_layer_reuse_buffer_zero_submit_ms = float(
                        token_zero_submit_ms)
                if zero_end_event is not None:
                    zero_end_event.record(torch.cuda.current_stream(new_k.device))
                    t_zero_wait0 = time.perf_counter()
                    torch.cuda.synchronize(new_k.device)
                    diag_layer_reuse_buffer_zero_wait_ms += float(
                        (time.perf_counter() - t_zero_wait0) * 1000.0)
                    diag_layer_reuse_buffer_zero_device_ms = float(
                        zero_start_event.elapsed_time(zero_end_event))

            h2d_tokens = int(cpu_reuse_idx_dev_for_copy.numel())
            if overlap_config_enabled:
                if not device_is_cuda:
                    diag_layer_overlap_reason = "non_cuda"
                elif timing_cuda_event:
                    diag_layer_overlap_reason = "timing_cuda_event_mode"
                elif not overlap_runtime_eligible:
                    diag_layer_overlap_reason = overlap_runtime_reason
                elif int(keepold_switch_tokens) <= 0:
                    diag_layer_overlap_reason = "empty_gpu_rows"
                elif int(h2d_tokens) <= 0:
                    diag_layer_overlap_reason = "empty_cpu_rows"
                elif old_page_fused_k_mode not in (
                        _KEEPOLD_FUSED_K_TRITON, _KEEPOLD_FUSED_K_CUDA_OP):
                    diag_layer_overlap_reason = "requires_non_sync_old_page_mode"
                else:
                    overlap_used = True
                    diag_layer_overlap_reason = "ok"
                    old_page_work_stream = torch.cuda.Stream(device=new_k.device)
                    # Ensure old-page stream does not race with zero_ writes that
                    # were just submitted on the default stream. This dependency
                    # only covers already-enqueued default-stream work and does
                    # not block subsequent CPU fallback copy/scatter submission,
                    # preserving overlap.
                    old_page_work_stream.wait_stream(
                        torch.cuda.current_stream(new_k.device))
            if diag_layer_enabled:
                diag_layer_overlap_used = 1.0 if overlap_used else 0.0
            if h2d_tokens > 0:
                src_tokens_now = int(src_k_eff.shape[0])
                if int(src_v_eff.shape[0]) != src_tokens_now:
                    raise RuntimeError(
                        "keepold mixed source K/V token mismatch: "
                        f"layer={layer_id}, k={src_tokens_now}, "
                        f"v={int(src_v_eff.shape[0])}")
                use_cpu_fallback_span_direct = bool(
                    cpu_fallback_span_direct_ready
                    and not cpu_fallback_dense_applied
                    and src_tokens_now == full_reuse_tokens
                    and int(cpu_fallback_span_direct_lengths_cpu.numel()) > 0
                    and int(cpu_fallback_span_direct_tokens) == int(h2d_tokens))
                if use_cpu_fallback_span_direct:
                    span_src_starts_cpu = cpu_fallback_span_direct_src_starts_cpu
                    span_dst_starts_cpu = cpu_fallback_span_direct_dst_starts_cpu
                    span_lengths_cpu = cpu_fallback_span_direct_lengths_cpu
                    span_count = int(span_lengths_cpu.numel())
                    if (int(span_src_starts_cpu.numel()) != span_count
                            or int(span_dst_starts_cpu.numel()) != span_count):
                        use_cpu_fallback_span_direct = False
                        cpu_fallback_span_direct_reason = (
                            "span_direct_metadata_len_mismatch")
                    else:
                        span_total = int(span_lengths_cpu.sum().item())
                        if span_total != h2d_tokens:
                            use_cpu_fallback_span_direct = False
                            cpu_fallback_span_direct_reason = (
                                "span_direct_token_mismatch")
                if use_cpu_fallback_span_direct:
                    src_is_pinned_k = bool(src_k_eff.is_pinned())
                    src_is_pinned_v = bool(src_v_eff.is_pinned())
                    non_blocking_k = (device_is_cuda
                                      and src_k_eff.device.type == "cpu"
                                      and src_is_pinned_k)
                    non_blocking_v = (device_is_cuda
                                      and src_v_eff.device.type == "cpu"
                                      and src_is_pinned_v)
                    if device_is_cuda:
                        h2d_start_event = torch.cuda.Event(enable_timing=True)
                        h2d_end_event = torch.cuda.Event(enable_timing=True)
                        h2d_start_event.record(torch.cuda.current_stream(new_k.device))
                    t_copy0 = time.perf_counter()
                    kv_h2d_start_event: Optional[torch.cuda.Event] = None
                    kv_h2d_end_event: Optional[torch.cuda.Event] = None
                    if timing_cuda_event and device_is_cuda:
                        kv_h2d_start_event = torch.cuda.Event(enable_timing=True)
                        kv_h2d_end_event = torch.cuda.Event(enable_timing=True)
                        kv_h2d_start_event.record(torch.cuda.current_stream(new_k.device))
                    t_copy_k0 = t_copy0
                    for i in range(int(cpu_fallback_span_direct_span_count)):
                        seg_len = int(cpu_fallback_span_direct_lengths_cpu[i].item())
                        if seg_len <= 0:
                            continue
                        src_st = int(cpu_fallback_span_direct_src_starts_cpu[i].item())
                        dst_st = int(cpu_fallback_span_direct_dst_starts_cpu[i].item())
                        src_ed = src_st + seg_len
                        dst_ed = dst_st + seg_len
                        reuse_k[dst_st:dst_ed].copy_(
                            src_k_eff[src_st:src_ed], non_blocking=non_blocking_k)
                    t_copy_k1 = time.perf_counter()
                    t_cpu_subset_to_staging_copy_k_ms = (
                        t_copy_k1 - t_copy_k0) * 1000.0
                    timing_v2_kv_h2d_k_submit_ms = float(
                        t_cpu_subset_to_staging_copy_k_ms)
                    t_copy_v0 = t_copy_k1
                    for i in range(int(cpu_fallback_span_direct_span_count)):
                        seg_len = int(cpu_fallback_span_direct_lengths_cpu[i].item())
                        if seg_len <= 0:
                            continue
                        src_st = int(cpu_fallback_span_direct_src_starts_cpu[i].item())
                        dst_st = int(cpu_fallback_span_direct_dst_starts_cpu[i].item())
                        src_ed = src_st + seg_len
                        dst_ed = dst_st + seg_len
                        reuse_v[dst_st:dst_ed].copy_(
                            src_v_eff[src_st:src_ed], non_blocking=non_blocking_v)
                    t_copy_v1 = time.perf_counter()
                    t_cpu_subset_to_staging_copy_v_ms = (
                        t_copy_v1 - t_copy_v0) * 1000.0
                    timing_v2_kv_h2d_v_submit_ms = float(
                        t_cpu_subset_to_staging_copy_v_ms)
                    if kv_h2d_end_event is not None:
                        kv_h2d_end_event.record(torch.cuda.current_stream(new_k.device))
                        t_h2d_wait0 = time.perf_counter()
                        torch.cuda.synchronize(new_k.device)
                        diag_layer_cpu_kv_h2d_wait_ms += float(
                            (time.perf_counter() - t_h2d_wait0) * 1000.0)
                        timing_v2_kv_h2d_device_ms = float(
                            kv_h2d_start_event.elapsed_time(kv_h2d_end_event))
                    if h2d_end_event is not None:
                        h2d_end_event.record(torch.cuda.current_stream(new_k.device))
                    t_copy1 = t_copy_v1
                    t_pure_h2d_copy_ms = (t_copy1 - t_copy0) * 1000.0
                    t_h2d_submit_ms = t_pure_h2d_copy_ms
                    diag_layer_cpu_fallback_span_direct_used = 1.0
                    diag_layer_cpu_fallback_span_count = float(
                        cpu_fallback_span_direct_span_count)
                    diag_layer_cpu_fallback_span_tokens = float(
                        cpu_fallback_span_direct_tokens)
                    diag_layer_cpu_fallback_span_copy_submit_ms = float(
                        t_pure_h2d_copy_ms)
                    if diag_layer_enabled:
                        diag_layer_cpu_subset_to_staging_submit_ms += float(
                            t_pure_h2d_copy_ms)
                    cpu_fallback_subset_skipped = True
                else:
                    if cpu_fallback_span_direct_ready:
                        cpu_fallback_span_direct_reason = (
                            cpu_fallback_span_direct_reason
                            if cpu_fallback_span_direct_reason else
                            "span_direct_disabled_runtime")

                if not use_cpu_fallback_span_direct:
                    if cpu_fallback_dense_applied:
                        if src_tokens_now == full_reuse_tokens:
                            src_k_cpu_subset = src_k_eff
                            src_v_cpu_subset = src_v_eff
                        else:
                            cpu_fallback_dense_applied = False
                            cpu_fallback_dense_reason_runtime = (
                                "dense_src_tokens_mismatch_fallback_subset")
                            cpu_fallback_subset_skipped = False
                            t_subset0 = time.perf_counter()
                            if src_tokens_now == h2d_tokens:
                                src_k_cpu_subset = src_k_eff
                                src_v_cpu_subset = src_v_eff
                            elif src_tokens_now == full_reuse_tokens:
                                src_k_cpu_subset = src_k_eff.index_select(
                                    0, cpu_reuse_idx_cpu_for_copy).contiguous()
                                src_v_cpu_subset = src_v_eff.index_select(
                                    0, cpu_reuse_idx_cpu_for_copy).contiguous()
                            else:
                                raise RuntimeError(
                                    "Unexpected keepold mixed source token shape: "
                                    f"layer={layer_id}, src_tokens={src_tokens_now}, "
                                    f"full_reuse_tokens={full_reuse_tokens}, "
                                    f"cpu_reuse_tokens={h2d_tokens}")
                            t_subset1 = time.perf_counter()
                            t_cpu_subset_prepare_ms = (t_subset1 - t_subset0) * 1000.0
                            if diag_layer_enabled:
                                diag_layer_cpu_subset_fetch_ms += float(
                                    t_cpu_subset_prepare_ms)
                            t_pin0 = time.perf_counter()
                            if new_k.device.type == "cuda":
                                if not src_k_cpu_subset.is_pinned():
                                    src_k_cpu_subset = src_k_cpu_subset.pin_memory()
                                if not src_v_cpu_subset.is_pinned():
                                    src_v_cpu_subset = src_v_cpu_subset.pin_memory()
                            t_pin1 = time.perf_counter()
                            t_cpu_subset_pin_ms = (t_pin1 - t_pin0) * 1000.0
                    else:
                        t_fetch0 = time.perf_counter() if diag_layer_enabled else 0.0
                        prepared_subset = self._keepold_cpu_subset_layers.get(layer_id)
                        if (prepared_subset is not None
                                and int(prepared_subset.k_cpu.shape[0]) == h2d_tokens
                                and int(prepared_subset.v_cpu.shape[0]) == h2d_tokens):
                            if diag_layer_enabled:
                                diag_layer_cpu_subset_fetch_ms += float(
                                    (time.perf_counter() - t_fetch0) * 1000.0)
                            t_reshape0 = (
                                time.perf_counter() if diag_layer_enabled else 0.0)
                            src_k_cpu_subset = self._reshape_to_kv_heads(
                                prepared_subset.k_cpu,
                                num_kv_heads,
                                head_size,
                            ).contiguous()
                            src_v_cpu_subset = self._reshape_to_kv_heads(
                                prepared_subset.v_cpu,
                                int(new_v.shape[1]),
                                int(new_v.shape[2]),
                            ).contiguous()
                            if diag_layer_enabled:
                                diag_layer_cpu_subset_reshape_ms += float(
                                    (time.perf_counter() - t_reshape0) * 1000.0)
                            used_prepared_cpu_subset = True
                        else:
                            t_subset0 = time.perf_counter()
                            if src_tokens_now == h2d_tokens:
                                # src_*_eff already contains only the CPU subset.
                                src_k_cpu_subset = src_k_eff
                                src_v_cpu_subset = src_v_eff
                            elif src_tokens_now == full_reuse_tokens:
                                # src_*_eff is still full reuse order, gather CPU subset.
                                src_k_cpu_subset = src_k_eff.index_select(
                                    0, cpu_reuse_idx_cpu_for_copy).contiguous()
                                src_v_cpu_subset = src_v_eff.index_select(
                                    0, cpu_reuse_idx_cpu_for_copy).contiguous()
                            else:
                                raise RuntimeError(
                                    "Unexpected keepold mixed source token shape: "
                                    f"layer={layer_id}, src_tokens={src_tokens_now}, "
                                    f"full_reuse_tokens={full_reuse_tokens}, "
                                    f"cpu_reuse_tokens={h2d_tokens}")
                            t_subset1 = time.perf_counter()
                            t_cpu_subset_prepare_ms = (
                                t_subset1 - t_subset0) * 1000.0
                            if diag_layer_enabled:
                                diag_layer_cpu_subset_fetch_ms += float(
                                    t_cpu_subset_prepare_ms)

                            t_pin0 = time.perf_counter()
                            if new_k.device.type == "cuda":
                                if not src_k_cpu_subset.is_pinned():
                                    src_k_cpu_subset = src_k_cpu_subset.pin_memory()
                                if not src_v_cpu_subset.is_pinned():
                                    src_v_cpu_subset = src_v_cpu_subset.pin_memory()
                            t_pin1 = time.perf_counter()
                            t_cpu_subset_pin_ms = (t_pin1 - t_pin0) * 1000.0
                    src_is_pinned_k = bool(src_k_cpu_subset.is_pinned())
                    src_is_pinned_v = bool(src_v_cpu_subset.is_pinned())
                    non_blocking_k = (new_k.device.type == "cuda"
                                      and src_k_cpu_subset.device.type == "cpu"
                                      and src_k_cpu_subset.is_pinned())
                    non_blocking_v = (new_v.device.type == "cuda"
                                      and src_v_cpu_subset.device.type == "cpu"
                                      and src_v_cpu_subset.is_pinned())

                if not use_cpu_fallback_span_direct:
                    cpu_k_shape = (h2d_tokens, num_kv_heads, head_size)
                    cpu_v_shape = (h2d_tokens, int(new_v.shape[1]),
                                   int(new_v.shape[2]))
                    t_cpu_staging_check0 = time.perf_counter() if diag_layer_enabled else 0.0
                    need_cpu_k_alloc = (
                        self._reuse_cpu_k_staging is None
                        or self._reuse_cpu_k_staging.shape != cpu_k_shape
                        or self._reuse_cpu_k_staging.dtype != new_k.dtype
                        or self._reuse_cpu_k_staging.device != new_k.device)
                    need_cpu_v_alloc = (
                        self._reuse_cpu_v_staging is None
                        or self._reuse_cpu_v_staging.shape != cpu_v_shape
                        or self._reuse_cpu_v_staging.dtype != new_v.dtype
                        or self._reuse_cpu_v_staging.device != new_v.device)
                    if diag_layer_enabled:
                        diag_layer_cpu_staging_shape_check_ms += float(
                            (time.perf_counter() - t_cpu_staging_check0) * 1000.0)
                    if need_cpu_k_alloc:
                        t_alloc0 = time.perf_counter() if diag_layer_enabled else 0.0
                        self._reuse_cpu_k_staging = torch.empty(
                            cpu_k_shape,
                            dtype=new_k.dtype,
                            device=new_k.device,
                        )
                        if diag_layer_enabled:
                            diag_layer_staging_alloc_ms += float(
                                (time.perf_counter() - t_alloc0) * 1000.0)
                    if need_cpu_v_alloc:
                        t_alloc0 = time.perf_counter() if diag_layer_enabled else 0.0
                        self._reuse_cpu_v_staging = torch.empty(
                            cpu_v_shape,
                            dtype=new_v.dtype,
                            device=new_v.device,
                        )
                        if diag_layer_enabled:
                            diag_layer_staging_alloc_ms += float(
                                (time.perf_counter() - t_alloc0) * 1000.0)
                    cpu_k_staging = self._reuse_cpu_k_staging
                    cpu_v_staging = self._reuse_cpu_v_staging
                    if new_k.device.type == "cuda":
                        h2d_start_event = torch.cuda.Event(enable_timing=True)
                        h2d_end_event = torch.cuda.Event(enable_timing=True)
                        h2d_start_event.record(
                            torch.cuda.current_stream(new_k.device))
                    t_copy0 = time.perf_counter()
                    kv_h2d_start_event: Optional[torch.cuda.Event] = None
                    kv_h2d_end_event: Optional[torch.cuda.Event] = None
                    if timing_cuda_event and new_k.device.type == "cuda":
                        kv_h2d_start_event = torch.cuda.Event(enable_timing=True)
                        kv_h2d_end_event = torch.cuda.Event(enable_timing=True)
                        kv_h2d_start_event.record(torch.cuda.current_stream(new_k.device))
                    t_copy_k0 = t_copy0
                    cpu_k_staging.copy_(src_k_cpu_subset,
                                        non_blocking=non_blocking_k)
                    t_copy_k1 = time.perf_counter()
                    t_cpu_subset_to_staging_copy_k_ms = (
                        t_copy_k1 - t_copy_k0) * 1000.0
                    timing_v2_kv_h2d_k_submit_ms = float(
                        t_cpu_subset_to_staging_copy_k_ms)
                    t_copy_v0 = t_copy_k1
                    cpu_v_staging.copy_(src_v_cpu_subset,
                                        non_blocking=non_blocking_v)
                    t_copy_v1 = time.perf_counter()
                    t_cpu_subset_to_staging_copy_v_ms = (
                        t_copy_v1 - t_copy_v0) * 1000.0
                    timing_v2_kv_h2d_v_submit_ms = float(
                        t_cpu_subset_to_staging_copy_v_ms)
                    if kv_h2d_end_event is not None:
                        kv_h2d_end_event.record(torch.cuda.current_stream(new_k.device))
                        t_h2d_wait0 = time.perf_counter()
                        torch.cuda.synchronize(new_k.device)
                        diag_layer_cpu_kv_h2d_wait_ms += float(
                            (time.perf_counter() - t_h2d_wait0) * 1000.0)
                        timing_v2_kv_h2d_device_ms = float(
                            kv_h2d_start_event.elapsed_time(kv_h2d_end_event))
                    if h2d_end_event is not None:
                        h2d_end_event.record(
                            torch.cuda.current_stream(new_k.device))
                    t_copy1 = t_copy_v1
                    t_pure_h2d_copy_ms = (t_copy1 - t_copy0) * 1000.0
                    if diag_layer_enabled:
                        diag_layer_cpu_subset_to_staging_submit_ms += float(
                            t_pure_h2d_copy_ms)

                    gpu_scatter_nvtx_pushed = self.maybe_nvtx_range_push(
                        "keepold_gpu_scatter",
                        layer_id=layer_id,
                        include_layer_id=True,
                    )
                    t_scatter0 = time.perf_counter()
                    idx_scatter_start_event: Optional[torch.cuda.Event] = None
                    idx_scatter_end_event: Optional[torch.cuda.Event] = None
                    if timing_cuda_event and new_k.device.type == "cuda":
                        idx_scatter_start_event = torch.cuda.Event(enable_timing=True)
                        idx_scatter_end_event = torch.cuda.Event(enable_timing=True)
                        idx_scatter_start_event.record(
                            torch.cuda.current_stream(new_k.device))
                    if old_page_fused_k_mode == _KEEPOLD_FUSED_K_CUDA_OP:
                        if (self._keepold_gpu_index_fast_ready
                                and self._keepold_gpu_index_idx_i64_dev is not None):
                            t_fused_idx_call0 = time.perf_counter()
                            fused_idx_kv_stats = (
                                self._fused_scatter_rows_to_reuse_kv_fast(
                                    src_k_rows=cpu_k_staging,
                                    src_v_rows=cpu_v_staging,
                                    dst_idx_i64=self._keepold_gpu_index_idx_i64_dev,
                                    reuse_k=reuse_k,
                                    reuse_v=reuse_v,
                                ))
                            t_fused_idx_call1 = time.perf_counter()
                            t_gpu_index_fused_kv_call_ms = (
                                t_fused_idx_call1 - t_fused_idx_call0) * 1000.0
                        else:
                            t_fused_idx_call0 = time.perf_counter()
                            fused_idx_kv_stats = self._fused_scatter_rows_to_reuse_kv(
                                src_k_rows=cpu_k_staging,
                                src_v_rows=cpu_v_staging,
                                dst_reuse_idx=cpu_reuse_idx_dev_for_copy,
                                reuse_k=reuse_k,
                                reuse_v=reuse_v,
                                mode=old_page_fused_k_mode,
                            )
                            t_fused_idx_call1 = time.perf_counter()
                            t_gpu_index_fused_kv_call_ms = (
                                t_fused_idx_call1 - t_fused_idx_call0) * 1000.0
                        gpu_index_fused_kv_used = bool(
                            fused_idx_kv_stats.get("used", False))
                        gpu_index_fused_kv_fallback_reason = str(
                            fused_idx_kv_stats.get("reason", "unknown"))
                        t_gpu_index_fused_kv_ms = float(
                            fused_idx_kv_stats.get("t_total_ms", 0.0) or 0.0)
                        if gpu_index_fused_kv_used:
                            t_staging_to_reuse_index_copy_k_ms = (
                                t_gpu_index_fused_kv_ms * 0.5)
                            t_staging_to_reuse_index_copy_v_ms = (
                                t_gpu_index_fused_kv_ms * 0.5)
                            t_scatter1 = time.perf_counter()
                        else:
                            t_scatter_k0 = time.perf_counter()
                            reuse_k.index_copy_(0, cpu_reuse_idx_dev_for_copy,
                                                cpu_k_staging)
                            t_scatter_k1 = time.perf_counter()
                            t_staging_to_reuse_index_copy_k_ms = (
                                t_scatter_k1 - t_scatter_k0) * 1000.0
                            t_scatter_v0 = t_scatter_k1
                            reuse_v.index_copy_(0, cpu_reuse_idx_dev_for_copy,
                                                cpu_v_staging)
                            t_scatter_v1 = time.perf_counter()
                            t_staging_to_reuse_index_copy_v_ms = (
                                t_scatter_v1 - t_scatter_v0) * 1000.0
                            t_scatter1 = t_scatter_v1
                    else:
                        t_scatter_k0 = time.perf_counter()
                        if (old_page_fused_k_mode in (
                                _KEEPOLD_FUSED_K_TRITON, _KEEPOLD_FUSED_K_CUDA_OP)
                                and h2d_tokens >= _KEEPOLD_TRITON_GPU_INDEX_MIN_TOKENS):
                            fused_idx_stats = self._fused_scatter_rows_to_reuse_k(
                                src_rows=cpu_k_staging,
                                dst_reuse_idx=cpu_reuse_idx_dev_for_copy,
                                reuse_k=reuse_k,
                                mode=old_page_fused_k_mode,
                            )
                            gpu_index_fused_k_used = bool(
                                fused_idx_stats.get("used", False))
                            gpu_index_fused_k_fallback_reason = str(
                                fused_idx_stats.get("reason", "unknown"))
                            t_gpu_index_fused_k_ms = float(
                                fused_idx_stats.get("t_total_ms", 0.0) or 0.0)
                            if gpu_index_fused_k_used:
                                t_staging_to_reuse_index_copy_k_ms = (
                                    t_gpu_index_fused_k_ms)
                                t_scatter_k1 = time.perf_counter()
                            else:
                                reuse_k.index_copy_(0, cpu_reuse_idx_dev_for_copy,
                                                    cpu_k_staging)
                                t_scatter_k1 = time.perf_counter()
                                t_staging_to_reuse_index_copy_k_ms = (
                                    t_scatter_k1 - t_scatter_k0) * 1000.0
                        else:
                            if old_page_fused_k_mode in (
                                    _KEEPOLD_FUSED_K_TRITON,
                                    _KEEPOLD_FUSED_K_CUDA_OP):
                                gpu_index_fused_k_fallback_reason = (
                                    "below_token_threshold_"
                                    f"{_KEEPOLD_TRITON_GPU_INDEX_MIN_TOKENS}")
                            reuse_k.index_copy_(0, cpu_reuse_idx_dev_for_copy,
                                                cpu_k_staging)
                            t_scatter_k1 = time.perf_counter()
                            t_staging_to_reuse_index_copy_k_ms = (
                                t_scatter_k1 - t_scatter_k0) * 1000.0
                        t_scatter_v0 = t_scatter_k1
                        reuse_v.index_copy_(0, cpu_reuse_idx_dev_for_copy,
                                            cpu_v_staging)
                        t_scatter_v1 = time.perf_counter()
                        t_staging_to_reuse_index_copy_v_ms = (
                            t_scatter_v1 - t_scatter_v0) * 1000.0
                        t_scatter1 = t_scatter_v1
                    t_gpu_index_copy_ms = (t_scatter1 - t_scatter0) * 1000.0
                    self.maybe_nvtx_range_pop(gpu_scatter_nvtx_pushed)
                    timing_v2_gpu_index_scatter_submit_ms = float(t_gpu_index_copy_ms)
                    if diag_layer_enabled:
                        diag_layer_staging_to_reuse_scatter_submit_ms += float(
                            t_gpu_index_copy_ms)
                    if idx_scatter_end_event is not None:
                        idx_scatter_end_event.record(
                            torch.cuda.current_stream(new_k.device))
                        t_scatter_wait0 = time.perf_counter()
                        torch.cuda.synchronize(new_k.device)
                        diag_layer_gpu_index_scatter_wait_ms += float(
                            (time.perf_counter() - t_scatter_wait0) * 1000.0)
                        timing_v2_gpu_index_scatter_device_ms = float(
                            idx_scatter_start_event.elapsed_time(
                                idx_scatter_end_event))
                    t_gpu_index_copy_accounted_ms = (
                        t_staging_to_reuse_index_copy_k_ms
                        + t_staging_to_reuse_index_copy_v_ms)
                    t_gpu_index_copy_unaccounted_ms = (
                        t_gpu_index_copy_ms - t_gpu_index_copy_accounted_ms)
            t_h2d_submit_ms = t_pure_h2d_copy_ms + t_gpu_index_copy_ms
            if diag_layer_enabled and overlap_used:
                diag_layer_cpu_stream_submit_ms = float(t_h2d_submit_ms)

            old_page_prev_stream: Optional[torch.cuda.Stream] = None
            t_old_page_stream_submit0 = (
                time.perf_counter() if (diag_layer_enabled and overlap_used) else 0.0)
            if overlap_used and old_page_work_stream is not None:
                old_page_prev_stream = torch.cuda.current_stream(new_k.device)
                torch.cuda.set_stream(old_page_work_stream)
            t_old_page_precheck0 = time.perf_counter() if diag_layer_enabled else 0.0
            old_slots_fast_i64 = None
            switch_reuse_idx_work = switch_reuse_idx
            if self._keepold_old_page_fast_ready:
                old_slots_fast_i64 = self._keepold_old_slots_i64_by_layer.get(
                    int(layer_id))
                if (old_slots_fast_i64 is not None
                        and self._keepold_switch_reuse_idx_i64_dev is not None):
                    switch_reuse_idx_work = self._keepold_switch_reuse_idx_i64_dev
            if (new_k.device.type == "cuda"
                    and old_page_fused_k_mode not in (
                        _KEEPOLD_FUSED_K_TRITON, _KEEPOLD_FUSED_K_CUDA_OP)):
                t_sync_pre0 = time.perf_counter()
                torch.cuda.synchronize(new_k.device)
                t_sync_pre1 = time.perf_counter()
                t_old_page_sync_pre_ms = (t_sync_pre1 - t_sync_pre0) * 1000.0
                # This sync drains queued GPU work before old-page gather,
                # mostly from CPU-subset staging + index_copy in mixed source.
                t_gpu_index_copy_sync_overhead_ms = t_old_page_sync_pre_ms
            if diag_layer_enabled:
                diag_layer_old_page_precheck_ms += float(
                    (time.perf_counter() - t_old_page_precheck0) * 1000.0)
            t_pg0 = time.perf_counter()
            old_page_start_event: Optional[torch.cuda.Event] = None
            old_page_end_event: Optional[torch.cuda.Event] = None
            if timing_cuda_event and new_k.device.type == "cuda":
                old_page_start_event = torch.cuda.Event(enable_timing=True)
                old_page_end_event = torch.cuda.Event(enable_timing=True)
                old_page_start_event.record(torch.cuda.current_stream(new_k.device))
            old_page_nvtx_pushed = self.maybe_nvtx_range_push(
                "keepold_old_page_gather",
                layer_id=layer_id,
                include_layer_id=True,
            )
            if old_slots_fast_i64 is not None:
                old_slots = old_slots_fast_i64
                t_old_slot_select_ms = 0.0
            else:
                if bool(getattr(self.config, "keepold_chunk_page_debug_checks",
                                False)):
                    max_switch_prompt = int(switch_prompt_pos.max().item())
                    if int(layer_snapshot.slot_mapping.numel()) <= max_switch_prompt:
                        raise ValueError(
                            "keepold switch prompt index out of range for "
                            f"layer={layer_id}: max_switch={max_switch_prompt} "
                            f"slot_count={int(layer_snapshot.slot_mapping.numel())}")
                t_slot0 = time.perf_counter()
                old_slots = layer_snapshot.slot_mapping.index_select(
                    0, switch_prompt_pos).contiguous()
                t_slot1 = time.perf_counter()
                t_old_slot_select_ms = (t_slot1 - t_slot0) * 1000.0
            old_page_fused_k_fallback_reason = "disabled"
            old_page_fused_kv_fallback_reason = "disabled"
            if old_page_fused_k_mode == _KEEPOLD_FUSED_K_CUDA_OP:
                t_fused_precheck0 = time.perf_counter() if diag_layer_enabled else 0.0
                fused_kv_stats = None
                fast_fused_ready = bool(
                    old_slots_fast_i64 is not None
                    and self._keepold_switch_reuse_idx_i64_dev is not None)
                if diag_layer_enabled:
                    diag_layer_fused_kv_precheck_ms += float(
                        (time.perf_counter() - t_fused_precheck0) * 1000.0)
                if fast_fused_ready:
                    t_layout0 = time.perf_counter() if diag_layer_enabled else 0.0
                    layout_meta = self.infer_paged_layout_meta_cached(
                        layer_id=layer_id,
                        cache=layer_snapshot.key_cache,
                        num_kv_heads=num_kv_heads,
                        head_size=head_size,
                    )
                    if diag_layer_enabled:
                        diag_layer_old_page_layout_infer_ms += float(
                            (time.perf_counter() - t_layout0) * 1000.0)
                    if layout_meta is not None:
                        layout_code, page_size = layout_meta
                        t_fused_kv_call0 = time.perf_counter()
                        fused_kv_stats = (
                            self._fused_extract_scatter_kv_from_paged_cache_fast(
                                key_cache=layer_snapshot.key_cache,
                                value_cache=layer_snapshot.value_cache,
                                old_slots_i64=old_slots_fast_i64,
                                dst_idx_i64=switch_reuse_idx_work,
                                reuse_k=reuse_k,
                                reuse_v=reuse_v,
                                layout_code=int(layout_code),
                                page_size=int(page_size),
                            ))
                        t_fused_kv_call1 = time.perf_counter()
                        t_old_page_fused_kv_call_ms = (
                            t_fused_kv_call1 - t_fused_kv_call0) * 1000.0
                        if not bool(fused_kv_stats.get("used", False)):
                            old_page_fused_kv_fallback_reason = str(
                                fused_kv_stats.get("reason", "unknown"))
                    else:
                        old_page_fused_kv_fallback_reason = (
                            "fast_layout_infer_failed")
                if fused_kv_stats is None or not bool(
                        fused_kv_stats.get("used", False)):
                    t_fused_kv_call0 = time.perf_counter()
                    fused_kv_stats = (
                        self._fused_extract_scatter_kv_from_paged_cache(
                            key_cache=layer_snapshot.key_cache,
                            value_cache=layer_snapshot.value_cache,
                            old_slots=old_slots,
                            dst_reuse_idx=switch_reuse_idx_work,
                            reuse_k=reuse_k,
                            reuse_v=reuse_v,
                            num_kv_heads=num_kv_heads,
                            head_size=head_size,
                            mode=old_page_fused_k_mode,
                        ))
                    t_fused_kv_call1 = time.perf_counter()
                    t_old_page_fused_kv_call_ms = (
                        t_fused_kv_call1 - t_fused_kv_call0) * 1000.0
                old_page_fused_kv_used = bool(
                    fused_kv_stats.get("used", False))
                old_page_fused_kv_fallback_reason = str(
                    fused_kv_stats.get("reason", "unknown"))
                t_old_page_fused_kv_total_ms = float(
                    fused_kv_stats.get("t_total_ms", 0.0) or 0.0)
                if old_page_fused_kv_used:
                    # Keep legacy K-fused fields populated for compatibility.
                    old_page_fused_k_used = True
                    old_page_fused_k_fallback_reason = "kv_kernel"
                    old_page_fused_k_chunks = int(
                        fused_kv_stats.get("chunk_count", 0) or 0)
                    t_old_page_fused_k_total_ms = t_old_page_fused_kv_total_ms
                    t_old_page_fused_k_gather_ms = t_old_page_fused_kv_total_ms
                    t_old_page_fused_k_cast_ms = 0.0
                    t_old_page_fused_k_scatter_ms = 0.0
                    # For fused KV single-kernel path, expose split as 50/50.
                    t_old_page_extract_k_ms += t_old_page_fused_kv_total_ms * 0.5
                    t_old_page_extract_v_ms += t_old_page_fused_kv_total_ms * 0.5
            elif old_page_fused_k_mode != _KEEPOLD_FUSED_K_OFF:
                fused_k_stats = self._fused_extract_scatter_k_from_paged_cache(
                    key_cache=layer_snapshot.key_cache,
                    old_slots=old_slots,
                    dst_reuse_idx=switch_reuse_idx_work,
                    reuse_k=reuse_k,
                    num_kv_heads=num_kv_heads,
                    head_size=head_size,
                    chunk_size=int(self.config.keepold_fused_k_chunk_size),
                    mode=old_page_fused_k_mode,
                )
                old_page_fused_k_used = bool(fused_k_stats.get("used", False))
                old_page_fused_k_fallback_reason = str(
                    fused_k_stats.get("reason", "unknown"))
                old_page_fused_k_chunks = int(
                    fused_k_stats.get("chunk_count", 0) or 0)
                t_old_page_fused_k_total_ms = float(
                    fused_k_stats.get("t_total_ms", 0.0) or 0.0)
                t_old_page_fused_k_gather_ms = float(
                    fused_k_stats.get("t_gather_ms", 0.0) or 0.0)
                t_old_page_fused_k_cast_ms = float(
                    fused_k_stats.get("t_cast_ms", 0.0) or 0.0)
                t_old_page_fused_k_scatter_ms = float(
                    fused_k_stats.get("t_scatter_ms", 0.0) or 0.0)
                if old_page_fused_k_used:
                    t_old_page_extract_k_ms += t_old_page_fused_k_gather_ms
                    t_old_page_dtype_cast_ms += t_old_page_fused_k_cast_ms
                    t_old_page_scatter_back_ms += t_old_page_fused_k_scatter_ms

            if not old_page_fused_k_used:
                t_extract_k0 = time.perf_counter()
                old_k = self._extract_tokens_from_paged_cache(
                    layer_snapshot.key_cache,
                    old_slots,
                    num_kv_heads,
                    head_size,
                )
                t_extract_k1 = time.perf_counter()
                t_old_page_extract_k_ms += (t_extract_k1 -
                                            t_extract_k0) * 1000.0
                t_cast_k0 = t_extract_k1
                if old_k.dtype != reuse_k.dtype:
                    old_k = old_k.to(dtype=reuse_k.dtype)
                t_cast_k1 = time.perf_counter()
                t_old_page_dtype_cast_ms += (t_cast_k1 - t_cast_k0) * 1000.0
                t_scatter_k0 = t_cast_k1
                reuse_k.index_copy_(0, switch_reuse_idx_work, old_k)
                t_scatter_k1 = time.perf_counter()
                t_old_page_scatter_back_ms += (t_scatter_k1 -
                                               t_scatter_k0) * 1000.0
                t_extract_v0 = t_scatter_k1
            else:
                t_extract_v0 = time.perf_counter()
            if not old_page_fused_kv_used:
                old_v = self._extract_tokens_from_paged_cache(
                    layer_snapshot.value_cache,
                    old_slots,
                    int(new_v.shape[1]),
                    int(new_v.shape[2]),
                )
                t_extract_v1 = time.perf_counter()
                t_old_page_extract_v_ms += (t_extract_v1 - t_extract_v0) * 1000.0
                t_cast0 = t_extract_v1
                if old_v.dtype != reuse_v.dtype:
                    old_v = old_v.to(dtype=reuse_v.dtype)
                t_cast1 = time.perf_counter()
                t_old_page_dtype_cast_ms += (t_cast1 - t_cast0) * 1000.0
                t_scatter_back_v0 = t_cast1
                reuse_v.index_copy_(0, switch_reuse_idx_work, old_v)
                t_scatter_back_v1 = time.perf_counter()
                t_old_page_scatter_back_ms += (t_scatter_back_v1 -
                                               t_scatter_back_v0) * 1000.0
            if (new_k.device.type == "cuda"
                    and old_page_fused_k_mode not in (
                        _KEEPOLD_FUSED_K_TRITON, _KEEPOLD_FUSED_K_CUDA_OP)):
                t_sync_post0 = time.perf_counter()
                torch.cuda.synchronize(new_k.device)
                t_sync_post1 = time.perf_counter()
                t_old_page_sync_post_ms = (
                    t_sync_post1 - t_sync_post0) * 1000.0
            t_old_page_sync_overhead_ms = (
                t_old_page_sync_pre_ms + t_old_page_sync_post_ms)
            if old_page_end_event is not None:
                old_page_end_event.record(torch.cuda.current_stream(new_k.device))
                t_old_page_wait0 = time.perf_counter()
                torch.cuda.synchronize(new_k.device)
                diag_layer_old_page_gather_wait_ms += float(
                    (time.perf_counter() - t_old_page_wait0) * 1000.0)
                timing_v2_old_page_gather_device_ms = float(
                    old_page_start_event.elapsed_time(old_page_end_event))
            t_pg1 = time.perf_counter()
            t_old_page_gather_ms = (t_pg1 - t_pg0) * 1000.0
            self.maybe_nvtx_range_pop(old_page_nvtx_pushed)
            if diag_layer_enabled:
                diag_layer_old_page_call_wrapper_ms += float(t_old_page_gather_ms)
            timing_v2_old_page_gather_submit_ms = float(
                max(0.0, t_old_page_gather_ms - t_old_page_sync_overhead_ms))
            t_old_page_gather_accounted_ms = (
                t_old_slot_select_ms
                + t_old_page_extract_k_ms
                + t_old_page_extract_v_ms
                + t_old_page_dtype_cast_ms
                + t_old_page_scatter_back_ms
                + t_old_page_sync_overhead_ms
            )
            t_old_page_gather_unaccounted_ms = (
                t_old_page_gather_ms - t_old_page_gather_accounted_ms)
            keepold_old_page_tokens = int(old_slots.numel())
            reuse_old_pos_for_rope = reuse_old_pos.clone().contiguous()
            reuse_old_pos_for_rope.index_copy_(0, switch_reuse_idx_work,
                                               switch_prompt_pos)
            h2d_source_mode = "mixed_cpu_plus_old_page"
            if emit_layer_metrics_full:
                _append_metrics({
                    "event": "selective_old_page_gather",
                    "layer_name": layer_name,
                    "layer_id": layer_id,
                    "keepold_bridge_key": self._keepold_bridge_key,
                    "switched_reuse_tokens": int(keepold_switch_tokens),
                    "cpu_reuse_tokens": int(h2d_tokens),
                    "old_page_tokens": int(keepold_old_page_tokens),
                    "t_old_page_gather_ms": float(t_old_page_gather_ms),
                    "timing_v2_old_page_gather_submit_ms": float(
                        timing_v2_old_page_gather_submit_ms),
                    "timing_v2_old_page_gather_device_ms": float(
                        timing_v2_old_page_gather_device_ms),
                    "t_old_slot_select_ms": float(t_old_slot_select_ms),
                    "t_old_page_extract_k_ms": float(t_old_page_extract_k_ms),
                    "t_old_page_extract_v_ms": float(t_old_page_extract_v_ms),
                    "t_old_page_dtype_cast_ms": float(t_old_page_dtype_cast_ms),
                    "t_old_page_scatter_back_ms":
                    float(t_old_page_scatter_back_ms),
                    "t_old_page_sync_overhead_ms":
                    float(t_old_page_sync_overhead_ms),
                    "t_old_page_gather_accounted_ms":
                    float(t_old_page_gather_accounted_ms),
                    "t_old_page_gather_unaccounted_ms":
                    float(t_old_page_gather_unaccounted_ms),
                    "t_old_page_sync_pre_ms": float(t_old_page_sync_pre_ms),
                    "t_old_page_sync_post_ms": float(t_old_page_sync_post_ms),
                    "old_page_fused_k_used": bool(old_page_fused_k_used),
                    "old_page_fused_k_mode": old_page_fused_k_mode,
                    "old_page_fused_k_fallback_reason":
                    old_page_fused_k_fallback_reason,
                    "old_page_fused_k_chunks": int(old_page_fused_k_chunks),
                    "t_old_page_fused_k_total_ms":
                    float(t_old_page_fused_k_total_ms),
                    "t_old_page_fused_k_gather_ms":
                    float(t_old_page_fused_k_gather_ms),
                    "t_old_page_fused_k_cast_ms":
                    float(t_old_page_fused_k_cast_ms),
                    "t_old_page_fused_k_scatter_ms":
                    float(t_old_page_fused_k_scatter_ms),
                    "old_page_fused_kv_used": bool(old_page_fused_kv_used),
                    "old_page_fused_kv_fallback_reason":
                    old_page_fused_kv_fallback_reason,
                    "t_old_page_fused_kv_total_ms":
                    float(t_old_page_fused_kv_total_ms),
                    "t_old_page_fused_kv_call_ms":
                    float(t_old_page_fused_kv_call_ms),
                    "gpu_index_fused_k_used": bool(gpu_index_fused_k_used),
                    "gpu_index_fused_k_fallback_reason":
                    gpu_index_fused_k_fallback_reason,
                    "t_gpu_index_fused_k_ms": float(t_gpu_index_fused_k_ms),
                    "gpu_index_fused_kv_used": bool(gpu_index_fused_kv_used),
                    "gpu_index_fused_kv_fallback_reason":
                    gpu_index_fused_kv_fallback_reason,
                    "t_gpu_index_fused_kv_ms": float(t_gpu_index_fused_kv_ms),
                    "t_gpu_index_fused_kv_call_ms":
                    float(t_gpu_index_fused_kv_call_ms),
                    "t_gpu_index_copy_accounted_ms":
                    float(t_gpu_index_copy_accounted_ms),
                    "t_gpu_index_copy_unaccounted_ms":
                    float(t_gpu_index_copy_unaccounted_ms),
                    "chunk_page_execution_enabled":
                    1 if chunk_page_exec_enabled else 0,
                    "chunk_page_exec_success":
                    1 if chunk_page_exec_success else 0,
                    "chunk_page_exec_fallback_reason":
                    chunk_page_exec_fallback_reason,
                    "chunk_page_exec_tokens": int(chunk_page_exec_tokens),
                    "chunk_page_exec_gpu_tokens":
                    int(chunk_page_exec_gpu_tokens),
                    "chunk_page_exec_cpu_tokens":
                    int(chunk_page_exec_cpu_tokens),
                    "chunk_page_exec_old_slots_build_ms":
                    float(chunk_page_exec_old_slots_build_ms),
                    "chunk_page_cpu_fallback_tokens":
                    int(chunk_page_cpu_fallback_tokens),
                    "cpu_fallback_mode": cpu_fallback_mode,
                    "cpu_fallback_dense_enabled":
                    1 if cpu_fallback_dense_enabled else 0,
                    "cpu_fallback_dense_applied":
                    1 if cpu_fallback_dense_applied else 0,
                    "cpu_fallback_dense_reason":
                    cpu_fallback_dense_reason_runtime,
                    "cpu_fallback_tokens": int(cpu_fallback_tokens),
                    "cpu_fallback_ratio": float(cpu_fallback_ratio),
                    "cpu_fallback_subset_skipped":
                    1 if cpu_fallback_subset_skipped else 0,
                    "cpu_fallback_span_direct_ready":
                    1 if cpu_fallback_span_direct_ready else 0,
                    "cpu_fallback_span_direct_reason":
                    cpu_fallback_span_direct_reason,
                    "cpu_fallback_span_direct_span_count":
                    int(cpu_fallback_span_direct_span_count),
                    "cpu_fallback_span_direct_tokens":
                    int(cpu_fallback_span_direct_tokens),
                    "timing_v2_index_tensor_h2d_submit_ms":
                    float(timing_v2_index_tensor_h2d_submit_ms),
                    "timing_v2_index_tensor_h2d_device_ms":
                    float(timing_v2_index_tensor_h2d_device_ms),
                    "ts_unix": time.time(),
                })
            if overlap_used and old_page_prev_stream is not None:
                torch.cuda.set_stream(old_page_prev_stream)
            if diag_layer_enabled and overlap_used:
                diag_layer_old_page_stream_submit_ms = float(
                    (time.perf_counter() - t_old_page_stream_submit0) * 1000.0)
    else:
        if new_k.device.type == "cuda" and h2d_tokens > 0:
            h2d_start_event = torch.cuda.Event(enable_timing=True)
            h2d_end_event = torch.cuda.Event(enable_timing=True)
            h2d_start_event.record(torch.cuda.current_stream(new_k.device))
        t_h2d0 = time.perf_counter()
        kv_h2d_start_event: Optional[torch.cuda.Event] = None
        kv_h2d_end_event: Optional[torch.cuda.Event] = None
        if timing_cuda_event and new_k.device.type == "cuda":
            kv_h2d_start_event = torch.cuda.Event(enable_timing=True)
            kv_h2d_end_event = torch.cuda.Event(enable_timing=True)
            kv_h2d_start_event.record(torch.cuda.current_stream(new_k.device))
        t_h2d_k0 = time.perf_counter()
        reuse_k.copy_(src_k_eff, non_blocking=non_blocking_k)
        t_h2d_k1 = time.perf_counter()
        timing_v2_kv_h2d_k_submit_ms = float((t_h2d_k1 - t_h2d_k0) * 1000.0)
        t_h2d_v0 = t_h2d_k1
        reuse_v.copy_(src_v_eff, non_blocking=non_blocking_v)
        t_h2d_v1 = time.perf_counter()
        timing_v2_kv_h2d_v_submit_ms = float((t_h2d_v1 - t_h2d_v0) * 1000.0)
        if kv_h2d_end_event is not None:
            kv_h2d_end_event.record(torch.cuda.current_stream(new_k.device))
            t_h2d_wait0 = time.perf_counter()
            torch.cuda.synchronize(new_k.device)
            diag_layer_cpu_kv_h2d_wait_ms += float(
                (time.perf_counter() - t_h2d_wait0) * 1000.0)
            timing_v2_kv_h2d_device_ms = float(
                kv_h2d_start_event.elapsed_time(kv_h2d_end_event))
        if h2d_end_event is not None:
            h2d_end_event.record(torch.cuda.current_stream(new_k.device))
        t_h2d1 = time.perf_counter()
        t_h2d_submit_ms = (t_h2d1 - t_h2d0) * 1000.0
        t_pure_h2d_copy_ms = t_h2d_submit_ms
    if overlap_used and device_is_cuda and old_page_work_stream is not None:
        t_wait0 = time.perf_counter() if diag_layer_enabled else 0.0
        torch.cuda.current_stream(new_k.device).wait_stream(old_page_work_stream)
        if diag_layer_enabled:
            diag_layer_overlap_wait_submit_ms = float(
                (time.perf_counter() - t_wait0) * 1000.0)
            diag_layer_overlap_saved_estimated_ms = float(
                max(0.0, min(diag_layer_cpu_stream_submit_ms,
                             diag_layer_old_page_stream_submit_ms)
                    - diag_layer_overlap_wait_submit_ms))
    if emit_layer_metrics_full:
        _append_metrics({
            "event": "selective_reuse_h2d",
        "layer_name": layer_name,
        "layer_id": layer_id,
        "reuse_tokens": int(reuse_pos.numel()),
        "reuse_tokens_h2d": int(h2d_tokens),
        "t_reuse_h2d_ms": t_h2d_submit_ms,
        "h2d_mode": "submit",
        "h2d_source_mode": h2d_source_mode,
        "dynamic_update_backend": dynamic_update_backend,
        "is_non_blocking": bool(non_blocking_k and non_blocking_v),
        "src_is_pinned_k": bool(src_is_pinned_k),
        "src_is_pinned_v": bool(src_is_pinned_v),
        "keepold_mixed_source": bool(keepold_mixed_source),
        "keepold_switch_tokens": int(keepold_switch_tokens),
        "keepold_old_page_tokens": int(keepold_old_page_tokens),
        "t_old_page_gather_ms": float(t_old_page_gather_ms),
        "t_cpu_subset_prepare_ms": float(t_cpu_subset_prepare_ms),
        "t_cpu_subset_pin_memory_ms": float(t_cpu_subset_pin_ms),
        "t_pure_h2d_copy_ms": float(t_pure_h2d_copy_ms),
        "t_gpu_index_copy_ms": float(t_gpu_index_copy_ms),
        "t_cpu_subset_to_staging_copy_k_ms":
        float(t_cpu_subset_to_staging_copy_k_ms),
        "t_cpu_subset_to_staging_copy_v_ms":
        float(t_cpu_subset_to_staging_copy_v_ms),
        "t_staging_to_reuse_index_copy_k_ms":
        float(t_staging_to_reuse_index_copy_k_ms),
        "t_staging_to_reuse_index_copy_v_ms":
        float(t_staging_to_reuse_index_copy_v_ms),
        "t_gpu_index_copy_accounted_ms":
        float(t_gpu_index_copy_accounted_ms),
        "t_gpu_index_copy_unaccounted_ms":
        float(t_gpu_index_copy_unaccounted_ms),
        "t_gpu_index_copy_sync_overhead_ms":
        float(t_gpu_index_copy_sync_overhead_ms),
        "t_old_slot_select_ms": float(t_old_slot_select_ms),
        "t_old_page_extract_k_ms": float(t_old_page_extract_k_ms),
        "t_old_page_extract_v_ms": float(t_old_page_extract_v_ms),
        "t_old_page_dtype_cast_ms": float(t_old_page_dtype_cast_ms),
        "t_old_page_scatter_back_ms": float(t_old_page_scatter_back_ms),
        "t_old_page_sync_overhead_ms": float(t_old_page_sync_overhead_ms),
        "t_old_page_gather_accounted_ms": float(
            t_old_slot_select_ms
            + t_old_page_extract_k_ms
            + t_old_page_extract_v_ms
            + t_old_page_dtype_cast_ms
            + t_old_page_scatter_back_ms
            + t_old_page_sync_overhead_ms),
        "t_old_page_gather_unaccounted_ms": float(
            t_old_page_gather_ms
            - (t_old_slot_select_ms
               + t_old_page_extract_k_ms
               + t_old_page_extract_v_ms
               + t_old_page_dtype_cast_ms
               + t_old_page_scatter_back_ms
               + t_old_page_sync_overhead_ms)),
        "old_page_fused_k_used": bool(old_page_fused_k_used),
        "old_page_fused_k_mode": old_page_fused_k_mode,
        "old_page_fused_k_fallback_reason":
        old_page_fused_k_fallback_reason,
        "old_page_fused_k_chunks": int(old_page_fused_k_chunks),
        "t_old_page_fused_k_total_ms": float(t_old_page_fused_k_total_ms),
        "t_old_page_fused_k_gather_ms": float(t_old_page_fused_k_gather_ms),
        "t_old_page_fused_k_cast_ms": float(t_old_page_fused_k_cast_ms),
        "t_old_page_fused_k_scatter_ms": float(t_old_page_fused_k_scatter_ms),
        "old_page_fused_kv_used": bool(old_page_fused_kv_used),
        "old_page_fused_kv_fallback_reason":
        old_page_fused_kv_fallback_reason,
        "t_old_page_fused_kv_total_ms": float(t_old_page_fused_kv_total_ms),
        "t_old_page_fused_kv_call_ms": float(t_old_page_fused_kv_call_ms),
        "gpu_index_fused_k_used": bool(gpu_index_fused_k_used),
        "gpu_index_fused_k_fallback_reason":
        gpu_index_fused_k_fallback_reason,
        "t_gpu_index_fused_k_ms": float(t_gpu_index_fused_k_ms),
        "gpu_index_fused_kv_used": bool(gpu_index_fused_kv_used),
        "gpu_index_fused_kv_fallback_reason":
        gpu_index_fused_kv_fallback_reason,
        "t_gpu_index_fused_kv_ms": float(t_gpu_index_fused_kv_ms),
        "t_gpu_index_fused_kv_call_ms": float(t_gpu_index_fused_kv_call_ms),
        "used_prepared_cpu_subset": bool(used_prepared_cpu_subset),
        "is_effective_timing": False,
        "chunk_page_execution_enabled": 1 if chunk_page_exec_enabled else 0,
        "chunk_page_exec_granularity": chunk_page_exec_granularity,
        "span_mode_active": 1 if span_mode_active else 0,
        "span_exec_ready": 1 if chunk_page_span_exec_ready else 0,
        "span_gpu_fused_used": int(span_gpu_fused_used),
        "span_gpu_fused_reason": str(span_gpu_fused_reason),
        "span_fused_span_count": int(span_fused_span_count),
        "span_fused_token_count": int(span_fused_token_count),
        "span_python_fallback_used": int(span_python_fallback_used),
        "span_python_fallback_reason": str(span_python_fallback_reason),
        "chunk_page_exec_success": 1 if chunk_page_exec_success else 0,
        "chunk_page_exec_fallback_reason": chunk_page_exec_fallback_reason,
        "chunk_page_exec_tokens": int(chunk_page_exec_tokens),
        "chunk_page_exec_gpu_tokens": int(chunk_page_exec_gpu_tokens),
        "chunk_page_exec_cpu_tokens": int(chunk_page_exec_cpu_tokens),
        "chunk_page_exec_old_slots_build_ms":
        float(chunk_page_exec_old_slots_build_ms),
        "chunk_page_cpu_fallback_tokens": int(chunk_page_cpu_fallback_tokens),
        "cpu_fallback_mode": cpu_fallback_mode,
        "cpu_fallback_dense_enabled": 1 if cpu_fallback_dense_enabled else 0,
        "cpu_fallback_dense_applied": 1 if cpu_fallback_dense_applied else 0,
        "cpu_fallback_dense_reason": cpu_fallback_dense_reason_runtime,
        "cpu_fallback_tokens": int(cpu_fallback_tokens),
        "cpu_fallback_ratio": float(cpu_fallback_ratio),
        "cpu_fallback_subset_skipped":
        1 if cpu_fallback_subset_skipped else 0,
        "cpu_fallback_span_direct_ready":
        1 if cpu_fallback_span_direct_ready else 0,
        "cpu_fallback_span_direct_reason": cpu_fallback_span_direct_reason,
        "cpu_fallback_span_direct_span_count":
        int(cpu_fallback_span_direct_span_count),
        "cpu_fallback_span_direct_tokens":
        int(cpu_fallback_span_direct_tokens),
        "overlap_cpu_gpu_compose_enabled":
        1 if overlap_config_enabled else 0,
        "overlap_cpu_gpu_compose_used":
        1 if overlap_used else 0,
        "overlap_cpu_gpu_compose_reason":
        str(diag_layer_overlap_reason),
        "t_overlap_cpu_stream_submit_ms":
        float(diag_layer_cpu_stream_submit_ms),
        "t_overlap_old_page_stream_submit_ms":
        float(diag_layer_old_page_stream_submit_ms),
        "t_overlap_wait_submit_ms":
        float(diag_layer_overlap_wait_submit_ms),
        "t_overlap_saved_estimated_ms":
        float(diag_layer_overlap_saved_estimated_ms),
        "span_gpu_gather_submit_ms": float(span_gpu_gather_submit_ms),
        "span_gpu_gather_device_ms": float(span_gpu_gather_device_ms),
        "span_cpu_copy_submit_ms": float(span_cpu_copy_submit_ms),
        "span_cpu_copy_device_ms": float(span_cpu_copy_device_ms),
        "span_cpu_copy_span_count": int(span_cpu_copy_span_count),
        "span_cpu_copy_tokens": int(span_cpu_copy_tokens),
        "span_no_dense_copy": int(span_no_dense_copy),
        "token_zero_submit_ms": float(token_zero_submit_ms),
        "token_zero_skipped": int(token_zero_skipped),
        "token_zero_skip_reason": str(token_zero_skip_reason),
        "token_full_coverage_ready": int(token_full_coverage_ready),
        "token_full_coverage_reason": str(token_full_coverage_reason),
        "token_full_coverage_gpu_rows": int(token_full_coverage_gpu_rows),
        "token_full_coverage_cpu_rows": int(token_full_coverage_cpu_rows),
        "token_full_coverage_total_rows": int(token_full_coverage_total_rows),
        "token_full_coverage_overlap_rows":
        int(token_full_coverage_overlap_rows),
        "token_full_coverage_missing_rows":
        int(token_full_coverage_missing_rows),
        "compose_fixed_total_ms": float(compose_fixed_total_ms),
        "compose_layout_cache_hit": int(compose_layout_cache_hit),
        "compose_bridge_cache_hit": int(compose_bridge_cache_hit),
        "compose_metrics_minimal_used": int(compose_metrics_minimal_used),
        "compose_return_pack_ms": float(compose_return_pack_ms),
        "timing_v2_index_tensor_h2d_submit_ms":
        float(timing_v2_index_tensor_h2d_submit_ms),
        "timing_v2_index_tensor_h2d_device_ms":
        float(timing_v2_index_tensor_h2d_device_ms),
        "timing_v2_kv_h2d_k_submit_ms": float(timing_v2_kv_h2d_k_submit_ms),
        "timing_v2_kv_h2d_v_submit_ms": float(timing_v2_kv_h2d_v_submit_ms),
        "timing_v2_kv_h2d_device_ms": float(timing_v2_kv_h2d_device_ms),
        "timing_v2_gpu_index_scatter_submit_ms":
        float(timing_v2_gpu_index_scatter_submit_ms),
        "timing_v2_gpu_index_scatter_device_ms":
        float(timing_v2_gpu_index_scatter_device_ms),
        "timing_v2_old_page_gather_submit_ms":
        float(timing_v2_old_page_gather_submit_ms),
        "timing_v2_old_page_gather_device_ms":
        float(timing_v2_old_page_gather_device_ms),
            "ts_unix": time.time(),
        })
    if emit_layer_metrics_full:
        _append_metrics({
            "event": "selective_reuse_h2d_submit",
        "layer_name": layer_name,
        "layer_id": layer_id,
        "reuse_tokens": int(reuse_pos.numel()),
        "reuse_tokens_h2d": int(h2d_tokens),
        "t_reuse_h2d_submit_ms": t_h2d_submit_ms,
        "h2d_source_mode": h2d_source_mode,
        "dynamic_update_backend": dynamic_update_backend,
        "is_non_blocking": bool(non_blocking_k and non_blocking_v),
        "src_is_pinned_k": bool(src_is_pinned_k),
        "src_is_pinned_v": bool(src_is_pinned_v),
        "keepold_mixed_source": bool(keepold_mixed_source),
        "keepold_switch_tokens": int(keepold_switch_tokens),
        "keepold_old_page_tokens": int(keepold_old_page_tokens),
        "t_old_page_gather_ms": float(t_old_page_gather_ms),
        "t_cpu_subset_prepare_ms": float(t_cpu_subset_prepare_ms),
        "t_cpu_subset_pin_memory_ms": float(t_cpu_subset_pin_ms),
        "t_pure_h2d_copy_ms": float(t_pure_h2d_copy_ms),
        "t_gpu_index_copy_ms": float(t_gpu_index_copy_ms),
        "t_cpu_subset_to_staging_copy_k_ms":
        float(t_cpu_subset_to_staging_copy_k_ms),
        "t_cpu_subset_to_staging_copy_v_ms":
        float(t_cpu_subset_to_staging_copy_v_ms),
        "t_staging_to_reuse_index_copy_k_ms":
        float(t_staging_to_reuse_index_copy_k_ms),
        "t_staging_to_reuse_index_copy_v_ms":
        float(t_staging_to_reuse_index_copy_v_ms),
        "t_gpu_index_copy_accounted_ms":
        float(t_gpu_index_copy_accounted_ms),
        "t_gpu_index_copy_unaccounted_ms":
        float(t_gpu_index_copy_unaccounted_ms),
        "t_gpu_index_copy_sync_overhead_ms":
        float(t_gpu_index_copy_sync_overhead_ms),
        "t_old_slot_select_ms": float(t_old_slot_select_ms),
        "t_old_page_extract_k_ms": float(t_old_page_extract_k_ms),
        "t_old_page_extract_v_ms": float(t_old_page_extract_v_ms),
        "t_old_page_dtype_cast_ms": float(t_old_page_dtype_cast_ms),
        "t_old_page_scatter_back_ms": float(t_old_page_scatter_back_ms),
        "t_old_page_sync_overhead_ms": float(t_old_page_sync_overhead_ms),
        "t_old_page_gather_accounted_ms": float(
            t_old_slot_select_ms
            + t_old_page_extract_k_ms
            + t_old_page_extract_v_ms
            + t_old_page_dtype_cast_ms
            + t_old_page_scatter_back_ms
            + t_old_page_sync_overhead_ms),
        "t_old_page_gather_unaccounted_ms": float(
            t_old_page_gather_ms
            - (t_old_slot_select_ms
               + t_old_page_extract_k_ms
               + t_old_page_extract_v_ms
               + t_old_page_dtype_cast_ms
               + t_old_page_scatter_back_ms
               + t_old_page_sync_overhead_ms)),
        "old_page_fused_k_used": bool(old_page_fused_k_used),
        "old_page_fused_k_mode": old_page_fused_k_mode,
        "old_page_fused_k_fallback_reason":
        old_page_fused_k_fallback_reason,
        "old_page_fused_k_chunks": int(old_page_fused_k_chunks),
        "t_old_page_fused_k_total_ms": float(t_old_page_fused_k_total_ms),
        "t_old_page_fused_k_gather_ms": float(t_old_page_fused_k_gather_ms),
        "t_old_page_fused_k_cast_ms": float(t_old_page_fused_k_cast_ms),
        "t_old_page_fused_k_scatter_ms": float(t_old_page_fused_k_scatter_ms),
        "old_page_fused_kv_used": bool(old_page_fused_kv_used),
        "old_page_fused_kv_fallback_reason":
        old_page_fused_kv_fallback_reason,
        "t_old_page_fused_kv_total_ms": float(t_old_page_fused_kv_total_ms),
        "t_old_page_fused_kv_call_ms": float(t_old_page_fused_kv_call_ms),
        "gpu_index_fused_k_used": bool(gpu_index_fused_k_used),
        "gpu_index_fused_k_fallback_reason":
        gpu_index_fused_k_fallback_reason,
        "t_gpu_index_fused_k_ms": float(t_gpu_index_fused_k_ms),
        "gpu_index_fused_kv_used": bool(gpu_index_fused_kv_used),
        "gpu_index_fused_kv_fallback_reason":
        gpu_index_fused_kv_fallback_reason,
        "t_gpu_index_fused_kv_ms": float(t_gpu_index_fused_kv_ms),
        "t_gpu_index_fused_kv_call_ms": float(t_gpu_index_fused_kv_call_ms),
        "used_prepared_cpu_subset": bool(used_prepared_cpu_subset),
        "chunk_page_execution_enabled": 1 if chunk_page_exec_enabled else 0,
        "chunk_page_exec_granularity": chunk_page_exec_granularity,
        "span_mode_active": 1 if span_mode_active else 0,
        "span_exec_ready": 1 if chunk_page_span_exec_ready else 0,
        "span_gpu_fused_used": int(span_gpu_fused_used),
        "span_gpu_fused_reason": str(span_gpu_fused_reason),
        "span_fused_span_count": int(span_fused_span_count),
        "span_fused_token_count": int(span_fused_token_count),
        "span_python_fallback_used": int(span_python_fallback_used),
        "span_python_fallback_reason": str(span_python_fallback_reason),
        "chunk_page_exec_success": 1 if chunk_page_exec_success else 0,
        "chunk_page_exec_fallback_reason": chunk_page_exec_fallback_reason,
        "chunk_page_exec_tokens": int(chunk_page_exec_tokens),
        "chunk_page_exec_gpu_tokens": int(chunk_page_exec_gpu_tokens),
        "chunk_page_exec_cpu_tokens": int(chunk_page_exec_cpu_tokens),
        "chunk_page_exec_old_slots_build_ms":
        float(chunk_page_exec_old_slots_build_ms),
        "chunk_page_cpu_fallback_tokens": int(chunk_page_cpu_fallback_tokens),
        "cpu_fallback_mode": cpu_fallback_mode,
        "cpu_fallback_dense_enabled": 1 if cpu_fallback_dense_enabled else 0,
        "cpu_fallback_dense_applied": 1 if cpu_fallback_dense_applied else 0,
        "cpu_fallback_dense_reason": cpu_fallback_dense_reason_runtime,
        "cpu_fallback_tokens": int(cpu_fallback_tokens),
        "cpu_fallback_ratio": float(cpu_fallback_ratio),
        "cpu_fallback_subset_skipped":
        1 if cpu_fallback_subset_skipped else 0,
        "cpu_fallback_span_direct_ready":
        1 if cpu_fallback_span_direct_ready else 0,
        "cpu_fallback_span_direct_reason": cpu_fallback_span_direct_reason,
        "cpu_fallback_span_direct_span_count":
        int(cpu_fallback_span_direct_span_count),
        "cpu_fallback_span_direct_tokens":
        int(cpu_fallback_span_direct_tokens),
        "overlap_cpu_gpu_compose_enabled":
        1 if overlap_config_enabled else 0,
        "overlap_cpu_gpu_compose_used":
        1 if overlap_used else 0,
        "overlap_cpu_gpu_compose_reason":
        str(diag_layer_overlap_reason),
        "t_overlap_cpu_stream_submit_ms":
        float(diag_layer_cpu_stream_submit_ms),
        "t_overlap_old_page_stream_submit_ms":
        float(diag_layer_old_page_stream_submit_ms),
        "t_overlap_wait_submit_ms":
        float(diag_layer_overlap_wait_submit_ms),
        "t_overlap_saved_estimated_ms":
        float(diag_layer_overlap_saved_estimated_ms),
        "span_gpu_gather_submit_ms": float(span_gpu_gather_submit_ms),
        "span_gpu_gather_device_ms": float(span_gpu_gather_device_ms),
        "span_cpu_copy_submit_ms": float(span_cpu_copy_submit_ms),
        "span_cpu_copy_device_ms": float(span_cpu_copy_device_ms),
        "span_cpu_copy_span_count": int(span_cpu_copy_span_count),
        "span_cpu_copy_tokens": int(span_cpu_copy_tokens),
        "span_no_dense_copy": int(span_no_dense_copy),
        "token_zero_submit_ms": float(token_zero_submit_ms),
        "token_zero_skipped": int(token_zero_skipped),
        "token_zero_skip_reason": str(token_zero_skip_reason),
        "token_full_coverage_ready": int(token_full_coverage_ready),
        "token_full_coverage_reason": str(token_full_coverage_reason),
        "token_full_coverage_gpu_rows": int(token_full_coverage_gpu_rows),
        "token_full_coverage_cpu_rows": int(token_full_coverage_cpu_rows),
        "token_full_coverage_total_rows": int(token_full_coverage_total_rows),
        "token_full_coverage_overlap_rows":
        int(token_full_coverage_overlap_rows),
        "token_full_coverage_missing_rows":
        int(token_full_coverage_missing_rows),
        "compose_fixed_total_ms": float(compose_fixed_total_ms),
        "compose_layout_cache_hit": int(compose_layout_cache_hit),
        "compose_bridge_cache_hit": int(compose_bridge_cache_hit),
        "compose_metrics_minimal_used": int(compose_metrics_minimal_used),
        "compose_return_pack_ms": float(compose_return_pack_ms),
        "timing_v2_index_tensor_h2d_submit_ms":
        float(timing_v2_index_tensor_h2d_submit_ms),
        "timing_v2_index_tensor_h2d_device_ms":
        float(timing_v2_index_tensor_h2d_device_ms),
        "timing_v2_kv_h2d_k_submit_ms": float(timing_v2_kv_h2d_k_submit_ms),
        "timing_v2_kv_h2d_v_submit_ms": float(timing_v2_kv_h2d_v_submit_ms),
        "timing_v2_kv_h2d_device_ms": float(timing_v2_kv_h2d_device_ms),
        "timing_v2_gpu_index_scatter_submit_ms":
        float(timing_v2_gpu_index_scatter_submit_ms),
        "timing_v2_gpu_index_scatter_device_ms":
        float(timing_v2_gpu_index_scatter_device_ms),
        "timing_v2_old_page_gather_submit_ms":
        float(timing_v2_old_page_gather_submit_ms),
        "timing_v2_old_page_gather_device_ms":
        float(timing_v2_old_page_gather_device_ms),
            "ts_unix": time.time(),
        })

    h2d_effective_recorded = False
    if reuse_rope_impl != _ROPE_IMPL_NONE:
        if reuse_rope_impl != _ROPE_IMPL_LMCACHE_COPS_DIRECT:
            raise ValueError(
                "Unsupported selective reuse_rope_impl: "
                f"{reuse_rope_impl}")
        if reuse_rope_cos_sin_cache is None:
            raise RuntimeError(
                "reuse_rope_cos_sin_cache is required for selective fused rope"
            )
        if (reuse_rope_cos_sin_cache.device != reuse_k.device
                or reuse_rope_cos_sin_cache.dtype != reuse_k.dtype):
            t_rope_prepare0 = time.perf_counter() if diag_layer_enabled else 0.0
            reuse_rope_cos_sin_cache = reuse_rope_cos_sin_cache.to(
                reuse_k.device, dtype=reuse_k.dtype)
            if diag_layer_enabled:
                diag_layer_rope_prepare_ms += float(
                    (time.perf_counter() - t_rope_prepare0) * 1000.0)
        lmc_ops = self._get_lmcache_cops()
        rope_start_event: Optional[torch.cuda.Event] = None
        rope_end_event: Optional[torch.cuda.Event] = None
        if timing_cuda_event and reuse_k.device.type == "cuda":
            rope_start_event = torch.cuda.Event(enable_timing=True)
            rope_end_event = torch.cuda.Event(enable_timing=True)
            rope_start_event.record(torch.cuda.current_stream(reuse_k.device))
        t_rope0 = time.perf_counter()
        t_rope_submit0 = t_rope0
        lmc_ops.rotary_embedding_k_fused(
            reuse_old_pos_for_rope,
            reuse_new_pos_for_rope,
            reuse_k,
            head_size,
            reuse_rope_cos_sin_cache,
            reuse_rope_is_neox_style,
        )
        t_rope_submit1 = time.perf_counter()
        timing_v2_reuse_rope_submit_ms = float((t_rope_submit1 - t_rope_submit0) *
                                               1000.0)
        if rope_end_event is not None:
            rope_end_event.record(torch.cuda.current_stream(reuse_k.device))
        t_rope_wait0 = time.perf_counter()
        if reuse_k.device.type == "cuda":
            torch.cuda.synchronize(reuse_k.device)
        t_rope_wait1 = time.perf_counter()
        timing_v2_reuse_rope_wait_ms = float((t_rope_wait1 - t_rope_wait0) *
                                             1000.0)
        t_rope1 = time.perf_counter()
        diag_layer_rope_total_ms = float((t_rope1 - t_rope0) * 1000.0)
        t_rope_kernel_ms: Optional[float] = None
        if rope_start_event is not None and rope_end_event is not None:
            t_rope_kernel_ms = float(
                rope_start_event.elapsed_time(rope_end_event))
            timing_v2_reuse_rope_device_ms = float(t_rope_kernel_ms)
        if emit_layer_metrics_full:
            _append_metrics({
                "event": "selective_reuse_rope",
                "layer_name": layer_name,
                "layer_id": layer_id,
                "reuse_tokens": int(reuse_pos.numel()),
                "reuse_tokens_rope": int(reuse_k.shape[0]),
                "reuse_rope_impl": reuse_rope_impl,
                "h2d_source_mode": h2d_source_mode,
                "dynamic_update_backend": dynamic_update_backend,
                "t_reuse_rope_ms": (t_rope1 - t_rope0) * 1000.0,
                "t_reuse_rope_with_wait_ms": (t_rope1 - t_rope0) * 1000.0,
                "t_reuse_rope_kernel_ms": t_rope_kernel_ms,
                "timing_v2_reuse_rope_submit_ms":
                float(timing_v2_reuse_rope_submit_ms),
                "timing_v2_reuse_rope_device_ms":
                float(timing_v2_reuse_rope_device_ms),
                "timing_v2_reuse_rope_wait_ms":
                float(timing_v2_reuse_rope_wait_ms),
                "includes_prior_h2d_wait": True,
                "ts_unix": time.time(),
            })
        if h2d_start_event is not None and h2d_end_event is not None:
            if emit_layer_metrics_full:
                _append_metrics({
                    "event": "selective_reuse_h2d_effective",
                    "layer_name": layer_name,
                    "layer_id": layer_id,
                    "reuse_tokens": int(reuse_pos.numel()),
                    "reuse_tokens_h2d": int(h2d_tokens),
                    "t_reuse_h2d_effective_ms": float(
                        h2d_start_event.elapsed_time(h2d_end_event)),
                    "t_pure_h2d_copy_ms": float(t_pure_h2d_copy_ms),
                    "t_cpu_subset_prepare_ms": float(t_cpu_subset_prepare_ms),
                    "t_cpu_subset_pin_memory_ms": float(t_cpu_subset_pin_ms),
                    "effective_scope": "pure_h2d_copy",
                    "effective_sync_point": "reuse_rope_sync",
                    "overlap_with_reuse_rope_ms": False,
                    "h2d_source_mode": h2d_source_mode,
                    "dynamic_update_backend": dynamic_update_backend,
                    "ts_unix": time.time(),
                })
            h2d_effective_recorded = True

    if not h2d_effective_recorded:
        if emit_layer_metrics_full:
            _append_metrics({
                "event": "selective_reuse_h2d_effective",
                "layer_name": layer_name,
                "layer_id": layer_id,
                "reuse_tokens": int(reuse_pos.numel()),
                "reuse_tokens_h2d": int(h2d_tokens),
                "t_reuse_h2d_effective_ms": None,
                "t_pure_h2d_copy_ms": float(t_pure_h2d_copy_ms),
                "t_cpu_subset_prepare_ms": float(t_cpu_subset_prepare_ms),
                "t_cpu_subset_pin_memory_ms": float(t_cpu_subset_pin_ms),
                "effective_scope": "unavailable",
                "effective_sync_point": "unavailable_in_compose",
                "overlap_with_reuse_rope_ms": False,
                "h2d_source_mode": h2d_source_mode,
                "dynamic_update_backend": dynamic_update_backend,
                "ts_unix": time.time(),
            })

    if getattr(self, "_keepold_diag_layer_breakdown_enabled", False):
        cpu_kv_h2d_submit_ms = float(
            timing_v2_kv_h2d_k_submit_ms + timing_v2_kv_h2d_v_submit_ms)
        old_page_fused_call_ms = float(
            t_old_page_fused_kv_call_ms if t_old_page_fused_kv_call_ms > 0.0
            else t_old_page_fused_k_total_ms)
        old_page_fused_total_ms = float(
            t_old_page_fused_kv_total_ms if t_old_page_fused_kv_total_ms > 0.0
            else t_old_page_fused_k_total_ms)
        old_page_fused_reason = str(
            old_page_fused_kv_fallback_reason
            if old_page_fused_kv_fallback_reason != "disabled"
            else old_page_fused_k_fallback_reason)
        token_mode_active = bool(
            keepold_mixed_source
            and chunk_page_exec_enabled
            and chunk_page_exec_granularity != _CHUNK_PAGE_EXEC_GRANULARITY_SPAN)
        span_mode_for_diag = bool(span_mode_active)
        full_cpu_copy_active = bool(
            (not keepold_mixed_source)
            and int(h2d_tokens) > 0
            and int(keepold_switch_tokens) == 0)
        diag_compose_fields = {
            "diag_layer_source_prepare_ms": 0.0,
            "diag_layer_cpu_subset_select_ms": float(t_cpu_subset_prepare_ms),
            "diag_layer_cpu_subset_pin_ms": float(t_cpu_subset_pin_ms),
            "diag_layer_cpu_kv_h2d_submit_ms": float(cpu_kv_h2d_submit_ms),
            "diag_layer_cpu_kv_h2d_device_ms": float(
                timing_v2_kv_h2d_device_ms),
            "diag_layer_cpu_kv_h2d_wait_ms": float(
                diag_layer_cpu_kv_h2d_wait_ms),
            "diag_layer_gpu_index_scatter_submit_ms": float(
                timing_v2_gpu_index_scatter_submit_ms),
            "diag_layer_gpu_index_scatter_device_ms": float(
                timing_v2_gpu_index_scatter_device_ms),
            "diag_layer_gpu_index_scatter_wait_ms": float(
                diag_layer_gpu_index_scatter_wait_ms),
            "diag_layer_reuse_buffer_zero_submit_ms": float(
                diag_layer_reuse_buffer_zero_submit_ms),
            "diag_layer_reuse_buffer_zero_device_ms": float(
                diag_layer_reuse_buffer_zero_device_ms),
            "diag_layer_reuse_buffer_zero_wait_ms": float(
                diag_layer_reuse_buffer_zero_wait_ms),
            "diag_layer_reuse_staging_shape_check_ms": float(
                diag_layer_reuse_staging_shape_check_ms),
            "diag_layer_cpu_staging_shape_check_ms": float(
                diag_layer_cpu_staging_shape_check_ms),
            "diag_layer_staging_alloc_ms": float(
                diag_layer_staging_alloc_ms),
            "diag_layer_src_reshape_ms": float(diag_layer_src_reshape_ms),
            "diag_layer_position_index_select_ms": float(
                diag_layer_position_index_select_ms),
            "diag_layer_get_positions_ms": float(diag_layer_get_positions_ms),
            "diag_layer_cpu_subset_fetch_ms": float(
                diag_layer_cpu_subset_fetch_ms),
            "diag_layer_cpu_subset_reshape_ms": float(
                diag_layer_cpu_subset_reshape_ms),
            "diag_layer_cpu_subset_to_staging_submit_ms": float(
                diag_layer_cpu_subset_to_staging_submit_ms),
            "diag_layer_staging_to_reuse_scatter_submit_ms": float(
                diag_layer_staging_to_reuse_scatter_submit_ms),
            "diag_layer_old_page_gather_submit_ms": float(
                timing_v2_old_page_gather_submit_ms),
            "diag_layer_old_page_gather_device_ms": float(
                timing_v2_old_page_gather_device_ms),
            "diag_layer_old_page_gather_wait_ms": float(
                diag_layer_old_page_gather_wait_ms),
            "diag_layer_old_page_precheck_ms": float(
                diag_layer_old_page_precheck_ms),
            "diag_layer_old_page_call_wrapper_ms": float(
                diag_layer_old_page_call_wrapper_ms),
            "diag_layer_old_page_fused_call_ms": float(
                old_page_fused_call_ms),
            "diag_layer_old_page_fused_total_ms": float(
                old_page_fused_total_ms),
            "diag_layer_rope_reposition_submit_ms": float(
                timing_v2_reuse_rope_submit_ms),
            "diag_layer_rope_reposition_device_ms": float(
                timing_v2_reuse_rope_device_ms),
            "diag_layer_rope_wait_ms": float(timing_v2_reuse_rope_wait_ms),
            "diag_layer_rope_prepare_ms": float(diag_layer_rope_prepare_ms),
            "diag_layer_rope_total_ms": float(diag_layer_rope_total_ms),
            "diag_layer_compose_python_overhead_ms": 0.0,
            "diag_layer_compose_total_ms": 0.0,
            "diag_layer_compose_known_sum_ms": 0.0,
            "diag_layer_compose_residual_ms": 0.0,
            "diag_layer_compose_wall_control_ms": 0.0,
            "diag_layer_compose_wall_zero_ms": 0.0,
            "diag_layer_compose_wall_cpu_source_ms": 0.0,
            "diag_layer_compose_wall_cpu_h2d_ms": 0.0,
            "diag_layer_compose_wall_cpu_scatter_ms": 0.0,
            "diag_layer_compose_wall_gpu_source_ms": 0.0,
            "diag_layer_compose_wall_rope_ms": 0.0,
            "diag_layer_compose_wall_return_pack_ms": 0.0,
            "diag_layer_compose_wall_known_sum_ms": 0.0,
            "diag_layer_compose_wall_residual_ms": 0.0,
            "diag_layer_token_branch_enter_ms": float(
                diag_layer_token_branch_enter_ms if token_mode_active else 0.0),
            "diag_layer_bridge_lookup_ms": float(
                diag_layer_bridge_lookup_ms if token_mode_active else 0.0),
            "diag_layer_switch_index_prepare_ms": float(
                diag_layer_switch_index_prepare_ms if token_mode_active else 0.0),
            "diag_layer_cpu_reuse_idx_prepare_ms": float(
                diag_layer_cpu_reuse_idx_prepare_ms if token_mode_active else 0.0),
            "diag_layer_dense_check_ms": float(
                diag_layer_dense_check_ms if token_mode_active else 0.0),
            "diag_layer_token_coverage_check_ms": float(
                diag_layer_token_coverage_check_ms if token_mode_active else 0.0),
            "diag_layer_old_page_layout_infer_ms": float(
                diag_layer_old_page_layout_infer_ms if token_mode_active else 0.0),
            "diag_layer_fused_kv_precheck_ms": float(
                diag_layer_fused_kv_precheck_ms if token_mode_active else 0.0),
            "diag_layer_overlap_enabled": float(
                diag_layer_overlap_enabled if token_mode_active else 0.0),
            "diag_layer_overlap_used": float(
                diag_layer_overlap_used if token_mode_active else 0.0),
            "diag_layer_overlap_reason": str(
                diag_layer_overlap_reason if token_mode_active else "inactive"),
            "diag_layer_cpu_stream_submit_ms": float(
                diag_layer_cpu_stream_submit_ms if token_mode_active else 0.0),
            "diag_layer_old_page_stream_submit_ms": float(
                diag_layer_old_page_stream_submit_ms
                if token_mode_active else 0.0),
            "diag_layer_overlap_wait_submit_ms": float(
                diag_layer_overlap_wait_submit_ms if token_mode_active else 0.0),
            "diag_layer_overlap_saved_estimated_ms": float(
                diag_layer_overlap_saved_estimated_ms
                if token_mode_active else 0.0),
            "diag_layer_cpu_fallback_span_direct_used": float(
                diag_layer_cpu_fallback_span_direct_used
                if token_mode_active else 0.0),
            "diag_layer_cpu_fallback_span_count": float(
                diag_layer_cpu_fallback_span_count
                if token_mode_active else 0.0),
            "diag_layer_cpu_fallback_span_tokens": float(
                diag_layer_cpu_fallback_span_tokens
                if token_mode_active else 0.0),
            "diag_layer_cpu_fallback_span_copy_submit_ms": float(
                diag_layer_cpu_fallback_span_copy_submit_ms
                if token_mode_active else 0.0),
            "diag_layer_metrics_append_ms": float(
                diag_layer_metrics_append_ms if token_mode_active else 0.0),
            "diag_layer_return_pack_ms": 0.0,
            "diag_layer_rope_impl": str(reuse_rope_impl),
            "diag_layer_rope_tokens": int(reuse_k.shape[0]),
            "diag_old_page_tokens": int(keepold_old_page_tokens),
            "diag_cpu_source_tokens": int(h2d_tokens),
            "diag_gpu_target_tokens": int(keepold_switch_tokens),
            "diag_old_page_fused_kv_used":
            1 if bool(old_page_fused_kv_used) else 0,
            "diag_old_page_fused_kv_reason": old_page_fused_reason,
            "diag_old_page_fused_span_count": int(span_fused_span_count),
            "diag_old_page_fused_token_count": int(
                span_fused_token_count
                if span_fused_token_count > 0 else keepold_old_page_tokens),
            "diag_old_page_fused_call_ms": float(old_page_fused_call_ms),
            "diag_old_page_fused_total_ms": float(old_page_fused_total_ms),
            "diag_token_old_slots_build_ms": float(
                chunk_page_exec_old_slots_build_ms if token_mode_active else 0.0),
            "diag_token_old_slots_count": int(
                keepold_old_page_tokens if token_mode_active else 0),
            "diag_token_dst_idx_count": int(
                keepold_switch_tokens if token_mode_active else 0),
            "diag_token_zero_submit_ms": float(
                token_zero_submit_ms if token_mode_active else 0.0),
            "diag_token_zero_skipped": int(
                token_zero_skipped if token_mode_active else 0),
            "diag_token_zero_skip_reason": str(
                token_zero_skip_reason if token_mode_active else "inactive"),
            "diag_token_full_coverage_ready": int(
                token_full_coverage_ready if token_mode_active else 0),
            "diag_token_full_coverage_reason": str(
                token_full_coverage_reason
                if token_mode_active else "inactive"),
            "diag_token_full_coverage_gpu_rows": int(
                token_full_coverage_gpu_rows if token_mode_active else 0),
            "diag_token_full_coverage_cpu_rows": int(
                token_full_coverage_cpu_rows if token_mode_active else 0),
            "diag_token_full_coverage_total_rows": int(
                token_full_coverage_total_rows if token_mode_active else 0),
            "diag_token_full_coverage_overlap_rows": int(
                token_full_coverage_overlap_rows if token_mode_active else 0),
            "diag_token_full_coverage_missing_rows": int(
                token_full_coverage_missing_rows if token_mode_active else 0),
            "diag_layer_compose_fixed_total_ms": float(compose_fixed_total_ms),
            "diag_layer_compose_layout_cache_hit": int(
                compose_layout_cache_hit),
            "diag_layer_compose_bridge_cache_hit": int(
                compose_bridge_cache_hit),
            "diag_layer_compose_metrics_minimal_used": int(
                compose_metrics_minimal_used),
            "diag_span_plan_build_ms": float(
                getattr(self, "_keepold_chunk_page_span_plan_build_ms", 0.0)
                if span_mode_for_diag else 0.0),
            "diag_span_gpu_span_count": int(
                getattr(self, "_keepold_chunk_page_gpu_span_count", 0)
                if span_mode_for_diag else 0),
            "diag_span_cpu_span_count": int(
                getattr(self, "_keepold_chunk_page_cpu_span_count", 0)
                if span_mode_for_diag else 0),
            "diag_span_cpu_copy_tokens": int(
                span_cpu_copy_tokens if span_mode_for_diag else 0),
            "diag_span_no_dense_copy": int(
                span_no_dense_copy if span_mode_for_diag else 0),
            "diag_span_python_fallback_used": int(
                span_python_fallback_used if span_mode_for_diag else 0),
            "diag_full_cpu_copy_tokens": int(
                h2d_tokens if full_cpu_copy_active else 0),
            "diag_full_cpu_copy_submit_ms": float(
                cpu_kv_h2d_submit_ms if full_cpu_copy_active else 0.0),
            "diag_full_cpu_copy_device_ms": float(
                timing_v2_kv_h2d_device_ms if full_cpu_copy_active else 0.0),
        }
    t_return_pack0 = time.perf_counter()
    full_len = self._forward_prompt_len
    full_k = torch.empty(
        (full_len, num_kv_heads, head_size),
        dtype=new_k.dtype,
        device=new_k.device,
    )
    full_v = torch.empty(
        (full_len, int(new_v.shape[1]), int(new_v.shape[2])),
        dtype=new_v.dtype,
        device=new_v.device,
    )

    probe_reuse_k: Optional[torch.Tensor] = None
    probe_reuse_v: Optional[torch.Tensor] = None
    if probe_positions is not None:
        probe_pos = probe_positions.detach().to(
            device=new_k.device, dtype=torch.long).contiguous()
        if int(probe_pos.numel()) > 0:
            if int(probe_pos.min().item()) < 0 or int(
                    probe_pos.max().item()) >= int(full_len):
                raise ValueError(
                    f"probe_positions out of range for prompt_len={full_len}")
            reuse_pos_to_idx = torch.full(
                (int(full_len),),
                -1,
                dtype=torch.long,
                device=new_k.device,
            )
            reuse_pos_to_idx[full_reuse_pos_for_index_copy] = torch.arange(
                int(full_reuse_pos_for_index_copy.numel()),
                dtype=torch.long,
                device=new_k.device,
            )
            probe_reuse_idx = reuse_pos_to_idx.index_select(0, probe_pos)
            if bool((probe_reuse_idx < 0).any().item()):
                raise ValueError(
                    "probe_positions contains token(s) not available in reuse source"
                )
            probe_reuse_k = reuse_k.index_select(0, probe_reuse_idx).contiguous()
            probe_reuse_v = reuse_v.index_select(0, probe_reuse_idx).contiguous()

    full_k.index_copy_(0, full_reuse_pos_for_index_copy, reuse_k)
    full_v.index_copy_(0, full_reuse_pos_for_index_copy, reuse_v)
    full_k.index_copy_(0, recompute_pos, new_k)
    full_v.index_copy_(0, recompute_pos, new_v)
    compose_return_pack_ms = float((time.perf_counter() - t_return_pack0) *
                                   1000.0)
    compose_bridge_cache_hit = (
        1 if getattr(self, "_keepold_cached_bridge_layers", None) is not None else 0)
    compose_layout_cache_hit = (
        1 if int(layer_id) in getattr(self, "_keepold_paged_layout_meta_by_layer", {})
        else 0)
    compose_fixed_total_ms = float(
        token_zero_submit_ms
        + compose_return_pack_ms
        + diag_layer_reuse_staging_shape_check_ms
        + diag_layer_cpu_staging_shape_check_ms)
    if not emit_layer_metrics_full:
        compose_metrics_minimal_used = 1
        self.record_compose_layer_metrics_aggregate(
            reuse_tokens_h2d=int(h2d_tokens),
            timing_v2_kv_h2d_k_submit_ms=float(timing_v2_kv_h2d_k_submit_ms),
            timing_v2_kv_h2d_v_submit_ms=float(timing_v2_kv_h2d_v_submit_ms),
            timing_v2_kv_h2d_device_ms=float(timing_v2_kv_h2d_device_ms),
            timing_v2_gpu_index_scatter_submit_ms=float(
                timing_v2_gpu_index_scatter_submit_ms),
            timing_v2_gpu_index_scatter_device_ms=float(
                timing_v2_gpu_index_scatter_device_ms),
            timing_v2_old_page_gather_submit_ms=float(
                timing_v2_old_page_gather_submit_ms),
            timing_v2_old_page_gather_device_ms=float(
                timing_v2_old_page_gather_device_ms),
            timing_v2_reuse_rope_submit_ms=float(
                timing_v2_reuse_rope_submit_ms),
            timing_v2_reuse_rope_device_ms=float(
                timing_v2_reuse_rope_device_ms),
            timing_v2_reuse_rope_wait_ms=float(timing_v2_reuse_rope_wait_ms),
            t_gpu_index_copy_sync_overhead_ms=float(
                t_gpu_index_copy_sync_overhead_ms),
            span_gpu_gather_submit_ms=float(span_gpu_gather_submit_ms),
            span_gpu_gather_device_ms=float(span_gpu_gather_device_ms),
            span_cpu_copy_submit_ms=float(span_cpu_copy_submit_ms),
            span_cpu_copy_device_ms=float(span_cpu_copy_device_ms),
            span_cpu_copy_span_count=int(span_cpu_copy_span_count),
            span_cpu_copy_tokens=int(span_cpu_copy_tokens),
            span_no_dense_copy=int(span_no_dense_copy),
            cpu_fallback_span_direct_used=float(
                diag_layer_cpu_fallback_span_direct_used),
            cpu_fallback_span_direct_span_count=float(
                diag_layer_cpu_fallback_span_count),
            cpu_fallback_span_direct_tokens=float(
                diag_layer_cpu_fallback_span_tokens),
            cpu_fallback_span_direct_copy_submit_ms=float(
                diag_layer_cpu_fallback_span_copy_submit_ms),
            cpu_fallback_dense_enabled=bool(cpu_fallback_dense_enabled),
            cpu_fallback_tokens=int(cpu_fallback_tokens),
            cpu_fallback_ratio=float(cpu_fallback_ratio),
            cpu_fallback_subset_skipped=bool(cpu_fallback_subset_skipped),
            chunk_page_exec_granularity=str(chunk_page_exec_granularity),
            cpu_fallback_mode=str(cpu_fallback_mode),
            cpu_fallback_dense_reason=str(cpu_fallback_dense_reason_runtime),
            token_zero_submit_ms=float(token_zero_submit_ms),
            token_zero_skipped=int(token_zero_skipped),
            token_zero_skip_reason=str(token_zero_skip_reason),
            token_full_coverage_ready=int(token_full_coverage_ready),
            token_full_coverage_reason=str(token_full_coverage_reason),
            token_full_coverage_gpu_rows=int(token_full_coverage_gpu_rows),
            token_full_coverage_cpu_rows=int(token_full_coverage_cpu_rows),
            token_full_coverage_total_rows=int(token_full_coverage_total_rows),
            token_full_coverage_overlap_rows=int(
                token_full_coverage_overlap_rows),
            token_full_coverage_missing_rows=int(
                token_full_coverage_missing_rows),
            compose_fixed_total_ms=float(compose_fixed_total_ms),
            compose_layout_cache_hit=int(compose_layout_cache_hit),
            compose_bridge_cache_hit=int(compose_bridge_cache_hit),
            compose_metrics_minimal_used=int(compose_metrics_minimal_used),
            compose_return_pack_ms=float(compose_return_pack_ms),
        )
    if emit_layer_metrics_full:
        _append_metrics({
            "event": "selective_compose_fixed",
            "layer_name": layer_name,
            "layer_id": layer_id,
            "compose_fixed_total_ms": float(compose_fixed_total_ms),
            "compose_layout_cache_hit": int(compose_layout_cache_hit),
            "compose_bridge_cache_hit": int(compose_bridge_cache_hit),
            "compose_metrics_minimal_used": int(compose_metrics_minimal_used),
            "compose_return_pack_ms": float(compose_return_pack_ms),
            "token_zero_submit_ms": float(token_zero_submit_ms),
            "token_zero_skipped": int(token_zero_skipped),
            "token_zero_skip_reason": str(token_zero_skip_reason),
            "token_full_coverage_ready": int(token_full_coverage_ready),
            "token_full_coverage_reason": str(token_full_coverage_reason),
            "token_full_coverage_gpu_rows": int(token_full_coverage_gpu_rows),
            "token_full_coverage_cpu_rows": int(token_full_coverage_cpu_rows),
            "token_full_coverage_total_rows": int(token_full_coverage_total_rows),
            "token_full_coverage_overlap_rows":
            int(token_full_coverage_overlap_rows),
            "token_full_coverage_missing_rows":
            int(token_full_coverage_missing_rows),
            "ts_unix": time.time(),
        })
    if diag_layer_enabled:
        diag_layer_return_pack_ms = float(compose_return_pack_ms)
        diag_layer_compose_total_ms = float(
            (time.perf_counter() - diag_compose_t0) * 1000.0)
        diag_layer_compose_wall_control_ms = float(
            diag_layer_src_reshape_ms
            + diag_layer_get_positions_ms
            + diag_layer_position_index_select_ms
            + diag_layer_reuse_staging_shape_check_ms
            + diag_layer_cpu_staging_shape_check_ms
            + diag_layer_staging_alloc_ms
            + diag_layer_bridge_lookup_ms
            + diag_layer_switch_index_prepare_ms
            + diag_layer_cpu_reuse_idx_prepare_ms
            + diag_layer_dense_check_ms
            + diag_layer_token_coverage_check_ms
            + diag_layer_old_page_layout_infer_ms
            + diag_layer_fused_kv_precheck_ms
            + diag_layer_metrics_append_ms)
        diag_layer_compose_wall_zero_ms = float(
            diag_layer_reuse_buffer_zero_submit_ms
            + diag_layer_reuse_buffer_zero_wait_ms)
        diag_layer_compose_wall_cpu_h2d_ms = float(
            diag_layer_cpu_subset_to_staging_submit_ms
            + diag_layer_cpu_kv_h2d_wait_ms)
        diag_layer_compose_wall_cpu_scatter_ms = float(
            diag_layer_staging_to_reuse_scatter_submit_ms
            + diag_layer_gpu_index_scatter_wait_ms)
        diag_layer_compose_wall_cpu_source_ms = float(
            diag_layer_cpu_subset_fetch_ms
            + diag_layer_cpu_subset_reshape_ms
            + diag_layer_cpu_subset_to_staging_submit_ms
            + diag_layer_cpu_kv_h2d_wait_ms
            + diag_layer_staging_to_reuse_scatter_submit_ms
            + diag_layer_gpu_index_scatter_wait_ms)
        diag_layer_compose_wall_gpu_source_ms = float(
            diag_layer_old_page_precheck_ms
            + diag_layer_old_page_call_wrapper_ms)
        diag_layer_compose_wall_rope_ms = float(
            diag_layer_rope_prepare_ms + diag_layer_rope_total_ms)
        diag_layer_compose_wall_return_pack_ms = float(
            diag_layer_return_pack_ms)
        diag_layer_compose_wall_known_sum_ms = float(
            diag_layer_compose_wall_control_ms
            + diag_layer_compose_wall_zero_ms
            + diag_layer_compose_wall_cpu_source_ms
            + diag_layer_compose_wall_gpu_source_ms
            + diag_layer_compose_wall_rope_ms
            + diag_layer_compose_wall_return_pack_ms)
        diag_layer_compose_wall_residual_ms = float(
            max(0.0,
                diag_layer_compose_total_ms
                - diag_layer_compose_wall_known_sum_ms))
        diag_layer_compose_known_sum_ms = float(
            diag_layer_src_reshape_ms
            + diag_layer_get_positions_ms
            + diag_layer_position_index_select_ms
            + diag_layer_reuse_staging_shape_check_ms
            + diag_layer_cpu_staging_shape_check_ms
            + diag_layer_staging_alloc_ms
            + diag_layer_reuse_buffer_zero_submit_ms
            + diag_layer_reuse_buffer_zero_wait_ms
            + diag_layer_cpu_subset_fetch_ms
            + diag_layer_cpu_subset_reshape_ms
            + diag_layer_cpu_subset_to_staging_submit_ms
            + diag_layer_cpu_kv_h2d_wait_ms
            + diag_layer_staging_to_reuse_scatter_submit_ms
            + diag_layer_gpu_index_scatter_wait_ms
            + diag_layer_old_page_precheck_ms
            + diag_layer_old_page_call_wrapper_ms
            + diag_layer_rope_prepare_ms
            + diag_layer_rope_total_ms
            + diag_layer_bridge_lookup_ms
            + diag_layer_switch_index_prepare_ms
            + diag_layer_cpu_reuse_idx_prepare_ms
            + diag_layer_dense_check_ms
            + diag_layer_token_coverage_check_ms
            + diag_layer_old_page_layout_infer_ms
            + diag_layer_fused_kv_precheck_ms
            + diag_layer_metrics_append_ms
            + diag_layer_return_pack_ms)
        diag_layer_compose_residual_ms = float(
            max(0.0, diag_layer_compose_total_ms - diag_layer_compose_known_sum_ms))
        diag_layer_compose_python_overhead_ms = float(
            diag_layer_compose_residual_ms)
        if diag_compose_fields is not None:
            diag_compose_fields["diag_layer_metrics_append_ms"] = float(
                diag_layer_metrics_append_ms)
            diag_compose_fields["diag_layer_return_pack_ms"] = float(
                diag_layer_return_pack_ms)
            diag_compose_fields["diag_layer_compose_total_ms"] = float(
                diag_layer_compose_total_ms)
            diag_compose_fields["diag_layer_compose_known_sum_ms"] = float(
                diag_layer_compose_known_sum_ms)
            diag_compose_fields["diag_layer_compose_residual_ms"] = float(
                diag_layer_compose_residual_ms)
            diag_compose_fields["diag_layer_compose_python_overhead_ms"] = float(
                diag_layer_compose_python_overhead_ms)
            diag_compose_fields["diag_layer_compose_wall_control_ms"] = float(
                diag_layer_compose_wall_control_ms)
            diag_compose_fields["diag_layer_compose_wall_zero_ms"] = float(
                diag_layer_compose_wall_zero_ms)
            diag_compose_fields["diag_layer_compose_wall_cpu_source_ms"] = float(
                diag_layer_compose_wall_cpu_source_ms)
            diag_compose_fields["diag_layer_compose_wall_cpu_h2d_ms"] = float(
                diag_layer_compose_wall_cpu_h2d_ms)
            diag_compose_fields["diag_layer_compose_wall_cpu_scatter_ms"] = float(
                diag_layer_compose_wall_cpu_scatter_ms)
            diag_compose_fields["diag_layer_compose_wall_gpu_source_ms"] = float(
                diag_layer_compose_wall_gpu_source_ms)
            diag_compose_fields["diag_layer_compose_wall_rope_ms"] = float(
                diag_layer_compose_wall_rope_ms)
            diag_compose_fields["diag_layer_compose_wall_return_pack_ms"] = float(
                diag_layer_compose_wall_return_pack_ms)
            diag_compose_fields["diag_layer_compose_wall_known_sum_ms"] = float(
                diag_layer_compose_wall_known_sum_ms)
            diag_compose_fields["diag_layer_compose_wall_residual_ms"] = float(
                diag_layer_compose_wall_residual_ms)
            self.record_layer_compose_diag(
                layer_id=layer_id,
                layer_name=layer_name,
                fields=diag_compose_fields,
            )
    self.maybe_nvtx_range_pop(layer_pipeline_nvtx_pushed)
    return full_k, full_v, probe_reuse_k, probe_reuse_v
