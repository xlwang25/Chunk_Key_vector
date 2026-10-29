# SPDX-License-Identifier: Apache-2.0
"""Chunk-page record/lookup plan helpers (dry-run)."""

from __future__ import annotations

from typing import Optional, Sequence
import time

import torch

from .chunk_page_index_hash import compute_token_checksum, hash_chunk_tokens
from .chunk_page_index_manager import ChunkPageIndex
from .chunk_page_index_types import (ChunkPageEntry, GatherSpanPlan, PageReusePlan,
                                     PageSpan)


def _normalize_chunk_infos(
    chunk_infos: Sequence[dict[str, int]],
    *,
    prompt_len: int,
) -> list[dict[str, int]]:
    out: list[dict[str, int]] = []
    for raw in chunk_infos:
        start = int(raw.get("start", -1))
        end = int(raw.get("end", -1))
        if start < 0 or end <= start or end > int(prompt_len):
            continue
        out.append({
            "orig_idx": int(raw.get("orig_idx", -1)),
            "start": start,
            "end": end,
        })
    out.sort(key=lambda x: (int(x["start"]), int(x["end"]), int(x["orig_idx"])))
    return out


def _slice_chunk_tokens(
    prompt_token_ids: Sequence[int],
    *,
    start: int,
    end: int,
) -> Sequence[int]:
    """Fast token span extraction without redundant per-token int() casts."""
    sliced = prompt_token_ids[start:end]
    if isinstance(sliced, (list, tuple)):
        return sliced
    return [int(x) for x in sliced]


def build_page_spans(
    old_start: int,
    old_end: int,
    block_size: int,
    logical_to_physical: Sequence[int],
) -> list[PageSpan]:
    spans: list[PageSpan] = []
    start = int(old_start)
    end = int(old_end)
    bs = int(block_size)
    if start < 0 or end <= start:
        return spans
    cur = start
    while cur < end:
        logical_block = int(cur // bs)
        if logical_block < 0 or logical_block >= len(logical_to_physical):
            raise IndexError(
                "logical_to_physical index out of range: "
                f"logical_block={logical_block}, len={len(logical_to_physical)}")
        block_start = logical_block * bs
        block_end = block_start + bs
        page_offset = cur - block_start
        take = min(end, block_end) - cur
        if take <= 0:
            raise RuntimeError("build_page_spans computed non-positive token_len")
        physical_block_id = int(logical_to_physical[logical_block])
        chunk_token_offset = cur - start
        is_full_page = (page_offset == 0 and take == bs)
        spans.append(
            PageSpan(
                physical_block_id=physical_block_id,
                page_offset=int(page_offset),
                token_len=int(take),
                old_abs_start=int(cur),
                old_abs_end=int(cur + take),
                chunk_token_offset=int(chunk_token_offset),
                is_full_page=bool(is_full_page),
                logical_block_id=int(logical_block),
            ))
        cur += take
    return spans


def build_record_entries(
    *,
    index: Optional[ChunkPageIndex],
    source_id: str,
    prompt_token_ids: Sequence[int],
    chunk_infos: Sequence[dict[str, int]],
    logical_to_physical: Sequence[int],
    block_size: int,
    hash_algo: str,
    model_id: Optional[str],
    dtype: Optional[str],
    kv_layout: Optional[str],
    no_replace_existing_hash: bool = False,
    max_entries_per_hash: int = 8,
) -> tuple[list[ChunkPageEntry], dict[str, int]]:
    normalized = _normalize_chunk_infos(chunk_infos, prompt_len=len(prompt_token_ids))
    entries: list[ChunkPageEntry] = []
    indexed_chunks = 0
    indexed_spans = 0
    indexed_full_page_spans = 0
    indexed_partial_page_spans = 0
    indexed_full_page_tokens = 0
    indexed_partial_tokens = 0
    register_attempts = 0
    skipped_duplicate_hash = 0
    skipped_bucket_full = 0
    for info in normalized:
        start = int(info["start"])
        end = int(info["end"])
        tokens = _slice_chunk_tokens(prompt_token_ids, start=start, end=end)
        chunk_hash = hash_chunk_tokens(
            tokens,
            block_size=block_size,
            model_id=model_id,
            dtype=dtype,
            kv_layout=kv_layout,
            hash_algo=hash_algo,
        )
        token_checksum = compute_token_checksum(tokens)
        register_attempts += 1
        if (no_replace_existing_hash and index is not None
                and int(max_entries_per_hash) > 0):
            valid_bucket_size = index.count_valid_entries_for_hash(
                chunk_hash=int(chunk_hash),
                token_count=int(end - start),
                token_checksum=int(token_checksum),
                model_id=model_id,
                dtype=dtype,
                kv_layout=kv_layout,
                block_size=int(block_size),
            )
            if valid_bucket_size >= int(max_entries_per_hash):
                skipped_duplicate_hash += 1
                skipped_bucket_full += 1
                index.note_register_skipped_duplicate_hash()
                continue
        spans = build_page_spans(
            old_start=start,
            old_end=end,
            block_size=block_size,
            logical_to_physical=logical_to_physical,
        )
        entry_id = (
            f"{source_id}:chunk{int(info['orig_idx'])}:{start}-{end}:"
            f"{chunk_hash & 0xFFFFFFFFFFFFFFFF:016x}"
        )
        entries.append(
            ChunkPageEntry(
                entry_id=entry_id,
                source_id=str(source_id),
                chunk_token_hash=int(chunk_hash),
                token_count=int(end - start),
                old_prompt_start=start,
                old_prompt_end=end,
                block_size=int(block_size),
                page_spans=spans,
                model_id=model_id,
                dtype=dtype,
                kv_layout=kv_layout,
                token_checksum=token_checksum,
            ))
        indexed_chunks += 1
        indexed_spans += len(spans)
        for span in spans:
            if span.is_full_page:
                indexed_full_page_spans += 1
                indexed_full_page_tokens += int(span.token_len)
            else:
                indexed_partial_page_spans += 1
                indexed_partial_tokens += int(span.token_len)
    stats = {
        "indexed_chunks": int(indexed_chunks),
        "indexed_page_spans": int(indexed_spans),
        "indexed_full_page_spans": int(indexed_full_page_spans),
        "indexed_partial_page_spans": int(indexed_partial_page_spans),
        "indexed_full_page_tokens": int(indexed_full_page_tokens),
        "indexed_partial_tokens": int(indexed_partial_tokens),
        "register_attempts": int(register_attempts),
        "register_skipped_duplicate_hash": int(skipped_duplicate_hash),
        "register_skipped_bucket_full": int(skipped_bucket_full),
    }
    return entries, stats


def build_page_reuse_plan(
    *,
    index: ChunkPageIndex,
    source_id: str,
    global_lookup: bool,
    prompt_token_ids: Sequence[int],
    chunk_infos: Sequence[dict[str, int]],
    block_size: int,
    hash_algo: str,
    model_id: Optional[str],
    dtype: Optional[str],
    kv_layout: Optional[str],
    touch_on_hit: bool = True,
) -> PageReusePlan:
    normalized = _normalize_chunk_infos(chunk_infos, prompt_len=len(prompt_token_ids))
    plan = PageReusePlan()
    plan.indexed_chunks = int(len(normalized))
    lookup_profile: dict[str, float] = {
        "lookup_hash_ms": 0.0,
        "lookup_candidate_scan_ms": 0.0,
        "lookup_block_validity_check_ms": 0.0,
        "lookup_invalid_filter_ms": 0.0,
        "lookup_touch_ms": 0.0,
        "lookup_materialize_ms": 0.0,
        "lookup_total_ms": 0.0,
        "lookup_bucket_size_total": 0.0,
        "lookup_candidates_scanned": 0.0,
        "lookup_candidates_valid": 0.0,
        "lookup_candidates_invalid": 0.0,
        "metadata_only_candidate_count": 0.0,
        "metadata_only_filtered_count": 0.0,
        "metadata_only_false_hit_count": 0.0,
    }
    t_lookup_total0 = time.perf_counter()
    for info in normalized:
        start = int(info["start"])
        end = int(info["end"])
        chunk_key = (int(info.get("orig_idx", -1)), start, end)
        chunk_tokens = _slice_chunk_tokens(
            prompt_token_ids, start=start, end=end)
        t_hash0 = time.perf_counter()
        chunk_checksum = compute_token_checksum(chunk_tokens)
        chunk_hash = hash_chunk_tokens(
            chunk_tokens,
            block_size=block_size,
            model_id=model_id,
            dtype=dtype,
            kv_layout=kv_layout,
            hash_algo=hash_algo,
        )
        lookup_profile["lookup_hash_ms"] += float(
            (time.perf_counter() - t_hash0) * 1000.0)
        lookup_diag: dict[str, float] = {}
        candidates = index.lookup(
            chunk_hash=int(chunk_hash),
            token_count=int(end - start),
            token_checksum=int(chunk_checksum),
            source_id=(None if bool(global_lookup) else str(source_id)),
            model_id=model_id,
            dtype=dtype,
            kv_layout=kv_layout,
            block_size=int(block_size),
            touch=bool(touch_on_hit),
            diag=lookup_diag,
        )
        lookup_profile["lookup_candidate_scan_ms"] += float(
            lookup_diag.get("candidate_scan_ms", 0.0))
        lookup_profile["lookup_block_validity_check_ms"] += float(
            lookup_diag.get("block_validity_check_ms", 0.0))
        lookup_profile["lookup_invalid_filter_ms"] += float(
            lookup_diag.get("invalid_filter_ms", 0.0))
        lookup_profile["lookup_touch_ms"] += float(
            lookup_diag.get("touch_ms", 0.0))
        lookup_profile["lookup_bucket_size_total"] += float(
            lookup_diag.get("bucket_size", 0.0))
        lookup_profile["lookup_candidates_scanned"] += float(
            lookup_diag.get("candidates_scanned", 0.0))
        lookup_profile["lookup_candidates_valid"] += float(
            lookup_diag.get("candidates_kept", 0.0))
        lookup_profile["lookup_candidates_invalid"] += float(
            lookup_diag.get("candidates_invalid", 0.0))
        lookup_profile["metadata_only_candidate_count"] += float(
            lookup_diag.get("metadata_only_candidate_count", 0.0))
        lookup_profile["metadata_only_filtered_count"] += float(
            lookup_diag.get("metadata_only_filtered_count", 0.0))
        lookup_profile["metadata_only_false_hit_count"] += float(
            lookup_diag.get("metadata_only_false_hit_count", 0.0))
        t_materialize0 = time.perf_counter()
        plan.candidate_count += int(len(candidates))
        best = _pick_best_candidate(candidates, start=start, end=end)
        if best is None:
            plan.miss_chunks += 1
            plan.miss_chunk_keys.append(chunk_key)
            plan.cpu_fallback_spans.append((start, end))
            lookup_profile["lookup_materialize_ms"] += float(
                (time.perf_counter() - t_materialize0) * 1000.0)
            continue
        plan.hit_chunks += 1
        plan.hit_chunk_keys.append(chunk_key)
        if str(best.source_id) == str(source_id):
            plan.same_source_hit_count += 1
        else:
            plan.global_cross_source_hit_count += 1
        for span in best.page_spans:
            new_abs_start = start + int(span.chunk_token_offset)
            new_abs_end = new_abs_start + int(span.token_len)
            gather = GatherSpanPlan(
                source_id=str(best.source_id),
                physical_block_id=int(span.physical_block_id),
                page_offset=int(span.page_offset),
                token_len=int(span.token_len),
                old_abs_start=int(span.old_abs_start),
                old_abs_end=int(span.old_abs_end),
                new_abs_start=int(new_abs_start),
                new_abs_end=int(new_abs_end),
                is_full_page=bool(span.is_full_page),
            )
            plan.gather_spans.append(gather)
            if span.is_full_page:
                plan.gpu_gather_full_tokens += int(span.token_len)
            else:
                plan.gpu_gather_partial_tokens += int(span.token_len)
        lookup_profile["lookup_materialize_ms"] += float(
            (time.perf_counter() - t_materialize0) * 1000.0)
    lookup_profile["lookup_total_ms"] = float(
        (time.perf_counter() - t_lookup_total0) * 1000.0)
    plan.lookup_profile = lookup_profile
    return plan


def build_all_miss_page_reuse_plan(
    *,
    prompt_token_ids: Sequence[int],
    chunk_infos: Sequence[dict[str, int]],
) -> PageReusePlan:
    """Build a cold-start plan when no compatible global source metadata exists."""
    normalized = _normalize_chunk_infos(chunk_infos, prompt_len=len(prompt_token_ids))
    plan = PageReusePlan()
    plan.lookup_profile = {
        "lookup_hash_ms": 0.0,
        "lookup_candidate_scan_ms": 0.0,
        "lookup_block_validity_check_ms": 0.0,
        "lookup_invalid_filter_ms": 0.0,
        "lookup_touch_ms": 0.0,
        "lookup_materialize_ms": 0.0,
        "lookup_total_ms": 0.0,
        "lookup_bucket_size_total": 0.0,
        "lookup_candidates_scanned": 0.0,
        "lookup_candidates_valid": 0.0,
        "lookup_candidates_invalid": 0.0,
        "metadata_only_candidate_count": 0.0,
        "metadata_only_filtered_count": 0.0,
        "metadata_only_false_hit_count": 0.0,
    }
    plan.indexed_chunks = int(len(normalized))
    for info in normalized:
        start = int(info["start"])
        end = int(info["end"])
        plan.miss_chunks += 1
        plan.miss_chunk_keys.append((int(info.get("orig_idx", -1)), start, end))
        plan.cpu_fallback_spans.append((start, end))
    return plan


def _pick_best_candidate(
    candidates: Sequence[ChunkPageEntry],
    *,
    start: int,
    end: int,
) -> Optional[ChunkPageEntry]:
    if not candidates:
        return None
    # Prefer exact same absolute span; tie-break by most recently touched.
    best = None
    best_score = None
    for cand in candidates:
        same_span = (int(cand.old_prompt_start) == int(start)
                     and int(cand.old_prompt_end) == int(end))
        score = (
            1 if same_span else 0,
            float(cand.last_access_ts),
            int(cand.access_count),
        )
        if best is None or score > best_score:
            best = cand
            best_score = score
    return best


def _normalize_old_slots_build_mode(mode: str) -> str:
    norm = str(mode).strip().lower()
    if norm not in ("auto", "fast", "slow"):
        raise ValueError(
            "Invalid old-slots build mode: "
            f"{mode}. Expected one of auto/fast/slow.")
    return norm


def _pack_old_slots_build_output(
    *,
    old_slots: torch.Tensor,
    dst_idx: torch.Tensor,
    old_positions: torch.Tensor,
    new_positions: torch.Tensor,
    target_reuse_tokens: int,
    gathered_target_tokens: int,
    cpu_fallback_tokens: int,
    skipped_out_of_range: int,
    skipped_missing_dst: int,
    skipped_duplicate: int,
    noncontiguous_span_count: int,
    old_slots_total_ms: float,
    old_slots_map_build_ms: float,
    old_slots_span_iter_ms: float,
    old_slots_tensor_materialize_ms: float,
    old_slots_position_build_ms: float,
    old_slots_check_ms: float,
    old_slots_fastpath_used: int,
    old_slots_fastpath_reason: str,
    old_slots_span_count: int,
) -> dict[str, torch.Tensor | int | float | str]:
    old_slots_out = old_slots.to(dtype=torch.long, device="cpu").contiguous()
    dst_idx_out = dst_idx.to(dtype=torch.long, device="cpu").contiguous()
    old_pos_out = old_positions.to(dtype=torch.long, device="cpu").contiguous()
    new_pos_out = new_positions.to(dtype=torch.long, device="cpu").contiguous()
    return {
        "old_slots": old_slots_out,
        "dst_idx": dst_idx_out,
        "old_positions": old_pos_out,
        "new_positions": new_pos_out,
        "target_reuse_tokens": int(target_reuse_tokens),
        "gathered_target_tokens": int(gathered_target_tokens),
        "cpu_fallback_tokens": int(max(0, cpu_fallback_tokens)),
        "skipped_out_of_range": int(skipped_out_of_range),
        "skipped_missing_dst": int(skipped_missing_dst),
        "skipped_duplicate": int(skipped_duplicate),
        "noncontiguous_span_count": int(noncontiguous_span_count),
        "old_slots_build_ms": float(old_slots_total_ms),
        "old_slots_total_ms": float(old_slots_total_ms),
        "old_slots_map_build_ms": float(old_slots_map_build_ms),
        "old_slots_span_iter_ms": float(old_slots_span_iter_ms),
        "old_slots_tensor_materialize_ms": float(
            old_slots_tensor_materialize_ms),
        "old_slots_position_build_ms": float(old_slots_position_build_ms),
        "old_slots_check_ms": float(old_slots_check_ms),
        "old_slots_fastpath_used": int(old_slots_fastpath_used),
        "old_slots_fastpath_reason": str(old_slots_fastpath_reason),
        "old_slots_span_count": int(old_slots_span_count),
        "old_slots_token_count": int(old_slots_out.numel()),
    }


def _stable_dedup_by_first_dst(
    *,
    dst_idx: torch.Tensor,
    reuse_count: int,
) -> tuple[torch.Tensor, int]:
    num_candidates = int(dst_idx.numel())
    order = torch.arange(num_candidates, dtype=torch.long)
    first_order = torch.full((reuse_count, ), num_candidates, dtype=torch.long)
    first_order.scatter_reduce_(
        0,
        dst_idx,
        order,
        reduce="amin",
        include_self=True,
    )
    keep_mask = (order == first_order.index_select(0, dst_idx))
    skipped_duplicate = int((~keep_mask).sum().item())
    return keep_mask, skipped_duplicate


def _first_tensor_mismatch(
    left: torch.Tensor,
    right: torch.Tensor,
) -> tuple[int, int, int]:
    left_flat = left.view(-1)
    right_flat = right.view(-1)
    neq = torch.nonzero(left_flat != right_flat, as_tuple=False)
    if int(neq.numel()) <= 0:
        return -1, 0, 0
    idx = int(neq[0].item())
    lv = int(left_flat[idx].item())
    rv = int(right_flat[idx].item())
    return idx, lv, rv


def _assert_old_slots_build_match(
    *,
    fast: dict[str, torch.Tensor | int | float | str],
    slow: dict[str, torch.Tensor | int | float | str],
    gather_spans: Sequence[GatherSpanPlan],
) -> None:
    tensor_keys = ("old_slots", "dst_idx", "old_positions", "new_positions")
    scalar_keys = (
        "target_reuse_tokens",
        "gathered_target_tokens",
        "cpu_fallback_tokens",
        "skipped_out_of_range",
        "skipped_missing_dst",
        "skipped_duplicate",
        "noncontiguous_span_count",
    )
    for key in tensor_keys:
        f = fast.get(key)
        s = slow.get(key)
        if not isinstance(f, torch.Tensor) or not isinstance(s, torch.Tensor):
            raise AssertionError(f"old-slots debug compare invalid type for {key}")
        if f.shape != s.shape:
            first_span = gather_spans[0] if gather_spans else None
            raise AssertionError(
                "old-slots debug compare shape mismatch: "
                f"field={key}, fast_shape={tuple(f.shape)}, "
                f"slow_shape={tuple(s.shape)}, span_count={len(gather_spans)}, "
                f"first_span={first_span}")
        if not torch.equal(f, s):
            mismatch_idx, fast_val, slow_val = _first_tensor_mismatch(f, s)
            first_span = gather_spans[0] if gather_spans else None
            raise AssertionError(
                "old-slots debug compare value mismatch: "
                f"field={key}, mismatch_idx={mismatch_idx}, "
                f"fast={fast_val}, slow={slow_val}, "
                f"span_count={len(gather_spans)}, first_span={first_span}")
    for key in scalar_keys:
        fv = int(fast.get(key, -1))
        sv = int(slow.get(key, -1))
        if fv != sv:
            first_span = gather_spans[0] if gather_spans else None
            raise AssertionError(
                "old-slots debug compare scalar mismatch: "
                f"field={key}, fast={fv}, slow={sv}, "
                f"span_count={len(gather_spans)}, first_span={first_span}")


def build_old_slots_and_dst_idx_slow(
    *,
    plan: PageReusePlan,
    reuse_positions_cpu: torch.Tensor,
    prompt_len: int,
    block_size: int,
    debug_build_positions: bool = False,
    debug_check_duplicates: bool = False,
) -> dict[str, torch.Tensor | int | float | str]:
    t_total0 = time.perf_counter()
    t_map0 = time.perf_counter()
    reuse_pos = reuse_positions_cpu.to(dtype=torch.long, device="cpu").contiguous()
    reuse_count = int(reuse_pos.numel())
    reuse_pos_to_idx = torch.full((int(prompt_len), ), -1, dtype=torch.long)
    if reuse_count > 0:
        reuse_pos_to_idx[reuse_pos] = torch.arange(reuse_count, dtype=torch.long)
    t_map1 = time.perf_counter()
    old_slots_map_build_ms = float((t_map1 - t_map0) * 1000.0)

    gather_spans = [span for span in plan.gather_spans if int(span.token_len) > 0]
    target_gather_tokens = int(sum(int(span.token_len) for span in gather_spans))
    target_fallback_tokens = int(
        sum(max(0, int(end) - int(start)) for start, end in plan.cpu_fallback_spans))
    target_reuse_tokens = int(max(0, target_gather_tokens + target_fallback_tokens))

    skipped_out_of_range = 0
    skipped_missing_dst = 0
    skipped_duplicate = 0
    noncontiguous_span_count = 0
    old_slots_position_build_ms = 0.0
    old_slots_check_ms = 0.0
    old_slots_span_iter_ms = 0.0
    old_slots_tensor_materialize_ms = 0.0

    if target_gather_tokens <= 0:
        t_total1 = time.perf_counter()
        old_slots_total_ms = float((t_total1 - t_total0) * 1000.0)
        return _pack_old_slots_build_output(
            old_slots=torch.empty(0, dtype=torch.long),
            dst_idx=torch.empty(0, dtype=torch.long),
            old_positions=torch.empty(0, dtype=torch.long),
            new_positions=torch.empty(0, dtype=torch.long),
            target_reuse_tokens=int(target_reuse_tokens),
            gathered_target_tokens=0,
            cpu_fallback_tokens=int(max(0, target_reuse_tokens)),
            skipped_out_of_range=0,
            skipped_missing_dst=0,
            skipped_duplicate=0,
            noncontiguous_span_count=0,
            old_slots_total_ms=old_slots_total_ms,
            old_slots_map_build_ms=old_slots_map_build_ms,
            old_slots_span_iter_ms=0.0,
            old_slots_tensor_materialize_ms=0.0,
            old_slots_position_build_ms=0.0,
            old_slots_check_ms=0.0,
            old_slots_fastpath_used=0,
            old_slots_fastpath_reason="slow_path_empty",
            old_slots_span_count=0,
        )

    bs = int(block_size)
    write_ptr = 0
    arange_cache: dict[int, torch.Tensor] = {}

    def _get_offsets(token_len: int) -> torch.Tensor:
        cached = arange_cache.get(token_len)
        if cached is None:
            cached = torch.arange(token_len, dtype=torch.long)
            arange_cache[token_len] = cached
        return cached

    t_mat0 = time.perf_counter()
    old_slots = torch.empty(target_gather_tokens, dtype=torch.long)
    dst_idx = torch.empty(target_gather_tokens, dtype=torch.long)
    if debug_build_positions:
        old_positions = torch.empty(target_gather_tokens, dtype=torch.long)
        new_positions = torch.empty(target_gather_tokens, dtype=torch.long)
    else:
        old_positions = torch.empty(0, dtype=torch.long)
        new_positions = torch.empty(0, dtype=torch.long)
    t_mat1 = time.perf_counter()
    old_slots_tensor_materialize_ms += float((t_mat1 - t_mat0) * 1000.0)

    t_span0 = time.perf_counter()
    for span in gather_spans:
        token_len = int(span.token_len)
        offsets = _get_offsets(token_len)
        base_slot = int(span.physical_block_id) * bs + int(span.page_offset)
        old_start = int(span.old_abs_start)
        new_start = int(span.new_abs_start)
        new_end = new_start + token_len

        contiguous = False
        start_in_range = (0 <= new_start < int(prompt_len))
        end_in_range = (0 <= (new_end - 1) < int(prompt_len))
        if start_in_range and end_in_range:
            start_idx = int(reuse_pos_to_idx[new_start].item())
            end_idx = int(reuse_pos_to_idx[new_end - 1].item())
            if (start_idx >= 0 and end_idx >= 0
                    and end_idx == (start_idx + token_len - 1)):
                contiguous = True
                sl = slice(write_ptr, write_ptr + token_len)
                old_slots[sl] = base_slot + offsets
                dst_idx[sl] = start_idx + offsets
                if debug_build_positions:
                    t_pos0 = time.perf_counter()
                    old_positions[sl] = old_start + offsets
                    new_positions[sl] = new_start + offsets
                    t_pos1 = time.perf_counter()
                    old_slots_position_build_ms += float((t_pos1 - t_pos0) *
                                                         1000.0)
                write_ptr += token_len

        if contiguous:
            continue

        noncontiguous_span_count += 1
        span_new_positions = new_start + offsets
        in_range = ((span_new_positions >= 0)
                    & (span_new_positions < int(prompt_len)))
        skipped_out_of_range += int((~in_range).sum().item())
        if not bool(in_range.any().item()):
            continue

        valid_offsets = offsets[in_range]
        valid_new_positions = span_new_positions[in_range]
        span_dst_idx = reuse_pos_to_idx[valid_new_positions]
        valid_dst = ((span_dst_idx >= 0) & (span_dst_idx < reuse_count))
        skipped_missing_dst += int((~valid_dst).sum().item())
        if not bool(valid_dst.any().item()):
            continue

        valid_offsets = valid_offsets[valid_dst]
        valid_new_positions = valid_new_positions[valid_dst]
        span_dst_idx = span_dst_idx[valid_dst]

        n = int(span_dst_idx.numel())
        sl = slice(write_ptr, write_ptr + n)
        old_slots[sl] = base_slot + valid_offsets
        dst_idx[sl] = span_dst_idx
        if debug_build_positions:
            t_pos0 = time.perf_counter()
            old_positions[sl] = old_start + valid_offsets
            new_positions[sl] = valid_new_positions
            t_pos1 = time.perf_counter()
            old_slots_position_build_ms += float((t_pos1 - t_pos0) * 1000.0)
        write_ptr += n
    t_span1 = time.perf_counter()
    old_slots_span_iter_ms = float((t_span1 - t_span0) * 1000.0)

    t_mat0 = time.perf_counter()
    old_slots = old_slots[:write_ptr].contiguous()
    dst_idx = dst_idx[:write_ptr].contiguous()
    if debug_build_positions:
        old_positions = old_positions[:write_ptr].contiguous()
        new_positions = new_positions[:write_ptr].contiguous()
    t_mat1 = time.perf_counter()
    old_slots_tensor_materialize_ms += float((t_mat1 - t_mat0) * 1000.0)

    t_check0 = time.perf_counter()
    if debug_check_duplicates and reuse_count > 0 and int(dst_idx.numel()) > 0:
        keep_mask, skipped_duplicate = _stable_dedup_by_first_dst(
            dst_idx=dst_idx,
            reuse_count=reuse_count,
        )
        if skipped_duplicate > 0:
            old_slots = old_slots[keep_mask].contiguous()
            dst_idx = dst_idx[keep_mask].contiguous()
            if debug_build_positions:
                old_positions = old_positions[keep_mask].contiguous()
                new_positions = new_positions[keep_mask].contiguous()

    seen = torch.zeros(reuse_count, dtype=torch.bool)
    if reuse_count > 0 and int(dst_idx.numel()) > 0:
        seen[dst_idx] = True
    gathered_target_tokens = int(seen.sum().item())
    cpu_fallback_tokens = int(max(0, target_reuse_tokens - gathered_target_tokens))
    t_check1 = time.perf_counter()
    old_slots_check_ms = float((t_check1 - t_check0) * 1000.0)

    t_total1 = time.perf_counter()
    old_slots_total_ms = float((t_total1 - t_total0) * 1000.0)
    return _pack_old_slots_build_output(
        old_slots=old_slots,
        dst_idx=dst_idx,
        old_positions=old_positions,
        new_positions=new_positions,
        target_reuse_tokens=int(target_reuse_tokens),
        gathered_target_tokens=int(gathered_target_tokens),
        cpu_fallback_tokens=int(cpu_fallback_tokens),
        skipped_out_of_range=int(skipped_out_of_range),
        skipped_missing_dst=int(skipped_missing_dst),
        skipped_duplicate=int(skipped_duplicate),
        noncontiguous_span_count=int(noncontiguous_span_count),
        old_slots_total_ms=old_slots_total_ms,
        old_slots_map_build_ms=old_slots_map_build_ms,
        old_slots_span_iter_ms=old_slots_span_iter_ms,
        old_slots_tensor_materialize_ms=old_slots_tensor_materialize_ms,
        old_slots_position_build_ms=old_slots_position_build_ms,
        old_slots_check_ms=old_slots_check_ms,
        old_slots_fastpath_used=0,
        old_slots_fastpath_reason="slow_path",
        old_slots_span_count=len(gather_spans),
    )


def build_old_slots_and_dst_idx_fast(
    *,
    plan: PageReusePlan,
    reuse_positions_cpu: torch.Tensor,
    prompt_len: int,
    block_size: int,
    debug_build_positions: bool = False,
    debug_check_duplicates: bool = False,
    max_dense_pos_map_size: int = 1_000_000,
) -> dict[str, torch.Tensor | int | float | str]:
    t_total0 = time.perf_counter()
    gather_spans = [span for span in plan.gather_spans if int(span.token_len) > 0]
    span_count = len(gather_spans)
    target_gather_tokens = int(sum(int(span.token_len) for span in gather_spans))
    target_fallback_tokens = int(
        sum(max(0, int(end) - int(start)) for start, end in plan.cpu_fallback_spans))
    target_reuse_tokens = int(max(0, target_gather_tokens + target_fallback_tokens))

    skipped_out_of_range = 0
    skipped_missing_dst = 0
    skipped_duplicate = 0
    noncontiguous_span_count = 0

    old_slots_map_build_ms = 0.0
    old_slots_span_iter_ms = 0.0
    old_slots_tensor_materialize_ms = 0.0
    old_slots_position_build_ms = 0.0
    old_slots_check_ms = 0.0

    if target_gather_tokens <= 0:
        t_total1 = time.perf_counter()
        old_slots_total_ms = float((t_total1 - t_total0) * 1000.0)
        return _pack_old_slots_build_output(
            old_slots=torch.empty(0, dtype=torch.long),
            dst_idx=torch.empty(0, dtype=torch.long),
            old_positions=torch.empty(0, dtype=torch.long),
            new_positions=torch.empty(0, dtype=torch.long),
            target_reuse_tokens=int(target_reuse_tokens),
            gathered_target_tokens=0,
            cpu_fallback_tokens=int(max(0, target_reuse_tokens)),
            skipped_out_of_range=0,
            skipped_missing_dst=0,
            skipped_duplicate=0,
            noncontiguous_span_count=0,
            old_slots_total_ms=old_slots_total_ms,
            old_slots_map_build_ms=0.0,
            old_slots_span_iter_ms=0.0,
            old_slots_tensor_materialize_ms=0.0,
            old_slots_position_build_ms=0.0,
            old_slots_check_ms=0.0,
            old_slots_fastpath_used=1,
            old_slots_fastpath_reason="ok_empty",
            old_slots_span_count=0,
        )

    t_map0 = time.perf_counter()
    reuse_pos = reuse_positions_cpu.to(dtype=torch.long, device="cpu").contiguous()
    reuse_count = int(reuse_pos.numel())
    max_reuse_pos = int(reuse_pos.max().item()) if reuse_count > 0 else -1
    max_needed_new_pos = max(
        int(span.new_abs_end) - 1 for span in gather_spans)
    map_size = max(int(prompt_len), max_reuse_pos + 1, max_needed_new_pos + 1)
    if map_size > int(max_dense_pos_map_size):
        raise RuntimeError(
            "dense_map_too_large:"
            f"map_size={map_size},limit={int(max_dense_pos_map_size)}")
    reuse_pos_to_idx = torch.full((int(map_size), ), -1, dtype=torch.long)
    if reuse_count > 0:
        reuse_pos_to_idx[reuse_pos] = torch.arange(reuse_count, dtype=torch.long)
    t_map1 = time.perf_counter()
    old_slots_map_build_ms = float((t_map1 - t_map0) * 1000.0)

    t_span0 = time.perf_counter()
    bs = int(block_size)
    token_lens = torch.tensor([int(span.token_len) for span in gather_spans],
                              dtype=torch.long)
    new_starts = torch.tensor([int(span.new_abs_start) for span in gather_spans],
                              dtype=torch.long)
    old_starts = torch.tensor([int(span.old_abs_start) for span in gather_spans],
                              dtype=torch.long)
    base_slots = torch.tensor([
        int(span.physical_block_id) * bs + int(span.page_offset)
        for span in gather_spans
    ],
                              dtype=torch.long)
    new_ends = new_starts + token_lens
    span_start_in_range = ((new_starts >= 0) & (new_starts < int(prompt_len)))
    span_end_pos = new_ends - 1
    span_end_in_range = ((span_end_pos >= 0) & (span_end_pos < int(prompt_len)))
    span_in_range = span_start_in_range & span_end_in_range
    span_start_idx = torch.full((span_count, ), -1, dtype=torch.long)
    span_end_idx = torch.full((span_count, ), -1, dtype=torch.long)
    if bool(span_in_range.any().item()):
        span_start_idx[span_in_range] = reuse_pos_to_idx.index_select(
            0, new_starts[span_in_range])
        span_end_idx[span_in_range] = reuse_pos_to_idx.index_select(
            0, span_end_pos[span_in_range])
    span_contiguous = (
        span_in_range
        & (span_start_idx >= 0)
        & (span_end_idx >= 0)
        & (span_end_idx == (span_start_idx + token_lens - 1)))
    noncontiguous_span_count = int(span_count - int(span_contiguous.sum().item()))
    t_span1 = time.perf_counter()
    old_slots_span_iter_ms = float((t_span1 - t_span0) * 1000.0)

    t_mat0 = time.perf_counter()
    span_prefix = torch.cumsum(token_lens, dim=0) - token_lens
    span_ids = torch.repeat_interleave(torch.arange(span_count,
                                                    dtype=torch.long),
                                       token_lens)
    total_tokens = int(span_ids.numel())
    token_offsets = (torch.arange(total_tokens, dtype=torch.long)
                     - span_prefix.index_select(0, span_ids))
    old_slots_all = base_slots.index_select(0, span_ids) + token_offsets
    new_positions_all = new_starts.index_select(0, span_ids) + token_offsets
    t_mat1 = time.perf_counter()
    old_slots_tensor_materialize_ms = float((t_mat1 - t_mat0) * 1000.0)

    t_pos0 = time.perf_counter()
    if debug_build_positions:
        old_positions_all = old_starts.index_select(0, span_ids) + token_offsets
    else:
        old_positions_all = torch.empty(0, dtype=torch.long)
    t_pos1 = time.perf_counter()
    old_slots_position_build_ms = float((t_pos1 - t_pos0) * 1000.0)

    in_range = ((new_positions_all >= 0) & (new_positions_all < int(prompt_len)))
    skipped_out_of_range = int((~in_range).sum().item())
    dst_idx_all = torch.full((total_tokens, ), -1, dtype=torch.long)
    if bool(in_range.any().item()):
        dst_idx_all[in_range] = reuse_pos_to_idx.index_select(
            0, new_positions_all[in_range])
    valid_dst = ((dst_idx_all >= 0) & (dst_idx_all < reuse_count))
    skipped_missing_dst = int((in_range & (~valid_dst)).sum().item())

    t_mat0 = time.perf_counter()
    old_slots = old_slots_all[valid_dst].contiguous()
    dst_idx = dst_idx_all[valid_dst].contiguous()
    if debug_build_positions:
        old_positions = old_positions_all[valid_dst].contiguous()
        new_positions = new_positions_all[valid_dst].contiguous()
    else:
        old_positions = torch.empty(0, dtype=torch.long)
        new_positions = torch.empty(0, dtype=torch.long)
    t_mat1 = time.perf_counter()
    old_slots_tensor_materialize_ms += float((t_mat1 - t_mat0) * 1000.0)

    t_check0 = time.perf_counter()
    if debug_check_duplicates and reuse_count > 0 and int(dst_idx.numel()) > 0:
        keep_mask, skipped_duplicate = _stable_dedup_by_first_dst(
            dst_idx=dst_idx,
            reuse_count=reuse_count,
        )
        if skipped_duplicate > 0:
            old_slots = old_slots[keep_mask].contiguous()
            dst_idx = dst_idx[keep_mask].contiguous()
            if debug_build_positions:
                old_positions = old_positions[keep_mask].contiguous()
                new_positions = new_positions[keep_mask].contiguous()

    seen = torch.zeros(reuse_count, dtype=torch.bool)
    if reuse_count > 0 and int(dst_idx.numel()) > 0:
        seen[dst_idx] = True
    gathered_target_tokens = int(seen.sum().item())
    cpu_fallback_tokens = int(max(0, target_reuse_tokens - gathered_target_tokens))
    t_check1 = time.perf_counter()
    old_slots_check_ms = float((t_check1 - t_check0) * 1000.0)

    t_total1 = time.perf_counter()
    old_slots_total_ms = float((t_total1 - t_total0) * 1000.0)
    return _pack_old_slots_build_output(
        old_slots=old_slots,
        dst_idx=dst_idx,
        old_positions=old_positions,
        new_positions=new_positions,
        target_reuse_tokens=int(target_reuse_tokens),
        gathered_target_tokens=int(gathered_target_tokens),
        cpu_fallback_tokens=int(cpu_fallback_tokens),
        skipped_out_of_range=int(skipped_out_of_range),
        skipped_missing_dst=int(skipped_missing_dst),
        skipped_duplicate=int(skipped_duplicate),
        noncontiguous_span_count=int(noncontiguous_span_count),
        old_slots_total_ms=old_slots_total_ms,
        old_slots_map_build_ms=old_slots_map_build_ms,
        old_slots_span_iter_ms=old_slots_span_iter_ms,
        old_slots_tensor_materialize_ms=old_slots_tensor_materialize_ms,
        old_slots_position_build_ms=old_slots_position_build_ms,
        old_slots_check_ms=old_slots_check_ms,
        old_slots_fastpath_used=1,
        old_slots_fastpath_reason="ok",
        old_slots_span_count=span_count,
    )


def build_old_slots_and_dst_idx(
    *,
    plan: PageReusePlan,
    reuse_positions_cpu: torch.Tensor,
    prompt_len: int,
    block_size: int,
    debug_build_positions: bool = False,
    debug_check_duplicates: bool = False,
    mode: str = "auto",
    debug_compare_fast_slow: bool = False,
    max_dense_pos_map_size: int = 1_000_000,
) -> dict[str, torch.Tensor | int | float | str]:
    mode_norm = _normalize_old_slots_build_mode(mode)
    gather_spans = [span for span in plan.gather_spans if int(span.token_len) > 0]

    if mode_norm == "slow":
        slow = build_old_slots_and_dst_idx_slow(
            plan=plan,
            reuse_positions_cpu=reuse_positions_cpu,
            prompt_len=prompt_len,
            block_size=block_size,
            debug_build_positions=debug_build_positions,
            debug_check_duplicates=debug_check_duplicates,
        )
        slow["old_slots_fastpath_used"] = 0
        slow["old_slots_fastpath_reason"] = "forced_slow_mode"
        return slow

    try:
        fast = build_old_slots_and_dst_idx_fast(
            plan=plan,
            reuse_positions_cpu=reuse_positions_cpu,
            prompt_len=prompt_len,
            block_size=block_size,
            debug_build_positions=debug_build_positions,
            debug_check_duplicates=debug_check_duplicates,
            max_dense_pos_map_size=max_dense_pos_map_size,
        )
    except Exception as exc:
        if mode_norm == "fast":
            raise
        if debug_compare_fast_slow:
            raise RuntimeError(
                "old-slots fast path failed in debug compare mode: "
                f"{type(exc).__name__}: {exc}") from exc
        slow = build_old_slots_and_dst_idx_slow(
            plan=plan,
            reuse_positions_cpu=reuse_positions_cpu,
            prompt_len=prompt_len,
            block_size=block_size,
            debug_build_positions=debug_build_positions,
            debug_check_duplicates=debug_check_duplicates,
        )
        slow["old_slots_fastpath_used"] = 0
        slow["old_slots_fastpath_reason"] = (
            f"fallback:{type(exc).__name__}:{exc}")
        return slow

    if debug_compare_fast_slow:
        slow_ref = build_old_slots_and_dst_idx_slow(
            plan=plan,
            reuse_positions_cpu=reuse_positions_cpu,
            prompt_len=prompt_len,
            block_size=block_size,
            debug_build_positions=debug_build_positions,
            debug_check_duplicates=debug_check_duplicates,
        )
        _assert_old_slots_build_match(
            fast=fast,
            slow=slow_ref,
            gather_spans=gather_spans,
        )
    return fast


def build_span_reuse_plan(
    *,
    plan: PageReusePlan,
    reuse_positions_cpu: torch.Tensor,
    prompt_len: int,
    block_size: int,
    debug_check_duplicates: bool = False,
) -> dict[str, torch.Tensor | int | float | bool | str]:
    """Build span-level gather/copy metadata for chunk-page execution.

    This keeps metadata in span form instead of expanding to token-level
    old_slots/dst_idx arrays.
    """
    t0 = time.perf_counter()

    def _stable_first_occurrence_mask(values: torch.Tensor) -> torch.Tensor:
        n = int(values.numel())
        if n <= 1:
            return torch.ones(n, dtype=torch.bool)
        perm = torch.argsort(values, stable=True)
        sorted_vals = values.index_select(0, perm)
        first_sorted = torch.ones(n, dtype=torch.bool)
        first_sorted[1:] = sorted_vals[1:] != sorted_vals[:-1]
        keep_pos = perm[first_sorted]
        keep_mask = torch.zeros(n, dtype=torch.bool)
        keep_mask[keep_pos] = True
        return keep_mask

    span_plan_make_pos_map_ms = 0.0
    span_plan_iter_gpu_spans_ms = 0.0
    span_plan_merge_gpu_spans_ms = 0.0
    span_plan_build_cpu_spans_ms = 0.0
    span_plan_tensor_materialize_ms = 0.0
    span_plan_debug_check_ms = 0.0
    span_plan_other_ms = 0.0
    gpu_span_count_before_merge = 0
    gpu_span_count_after_merge = 0
    cpu_span_count_before_merge = 0
    cpu_span_count_after_merge = 0

    t_map0 = time.perf_counter()
    reuse_pos = reuse_positions_cpu.to(dtype=torch.long, device="cpu").contiguous()
    reuse_count = int(reuse_pos.numel())
    reuse_pos_to_idx = torch.full((int(prompt_len), ), -1, dtype=torch.long)
    if reuse_count > 0:
        reuse_pos_to_idx[reuse_pos] = torch.arange(reuse_count, dtype=torch.long)
    t_map1 = time.perf_counter()
    span_plan_make_pos_map_ms = float((t_map1 - t_map0) * 1000.0)

    gather_spans = [span for span in plan.gather_spans if int(span.token_len) > 0]
    target_gather_tokens = int(sum(int(span.token_len) for span in gather_spans))
    target_fallback_tokens = int(
        sum(max(0, int(end) - int(start)) for start, end in plan.cpu_fallback_spans))
    target_reuse_tokens = int(max(0, target_gather_tokens + target_fallback_tokens))

    skipped_out_of_range = 0
    skipped_missing_dst = 0
    skipped_duplicate = 0
    noncontiguous_span_count = 0

    bs = int(block_size)
    if bs <= 0:
        t1 = time.perf_counter()
        span_plan_total_ms = float((t1 - t0) * 1000.0)
        span_plan_known_ms = (
            span_plan_make_pos_map_ms
            + span_plan_iter_gpu_spans_ms
            + span_plan_merge_gpu_spans_ms
            + span_plan_build_cpu_spans_ms
            + span_plan_tensor_materialize_ms
            + span_plan_debug_check_ms
        )
        span_plan_other_ms = float(max(0.0, span_plan_total_ms - span_plan_known_ms))
        return {
            "gpu_span_block_ids": torch.empty(0, dtype=torch.long),
            "gpu_span_page_offsets": torch.empty(0, dtype=torch.long),
            "gpu_span_dst_starts": torch.empty(0, dtype=torch.long),
            "gpu_span_lengths": torch.empty(0, dtype=torch.long),
            "gpu_span_old_abs_starts": torch.empty(0, dtype=torch.long),
            "gpu_span_new_abs_starts": torch.empty(0, dtype=torch.long),
            "cpu_span_src_starts": torch.empty(0, dtype=torch.long),
            "cpu_span_dst_starts": torch.empty(0, dtype=torch.long),
            "cpu_span_lengths": torch.empty(0, dtype=torch.long),
            "gpu_span_count": 0,
            "gpu_span_tokens": 0,
            "cpu_span_count": 0,
            "cpu_span_tokens": 0,
            "target_reuse_tokens": int(target_reuse_tokens),
            "covered_reuse_tokens": 0,
            "skipped_out_of_range": 0,
            "skipped_missing_dst": 0,
            "skipped_duplicate": 0,
            "noncontiguous_span_count": 0,
            "span_plan_build_ms": float(span_plan_total_ms),
            "span_plan_total_ms": float(span_plan_total_ms),
            "span_plan_make_pos_map_ms": float(span_plan_make_pos_map_ms),
            "span_plan_iter_gpu_spans_ms": float(span_plan_iter_gpu_spans_ms),
            "span_plan_build_cpu_spans_ms": float(span_plan_build_cpu_spans_ms),
            "span_plan_merge_gpu_spans_ms": float(span_plan_merge_gpu_spans_ms),
            "span_plan_tensor_materialize_ms": float(
                span_plan_tensor_materialize_ms),
            "span_plan_debug_check_ms": float(span_plan_debug_check_ms),
            "span_plan_other_ms": float(span_plan_other_ms),
            "span_plan_gpu_span_count_before_merge":
            int(gpu_span_count_before_merge),
            "span_plan_gpu_span_count_after_merge":
            int(gpu_span_count_after_merge),
            "span_plan_cpu_span_count_before_merge":
            int(cpu_span_count_before_merge),
            "span_plan_cpu_span_count_after_merge":
            int(cpu_span_count_after_merge),
            "requires_token_fallback": True,
            "fallback_reason": "invalid_block_size",
        }

    seen_dst = torch.zeros(reuse_count, dtype=torch.bool)

    gpu_span_block_ids = torch.empty(0, dtype=torch.long)
    gpu_span_page_offsets = torch.empty(0, dtype=torch.long)
    gpu_span_dst_starts = torch.empty(0, dtype=torch.long)
    gpu_span_lengths = torch.empty(0, dtype=torch.long)
    gpu_span_old_abs_starts = torch.empty(0, dtype=torch.long)
    gpu_span_new_abs_starts = torch.empty(0, dtype=torch.long)

    # GPU gather spans: vectorized mapping + stable dedup, then conservative merge.
    if gather_spans:
        token_lens_gpu = torch.tensor(
            [int(span.token_len) for span in gather_spans], dtype=torch.long)
        if int(token_lens_gpu.numel()) > 0 and int(token_lens_gpu.sum().item()) > 0:
            new_abs_starts_gpu = torch.tensor(
                [int(span.new_abs_start) for span in gather_spans],
                dtype=torch.long)
            old_abs_starts_gpu = torch.tensor(
                [int(span.old_abs_start) for span in gather_spans],
                dtype=torch.long)
            old_slot_starts_gpu = torch.tensor([
                int(span.physical_block_id) * bs + int(span.page_offset)
                for span in gather_spans
            ],
                                               dtype=torch.long)

            t_gpu_iter0 = time.perf_counter()
            span_count_gpu = int(token_lens_gpu.numel())
            prefix_gpu = torch.cumsum(token_lens_gpu, dim=0) - token_lens_gpu
            span_ids_gpu = torch.repeat_interleave(
                torch.arange(span_count_gpu, dtype=torch.long), token_lens_gpu)
            total_gpu_tokens = int(span_ids_gpu.numel())
            token_offsets_gpu = (
                torch.arange(total_gpu_tokens, dtype=torch.long)
                - prefix_gpu.index_select(0, span_ids_gpu))
            abs_new_gpu = (new_abs_starts_gpu.index_select(0, span_ids_gpu)
                           + token_offsets_gpu)
            abs_old_gpu = (old_abs_starts_gpu.index_select(0, span_ids_gpu)
                           + token_offsets_gpu)
            old_slots_gpu = (old_slot_starts_gpu.index_select(0, span_ids_gpu)
                             + token_offsets_gpu)

            valid_range_gpu = ((abs_new_gpu >= 0)
                               & (abs_new_gpu < int(prompt_len)))
            skipped_out_of_range += int((~valid_range_gpu).sum().item())

            dst_idx_gpu = torch.full((total_gpu_tokens, ), -1, dtype=torch.long)
            if bool(valid_range_gpu.any().item()):
                dst_idx_gpu[valid_range_gpu] = reuse_pos_to_idx.index_select(
                    0, abs_new_gpu[valid_range_gpu])
            missing_dst_gpu = valid_range_gpu & (dst_idx_gpu < 0)
            skipped_missing_dst += int(missing_dst_gpu.sum().item())

            candidate_gpu = valid_range_gpu & (dst_idx_gpu >= 0)
            cand_gpu_idx = torch.nonzero(candidate_gpu,
                                         as_tuple=False).flatten()
            kept_gpu_idx = torch.empty(0, dtype=torch.long)
            if int(cand_gpu_idx.numel()) > 0:
                cand_gpu_dst = dst_idx_gpu.index_select(0, cand_gpu_idx)
                keep_mask = _stable_first_occurrence_mask(cand_gpu_dst)
                kept_rel = torch.nonzero(keep_mask, as_tuple=False).flatten()
                kept_gpu_idx = cand_gpu_idx.index_select(0, kept_rel)
                skipped_duplicate += int(
                    int(cand_gpu_idx.numel()) - int(kept_gpu_idx.numel()))
            t_gpu_iter1 = time.perf_counter()
            span_plan_iter_gpu_spans_ms = float(
                (t_gpu_iter1 - t_gpu_iter0) * 1000.0)

            t_gpu_merge0 = time.perf_counter()
            if int(kept_gpu_idx.numel()) > 0:
                kept_gpu_dst = dst_idx_gpu.index_select(0, kept_gpu_idx)
                kept_gpu_old_slots = old_slots_gpu.index_select(0, kept_gpu_idx)
                kept_gpu_old_abs = abs_old_gpu.index_select(0, kept_gpu_idx)
                kept_gpu_new_abs = abs_new_gpu.index_select(0, kept_gpu_idx)
                kept_gpu_span_ids = span_ids_gpu.index_select(0, kept_gpu_idx)
                seen_dst[kept_gpu_dst] = True

                k_gpu = int(kept_gpu_idx.numel())
                if k_gpu == 1:
                    before_start_pos = torch.zeros(1, dtype=torch.long)
                    after_start_pos = torch.zeros(1, dtype=torch.long)
                else:
                    cont_base = ((kept_gpu_old_slots[1:]
                                  == (kept_gpu_old_slots[:-1] + 1))
                                 & (kept_gpu_dst[1:]
                                    == (kept_gpu_dst[:-1] + 1))
                                 & (kept_gpu_new_abs[1:]
                                    == (kept_gpu_new_abs[:-1] + 1)))
                    before_cont = (cont_base
                                   & (kept_gpu_span_ids[1:]
                                      == kept_gpu_span_ids[:-1]))
                    before_start_mask = torch.ones(k_gpu, dtype=torch.bool)
                    before_start_mask[1:] = ~before_cont
                    before_start_pos = torch.nonzero(
                        before_start_mask, as_tuple=False).flatten()
                    after_start_mask = torch.ones(k_gpu, dtype=torch.bool)
                    after_start_mask[1:] = ~cont_base
                    after_start_pos = torch.nonzero(
                        after_start_mask, as_tuple=False).flatten()

                gpu_span_count_before_merge = int(before_start_pos.numel())
                gpu_span_count_after_merge = int(after_start_pos.numel())

                spans_with_kept = int(torch.unique(kept_gpu_span_ids).numel())
                noncontiguous_span_count += int(
                    max(0, gpu_span_count_before_merge - spans_with_kept))

                after_end_pos = torch.cat([
                    after_start_pos[1:],
                    torch.tensor([k_gpu], dtype=torch.long),
                ])
                gpu_span_lengths = (after_end_pos - after_start_pos).contiguous()
                gpu_old_slot_starts = kept_gpu_old_slots.index_select(
                    0, after_start_pos)
                gpu_span_block_ids = (gpu_old_slot_starts // bs).contiguous()
                gpu_span_page_offsets = (gpu_old_slot_starts
                                         - gpu_span_block_ids * bs).contiguous()
                gpu_span_dst_starts = kept_gpu_dst.index_select(
                    0, after_start_pos).contiguous()
                gpu_span_old_abs_starts = kept_gpu_old_abs.index_select(
                    0, after_start_pos).contiguous()
                gpu_span_new_abs_starts = kept_gpu_new_abs.index_select(
                    0, after_start_pos).contiguous()
            t_gpu_merge1 = time.perf_counter()
            span_plan_merge_gpu_spans_ms = float(
                (t_gpu_merge1 - t_gpu_merge0) * 1000.0)

    # CPU fallback spans: vectorized mapping + stable dedup, merge contiguous rows.
    cpu_span_src_starts = torch.empty(0, dtype=torch.long)
    cpu_span_dst_starts = torch.empty(0, dtype=torch.long)
    cpu_span_lengths = torch.empty(0, dtype=torch.long)
    fallback_spans = [(int(s), int(e)) for s, e in plan.cpu_fallback_spans
                      if int(e) > int(s)]
    if fallback_spans:
        token_lens_cpu = torch.tensor(
            [int(e - s) for s, e in fallback_spans], dtype=torch.long)
        if int(token_lens_cpu.numel()) > 0 and int(token_lens_cpu.sum().item()) > 0:
            new_abs_starts_cpu = torch.tensor([int(s) for s, _ in fallback_spans],
                                              dtype=torch.long)
            t_cpu0 = time.perf_counter()
            span_count_cpu = int(token_lens_cpu.numel())
            prefix_cpu = torch.cumsum(token_lens_cpu, dim=0) - token_lens_cpu
            span_ids_cpu = torch.repeat_interleave(
                torch.arange(span_count_cpu, dtype=torch.long), token_lens_cpu)
            total_cpu_tokens = int(span_ids_cpu.numel())
            token_offsets_cpu = (
                torch.arange(total_cpu_tokens, dtype=torch.long)
                - prefix_cpu.index_select(0, span_ids_cpu))
            abs_new_cpu = (new_abs_starts_cpu.index_select(0, span_ids_cpu)
                           + token_offsets_cpu)
            valid_range_cpu = ((abs_new_cpu >= 0)
                               & (abs_new_cpu < int(prompt_len)))
            skipped_out_of_range += int((~valid_range_cpu).sum().item())
            dst_idx_cpu = torch.full((total_cpu_tokens, ), -1, dtype=torch.long)
            if bool(valid_range_cpu.any().item()):
                dst_idx_cpu[valid_range_cpu] = reuse_pos_to_idx.index_select(
                    0, abs_new_cpu[valid_range_cpu])
            missing_dst_cpu = valid_range_cpu & (dst_idx_cpu < 0)
            skipped_missing_dst += int(missing_dst_cpu.sum().item())

            candidate_cpu = valid_range_cpu & (dst_idx_cpu >= 0)
            cand_cpu_idx = torch.nonzero(candidate_cpu,
                                         as_tuple=False).flatten()
            kept_cpu_idx = torch.empty(0, dtype=torch.long)
            if int(cand_cpu_idx.numel()) > 0:
                cand_cpu_dst = dst_idx_cpu.index_select(0, cand_cpu_idx)
                dup_with_seen = seen_dst.index_select(0, cand_cpu_dst)
                if debug_check_duplicates:
                    skipped_duplicate += int(dup_with_seen.sum().item())
                cand_cpu_idx2 = cand_cpu_idx[~dup_with_seen]
                if int(cand_cpu_idx2.numel()) > 0:
                    cand_cpu_dst2 = dst_idx_cpu.index_select(0, cand_cpu_idx2)
                    keep_mask2 = _stable_first_occurrence_mask(cand_cpu_dst2)
                    kept_rel2 = torch.nonzero(keep_mask2,
                                              as_tuple=False).flatten()
                    kept_cpu_idx = cand_cpu_idx2.index_select(0, kept_rel2)
                    if debug_check_duplicates:
                        skipped_duplicate += int(
                            int(cand_cpu_idx2.numel()) - int(kept_cpu_idx.numel()))
            t_cpu1 = time.perf_counter()

            t_cpu_merge0 = time.perf_counter()
            if int(kept_cpu_idx.numel()) > 0:
                kept_cpu_dst = dst_idx_cpu.index_select(0, kept_cpu_idx)
                kept_cpu_span_ids = span_ids_cpu.index_select(0, kept_cpu_idx)
                seen_dst[kept_cpu_dst] = True
                k_cpu = int(kept_cpu_idx.numel())
                if k_cpu == 1:
                    before_start_pos_cpu = torch.zeros(1, dtype=torch.long)
                    after_start_pos_cpu = torch.zeros(1, dtype=torch.long)
                else:
                    cont_cpu = (kept_cpu_dst[1:] == (kept_cpu_dst[:-1] + 1))
                    before_cont_cpu = (cont_cpu
                                       & (kept_cpu_span_ids[1:]
                                          == kept_cpu_span_ids[:-1]))
                    before_start_mask_cpu = torch.ones(k_cpu, dtype=torch.bool)
                    before_start_mask_cpu[1:] = ~before_cont_cpu
                    before_start_pos_cpu = torch.nonzero(
                        before_start_mask_cpu, as_tuple=False).flatten()
                    after_start_mask_cpu = torch.ones(k_cpu, dtype=torch.bool)
                    after_start_mask_cpu[1:] = ~cont_cpu
                    after_start_pos_cpu = torch.nonzero(
                        after_start_mask_cpu, as_tuple=False).flatten()

                cpu_span_count_before_merge = int(before_start_pos_cpu.numel())
                cpu_span_count_after_merge = int(after_start_pos_cpu.numel())
                after_end_pos_cpu = torch.cat([
                    after_start_pos_cpu[1:],
                    torch.tensor([k_cpu], dtype=torch.long),
                ])
                cpu_span_lengths = (
                    after_end_pos_cpu - after_start_pos_cpu).contiguous()
                cpu_span_dst_starts = kept_cpu_dst.index_select(
                    0, after_start_pos_cpu).contiguous()
                # Source KV is reuse-order for current system microbenchmark.
                cpu_span_src_starts = cpu_span_dst_starts.clone().contiguous()
            t_cpu_merge1 = time.perf_counter()
            span_plan_build_cpu_spans_ms = float(
                (t_cpu1 - t_cpu0 + t_cpu_merge1 - t_cpu_merge0) * 1000.0)

    t_mat0 = time.perf_counter()
    gpu_span_block_ids = gpu_span_block_ids.to(dtype=torch.long,
                                               device="cpu").contiguous()
    gpu_span_page_offsets = gpu_span_page_offsets.to(dtype=torch.long,
                                                     device="cpu").contiguous()
    gpu_span_dst_starts = gpu_span_dst_starts.to(dtype=torch.long,
                                                 device="cpu").contiguous()
    gpu_span_lengths = gpu_span_lengths.to(dtype=torch.long,
                                           device="cpu").contiguous()
    gpu_span_old_abs_starts = gpu_span_old_abs_starts.to(
        dtype=torch.long, device="cpu").contiguous()
    gpu_span_new_abs_starts = gpu_span_new_abs_starts.to(
        dtype=torch.long, device="cpu").contiguous()
    cpu_span_src_starts = cpu_span_src_starts.to(dtype=torch.long,
                                                 device="cpu").contiguous()
    cpu_span_dst_starts = cpu_span_dst_starts.to(dtype=torch.long,
                                                 device="cpu").contiguous()
    cpu_span_lengths = cpu_span_lengths.to(dtype=torch.long,
                                           device="cpu").contiguous()
    t_mat1 = time.perf_counter()
    span_plan_tensor_materialize_ms = float((t_mat1 - t_mat0) * 1000.0)

    gpu_span_tokens = int(gpu_span_lengths.sum().item()) if int(
        gpu_span_lengths.numel()) > 0 else 0
    cpu_span_tokens = int(cpu_span_lengths.sum().item()) if int(
        cpu_span_lengths.numel()) > 0 else 0
    covered_reuse_tokens = int(seen_dst.sum().item())

    requires_token_fallback = False
    fallback_reason = "ok"
    if skipped_duplicate > 0:
        requires_token_fallback = True
        fallback_reason = "span_duplicate_dst"
    elif covered_reuse_tokens <= 0 and target_reuse_tokens > 0:
        requires_token_fallback = True
        fallback_reason = "span_empty_coverage"

    t1 = time.perf_counter()
    span_plan_total_ms = float((t1 - t0) * 1000.0)
    span_plan_known_ms = (
        span_plan_make_pos_map_ms
        + span_plan_iter_gpu_spans_ms
        + span_plan_merge_gpu_spans_ms
        + span_plan_build_cpu_spans_ms
        + span_plan_tensor_materialize_ms
        + span_plan_debug_check_ms
    )
    span_plan_other_ms = float(max(0.0, span_plan_total_ms - span_plan_known_ms))

    return {
        "gpu_span_block_ids": gpu_span_block_ids,
        "gpu_span_page_offsets": gpu_span_page_offsets,
        "gpu_span_dst_starts": gpu_span_dst_starts,
        "gpu_span_lengths": gpu_span_lengths,
        "gpu_span_old_abs_starts": gpu_span_old_abs_starts,
        "gpu_span_new_abs_starts": gpu_span_new_abs_starts,
        "cpu_span_src_starts": cpu_span_src_starts,
        "cpu_span_dst_starts": cpu_span_dst_starts,
        "cpu_span_lengths": cpu_span_lengths,
        "gpu_span_count": int(gpu_span_count_after_merge),
        "gpu_span_tokens": int(gpu_span_tokens),
        "cpu_span_count": int(cpu_span_count_after_merge),
        "cpu_span_tokens": int(cpu_span_tokens),
        "target_reuse_tokens": int(target_reuse_tokens),
        "covered_reuse_tokens": int(covered_reuse_tokens),
        "skipped_out_of_range": int(skipped_out_of_range),
        "skipped_missing_dst": int(skipped_missing_dst),
        "skipped_duplicate": int(skipped_duplicate),
        "noncontiguous_span_count": int(noncontiguous_span_count),
        "span_plan_build_ms": float(span_plan_total_ms),
        "span_plan_total_ms": float(span_plan_total_ms),
        "span_plan_make_pos_map_ms": float(span_plan_make_pos_map_ms),
        "span_plan_iter_gpu_spans_ms": float(span_plan_iter_gpu_spans_ms),
        "span_plan_build_cpu_spans_ms": float(span_plan_build_cpu_spans_ms),
        "span_plan_merge_gpu_spans_ms": float(span_plan_merge_gpu_spans_ms),
        "span_plan_tensor_materialize_ms": float(
            span_plan_tensor_materialize_ms),
        "span_plan_debug_check_ms": float(span_plan_debug_check_ms),
        "span_plan_other_ms": float(span_plan_other_ms),
        "span_plan_gpu_span_count_before_merge":
        int(gpu_span_count_before_merge),
        "span_plan_gpu_span_count_after_merge":
        int(gpu_span_count_after_merge),
        "span_plan_cpu_span_count_before_merge":
        int(cpu_span_count_before_merge),
        "span_plan_cpu_span_count_after_merge":
        int(cpu_span_count_after_merge),
        "requires_token_fallback": bool(requires_token_fallback),
        "fallback_reason": str(fallback_reason),
    }
