# SPDX-License-Identifier: Apache-2.0
"""In-process chunk-page index manager (rank-local)."""

from __future__ import annotations

from collections import OrderedDict, deque
from dataclasses import replace
from typing import Optional, Sequence, Any
import time

import torch

from .chunk_page_index_types import ChunkPageEntry, OldPageSource

_LRU_UNIT_ENTRY = "entry"
_LRU_UNIT_CHUNK_HASH = "chunk_hash"


class ChunkPageIndex:
    """Chunk-centric index for old-page gather planning.

    Notes:
    - Rank-local only (no cross-rank sharing).
    - LRU is best-effort capacity protection for stage-1 dry-run.
    """

    def __init__(
        self,
        *,
        max_chunks: int = 128,
        max_entries_per_hash: int = 8,
        global_lookup_enabled: bool = False,
        no_replace_existing_hash: bool = False,
        block_invalidation_enabled: bool = False,
        lru_unit: str = _LRU_UNIT_ENTRY,
        lru_evict_batch_chunks: int = 0,
        lru_low_watermark_chunks: int = -1,
    ) -> None:
        self.max_chunks = int(max_chunks)
        self.max_entries_per_hash = int(max_entries_per_hash)
        self.global_lookup_enabled = bool(global_lookup_enabled)
        self.no_replace_existing_hash = bool(no_replace_existing_hash)
        self.block_invalidation_enabled = bool(block_invalidation_enabled)
        norm_unit = str(lru_unit).strip().lower()
        if norm_unit not in (_LRU_UNIT_ENTRY, _LRU_UNIT_CHUNK_HASH):
            norm_unit = _LRU_UNIT_ENTRY
        self.lru_unit = norm_unit
        self.lru_evict_batch_chunks = int(max(0, int(lru_evict_batch_chunks)))
        self.lru_low_watermark_chunks = int(lru_low_watermark_chunks)

        self.entries_by_hash: dict[int, deque[str]] = {}
        self.entries: dict[str, ChunkPageEntry] = {}
        self.sources: dict[str, OldPageSource] = {}
        self.entries_by_block: dict[int, set[str]] = {}
        self.blocks_by_entry: dict[str, set[int]] = {}
        self._lru: OrderedDict[str, None] = OrderedDict()
        self._hash_lru: OrderedDict[int, None] = OrderedDict()

        self.lru_evictions = 0
        self.lru_capacity_evictions = 0
        self.lru_bucket_evictions = 0
        self.lru_chunk_hash_capacity_evictions = 0
        self.lru_evicted_chunk_hashes = 0
        self.lru_chunk_hash_evicted_entries_sum = 0
        self.lru_chunk_hash_evicted_entries_max_per_hash = 0
        self.capacity_full_events = 0
        self.register_attempts = 0
        self.register_success = 0
        self.register_skipped_duplicate_hash = 0
        self.register_skipped_bucket_full = 0
        self.block_invalidations = 0
        self.block_invalidated_entries = 0
        self.block_invalidated_blocks = 0
        self.metadata_only_entries_current = 0
        self.real_entries_current = 0
        self.last_register_lru_evicted_entries = 0
        self.last_register_lru_evicted_chunk_hashes = 0
        self.last_register_lru_evicted_entries_per_chunk_max = 0
        self.last_register_lru_trigger_reason = "none"

    def configure(
        self,
        *,
        max_chunks: int,
        max_entries_per_hash: int,
        global_lookup_enabled: Optional[bool] = None,
        no_replace_existing_hash: Optional[bool] = None,
        block_invalidation_enabled: Optional[bool] = None,
        lru_unit: Optional[str] = None,
        lru_evict_batch_chunks: Optional[int] = None,
        lru_low_watermark_chunks: Optional[int] = None,
    ) -> None:
        self.max_chunks = int(max_chunks)
        self.max_entries_per_hash = int(max_entries_per_hash)
        if global_lookup_enabled is not None:
            self.global_lookup_enabled = bool(global_lookup_enabled)
        if no_replace_existing_hash is not None:
            self.no_replace_existing_hash = bool(no_replace_existing_hash)
        if block_invalidation_enabled is not None:
            self.block_invalidation_enabled = bool(block_invalidation_enabled)
        if lru_unit is not None:
            norm_unit = str(lru_unit).strip().lower()
            if norm_unit in (_LRU_UNIT_ENTRY, _LRU_UNIT_CHUNK_HASH):
                self.lru_unit = norm_unit
        if lru_evict_batch_chunks is not None:
            self.lru_evict_batch_chunks = int(max(0, int(lru_evict_batch_chunks)))
        if lru_low_watermark_chunks is not None:
            self.lru_low_watermark_chunks = int(lru_low_watermark_chunks)
        self._evict_lru_until_capacity()

    def clear(self) -> None:
        self.entries_by_hash.clear()
        self.entries.clear()
        self.sources.clear()
        self.entries_by_block.clear()
        self.blocks_by_entry.clear()
        self._lru.clear()
        self._hash_lru.clear()
        self.lru_evictions = 0
        self.lru_capacity_evictions = 0
        self.lru_bucket_evictions = 0
        self.lru_chunk_hash_capacity_evictions = 0
        self.lru_evicted_chunk_hashes = 0
        self.lru_chunk_hash_evicted_entries_sum = 0
        self.lru_chunk_hash_evicted_entries_max_per_hash = 0
        self.capacity_full_events = 0
        self.register_attempts = 0
        self.register_success = 0
        self.register_skipped_duplicate_hash = 0
        self.register_skipped_bucket_full = 0
        self.block_invalidations = 0
        self.block_invalidated_entries = 0
        self.block_invalidated_blocks = 0
        self.metadata_only_entries_current = 0
        self.real_entries_current = 0
        self.last_register_lru_evicted_entries = 0
        self.last_register_lru_evicted_chunk_hashes = 0
        self.last_register_lru_evicted_entries_per_chunk_max = 0
        self.last_register_lru_trigger_reason = "none"

    def register_source(
        self,
        *,
        source_id: str,
        block_size: int,
        layout_code: int,
        model_id: Optional[str],
        dtype: Optional[str],
        kv_layout: Optional[str],
        is_metadata_only_source: bool = False,
        valid_for_lookup_source: bool = True,
    ) -> OldPageSource:
        sid = str(source_id)
        src = self.sources.get(sid)
        if src is None:
            src = OldPageSource(
                source_id=sid,
                block_size=int(block_size),
                layout_code=int(layout_code),
                model_id=model_id,
                dtype=dtype,
                kv_layout=kv_layout,
                created_ts=time.time(),
                valid=True,
                is_metadata_only_source=bool(is_metadata_only_source),
                valid_for_lookup_source=bool(valid_for_lookup_source),
            )
            self.sources[sid] = src
            return src

        src.block_size = int(block_size)
        src.layout_code = int(layout_code)
        src.model_id = model_id
        src.dtype = dtype
        src.kv_layout = kv_layout
        src.valid = True
        src.is_metadata_only_source = bool(is_metadata_only_source)
        src.valid_for_lookup_source = bool(valid_for_lookup_source)
        return src

    def update_source_layer_cache(
        self,
        *,
        source_id: str,
        layer_id: int,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
    ) -> bool:
        src = self.sources.get(str(source_id))
        if src is None or not src.valid:
            return False
        src.layer_caches[int(layer_id)] = (key_cache, value_cache)
        return True

    def invalidate_source(self, source_id: str) -> None:
        sid = str(source_id)
        src = self.sources.get(sid)
        if src is not None:
            src.valid = False
        to_remove = [
            entry_id for entry_id, entry in self.entries.items()
            if entry.source_id == sid
        ]
        for entry_id in to_remove:
            self._remove_entry(entry_id)
        self.sources.pop(sid, None)

    def count_valid_entries_for_hash(
        self,
        *,
        chunk_hash: int,
        token_count: Optional[int] = None,
        token_checksum: Optional[int] = None,
        model_id: Optional[str] = None,
        dtype: Optional[str] = None,
        kv_layout: Optional[str] = None,
        block_size: Optional[int] = None,
    ) -> int:
        bucket = self.entries_by_hash.get(int(chunk_hash))
        if not bucket:
            return 0
        count = 0
        for entry_id in list(bucket):
            entry = self.entries.get(entry_id)
            if entry is None or entry.state != "valid":
                continue
            src = self.sources.get(entry.source_id)
            if src is None or not src.valid:
                continue
            if model_id is not None and src.model_id not in (None, model_id):
                continue
            if dtype is not None and src.dtype not in (None, dtype):
                continue
            if kv_layout is not None and src.kv_layout not in (None, kv_layout):
                continue
            if block_size is not None and int(src.block_size) != int(
                    block_size):
                continue
            if token_count is not None and int(entry.token_count) != int(
                    token_count):
                continue
            if (token_checksum is not None
                    and int(entry.token_checksum
                            if entry.token_checksum is not None else -1)
                    != int(token_checksum)):
                continue
            if model_id is not None and entry.model_id not in (None, model_id):
                continue
            if dtype is not None and entry.dtype not in (None, dtype):
                continue
            if kv_layout is not None and entry.kv_layout not in (None,
                                                                  kv_layout):
                continue
            if block_size is not None and int(entry.block_size) != int(
                    block_size):
                continue
            count += 1
        return int(count)

    def register_entry(self, entry: ChunkPageEntry) -> bool:
        self._reset_last_register_lru_stats()
        self.register_attempts += 1
        entry_id = str(entry.entry_id)
        if entry_id in self.entries:
            self._remove_entry(entry_id)

        if self.no_replace_existing_hash and self.max_entries_per_hash > 0:
            valid_bucket_size = self.count_valid_entries_for_hash(
                chunk_hash=int(entry.chunk_token_hash),
                token_count=int(entry.token_count),
                token_checksum=int(entry.token_checksum)
                if entry.token_checksum is not None else None,
                model_id=entry.model_id,
                dtype=entry.dtype,
                kv_layout=entry.kv_layout,
                block_size=int(entry.block_size),
            )
            if valid_bucket_size >= int(self.max_entries_per_hash):
                self.register_skipped_duplicate_hash += 1
                self.register_skipped_bucket_full += 1
                return False

        entry = replace(
            entry,
            state="valid",
            last_access_ts=time.time(),
            access_count=max(0, int(entry.access_count)),
        )
        self.entries[entry_id] = entry
        if bool(entry.is_metadata_only):
            self.metadata_only_entries_current += 1
        else:
            self.real_entries_current += 1
        bucket = self.entries_by_hash.setdefault(int(entry.chunk_token_hash),
                                                 deque())
        bucket.append(entry_id)
        self._touch_hash(int(entry.chunk_token_hash))

        block_ids: set[int] = set()
        for span in entry.page_spans:
            block_ids.add(int(span.physical_block_id))
        if block_ids:
            self.blocks_by_entry[entry_id] = set(block_ids)
            for block_id in block_ids:
                self.entries_by_block.setdefault(int(block_id), set()).add(
                    entry_id)

        self.register_success += 1
        self._touch(entry_id)
        self._evict_hash_bucket_if_needed(int(entry.chunk_token_hash))
        self._evict_lru_until_capacity()
        return True

    def note_register_skipped_duplicate_hash(self) -> None:
        """Account for a chunk skipped before PageSpan materialization."""
        self.register_attempts += 1
        self.register_skipped_duplicate_hash += 1
        self.register_skipped_bucket_full += 1

    def touch(self, entry_id: str) -> None:
        self._touch(str(entry_id))

    def lookup(
        self,
        *,
        chunk_hash: int,
        token_count: Optional[int] = None,
        token_checksum: Optional[int] = None,
        source_id: Optional[str] = None,
        model_id: Optional[str] = None,
        dtype: Optional[str] = None,
        kv_layout: Optional[str] = None,
        block_size: Optional[int] = None,
        touch: bool = True,
        diag: Optional[dict[str, Any]] = None,
    ) -> list[ChunkPageEntry]:
        bucket = self.entries_by_hash.get(int(chunk_hash))
        if not bucket:
            if diag is not None:
                diag.update({
                    "bucket_size": 0,
                    "candidates_scanned": 0,
                    "candidates_kept": 0,
                    "candidates_invalid": 0,
                    "metadata_only_candidate_count": 0,
                    "metadata_only_filtered_count": 0,
                    "metadata_only_false_hit_count": 0,
                    "candidate_scan_ms": 0.0,
                    "block_validity_check_ms": 0.0,
                    "invalid_filter_ms": 0.0,
                    "touch_ms": 0.0,
                })
            return []
        out: list[ChunkPageEntry] = []
        src_filter = str(source_id) if source_id is not None else None
        scan_started = time.perf_counter()
        validity_ms = 0.0
        invalid_filter_ms = 0.0
        scanned = 0
        kept = 0
        invalid = 0
        metadata_only_candidate_count = 0
        metadata_only_filtered_count = 0
        metadata_only_false_hit_count = 0
        for entry_id in list(bucket):
            scanned += 1
            t_valid0 = time.perf_counter()
            entry = self.entries.get(entry_id)
            if entry is None or entry.state != "valid":
                validity_ms += float((time.perf_counter() - t_valid0) * 1000.0)
                invalid += 1
                continue
            if bool(getattr(entry, "is_metadata_only", False)):
                metadata_only_candidate_count += 1
            src = self.sources.get(entry.source_id)
            if src is None or not src.valid:
                validity_ms += float((time.perf_counter() - t_valid0) * 1000.0)
                invalid += 1
                continue
            validity_ms += float((time.perf_counter() - t_valid0) * 1000.0)

            t_filter0 = time.perf_counter()
            if not bool(getattr(entry, "valid_for_lookup", True)):
                invalid_filter_ms += float((time.perf_counter() - t_filter0) * 1000.0)
                invalid += 1
                if bool(getattr(entry, "is_metadata_only", False)):
                    metadata_only_filtered_count += 1
                continue
            if src_filter is not None and entry.source_id != src_filter:
                invalid_filter_ms += float((time.perf_counter() - t_filter0) * 1000.0)
                invalid += 1
                continue
            if model_id is not None and src.model_id not in (None, model_id):
                invalid_filter_ms += float((time.perf_counter() - t_filter0) * 1000.0)
                invalid += 1
                continue
            if dtype is not None and src.dtype not in (None, dtype):
                invalid_filter_ms += float((time.perf_counter() - t_filter0) * 1000.0)
                invalid += 1
                continue
            if kv_layout is not None and src.kv_layout not in (None, kv_layout):
                invalid_filter_ms += float((time.perf_counter() - t_filter0) * 1000.0)
                invalid += 1
                continue
            if block_size is not None and int(src.block_size) != int(
                    block_size):
                invalid_filter_ms += float((time.perf_counter() - t_filter0) * 1000.0)
                invalid += 1
                continue
            if token_count is not None and int(entry.token_count) != int(
                    token_count):
                invalid_filter_ms += float((time.perf_counter() - t_filter0) * 1000.0)
                invalid += 1
                continue
            if (token_checksum is not None
                    and int(entry.token_checksum
                            if entry.token_checksum is not None else -1)
                    != int(token_checksum)):
                invalid_filter_ms += float((time.perf_counter() - t_filter0) * 1000.0)
                invalid += 1
                continue
            if model_id is not None and entry.model_id not in (None, model_id):
                invalid_filter_ms += float((time.perf_counter() - t_filter0) * 1000.0)
                invalid += 1
                continue
            if dtype is not None and entry.dtype not in (None, dtype):
                invalid_filter_ms += float((time.perf_counter() - t_filter0) * 1000.0)
                invalid += 1
                continue
            if kv_layout is not None and entry.kv_layout not in (None,
                                                                  kv_layout):
                invalid_filter_ms += float((time.perf_counter() - t_filter0) * 1000.0)
                invalid += 1
                continue
            if block_size is not None and int(entry.block_size) != int(
                    block_size):
                invalid_filter_ms += float((time.perf_counter() - t_filter0) * 1000.0)
                invalid += 1
                continue
            invalid_filter_ms += float((time.perf_counter() - t_filter0) * 1000.0)
            out.append(entry)
            if bool(getattr(entry, "is_metadata_only", False)):
                metadata_only_false_hit_count += 1
            kept += 1
        scan_ms = float((time.perf_counter() - scan_started) * 1000.0)
        touch_ms = 0.0
        if touch:
            t_touch0 = time.perf_counter()
            for entry in out:
                self._touch(entry.entry_id)
            touch_ms = float((time.perf_counter() - t_touch0) * 1000.0)
        if diag is not None:
            diag.update({
                "bucket_size": int(len(bucket)),
                "candidates_scanned": int(scanned),
                "candidates_kept": int(kept),
                "candidates_invalid": int(invalid),
                "metadata_only_candidate_count": int(metadata_only_candidate_count),
                "metadata_only_filtered_count": int(metadata_only_filtered_count),
                "metadata_only_false_hit_count": int(
                    metadata_only_false_hit_count),
                "candidate_scan_ms": float(scan_ms),
                "block_validity_check_ms": float(validity_ms),
                "invalid_filter_ms": float(invalid_filter_ms),
                "touch_ms": float(touch_ms),
            })
        return out

    def invalidate_block(self, block_id: int) -> dict[str, int]:
        if not self.block_invalidation_enabled:
            return {"invalidated_entries": 0, "invalidated_blocks": 0}
        block_id_int = int(block_id)
        affected_ids = list(self.entries_by_block.get(block_id_int, set()))
        invalidated_entries = 0
        for entry_id in affected_ids:
            if entry_id not in self.entries:
                continue
            self._remove_entry(entry_id)
            invalidated_entries += 1
        invalidated_blocks = 1 if affected_ids else 0
        self.block_invalidations += 1
        self.block_invalidated_entries += int(invalidated_entries)
        self.block_invalidated_blocks += int(invalidated_blocks)
        return {
            "invalidated_entries": int(invalidated_entries),
            "invalidated_blocks": int(invalidated_blocks),
        }

    def invalidate_blocks(self, block_ids: Sequence[int]) -> dict[str, int]:
        if not self.block_invalidation_enabled:
            return {"invalidated_entries": 0, "invalidated_blocks": 0}
        total_entries = 0
        total_blocks = 0
        seen: set[int] = set()
        for block_id in block_ids:
            block_id_int = int(block_id)
            if block_id_int in seen:
                continue
            seen.add(block_id_int)
            stat = self.invalidate_block(block_id_int)
            total_entries += int(stat.get("invalidated_entries", 0))
            total_blocks += int(stat.get("invalidated_blocks", 0))
        return {
            "invalidated_entries": int(total_entries),
            "invalidated_blocks": int(total_blocks),
        }

    def stats(self) -> dict[str, Any]:
        max_bucket = 0
        for bucket in self.entries_by_hash.values():
            if len(bucket) > max_bucket:
                max_bucket = len(bucket)
        return {
            "entries": int(len(self.entries)),
            "current_entry_count": int(len(self.entries)),
            "chunk_hash_count": int(len(self.entries_by_hash)),
            "current_chunk_hash_count": int(len(self.entries_by_hash)),
            "max_chunks_raw_config": int(self.max_chunks),
            "max_entries_per_hash_config": int(self.max_entries_per_hash),
            "lru_evict_batch_chunks": int(self.lru_evict_batch_chunks),
            "lru_low_watermark_chunks": int(self.lru_low_watermark_chunks),
            "sources": int(len(self.sources)),
            "entries_per_hash_max": int(max_bucket),
            "entries_by_block_size": int(len(self.entries_by_block)),
            "blocks_by_entry_size": int(len(self.blocks_by_entry)),
            "lru_evictions": int(self.lru_evictions),
            "lru_capacity_evictions": int(self.lru_capacity_evictions),
            "lru_bucket_evictions": int(self.lru_bucket_evictions),
            "lru_chunk_hash_capacity_evictions": int(
                self.lru_chunk_hash_capacity_evictions),
            "lru_evicted_chunk_hashes": int(self.lru_evicted_chunk_hashes),
            "lru_chunk_hash_evicted_entries_sum": int(
                self.lru_chunk_hash_evicted_entries_sum),
            "lru_chunk_hash_evicted_entries_max_per_hash": int(
                self.lru_chunk_hash_evicted_entries_max_per_hash),
            "capacity_full_events": int(self.capacity_full_events),
            "register_attempts": int(self.register_attempts),
            "register_success": int(self.register_success),
            "register_skipped_duplicate_hash": int(
                self.register_skipped_duplicate_hash),
            "register_skipped_bucket_full": int(
                self.register_skipped_bucket_full),
            "block_invalidations": int(self.block_invalidations),
            "block_invalidated_entries": int(self.block_invalidated_entries),
            "block_invalidated_blocks": int(self.block_invalidated_blocks),
            "metadata_only_entries_current": int(
                self.metadata_only_entries_current),
            "real_entries_current": int(self.real_entries_current),
            "global_lookup_enabled": int(self.global_lookup_enabled),
            "no_replace_existing_hash": int(self.no_replace_existing_hash),
            "block_invalidation_enabled": int(self.block_invalidation_enabled),
            "lru_unit": str(self.lru_unit),
        }

    def get_last_register_lru_stats(self) -> dict[str, Any]:
        return {
            "lru_evicted_entries": int(self.last_register_lru_evicted_entries),
            "lru_evicted_chunk_hashes": int(
                self.last_register_lru_evicted_chunk_hashes),
            "lru_evicted_entries_per_chunk_max": int(
                self.last_register_lru_evicted_entries_per_chunk_max),
            "lru_trigger_reason": str(self.last_register_lru_trigger_reason),
        }

    def _touch(self, entry_id: str) -> None:
        entry = self.entries.get(entry_id)
        if entry is None:
            return
        entry.last_access_ts = time.time()
        entry.access_count += 1
        self._lru.pop(entry_id, None)
        self._lru[entry_id] = None
        self._touch_hash(int(entry.chunk_token_hash))

    def _touch_hash(self, chunk_hash: int) -> None:
        h = int(chunk_hash)
        self._hash_lru.pop(h, None)
        self._hash_lru[h] = None

    def _remove_entry(self, entry_id: str) -> None:
        entry = self.entries.pop(entry_id, None)
        if entry is None:
            self._lru.pop(entry_id, None)
            self.blocks_by_entry.pop(entry_id, None)
            return
        if bool(getattr(entry, "is_metadata_only", False)):
            self.metadata_only_entries_current = max(
                0, int(self.metadata_only_entries_current) - 1)
        else:
            self.real_entries_current = max(
                0, int(self.real_entries_current) - 1)
        bucket = self.entries_by_hash.get(int(entry.chunk_token_hash))
        if bucket is not None:
            try:
                bucket.remove(entry_id)
            except ValueError:
                pass
            if not bucket:
                self.entries_by_hash.pop(int(entry.chunk_token_hash), None)
                self._hash_lru.pop(int(entry.chunk_token_hash), None)
        block_ids = self.blocks_by_entry.pop(entry_id, None)
        if block_ids:
            for block_id in list(block_ids):
                ids = self.entries_by_block.get(int(block_id))
                if ids is None:
                    continue
                ids.discard(entry_id)
                if not ids:
                    self.entries_by_block.pop(int(block_id), None)
        self._lru.pop(entry_id, None)

    def _evict_hash_bucket_if_needed(self, chunk_hash: int) -> None:
        if self.max_entries_per_hash <= 0:
            return
        bucket = self.entries_by_hash.get(int(chunk_hash))
        if bucket is None:
            return
        while len(bucket) > self.max_entries_per_hash:
            evict_id = bucket[0]
            self._remove_entry(evict_id)
            self.lru_evictions += 1
            self.lru_bucket_evictions += 1
            self._note_last_register_lru_eviction(
                entries_removed=1,
                chunk_hashes_removed=0,
                max_entries_per_chunk=1,
                reason="bucket_capacity",
            )
            bucket = self.entries_by_hash.get(int(chunk_hash))
            if bucket is None:
                break

    def _evict_lru_until_capacity(self) -> None:
        if self.max_chunks <= 0:
            return
        if self.lru_unit == _LRU_UNIT_CHUNK_HASH:
            self._evict_hash_lru_until_capacity()
        else:
            self._evict_entry_lru_until_capacity()

    def _evict_entry_lru_until_capacity(self) -> None:
        if len(self.entries) > self.max_chunks:
            self.capacity_full_events += 1
        while len(self.entries) > self.max_chunks:
            oldest_id = next(iter(self._lru), None)
            if oldest_id is None:
                break
            self._remove_entry(oldest_id)
            self.lru_evictions += 1
            self.lru_capacity_evictions += 1
            self._note_last_register_lru_eviction(
                entries_removed=1,
                chunk_hashes_removed=0,
                max_entries_per_chunk=1,
                reason="entry_capacity",
            )

    def _remove_hash_bucket(self, chunk_hash: int) -> int:
        chunk_hash_int = int(chunk_hash)
        bucket = self.entries_by_hash.get(chunk_hash_int)
        if not bucket:
            self._hash_lru.pop(chunk_hash_int, None)
            return 0
        entry_ids = list(bucket)
        removed = 0
        for entry_id in entry_ids:
            if entry_id in self.entries:
                self._remove_entry(entry_id)
                removed += 1
        self._hash_lru.pop(chunk_hash_int, None)
        return int(removed)

    def _evict_oldest_hashes(self, num_hashes: int) -> int:
        evicted_hashes = 0
        remaining = int(max(0, num_hashes))
        while remaining > 0 and len(self.entries_by_hash) > 0:
            oldest_hash = next(iter(self._hash_lru), None)
            if oldest_hash is None:
                oldest_hash = next(iter(self.entries_by_hash), None)
            if oldest_hash is None:
                break
            removed_entries = int(self._remove_hash_bucket(int(oldest_hash)))
            if removed_entries <= 0:
                self._hash_lru.pop(int(oldest_hash), None)
                remaining -= 1
                continue
            self.lru_evicted_chunk_hashes += 1
            self.lru_chunk_hash_capacity_evictions += 1
            self.lru_evictions += int(removed_entries)
            self.lru_chunk_hash_evicted_entries_sum += int(removed_entries)
            self.lru_chunk_hash_evicted_entries_max_per_hash = max(
                int(self.lru_chunk_hash_evicted_entries_max_per_hash),
                int(removed_entries),
            )
            self._note_last_register_lru_eviction(
                entries_removed=int(removed_entries),
                chunk_hashes_removed=1,
                max_entries_per_chunk=int(removed_entries),
                reason="chunk_hash_capacity",
            )
            evicted_hashes += 1
            remaining -= 1
        return int(evicted_hashes)

    def _evict_hash_lru_until_capacity(self) -> None:
        hash_count = int(len(self.entries_by_hash))
        if hash_count > int(self.max_chunks):
            self.capacity_full_events += 1
        while int(len(self.entries_by_hash)) > int(self.max_chunks):
            target = int(self.max_chunks)
            if (int(self.lru_low_watermark_chunks) > 0
                    and int(self.lru_low_watermark_chunks) < int(self.max_chunks)):
                target = int(self.lru_low_watermark_chunks)
            if int(target) < 0:
                target = 0
            if target < int(self.max_chunks):
                remove_hashes = int(len(self.entries_by_hash) - target)
            else:
                batch = int(self.lru_evict_batch_chunks)
                remove_hashes = int(batch if batch > 0 else 1)
            remove_hashes = int(max(1, remove_hashes))
            before = int(len(self.entries_by_hash))
            _ = self._evict_oldest_hashes(remove_hashes)
            after = int(len(self.entries_by_hash))
            if after >= before:
                break

    def _reset_last_register_lru_stats(self) -> None:
        self.last_register_lru_evicted_entries = 0
        self.last_register_lru_evicted_chunk_hashes = 0
        self.last_register_lru_evicted_entries_per_chunk_max = 0
        self.last_register_lru_trigger_reason = "none"

    def _note_last_register_lru_eviction(
        self,
        *,
        entries_removed: int,
        chunk_hashes_removed: int,
        max_entries_per_chunk: int,
        reason: str,
    ) -> None:
        self.last_register_lru_evicted_entries += int(max(0, entries_removed))
        self.last_register_lru_evicted_chunk_hashes += int(
            max(0, chunk_hashes_removed))
        self.last_register_lru_evicted_entries_per_chunk_max = max(
            int(self.last_register_lru_evicted_entries_per_chunk_max),
            int(max(0, max_entries_per_chunk)),
        )
        pri = {
            "none": 0,
            "bucket_capacity": 1,
            "entry_capacity": 2,
            "chunk_hash_capacity": 3,
        }
        cur = str(self.last_register_lru_trigger_reason)
        nxt = str(reason)
        if pri.get(nxt, 0) >= pri.get(cur, 0):
            self.last_register_lru_trigger_reason = nxt


_CHUNK_PAGE_INDEX_SINGLETON: Optional[ChunkPageIndex] = None


def get_chunk_page_index(
    *,
    max_chunks: int = 128,
    max_entries_per_hash: int = 8,
    global_lookup_enabled: Optional[bool] = None,
    no_replace_existing_hash: Optional[bool] = None,
    block_invalidation_enabled: Optional[bool] = None,
    lru_unit: Optional[str] = None,
    lru_evict_batch_chunks: Optional[int] = None,
    lru_low_watermark_chunks: Optional[int] = None,
) -> ChunkPageIndex:
    global _CHUNK_PAGE_INDEX_SINGLETON
    if _CHUNK_PAGE_INDEX_SINGLETON is None:
        _CHUNK_PAGE_INDEX_SINGLETON = ChunkPageIndex(
            max_chunks=max_chunks,
            max_entries_per_hash=max_entries_per_hash,
            global_lookup_enabled=bool(global_lookup_enabled)
            if global_lookup_enabled is not None else False,
            no_replace_existing_hash=bool(no_replace_existing_hash)
            if no_replace_existing_hash is not None else False,
            block_invalidation_enabled=bool(block_invalidation_enabled)
            if block_invalidation_enabled is not None else False,
            lru_unit=str(lru_unit).strip().lower()
            if lru_unit is not None else _LRU_UNIT_ENTRY,
            lru_evict_batch_chunks=int(lru_evict_batch_chunks)
            if lru_evict_batch_chunks is not None else 0,
            lru_low_watermark_chunks=int(lru_low_watermark_chunks)
            if lru_low_watermark_chunks is not None else -1,
        )
    else:
        _CHUNK_PAGE_INDEX_SINGLETON.configure(
            max_chunks=max_chunks,
            max_entries_per_hash=max_entries_per_hash,
            global_lookup_enabled=global_lookup_enabled,
            no_replace_existing_hash=no_replace_existing_hash,
            block_invalidation_enabled=block_invalidation_enabled,
            lru_unit=lru_unit,
            lru_evict_batch_chunks=lru_evict_batch_chunks,
            lru_low_watermark_chunks=lru_low_watermark_chunks,
        )
    return _CHUNK_PAGE_INDEX_SINGLETON


def clear_chunk_page_index(source_id: Optional[str] = None) -> None:
    global _CHUNK_PAGE_INDEX_SINGLETON
    if _CHUNK_PAGE_INDEX_SINGLETON is None:
        return
    if source_id is None:
        _CHUNK_PAGE_INDEX_SINGLETON.clear()
    else:
        _CHUNK_PAGE_INDEX_SINGLETON.invalidate_source(str(source_id))


def invalidate_chunk_page_block(block_id: int) -> dict[str, int]:
    singleton = _CHUNK_PAGE_INDEX_SINGLETON
    if singleton is None:
        return {"invalidated_entries": 0, "invalidated_blocks": 0}
    return singleton.invalidate_block(int(block_id))


def invalidate_chunk_page_blocks(block_ids: Sequence[int]) -> dict[str, int]:
    singleton = _CHUNK_PAGE_INDEX_SINGLETON
    if singleton is None:
        return {"invalidated_entries": 0, "invalidated_blocks": 0}
    return singleton.invalidate_blocks(block_ids)


def is_chunk_page_block_invalidation_enabled() -> bool:
    singleton = _CHUNK_PAGE_INDEX_SINGLETON
    return bool(singleton is not None and singleton.block_invalidation_enabled)
