# SPDX-License-Identifier: Apache-2.0
"""Types for chunk-centric old-page gather index (dry-run first)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch


@dataclass
class OldPageSource:
    source_id: str
    block_size: int
    layout_code: int
    layer_caches: dict[int, tuple[torch.Tensor, torch.Tensor]] = field(
        default_factory=dict)
    model_id: Optional[str] = None
    dtype: Optional[str] = None
    kv_layout: Optional[str] = None
    created_ts: float = 0.0
    valid: bool = True
    is_metadata_only_source: bool = False
    valid_for_lookup_source: bool = True


@dataclass
class PageSpan:
    physical_block_id: int
    page_offset: int
    token_len: int
    old_abs_start: int
    old_abs_end: int
    chunk_token_offset: int
    is_full_page: bool
    logical_block_id: Optional[int] = None
    block_generation: Optional[int] = None


@dataclass
class ChunkPageEntry:
    entry_id: str
    source_id: str
    chunk_token_hash: int
    token_count: int
    old_prompt_start: int
    old_prompt_end: int
    block_size: int
    page_spans: list[PageSpan]
    model_id: Optional[str] = None
    dtype: Optional[str] = None
    kv_layout: Optional[str] = None
    state: str = "valid"
    last_access_ts: float = 0.0
    access_count: int = 0
    token_checksum: Optional[int] = None
    is_metadata_only: bool = False
    valid_for_lookup: bool = True
    preemptible: bool = False


@dataclass
class GatherSpanPlan:
    source_id: str
    physical_block_id: int
    page_offset: int
    token_len: int
    old_abs_start: int
    old_abs_end: int
    new_abs_start: int
    new_abs_end: int
    is_full_page: bool


@dataclass
class PageReusePlan:
    gather_spans: list[GatherSpanPlan] = field(default_factory=list)
    cpu_fallback_spans: list[tuple[int, int]] = field(default_factory=list)
    lookup_profile: dict[str, float] = field(default_factory=dict)

    indexed_chunks: int = 0
    hit_chunks: int = 0
    miss_chunks: int = 0
    candidate_count: int = 0
    same_source_hit_count: int = 0
    global_cross_source_hit_count: int = 0
    hit_chunk_keys: list[tuple[int, int, int]] = field(default_factory=list)
    miss_chunk_keys: list[tuple[int, int, int]] = field(default_factory=list)

    gpu_gather_full_tokens: int = 0
    gpu_gather_partial_tokens: int = 0
    cpu_fallback_tokens: int = 0
