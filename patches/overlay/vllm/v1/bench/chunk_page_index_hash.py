# SPDX-License-Identifier: Apache-2.0
"""Hash helpers for chunk-centric old-page index."""

from __future__ import annotations

import hashlib
from typing import Optional, Sequence

_SUPPORTED_HASH_ALGOS = ("builtin", "sha256")


def normalize_chunk_page_hash_algo(algo: str) -> str:
    value = str(algo).strip().lower()
    if value not in _SUPPORTED_HASH_ALGOS:
        raise ValueError(
            "Invalid chunk-page hash algo: "
            f"{algo}. Expected one of {_SUPPORTED_HASH_ALGOS}."
        )
    return value


def _serialize_chunk_hash_input(
    *,
    token_ids: Sequence[int],
    block_size: int,
    model_id: Optional[str],
    dtype: Optional[str],
    kv_layout: Optional[str],
    lora_id: Optional[int],
    kv_cache_group_id: Optional[int],
) -> bytes:
    # Keep the payload explicit and readable for debugging.
    parts = [
        f"block_size={int(block_size)}",
        f"model_id={model_id or ''}",
        f"dtype={dtype or ''}",
        f"kv_layout={kv_layout or ''}",
        f"lora_id={'' if lora_id is None else int(lora_id)}",
        ("kv_cache_group_id="
         f"{'' if kv_cache_group_id is None else int(kv_cache_group_id)}"),
        f"token_count={len(token_ids)}",
        "tokens=" + ",".join(str(int(t)) for t in token_ids),
    ]
    return "|".join(parts).encode("utf-8", errors="strict")


def compute_token_checksum(token_ids: Sequence[int]) -> int:
    checksum = 0
    for tid in token_ids:
        checksum = (checksum * 1315423911 + int(tid)) & 0xFFFFFFFFFFFFFFFF
    return int(checksum)


def _stable_builtin_hash(payload: bytes) -> int:
    # Deterministic FNV-1a 64-bit hash for reproducible experiments.
    h = 0xCBF29CE484222325
    for b in payload:
        h ^= int(b)
        h = (h * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return int(h)


def hash_chunk_tokens(
    token_ids: Sequence[int],
    *,
    block_size: int,
    model_id: Optional[str] = None,
    dtype: Optional[str] = None,
    kv_layout: Optional[str] = None,
    lora_id: Optional[int] = None,
    kv_cache_group_id: Optional[int] = None,
    hash_algo: str = "sha256",
) -> int:
    algo = normalize_chunk_page_hash_algo(hash_algo)
    payload = _serialize_chunk_hash_input(
        token_ids=token_ids,
        block_size=block_size,
        model_id=model_id,
        dtype=dtype,
        kv_layout=kv_layout,
        lora_id=lora_id,
        kv_cache_group_id=kv_cache_group_id,
    )
    if algo == "sha256":
        digest = hashlib.sha256(payload).digest()
        return int.from_bytes(digest[:8], byteorder="little", signed=False)
    return _stable_builtin_hash(payload)
