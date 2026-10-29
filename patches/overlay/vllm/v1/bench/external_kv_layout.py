# SPDX-License-Identifier: Apache-2.0
"""Layout/shape helpers for external KV injector."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any, Optional
import time

import torch

try:
    import triton
    import triton.language as tl
    _TRITON_AVAILABLE = True
except Exception:
    triton = None
    tl = None
    _TRITON_AVAILABLE = False

_KEEPOLD_CUDA_OPS = None
_KEEPOLD_CUDA_OPS_ERR: Optional[str] = None


def _load_keepold_cuda_ops_module():
    global _KEEPOLD_CUDA_OPS, _KEEPOLD_CUDA_OPS_ERR
    if _KEEPOLD_CUDA_OPS is not None:
        return _KEEPOLD_CUDA_OPS
    if _KEEPOLD_CUDA_OPS_ERR is not None:
        return None
    try:
        # .../KeyChunk/vllm_keychunk/vllm/v1/bench/external_kv_layout.py
        # -> .../KeyChunk/keychunk/keepold/speed/kv_scatter_ops.py
        keychunk_root = Path(__file__).resolve().parents[4]
        speed_dir = keychunk_root / "keychunk" / "keepold" / "speed"
        module_path = speed_dir / "kv_scatter_ops.py"
        if not module_path.exists():
            # Fallback to legacy filename during migration.
            module_path = speed_dir / "keepold_cuda_ops.py"
        if not module_path.exists():
            _KEEPOLD_CUDA_OPS_ERR = "missing_kv_scatter_ops_py"
            return None
        spec = importlib.util.spec_from_file_location(
            "keychunk_keepold_cuda_ops", str(module_path))
        if spec is None or spec.loader is None:
            _KEEPOLD_CUDA_OPS_ERR = "cannot_create_module_spec"
            return None
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _KEEPOLD_CUDA_OPS = mod
        return _KEEPOLD_CUDA_OPS
    except Exception as exc:
        _KEEPOLD_CUDA_OPS_ERR = (
            f"cuda_ops_import_failed:{type(exc).__name__}:{exc}")
        return None


def prewarm_keepold_cuda_op() -> tuple[bool, str]:
    mod = _load_keepold_cuda_ops_module()
    if mod is None:
        return False, _KEEPOLD_CUDA_OPS_ERR or "cuda_ops_module_unavailable"
    get_extension = getattr(mod, "get_extension", None)
    if get_extension is None:
        return False, "missing_get_extension"
    try:
        ext, err = get_extension()
    except Exception as exc:
        return False, f"prewarm_exception:{type(exc).__name__}:{exc}"
    if ext is None:
        return False, err or "extension_unavailable"
    return True, "ok"


if _TRITON_AVAILABLE:

    @triton.jit
    def _old_page_gather_scatter_k_nhd_kernel(
        src_cache_ptr,
        dst_reuse_ptr,
        old_slots_ptr,
        dst_rows_ptr,
        page_size,
        src_s0,
        src_s1,
        src_s2,
        src_s3,
        dst_s0,
        dst_s1,
        dst_s2,
        head_size,
        total_hd,
        num_tokens,
        BLOCK_HD: tl.constexpr,
    ):
        tok = tl.program_id(0)
        hd_blk = tl.program_id(1)
        if tok >= num_tokens:
            return
        hd = hd_blk * BLOCK_HD + tl.arange(0, BLOCK_HD)
        mask = hd < total_hd

        slot = tl.load(old_slots_ptr + tok)
        page = slot // page_size
        off = slot - page * page_size
        h = hd // head_size
        d = hd - h * head_size

        src_off = page * src_s0 + off * src_s1 + h * src_s2 + d * src_s3
        vals = tl.load(src_cache_ptr + src_off, mask=mask, other=0)

        dst_row = tl.load(dst_rows_ptr + tok)
        dst_off = dst_row * dst_s0 + h * dst_s1 + d * dst_s2
        tl.store(dst_reuse_ptr + dst_off, vals, mask=mask)

    @triton.jit
    def _old_page_gather_scatter_k_hnd_kernel(
        src_cache_ptr,
        dst_reuse_ptr,
        old_slots_ptr,
        dst_rows_ptr,
        page_size,
        src_s0,
        src_s1,
        src_s2,
        src_s3,
        dst_s0,
        dst_s1,
        dst_s2,
        head_size,
        total_hd,
        num_tokens,
        BLOCK_HD: tl.constexpr,
    ):
        tok = tl.program_id(0)
        hd_blk = tl.program_id(1)
        if tok >= num_tokens:
            return
        hd = hd_blk * BLOCK_HD + tl.arange(0, BLOCK_HD)
        mask = hd < total_hd

        slot = tl.load(old_slots_ptr + tok)
        page = slot // page_size
        off = slot - page * page_size
        h = hd // head_size
        d = hd - h * head_size

        src_off = page * src_s0 + h * src_s1 + off * src_s2 + d * src_s3
        vals = tl.load(src_cache_ptr + src_off, mask=mask, other=0)

        dst_row = tl.load(dst_rows_ptr + tok)
        dst_off = dst_row * dst_s0 + h * dst_s1 + d * dst_s2
        tl.store(dst_reuse_ptr + dst_off, vals, mask=mask)

    @triton.jit
    def _scatter_rows_to_reuse_k_kernel(
        src_rows_ptr,
        dst_reuse_ptr,
        dst_rows_ptr,
        src_s0,
        src_s1,
        src_s2,
        dst_s0,
        dst_s1,
        dst_s2,
        head_size,
        total_hd,
        num_tokens,
        BLOCK_HD: tl.constexpr,
    ):
        tok = tl.program_id(0)
        hd_blk = tl.program_id(1)
        if tok >= num_tokens:
            return
        hd = hd_blk * BLOCK_HD + tl.arange(0, BLOCK_HD)
        mask = hd < total_hd
        h = hd // head_size
        d = hd - h * head_size

        src_off = tok * src_s0 + h * src_s1 + d * src_s2
        vals = tl.load(src_rows_ptr + src_off, mask=mask, other=0)

        dst_row = tl.load(dst_rows_ptr + tok)
        dst_off = dst_row * dst_s0 + h * dst_s1 + d * dst_s2
        tl.store(dst_reuse_ptr + dst_off, vals, mask=mask)


def reshape_to_kv_heads(
    tensor: torch.Tensor,
    num_kv_heads: int,
    head_size: int,
) -> torch.Tensor:
    if tensor.dim() == 3:
        if tensor.shape[1] != num_kv_heads or tensor.shape[2] != head_size:
            raise ValueError(
                "External KV 3D shape mismatch: "
                f"expected [T,{num_kv_heads},{head_size}], got {tuple(tensor.shape)}"
            )
        return tensor
    if tensor.dim() == 2:
        if tensor.shape[1] != num_kv_heads * head_size:
            raise ValueError(
                "External KV 2D shape mismatch: "
                f"expected [T,{num_kv_heads * head_size}], got {tuple(tensor.shape)}"
            )
        return tensor.view(tensor.shape[0], num_kv_heads, head_size)
    raise ValueError(
        f"Unsupported external KV tensor rank {tensor.dim()} for shape {tuple(tensor.shape)}"
    )


def extract_tokens_from_paged_cache(
    cache: torch.Tensor,
    slots: torch.Tensor,
    num_kv_heads: int,
    head_size: int,
) -> torch.Tensor:
    if cache.dim() != 4:
        raise ValueError(
            "Unsupported paged cache rank for token extraction: "
            f"shape={tuple(cache.shape)}")
    if slots.device != cache.device:
        slots = slots.to(device=cache.device, non_blocking=False)
    slots = slots.to(dtype=torch.long, non_blocking=False).contiguous()
    if slots.numel() == 0:
        return torch.empty((0, num_kv_heads, head_size),
                           dtype=cache.dtype,
                           device=cache.device)

    # NHD layout: [num_pages, page_size, num_kv_heads, head_size]
    if (int(cache.shape[2]) == int(num_kv_heads)
            and int(cache.shape[3]) == int(head_size)):
        flat = cache.reshape(-1, int(cache.shape[2]), int(cache.shape[3]))
        return flat.index_select(0, slots)

    # HND layout: [num_pages, num_kv_heads, page_size, head_size]
    if (int(cache.shape[1]) == int(num_kv_heads)
            and int(cache.shape[3]) == int(head_size)):
        flat = cache.permute(0, 2, 1, 3).contiguous().reshape(
            -1, int(cache.shape[1]), int(cache.shape[3]))
        return flat.index_select(0, slots)

    raise ValueError(
        "Cannot infer paged cache layout for token extraction: "
        f"shape={tuple(cache.shape)}, num_kv_heads={num_kv_heads}, "
        f"head_size={head_size}")


def _fused_extract_scatter_k_from_paged_cache_torch_chunked(
    *,
    key_cache: torch.Tensor,
    old_slots: torch.Tensor,
    dst_reuse_idx: torch.Tensor,
    reuse_k: torch.Tensor,
    num_kv_heads: int,
    head_size: int,
    chunk_size: int,
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "used": False,
        "reason": "unknown",
        "chunk_count": 0,
        "t_total_ms": 0.0,
        "t_gather_ms": 0.0,
        "t_cast_ms": 0.0,
        "t_scatter_ms": 0.0,
    }
    if key_cache.dim() != 4:
        out["reason"] = f"unsupported_cache_rank_{key_cache.dim()}"
        return out
    if old_slots.dim() != 1 or dst_reuse_idx.dim() != 1:
        out["reason"] = "slots_or_dst_not_1d"
        return out
    if int(old_slots.numel()) != int(dst_reuse_idx.numel()):
        out["reason"] = "slots_dst_size_mismatch"
        return out
    if reuse_k.dim() != 3:
        out["reason"] = f"unsupported_reuse_k_rank_{reuse_k.dim()}"
        return out
    if int(reuse_k.shape[1]) != int(num_kv_heads) or int(
            reuse_k.shape[2]) != int(head_size):
        out["reason"] = "reuse_k_shape_mismatch"
        return out
    if key_cache.device != reuse_k.device:
        out["reason"] = "device_mismatch_cache_reuse"
        return out
    if old_slots.numel() == 0:
        out["used"] = True
        out["reason"] = "empty_slots"
        return out

    layout: Optional[str] = None
    if (int(key_cache.shape[2]) == int(num_kv_heads)
            and int(key_cache.shape[3]) == int(head_size)):
        layout = "NHD"
        page_size = int(key_cache.shape[1])
        cache_nhd_view = key_cache
    elif (int(key_cache.shape[1]) == int(num_kv_heads)
            and int(key_cache.shape[3]) == int(head_size)):
        layout = "HND"
        page_size = int(key_cache.shape[2])
        # Keep as a view to avoid full contiguous materialization.
        cache_nhd_view = key_cache.permute(0, 2, 1, 3)
    else:
        out["reason"] = "unsupported_cache_layout"
        return out

    old_slots_dev = old_slots.to(device=reuse_k.device,
                                    dtype=torch.long,
                                    non_blocking=False).contiguous()
    dst_reuse_idx_dev = dst_reuse_idx.to(device=reuse_k.device,
                                            dtype=torch.long,
                                            non_blocking=False).contiguous()
    if int(old_slots_dev.numel()) > 0:
        max_slot = int(old_slots_dev.max().item())
        cap = int(cache_nhd_view.shape[0]) * int(page_size)
        if max_slot >= cap:
            out["reason"] = (
                "slot_out_of_range:"
                f"max_slot={max_slot}, slot_cap={cap}, layout={layout}")
            return out

    t0 = time.perf_counter()
    pages = torch.div(old_slots_dev, page_size, rounding_mode="floor")
    offsets = old_slots_dev - pages * page_size
    total_tokens = int(old_slots_dev.numel())
    step = int(chunk_size) if int(chunk_size) > 0 else total_tokens
    if step <= 0:
        step = total_tokens
    t_gather_ms = 0.0
    t_cast_ms = 0.0
    t_scatter_ms = 0.0
    chunk_count = 0
    for st in range(0, total_tokens, step):
        ed = min(total_tokens, st + step)
        p = pages[st:ed]
        o = offsets[st:ed]
        d = dst_reuse_idx_dev[st:ed]

        tg0 = time.perf_counter()
        # [chunk, num_kv_heads, head_size]
        k_chunk = cache_nhd_view[p, o, :, :]
        tg1 = time.perf_counter()
        t_gather_ms += (tg1 - tg0) * 1000.0

        tc0 = tg1
        if k_chunk.dtype != reuse_k.dtype:
            k_chunk = k_chunk.to(dtype=reuse_k.dtype)
        tc1 = time.perf_counter()
        t_cast_ms += (tc1 - tc0) * 1000.0

        ts0 = tc1
        reuse_k.index_copy_(0, d, k_chunk)
        ts1 = time.perf_counter()
        t_scatter_ms += (ts1 - ts0) * 1000.0
        chunk_count += 1

    t1 = time.perf_counter()
    out.update({
        "used": True,
        "reason": "ok",
        "chunk_count": int(chunk_count),
        "t_total_ms": float((t1 - t0) * 1000.0),
        "t_gather_ms": float(t_gather_ms),
        "t_cast_ms": float(t_cast_ms),
        "t_scatter_ms": float(t_scatter_ms),
    })
    return out


def _fused_extract_scatter_k_from_paged_cache_triton(
    *,
    key_cache: torch.Tensor,
    old_slots: torch.Tensor,
    dst_reuse_idx: torch.Tensor,
    reuse_k: torch.Tensor,
    num_kv_heads: int,
    head_size: int,
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "used": False,
        "reason": "unknown",
        "chunk_count": 1,
        "t_total_ms": 0.0,
        "t_gather_ms": 0.0,
        "t_cast_ms": 0.0,
        "t_scatter_ms": 0.0,
    }
    if not _TRITON_AVAILABLE:
        out["reason"] = "triton_unavailable"
        return out
    if key_cache.device.type != "cuda" or reuse_k.device.type != "cuda":
        out["reason"] = "cuda_only"
        return out
    if key_cache.dtype != reuse_k.dtype:
        out["reason"] = "dtype_mismatch"
        return out
    if key_cache.dim() != 4:
        out["reason"] = f"unsupported_cache_rank_{key_cache.dim()}"
        return out
    if reuse_k.dim() != 3:
        out["reason"] = f"unsupported_reuse_k_rank_{reuse_k.dim()}"
        return out
    if int(reuse_k.shape[1]) != int(num_kv_heads) or int(
            reuse_k.shape[2]) != int(head_size):
        out["reason"] = "reuse_k_shape_mismatch"
        return out
    if old_slots.dim() != 1 or dst_reuse_idx.dim() != 1:
        out["reason"] = "slots_or_dst_not_1d"
        return out
    if int(old_slots.numel()) != int(dst_reuse_idx.numel()):
        out["reason"] = "slots_dst_size_mismatch"
        return out
    if int(old_slots.numel()) == 0:
        out["used"] = True
        out["reason"] = "empty_slots"
        return out

    total_tokens = int(old_slots.numel())
    if total_tokens <= 0:
        out["used"] = True
        out["reason"] = "empty_slots"
        return out

    layout = None
    if (int(key_cache.shape[2]) == int(num_kv_heads)
            and int(key_cache.shape[3]) == int(head_size)):
        layout = "NHD"
        page_size = int(key_cache.shape[1])
    elif (int(key_cache.shape[1]) == int(num_kv_heads)
            and int(key_cache.shape[3]) == int(head_size)):
        layout = "HND"
        page_size = int(key_cache.shape[2])
    else:
        out["reason"] = "unsupported_cache_layout"
        return out

    slot_cap = int(key_cache.shape[0]) * int(page_size)
    if old_slots.device != reuse_k.device:
        old_slots = old_slots.to(device=reuse_k.device, non_blocking=False)
    if dst_reuse_idx.device != reuse_k.device:
        dst_reuse_idx = dst_reuse_idx.to(device=reuse_k.device,
                                         non_blocking=False)
    old_slots_i64 = old_slots.to(dtype=torch.long, non_blocking=False).contiguous()
    dst_idx_i64 = dst_reuse_idx.to(dtype=torch.long,
                                   non_blocking=False).contiguous()
    if int(old_slots_i64.max().item()) >= slot_cap:
        out["reason"] = "slot_out_of_range"
        return out
    if int(dst_idx_i64.max().item()) >= int(reuse_k.shape[0]):
        out["reason"] = "dst_out_of_range"
        return out
    max_i32 = 2**31 - 1
    if int(old_slots_i64.max().item()) > max_i32:
        out["reason"] = "slot_over_int32"
        return out
    if int(dst_idx_i64.max().item()) > max_i32:
        out["reason"] = "dst_over_int32"
        return out
    old_slots_i32 = old_slots_i64.to(dtype=torch.int32)
    dst_idx_i32 = dst_idx_i64.to(dtype=torch.int32)

    total_hd = int(num_kv_heads) * int(head_size)
    block_hd = 128
    grid = (total_tokens, triton.cdiv(total_hd, block_hd))
    t0 = time.perf_counter()
    if layout == "NHD":
        _old_page_gather_scatter_k_nhd_kernel[grid](
            key_cache,
            reuse_k,
            old_slots_i32,
            dst_idx_i32,
            int(page_size),
            int(key_cache.stride(0)),
            int(key_cache.stride(1)),
            int(key_cache.stride(2)),
            int(key_cache.stride(3)),
            int(reuse_k.stride(0)),
            int(reuse_k.stride(1)),
            int(reuse_k.stride(2)),
            int(head_size),
            int(total_hd),
            int(total_tokens),
            BLOCK_HD=block_hd,
            num_warps=4,
        )
    else:
        _old_page_gather_scatter_k_hnd_kernel[grid](
            key_cache,
            reuse_k,
            old_slots_i32,
            dst_idx_i32,
            int(page_size),
            int(key_cache.stride(0)),
            int(key_cache.stride(1)),
            int(key_cache.stride(2)),
            int(key_cache.stride(3)),
            int(reuse_k.stride(0)),
            int(reuse_k.stride(1)),
            int(reuse_k.stride(2)),
            int(head_size),
            int(total_hd),
            int(total_tokens),
            BLOCK_HD=block_hd,
            num_warps=4,
        )
    t1 = time.perf_counter()
    total_ms = (t1 - t0) * 1000.0
    out.update({
        "used": True,
        "reason": "ok",
        "chunk_count": 1,
        "t_total_ms": float(total_ms),
        "t_gather_ms": float(total_ms),
        "t_cast_ms": 0.0,
        "t_scatter_ms": 0.0,
    })
    return out


def _fused_extract_scatter_k_from_paged_cache_cuda_op(
    *,
    key_cache: torch.Tensor,
    old_slots: torch.Tensor,
    dst_reuse_idx: torch.Tensor,
    reuse_k: torch.Tensor,
    num_kv_heads: int,
    head_size: int,
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "used": False,
        "reason": "unknown",
        "chunk_count": 1,
        "t_total_ms": 0.0,
        "t_gather_ms": 0.0,
        "t_cast_ms": 0.0,
        "t_scatter_ms": 0.0,
    }
    mod = _load_keepold_cuda_ops_module()
    if mod is None:
        out["reason"] = _KEEPOLD_CUDA_OPS_ERR or "cuda_ops_module_unavailable"
        return out
    if key_cache.device.type != "cuda" or reuse_k.device.type != "cuda":
        out["reason"] = "cuda_only"
        return out
    if key_cache.dtype != reuse_k.dtype:
        out["reason"] = "dtype_mismatch"
        return out
    if key_cache.dim() != 4:
        out["reason"] = f"unsupported_cache_rank_{key_cache.dim()}"
        return out
    if reuse_k.dim() != 3:
        out["reason"] = f"unsupported_reuse_k_rank_{reuse_k.dim()}"
        return out
    if int(reuse_k.shape[1]) != int(num_kv_heads) or int(
            reuse_k.shape[2]) != int(head_size):
        out["reason"] = "reuse_k_shape_mismatch"
        return out
    if old_slots.dim() != 1 or dst_reuse_idx.dim() != 1:
        out["reason"] = "slots_or_dst_not_1d"
        return out
    if int(old_slots.numel()) != int(dst_reuse_idx.numel()):
        out["reason"] = "slots_dst_size_mismatch"
        return out
    if int(old_slots.numel()) == 0:
        out["used"] = True
        out["reason"] = "empty_slots"
        return out

    if (int(key_cache.shape[2]) == int(num_kv_heads)
            and int(key_cache.shape[3]) == int(head_size)):
        layout_code = 0
        page_size = int(key_cache.shape[1])
    elif (int(key_cache.shape[1]) == int(num_kv_heads)
            and int(key_cache.shape[3]) == int(head_size)):
        layout_code = 1
        page_size = int(key_cache.shape[2])
    else:
        out["reason"] = "unsupported_cache_layout"
        return out
    slot_cap = int(key_cache.shape[0]) * int(page_size)

    old_slots_i64 = old_slots.to(device=reuse_k.device,
                                 dtype=torch.long,
                                 non_blocking=False).contiguous()
    dst_idx_i64 = dst_reuse_idx.to(device=reuse_k.device,
                                   dtype=torch.long,
                                   non_blocking=False).contiguous()
    if int(old_slots_i64.max().item()) >= slot_cap:
        out["reason"] = "slot_out_of_range"
        return out
    if int(dst_idx_i64.max().item()) >= int(reuse_k.shape[0]):
        out["reason"] = "dst_out_of_range"
        return out

    t0 = time.perf_counter()
    ok, reason = mod.old_page_scatter_k(
        key_cache=key_cache,
        old_slots=old_slots_i64,
        dst_idx=dst_idx_i64,
        reuse_k=reuse_k,
        layout_code=int(layout_code),
        page_size=int(page_size),
    )
    t1 = time.perf_counter()
    total_ms = float((t1 - t0) * 1000.0)
    if not ok:
        out["reason"] = reason
        return out
    out.update({
        "used": True,
        "reason": "ok",
        "chunk_count": 1,
        "t_total_ms": total_ms,
        "t_gather_ms": total_ms,
        "t_cast_ms": 0.0,
        "t_scatter_ms": 0.0,
    })
    return out


def _fused_extract_scatter_kv_from_paged_cache_cuda_op(
    *,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    old_slots: torch.Tensor,
    dst_reuse_idx: torch.Tensor,
    reuse_k: torch.Tensor,
    reuse_v: torch.Tensor,
    num_kv_heads: int,
    head_size: int,
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "used": False,
        "reason": "unknown",
        "chunk_count": 1,
        "t_total_ms": 0.0,
        "t_gather_ms": 0.0,
        "t_cast_ms": 0.0,
        "t_scatter_ms": 0.0,
    }
    mod = _load_keepold_cuda_ops_module()
    if mod is None:
        out["reason"] = _KEEPOLD_CUDA_OPS_ERR or "cuda_ops_module_unavailable"
        return out
    if (key_cache.device.type != "cuda" or value_cache.device.type != "cuda"
            or reuse_k.device.type != "cuda" or reuse_v.device.type != "cuda"):
        out["reason"] = "cuda_only"
        return out
    if key_cache.dtype != reuse_k.dtype or value_cache.dtype != reuse_v.dtype:
        out["reason"] = "dtype_mismatch"
        return out
    if key_cache.dim() != 4 or value_cache.dim() != 4:
        out["reason"] = "unsupported_cache_rank"
        return out
    if reuse_k.dim() != 3 or reuse_v.dim() != 3:
        out["reason"] = "unsupported_reuse_rank"
        return out
    if int(reuse_k.shape[1]) != int(num_kv_heads) or int(
            reuse_k.shape[2]) != int(head_size):
        out["reason"] = "reuse_k_shape_mismatch"
        return out
    if int(reuse_v.shape[1]) != int(num_kv_heads) or int(
            reuse_v.shape[2]) != int(head_size):
        out["reason"] = "reuse_v_shape_mismatch"
        return out
    if old_slots.dim() != 1 or dst_reuse_idx.dim() != 1:
        out["reason"] = "slots_or_dst_not_1d"
        return out
    if int(old_slots.numel()) != int(dst_reuse_idx.numel()):
        out["reason"] = "slots_dst_size_mismatch"
        return out
    if int(old_slots.numel()) == 0:
        out["used"] = True
        out["reason"] = "empty_slots"
        return out
    if key_cache.shape != value_cache.shape:
        out["reason"] = "kv_cache_shape_mismatch"
        return out
    if reuse_k.shape != reuse_v.shape:
        out["reason"] = "reuse_kv_shape_mismatch"
        return out

    if (int(key_cache.shape[2]) == int(num_kv_heads)
            and int(key_cache.shape[3]) == int(head_size)):
        layout_code = 0
        page_size = int(key_cache.shape[1])
    elif (int(key_cache.shape[1]) == int(num_kv_heads)
            and int(key_cache.shape[3]) == int(head_size)):
        layout_code = 1
        page_size = int(key_cache.shape[2])
    else:
        out["reason"] = "unsupported_cache_layout"
        return out
    slot_cap = int(key_cache.shape[0]) * int(page_size)
    old_slots_i64 = old_slots.to(device=reuse_k.device,
                                 dtype=torch.long,
                                 non_blocking=False).contiguous()
    dst_idx_i64 = dst_reuse_idx.to(device=reuse_k.device,
                                   dtype=torch.long,
                                   non_blocking=False).contiguous()
    if int(old_slots_i64.max().item()) >= slot_cap:
        out["reason"] = "slot_out_of_range"
        return out
    if int(dst_idx_i64.max().item()) >= int(reuse_k.shape[0]):
        out["reason"] = "dst_out_of_range"
        return out

    t0 = time.perf_counter()
    ok, reason = mod.old_page_scatter_kv(
        key_cache=key_cache,
        value_cache=value_cache,
        old_slots=old_slots_i64,
        dst_idx=dst_idx_i64,
        reuse_k=reuse_k,
        reuse_v=reuse_v,
        layout_code=int(layout_code),
        page_size=int(page_size),
    )
    t1 = time.perf_counter()
    total_ms = float((t1 - t0) * 1000.0)
    if not ok:
        out["reason"] = reason
        return out
    out.update({
        "used": True,
        "reason": "ok",
        "chunk_count": 1,
        "t_total_ms": total_ms,
        "t_gather_ms": total_ms,
        "t_cast_ms": 0.0,
        "t_scatter_ms": 0.0,
    })
    return out


def fused_extract_scatter_kv_from_paged_cache_cuda_op_fast(
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
    """Fast-path fused KV call with prevalidated/prepared tensors.

    Caller guarantees:
    - CUDA tensors
    - dtype/layout/shape compatibility
    - old_slots_i64 and dst_idx_i64 are device long contiguous tensors
    - index bounds were checked upfront
    """
    out: dict[str, Any] = {
        "used": False,
        "reason": "unknown",
        "chunk_count": 1,
        "t_total_ms": 0.0,
        "t_gather_ms": 0.0,
        "t_cast_ms": 0.0,
        "t_scatter_ms": 0.0,
    }
    mod = _load_keepold_cuda_ops_module()
    if mod is None:
        out["reason"] = _KEEPOLD_CUDA_OPS_ERR or "cuda_ops_module_unavailable"
        return out

    t0 = time.perf_counter()
    ok, reason = mod.old_page_scatter_kv(
        key_cache=key_cache,
        value_cache=value_cache,
        old_slots=old_slots_i64,
        dst_idx=dst_idx_i64,
        reuse_k=reuse_k,
        reuse_v=reuse_v,
        layout_code=int(layout_code),
        page_size=int(page_size),
    )
    t1 = time.perf_counter()
    total_ms = float((t1 - t0) * 1000.0)
    if not ok:
        out["reason"] = reason
        return out
    out.update({
        "used": True,
        "reason": "ok",
        "chunk_count": 1,
        "t_total_ms": total_ms,
        "t_gather_ms": total_ms,
        "t_cast_ms": 0.0,
        "t_scatter_ms": 0.0,
    })
    return out


def fused_extract_scatter_kv_from_paged_cache_spans_cuda_op_fast(
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
    """Fast-path span-level fused KV gather/scatter with prepared tensors."""
    out: dict[str, Any] = {
        "used": False,
        "reason": "unknown",
        "chunk_count": 1,
        "span_count": int(span_lengths_i64.numel()),
        "token_count": int(
            span_lengths_i64.sum().item()) if int(span_lengths_i64.numel()) > 0 else 0,
        "t_total_ms": 0.0,
        "t_gather_ms": 0.0,
        "t_cast_ms": 0.0,
        "t_scatter_ms": 0.0,
    }
    mod = _load_keepold_cuda_ops_module()
    if mod is None:
        out["reason"] = _KEEPOLD_CUDA_OPS_ERR or "cuda_ops_module_unavailable"
        return out

    t0 = time.perf_counter()
    ok, reason = mod.old_page_scatter_kv_spans(
        key_cache=key_cache,
        value_cache=value_cache,
        span_block_ids=span_block_ids_i64,
        span_page_offsets=span_page_offsets_i64,
        span_dst_starts=span_dst_starts_i64,
        span_lengths=span_lengths_i64,
        reuse_k=reuse_k,
        reuse_v=reuse_v,
        layout_code=int(layout_code),
        page_size=int(page_size),
    )
    t1 = time.perf_counter()
    total_ms = float((t1 - t0) * 1000.0)
    if not ok:
        out["reason"] = reason
        return out
    out.update({
        "used": True,
        "reason": "ok",
        "chunk_count": 1,
        "span_count": int(span_lengths_i64.numel()),
        "token_count": int(span_lengths_i64.sum().item())
        if int(span_lengths_i64.numel()) > 0 else 0,
        "t_total_ms": total_ms,
        "t_gather_ms": total_ms,
        "t_cast_ms": 0.0,
        "t_scatter_ms": 0.0,
    })
    return out


def fused_extract_scatter_k_from_paged_cache(
    *,
    key_cache: torch.Tensor,
    old_slots: torch.Tensor,
    dst_reuse_idx: torch.Tensor,
    reuse_k: torch.Tensor,
    num_kv_heads: int,
    head_size: int,
    chunk_size: int,
    mode: str = "torch_chunked",
) -> dict[str, Any]:
    mode_norm = str(mode).strip().lower()
    if mode_norm == "cuda_op":
        return _fused_extract_scatter_k_from_paged_cache_cuda_op(
            key_cache=key_cache,
            old_slots=old_slots,
            dst_reuse_idx=dst_reuse_idx,
            reuse_k=reuse_k,
            num_kv_heads=num_kv_heads,
            head_size=head_size,
        )
    if mode_norm == "triton":
        return _fused_extract_scatter_k_from_paged_cache_triton(
            key_cache=key_cache,
            old_slots=old_slots,
            dst_reuse_idx=dst_reuse_idx,
            reuse_k=reuse_k,
            num_kv_heads=num_kv_heads,
            head_size=head_size,
        )
    if mode_norm == "torch_chunked":
        return _fused_extract_scatter_k_from_paged_cache_torch_chunked(
            key_cache=key_cache,
            old_slots=old_slots,
            dst_reuse_idx=dst_reuse_idx,
            reuse_k=reuse_k,
            num_kv_heads=num_kv_heads,
            head_size=head_size,
            chunk_size=chunk_size,
        )
    return {
        "used": False,
        "reason": f"unsupported_mode_{mode_norm}",
        "chunk_count": 0,
        "t_total_ms": 0.0,
        "t_gather_ms": 0.0,
        "t_cast_ms": 0.0,
        "t_scatter_ms": 0.0,
    }


def fused_extract_scatter_kv_from_paged_cache(
    *,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    old_slots: torch.Tensor,
    dst_reuse_idx: torch.Tensor,
    reuse_k: torch.Tensor,
    reuse_v: torch.Tensor,
    num_kv_heads: int,
    head_size: int,
    mode: str = "cuda_op",
) -> dict[str, Any]:
    mode_norm = str(mode).strip().lower()
    if mode_norm == "cuda_op":
        return _fused_extract_scatter_kv_from_paged_cache_cuda_op(
            key_cache=key_cache,
            value_cache=value_cache,
            old_slots=old_slots,
            dst_reuse_idx=dst_reuse_idx,
            reuse_k=reuse_k,
            reuse_v=reuse_v,
            num_kv_heads=num_kv_heads,
            head_size=head_size,
        )
    return {
        "used": False,
        "reason": f"unsupported_mode_{mode_norm}",
        "chunk_count": 0,
        "t_total_ms": 0.0,
        "t_gather_ms": 0.0,
        "t_cast_ms": 0.0,
        "t_scatter_ms": 0.0,
    }


def fused_scatter_rows_to_reuse_k(
    *,
    src_rows: torch.Tensor,
    dst_reuse_idx: torch.Tensor,
    reuse_k: torch.Tensor,
    mode: str = "triton",
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "used": False,
        "reason": "unknown",
        "t_total_ms": 0.0,
    }
    mode_norm = str(mode).strip().lower()
    if mode_norm == "cuda_op":
        mod = _load_keepold_cuda_ops_module()
        if mod is None:
            out["reason"] = (_KEEPOLD_CUDA_OPS_ERR
                             or "cuda_ops_module_unavailable")
            return out
        if (src_rows.device.type != "cuda" or reuse_k.device.type != "cuda"
                or dst_reuse_idx.device.type != "cuda"):
            out["reason"] = "cuda_only"
            return out
        if src_rows.dtype != reuse_k.dtype:
            out["reason"] = "dtype_mismatch"
            return out
        if src_rows.dim() != 3 or reuse_k.dim() != 3:
            out["reason"] = "rank_mismatch"
            return out
        if int(src_rows.shape[1]) != int(reuse_k.shape[1]) or int(
                src_rows.shape[2]) != int(reuse_k.shape[2]):
            out["reason"] = "shape_mismatch"
            return out
        if dst_reuse_idx.dim() != 1 or int(dst_reuse_idx.numel()) != int(
                src_rows.shape[0]):
            out["reason"] = "dst_size_mismatch"
            return out
        if int(src_rows.shape[0]) == 0:
            out["used"] = True
            out["reason"] = "empty_rows"
            return out
        dst_idx_i64 = dst_reuse_idx.to(device=src_rows.device,
                                       dtype=torch.long,
                                       non_blocking=False).contiguous()
        if int(dst_idx_i64.max().item()) >= int(reuse_k.shape[0]):
            out["reason"] = "dst_out_of_range"
            return out
        t0 = time.perf_counter()
        ok, reason = mod.scatter_rows_to_reuse_k(
            src_rows=src_rows,
            dst_idx=dst_idx_i64,
            reuse_k=reuse_k,
        )
        t1 = time.perf_counter()
        if not ok:
            out["reason"] = reason
            return out
        out.update({
            "used": True,
            "reason": "ok",
            "t_total_ms": float((t1 - t0) * 1000.0),
        })
        return out

    if mode_norm != "triton":
        out["reason"] = f"unsupported_mode_{mode_norm}"
        return out
    if not _TRITON_AVAILABLE:
        out["reason"] = "triton_unavailable"
        return out
    if (src_rows.device.type != "cuda" or reuse_k.device.type != "cuda"
            or dst_reuse_idx.device.type != "cuda"):
        out["reason"] = "cuda_only"
        return out
    if src_rows.dtype != reuse_k.dtype:
        out["reason"] = "dtype_mismatch"
        return out
    if src_rows.dim() != 3 or reuse_k.dim() != 3:
        out["reason"] = "rank_mismatch"
        return out
    if int(src_rows.shape[1]) != int(reuse_k.shape[1]) or int(
            src_rows.shape[2]) != int(reuse_k.shape[2]):
        out["reason"] = "shape_mismatch"
        return out
    if dst_reuse_idx.dim() != 1 or int(dst_reuse_idx.numel()) != int(
            src_rows.shape[0]):
        out["reason"] = "dst_size_mismatch"
        return out
    if int(src_rows.shape[0]) == 0:
        out["used"] = True
        out["reason"] = "empty_rows"
        return out

    dst_idx_i64 = dst_reuse_idx.to(dtype=torch.long, non_blocking=False).contiguous()
    if int(dst_idx_i64.max().item()) >= int(reuse_k.shape[0]):
        out["reason"] = "dst_out_of_range"
        return out
    max_i32 = 2**31 - 1
    if int(dst_idx_i64.max().item()) > max_i32:
        out["reason"] = "dst_over_int32"
        return out
    dst_idx_i32 = dst_idx_i64.to(dtype=torch.int32)
    num_tokens = int(src_rows.shape[0])
    head_size = int(src_rows.shape[2])
    total_hd = int(src_rows.shape[1]) * int(src_rows.shape[2])
    block_hd = 128
    grid = (num_tokens, triton.cdiv(total_hd, block_hd))
    t0 = time.perf_counter()
    _scatter_rows_to_reuse_k_kernel[grid](
        src_rows,
        reuse_k,
        dst_idx_i32,
        int(src_rows.stride(0)),
        int(src_rows.stride(1)),
        int(src_rows.stride(2)),
        int(reuse_k.stride(0)),
        int(reuse_k.stride(1)),
        int(reuse_k.stride(2)),
        int(head_size),
        int(total_hd),
        int(num_tokens),
        BLOCK_HD=block_hd,
        num_warps=4,
    )
    t1 = time.perf_counter()
    out.update({
        "used": True,
        "reason": "ok",
        "t_total_ms": float((t1 - t0) * 1000.0),
    })
    return out


def fused_scatter_rows_to_reuse_kv(
    *,
    src_k_rows: torch.Tensor,
    src_v_rows: torch.Tensor,
    dst_reuse_idx: torch.Tensor,
    reuse_k: torch.Tensor,
    reuse_v: torch.Tensor,
    mode: str = "cuda_op",
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "used": False,
        "reason": "unknown",
        "t_total_ms": 0.0,
    }
    mode_norm = str(mode).strip().lower()
    if mode_norm != "cuda_op":
        out["reason"] = f"unsupported_mode_{mode_norm}"
        return out
    mod = _load_keepold_cuda_ops_module()
    if mod is None:
        out["reason"] = (_KEEPOLD_CUDA_OPS_ERR or "cuda_ops_module_unavailable")
        return out
    if (src_k_rows.device.type != "cuda" or src_v_rows.device.type != "cuda"
            or reuse_k.device.type != "cuda" or reuse_v.device.type != "cuda"
            or dst_reuse_idx.device.type != "cuda"):
        out["reason"] = "cuda_only"
        return out
    if src_k_rows.dtype != reuse_k.dtype or src_v_rows.dtype != reuse_v.dtype:
        out["reason"] = "dtype_mismatch"
        return out
    if src_k_rows.dim() != 3 or src_v_rows.dim() != 3:
        out["reason"] = "rank_mismatch"
        return out
    if src_k_rows.shape != src_v_rows.shape:
        out["reason"] = "src_kv_shape_mismatch"
        return out
    if reuse_k.shape != reuse_v.shape:
        out["reason"] = "reuse_kv_shape_mismatch"
        return out
    if int(src_k_rows.shape[1]) != int(reuse_k.shape[1]) or int(
            src_k_rows.shape[2]) != int(reuse_k.shape[2]):
        out["reason"] = "shape_mismatch"
        return out
    if dst_reuse_idx.dim() != 1 or int(dst_reuse_idx.numel()) != int(
            src_k_rows.shape[0]):
        out["reason"] = "dst_size_mismatch"
        return out
    if int(src_k_rows.shape[0]) == 0:
        out["used"] = True
        out["reason"] = "empty_rows"
        return out
    dst_idx_i64 = dst_reuse_idx.to(device=src_k_rows.device,
                                   dtype=torch.long,
                                   non_blocking=False).contiguous()
    if int(dst_idx_i64.max().item()) >= int(reuse_k.shape[0]):
        out["reason"] = "dst_out_of_range"
        return out
    t0 = time.perf_counter()
    ok, reason = mod.scatter_rows_to_reuse_kv(
        src_k_rows=src_k_rows,
        src_v_rows=src_v_rows,
        dst_idx=dst_idx_i64,
        reuse_k=reuse_k,
        reuse_v=reuse_v,
    )
    t1 = time.perf_counter()
    if not ok:
        out["reason"] = reason
        return out
    out.update({
        "used": True,
        "reason": "ok",
        "t_total_ms": float((t1 - t0) * 1000.0),
    })
    return out


def fused_scatter_rows_to_reuse_kv_cuda_op_fast(
    *,
    src_k_rows: torch.Tensor,
    src_v_rows: torch.Tensor,
    dst_idx_i64: torch.Tensor,
    reuse_k: torch.Tensor,
    reuse_v: torch.Tensor,
) -> dict[str, Any]:
    """Fast-path scatter K/V rows with prevalidated/prepared inputs."""
    out: dict[str, Any] = {
        "used": False,
        "reason": "unknown",
        "t_total_ms": 0.0,
    }
    mod = _load_keepold_cuda_ops_module()
    if mod is None:
        out["reason"] = (_KEEPOLD_CUDA_OPS_ERR or "cuda_ops_module_unavailable")
        return out
    t0 = time.perf_counter()
    ok, reason = mod.scatter_rows_to_reuse_kv(
        src_k_rows=src_k_rows,
        src_v_rows=src_v_rows,
        dst_idx=dst_idx_i64,
        reuse_k=reuse_k,
        reuse_v=reuse_v,
    )
    t1 = time.perf_counter()
    if not ok:
        out["reason"] = reason
        return out
    out.update({
        "used": True,
        "reason": "ok",
        "t_total_ms": float((t1 - t0) * 1000.0),
    })
    return out
