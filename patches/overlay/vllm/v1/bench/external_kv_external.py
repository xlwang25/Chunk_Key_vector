# SPDX-License-Identifier: Apache-2.0
"""External KV injector benchmark path implementation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional
import os
import sys
import time

import torch

from vllm import _custom_ops as ops
from vllm.logger import init_logger
from vllm.model_executor.layers.rotary_embedding.common import apply_rotary_emb_torch

from .external_kv_common import (
    _ROPE_IMPL_NONE,
    _ROPE_IMPL_TORCH_TWO_STEP,
    _ROPE_IMPL_LMCACHE_WRAPPER,
    _ROPE_IMPL_LMCACHE_COPS_DIRECT,
    _normalize_rope_impl,
    _env_flag,
    _extract_layer_id,
)
from .external_kv_layout import reshape_to_kv_heads
from .external_kv_metrics import append_metrics_jsonl

logger = init_logger(__name__)

_INJECTOR_SINGLETON: Optional["ExternalKVInjector"] = None
_INJECTOR_INITED = False

@dataclass
class ExternalKVInjectorConfig:
    enabled: bool
    pt_path: Optional[Path]
    layer_id: Optional[int]
    max_tokens: int
    metrics_path: Optional[Path]
    inject_once: bool
    prefill_only: bool
    rope_impl: str

    @classmethod
    def from_env(cls) -> "ExternalKVInjectorConfig":
        enabled = _env_flag("VLLM_EXTERNAL_KV_BENCH", False)
        pt_path_raw = os.getenv("VLLM_EXTERNAL_KV_PT")
        layer_raw = os.getenv("VLLM_EXTERNAL_KV_LAYER_ID")
        max_tokens = int(os.getenv("VLLM_EXTERNAL_KV_MAX_TOKENS", "0"))
        metrics_raw = os.getenv("VLLM_EXTERNAL_KV_METRICS_PATH")
        inject_once = _env_flag("VLLM_EXTERNAL_KV_INJECT_ONCE", True)
        prefill_only = _env_flag("VLLM_EXTERNAL_KV_PREFILL_ONLY", True)
        rope_impl = _normalize_rope_impl(
            os.getenv("VLLM_EXTERNAL_KV_ROPE_IMPL", _ROPE_IMPL_TORCH_TWO_STEP))
        if rope_impl not in (
            _ROPE_IMPL_NONE,
            _ROPE_IMPL_TORCH_TWO_STEP,
            _ROPE_IMPL_LMCACHE_WRAPPER,
            _ROPE_IMPL_LMCACHE_COPS_DIRECT,
        ):
            raise ValueError(
                "Invalid VLLM_EXTERNAL_KV_ROPE_IMPL: "
                f"{rope_impl}. Expected one of "
                f"{_ROPE_IMPL_NONE}, {_ROPE_IMPL_TORCH_TWO_STEP}, "
                f"{_ROPE_IMPL_LMCACHE_WRAPPER}, {_ROPE_IMPL_LMCACHE_COPS_DIRECT}."
            )

        pt_path = Path(pt_path_raw) if pt_path_raw else None
        layer_id = int(layer_raw) if layer_raw is not None else None
        metrics_path = Path(metrics_raw) if metrics_raw else None

        return cls(
            enabled=enabled,
            pt_path=pt_path,
            layer_id=layer_id,
            max_tokens=max_tokens,
            metrics_path=metrics_path,
            inject_once=inject_once,
            prefill_only=prefill_only,
            rope_impl=rope_impl,
        )


class ExternalKVInjector:
    """Minimal external KV injector for vLLM v1."""

    def __init__(self, config: ExternalKVInjectorConfig):
        self.config = config
        self._done = False

        self._source_loaded = False
        self._src_k: Optional[torch.Tensor] = None
        self._src_v: Optional[torch.Tensor] = None
        self._source_tokens = 0
        self._src_layer_kv: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        self._src_layer_tokens: dict[int, int] = {}
        self._injected_layers: set[int] = set()

        self._k_staging: Optional[torch.Tensor] = None
        self._v_staging: Optional[torch.Tensor] = None
        # Cache (cos, sin, cache_len) keyed by (head_size, dtype, device).
        self._rope_cache: dict[tuple[int, torch.dtype, str],
                               tuple[torch.Tensor, torch.Tensor, int]] = {}
        self._lmcache_fused_rope_cache: dict[
            tuple[int, torch.dtype, bool], Any] = {}
        self._lmcache_cops_module: Optional[Any] = None
        self._cops_cos_sin_cache: dict[tuple[int, torch.dtype, str],
                                       tuple[torch.Tensor, int]] = {}
        self._direct_shape_logged: set[tuple[Any, ...]] = set()

        if self.config.metrics_path is not None:
            self.config.metrics_path.parent.mkdir(parents=True, exist_ok=True)

    def reconfigure(self, config: ExternalKVInjectorConfig) -> None:
        """Reconfigure source/settings for the next benchmark run."""
        self.config = config
        self._done = False
        self._source_loaded = False
        self._src_k = None
        self._src_v = None
        self._source_tokens = 0
        self._src_layer_kv = {}
        self._src_layer_tokens = {}
        self._injected_layers = set()
        if self.config.metrics_path is not None:
            self.config.metrics_path.parent.mkdir(parents=True, exist_ok=True)

    @classmethod
    def build_from_env(cls) -> Optional["ExternalKVInjector"]:
        cfg = ExternalKVInjectorConfig.from_env()
        if not cfg.enabled:
            return None

        if cfg.pt_path is None:
            logger.warning(
                "External KV bench is enabled but missing required env: "
                "VLLM_EXTERNAL_KV_PT.")
            return None

        if not cfg.pt_path.exists():
            logger.warning("External KV PT file not found: %s", cfg.pt_path)
            return None

        if cfg.layer_id is None:
            logger.info(
                "External KV injector enabled (layerwise): pt=%s max_tokens=%d",
                cfg.pt_path,
                cfg.max_tokens,
            )
        else:
            logger.info(
                "External KV injector enabled: pt=%s layer=%d max_tokens=%d",
                cfg.pt_path,
                cfg.layer_id,
                cfg.max_tokens,
            )
        return cls(cfg)

    def _load_source_once(self) -> None:
        if self._source_loaded:
            return

        assert self.config.pt_path is not None

        obj = torch.load(self.config.pt_path, map_location="cpu")
        if not isinstance(obj, dict):
            raise ValueError(
                f"Unsupported external KV file format: {type(obj)} "
                f"(file={self.config.pt_path})")

        if "layer_kv" not in obj:
            raise KeyError(
                f"External KV file misses key 'layer_kv': {self.config.pt_path}")

        layer_kv = obj["layer_kv"]
        if not isinstance(layer_kv, dict):
            raise ValueError("'layer_kv' should be a dict in external KV file")

        if self.config.layer_id is not None:
            entry: Optional[dict[str, Any]] = None
            if self.config.layer_id in layer_kv:
                entry = layer_kv[self.config.layer_id]
            elif str(self.config.layer_id) in layer_kv:
                entry = layer_kv[str(self.config.layer_id)]

            if entry is None:
                raise KeyError(
                    f"Layer {self.config.layer_id} not found in {self.config.pt_path}")
            if not isinstance(entry, dict) or "k" not in entry or "v" not in entry:
                raise ValueError(
                    "External KV layer entry should contain keys 'k' and 'v'")

            src_k = entry["k"]
            src_v = entry["v"]
            if (not isinstance(src_k, torch.Tensor)
                    or not isinstance(src_v, torch.Tensor)):
                raise ValueError("External KV values must be torch tensors")

            src_k = src_k.detach().cpu().contiguous()
            src_v = src_v.detach().cpu().contiguous()
            if src_k.shape[0] != src_v.shape[0]:
                raise ValueError("External K/V token dims mismatch")

            self._src_k = src_k
            self._src_v = src_v
            self._source_tokens = int(src_k.shape[0])
        else:
            layer_items: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
            layer_tokens: dict[int, int] = {}
            for lid_raw, entry in layer_kv.items():
                try:
                    lid = int(lid_raw)
                except Exception as e:
                    raise ValueError(
                        f"Invalid layer id key in layer_kv: {lid_raw}") from e
                if not isinstance(entry, dict) or "k" not in entry or "v" not in entry:
                    raise ValueError(
                        f"External KV layer entry should contain keys 'k' and 'v' (layer={lid})"
                    )
                src_k = entry["k"]
                src_v = entry["v"]
                if (not isinstance(src_k, torch.Tensor)
                        or not isinstance(src_v, torch.Tensor)):
                    raise ValueError(
                        f"External KV values must be torch tensors (layer={lid})")
                src_k = src_k.detach().cpu().contiguous()
                src_v = src_v.detach().cpu().contiguous()
                if src_k.shape[0] != src_v.shape[0]:
                    raise ValueError(
                        f"External K/V token dims mismatch (layer={lid})")
                layer_items[lid] = (src_k, src_v)
                layer_tokens[lid] = int(src_k.shape[0])

            if not layer_items:
                raise ValueError(
                    f"No layer_kv entries found in {self.config.pt_path}")
            self._src_layer_kv = layer_items
            self._src_layer_tokens = layer_tokens

        self._source_loaded = True

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

    @staticmethod
    def _default_max_position_for_lmcache() -> int:
        # Keep this bounded and stable for benchmark use.
        return 131072

    def _get_lmcache_fused_rope(self, head_size: int,
                                dtype: torch.dtype) -> Any:
        cache_key = (head_size, dtype, True)
        fused = self._lmcache_fused_rope_cache.get(cache_key)
        if fused is not None:
            return fused

        try:
            from lmcache.v1.compute.positional_encoding import get_fused_rope
        except Exception as e:
            raise RuntimeError(
                "Failed to import LMCache fused rope. "
                "Please ensure LMCache path and c_ops are available."
            ) from e

        fused = get_fused_rope(
            head_size=head_size,
            rotary_dim=head_size,
            max_position=self._default_max_position_for_lmcache(),
            base=10000.0,
            is_neox_style=True,
            rope_scaling=None,
            dtype=dtype,
            partial_rotary_factor=1.0,
        )
        if fused is None:
            raise RuntimeError("LMCache get_fused_rope returned None")
        self._lmcache_fused_rope_cache[cache_key] = fused
        return fused

    def _get_lmcache_cops(self) -> Any:
        if self._lmcache_cops_module is not None:
            return self._lmcache_cops_module
        try:
            import lmcache.c_ops as lmc_ops
        except Exception as e:
            raise RuntimeError(
                "Failed to import lmcache.c_ops for direct fused rope mode."
            ) from e
        self._lmcache_cops_module = lmc_ops
        return lmc_ops

    def _get_rope_cos_sin_cache(
        self,
        cache_len: int,
        head_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, float, bool]:
        cache_key = (head_size, dtype, str(device))
        cache = self._rope_cache.get(cache_key)
        if cache is not None:
            cos_cache, sin_cache, existing_len = cache
            if existing_len >= cache_len:
                return cos_cache, sin_cache, 0.0, True

        # Keep this benchmark path minimal:
        # - full rotary_dim == head_size
        # - base == 10000
        torch.cuda.synchronize(device)
        t_gen0 = time.perf_counter()
        inv_freq = 1.0 / (10000.0**(torch.arange(
            0, head_size, 2, dtype=torch.float32, device=device) / head_size))
        positions = torch.arange(cache_len, dtype=torch.float32, device=device)
        freqs = torch.einsum("i,j->ij", positions, inv_freq)
        cos_cache = freqs.cos().to(dtype=dtype)
        sin_cache = freqs.sin().to(dtype=dtype)
        torch.cuda.synchronize(device)
        t_gen1 = time.perf_counter()
        self._rope_cache[cache_key] = (cos_cache, sin_cache, cache_len)
        return cos_cache, sin_cache, (t_gen1 - t_gen0) * 1000.0, False

    def _get_cops_cos_sin_cache(
        self,
        max_position: int,
        head_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> tuple[torch.Tensor, float, bool]:
        cache_key = (head_size, dtype, str(device))
        cache = self._cops_cos_sin_cache.get(cache_key)
        if cache is not None:
            cache_tensor, cache_len = cache
            if cache_len >= max_position:
                return cache_tensor, 0.0, True

        cos_cache, sin_cache, t_gen_ms, cos_sin_cache_hit = (
            self._get_rope_cos_sin_cache(
                max_position,
                head_size,
                dtype,
                device,
            ))
        cos_sin_cache = torch.cat((cos_cache, sin_cache), dim=-1).contiguous()
        self._cops_cos_sin_cache[cache_key] = (cos_sin_cache, max_position)
        return cos_sin_cache, t_gen_ms, cos_sin_cache_hit

    def maybe_inject(
        self,
        layer_name: str,
        key: torch.Tensor,
        value: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
        num_actual_tokens: int,
        kv_cache_dtype: str,
        k_scale: torch.Tensor,
        v_scale: torch.Tensor,
    ) -> bool:
        if self.config.inject_once and self._done:
            return False

        layer_id = _extract_layer_id(layer_name)
        if layer_id is None:
            return False

        if self.config.layer_id is not None:
            if layer_id != self.config.layer_id:
                return False
        else:
            if layer_id in self._injected_layers:
                return False

        # Decode-only steps are usually length=1; skip by default.
        if self.config.prefill_only and num_actual_tokens <= 1:
            return False
        if num_actual_tokens <= 0:
            return False

        self._load_source_once()
        if self.config.layer_id is not None:
            assert self._src_k is not None and self._src_v is not None
            src_k_raw = self._src_k
            src_v_raw = self._src_v
            source_tokens = self._source_tokens
        else:
            src_pair = self._src_layer_kv.get(layer_id)
            if src_pair is None:
                return False
            src_k_raw, src_v_raw = src_pair
            source_tokens = self._src_layer_tokens.get(layer_id, 0)

        # Use only valid slots for the current forward.
        slots_all = slot_mapping[:num_actual_tokens]
        slots_all = slots_all[slots_all >= 0]
        if slots_all.numel() == 0:
            return False

        num_kv_heads = int(key.shape[1])
        head_size = int(key.shape[2])
        src_k = self._reshape_to_kv_heads(src_k_raw, num_kv_heads, head_size)
        src_v = self._reshape_to_kv_heads(src_v_raw, num_kv_heads, head_size)

        inject_tokens = min(
            int(slots_all.numel()),
            int(src_k.shape[0]),
            int(src_v.shape[0]),
        )
        if self.config.max_tokens > 0:
            inject_tokens = min(inject_tokens, self.config.max_tokens)
        if inject_tokens <= 0:
            return False

        src_k_slice = src_k[:inject_tokens].contiguous()
        src_v_slice = src_v[:inject_tokens].contiguous()
        if src_k_slice.dtype != key.dtype:
            src_k_slice = src_k_slice.to(dtype=key.dtype)
        if src_v_slice.dtype != value.dtype:
            src_v_slice = src_v_slice.to(dtype=value.dtype)

        target_k_shape = (inject_tokens, num_kv_heads, head_size)
        target_v_shape = (inject_tokens, int(value.shape[1]), int(value.shape[2]))

        if (
            self._k_staging is None
            or self._k_staging.shape != target_k_shape
            or self._k_staging.dtype != key.dtype
            or self._k_staging.device != key.device
        ):
            self._k_staging = torch.empty(target_k_shape, dtype=key.dtype, device=key.device)
        if (
            self._v_staging is None
            or self._v_staging.shape != target_v_shape
            or self._v_staging.dtype != value.dtype
            or self._v_staging.device != value.device
        ):
            self._v_staging = torch.empty(
                target_v_shape,
                dtype=value.dtype,
                device=value.device,
            )

        # 1) CPU -> GPU staging copy.
        torch.cuda.synchronize(key.device)
        t0 = time.perf_counter()
        self._k_staging.copy_(src_k_slice, non_blocking=False)
        self._v_staging.copy_(src_v_slice, non_blocking=False)
        torch.cuda.synchronize(key.device)
        t1 = time.perf_counter()

        rope_impl = _normalize_rope_impl(self.config.rope_impl)
        t_cos_sin_prepare_ms = 0.0
        t_cos_sin_gen_ms = 0.0
        cos_sin_cache_hit = True
        if rope_impl == _ROPE_IMPL_NONE:
            t_derope_ms = 0.0
            t_rerope_ms = 0.0
            t_rope_fused_wrapper_ms = 0.0
            t_rope_fused_direct_ms = 0.0
            t_rope_fused_ms = 0.0
            t_rope_total_ms = 0.0
            t2 = t1
        else:
            # 2) De-RoPE(old position) then Re-RoPE(new position) on K only.
            # First version uses simulated positions: arange(0, inject_tokens).
            old_positions = torch.arange(
                inject_tokens, dtype=torch.long, device=key.device)
            new_positions = torch.arange(
                inject_tokens, dtype=torch.long, device=key.device)

            # Ensure all prep/caching is done before timing for direct c_ops mode.
            direct_cos_sin_cache = None
            if rope_impl == _ROPE_IMPL_LMCACHE_COPS_DIRECT:
                _ = self._get_lmcache_cops()
                max_position = int(
                    max(old_positions.max().item(), new_positions.max().item()) + 1)
                torch.cuda.synchronize(key.device)
                t_prep0 = time.perf_counter()
                direct_cos_sin_cache, t_cos_sin_gen_ms, cos_sin_cache_hit = self._get_cops_cos_sin_cache(
                    max_position=max_position,
                    head_size=head_size,
                    dtype=self._k_staging.dtype,
                    device=key.device,
                )
                torch.cuda.synchronize(key.device)
                t_prep1 = time.perf_counter()
                t_cos_sin_prepare_ms = (t_prep1 - t_prep0) * 1000.0

            torch.cuda.synchronize(key.device)
            t_rope0 = time.perf_counter()

            if rope_impl == _ROPE_IMPL_TORCH_TWO_STEP:
                rope_cache_len = inject_tokens
                cos_cache, sin_cache, t_cos_sin_gen_ms, cos_sin_cache_hit = self._get_rope_cos_sin_cache(
                    rope_cache_len, head_size, self._k_staging.dtype, key.device)
                cos_old = cos_cache.index_select(0, old_positions)
                sin_old = sin_cache.index_select(0, old_positions)
                cos_new = cos_cache.index_select(0, new_positions)
                sin_new = sin_cache.index_select(0, new_positions)

                self._k_staging.copy_(
                    apply_rotary_emb_torch(
                        self._k_staging,
                        cos_old,
                        -sin_old,
                        is_neox_style=True,
                    ))
                torch.cuda.synchronize(key.device)
                t_rope1 = time.perf_counter()
                self._k_staging.copy_(
                    apply_rotary_emb_torch(
                        self._k_staging,
                        cos_new,
                        sin_new,
                        is_neox_style=True,
                    ))
                torch.cuda.synchronize(key.device)
                t2 = time.perf_counter()
                t_derope_ms = (t_rope1 - t_rope0) * 1000.0
                t_rerope_ms = (t2 - t_rope1) * 1000.0
                t_rope_fused_wrapper_ms = 0.0
                t_rope_fused_direct_ms = 0.0
            elif rope_impl == _ROPE_IMPL_LMCACHE_WRAPPER:
                fused_rope = self._get_lmcache_fused_rope(head_size=head_size,
                                                          dtype=self._k_staging.dtype)
                fused_rope(old_positions, new_positions,
                           self._k_staging.view(inject_tokens, -1))
                torch.cuda.synchronize(key.device)
                t2 = time.perf_counter()
                t_derope_ms = 0.0
                t_rerope_ms = 0.0
                t_rope_fused_wrapper_ms = (t2 - t_rope0) * 1000.0
                t_rope_fused_direct_ms = 0.0
            elif rope_impl == _ROPE_IMPL_LMCACHE_COPS_DIRECT:
                assert direct_cos_sin_cache is not None
                lmc_ops = self._get_lmcache_cops()
                lmc_ops.rotary_embedding_k_fused(
                    old_positions,
                    new_positions,
                    self._k_staging,
                    head_size,
                    direct_cos_sin_cache,
                    True,
                )
                torch.cuda.synchronize(key.device)
                t2 = time.perf_counter()
                t_derope_ms = 0.0
                t_rerope_ms = 0.0
                t_rope_fused_wrapper_ms = 0.0
                t_rope_fused_direct_ms = (t2 - t_rope0) * 1000.0

                shape_key = (
                    tuple(self._k_staging.shape),
                    tuple(old_positions.shape),
                    tuple(new_positions.shape),
                    tuple(direct_cos_sin_cache.shape),
                    str(self._k_staging.dtype),
                    str(old_positions.dtype),
                    str(new_positions.dtype),
                    str(direct_cos_sin_cache.dtype),
                )
                if shape_key not in self._direct_shape_logged:
                    self._direct_shape_logged.add(shape_key)
                    logger.info(
                        "LMCache c_ops direct fused rope inputs: "
                        "key.shape=%s old_positions.shape=%s new_positions.shape=%s "
                        "cos_sin_cache.shape=%s key.dtype=%s pos.dtype=%s cache.dtype=%s",
                        tuple(self._k_staging.shape),
                        tuple(old_positions.shape),
                        tuple(new_positions.shape),
                        tuple(direct_cos_sin_cache.shape),
                        self._k_staging.dtype,
                        old_positions.dtype,
                        direct_cos_sin_cache.dtype,
                    )
            else:
                raise ValueError(f"Unsupported rope_impl: {rope_impl}")

            t_rope_total_ms = (t2 - t_rope0) * 1000.0
            t_rope_fused_ms = t_rope_fused_wrapper_ms + t_rope_fused_direct_ms

        # 3) GPU staging -> vLLM paged KV.
        slots = slots_all[:inject_tokens].contiguous()
        ops.reshape_and_cache_flash(
            self._k_staging,
            self._v_staging,
            key_cache,
            value_cache,
            slots,
            kv_cache_dtype,
            k_scale,
            v_scale,
        )
        torch.cuda.synchronize(key.device)
        t3 = time.perf_counter()

        metrics = {
            "layer_name": layer_name,
            "layer_id": layer_id,
            "source_file": str(self.config.pt_path) if self.config.pt_path else None,
            "source_tokens": source_tokens,
            "num_actual_tokens": int(num_actual_tokens),
            "num_slots_valid": int(slots_all.numel()),
            "num_tokens_injected": int(inject_tokens),
            "slot_min": int(slots.min().item()) if slots.numel() else None,
            "slot_max": int(slots.max().item()) if slots.numel() else None,
            "rope_impl": rope_impl,
            "t_cos_sin_prepare_ms": t_cos_sin_prepare_ms,
            "t_cos_sin_gen_ms": t_cos_sin_gen_ms,
            "cos_sin_cache_hit": cos_sin_cache_hit,
            "t_h2d_copy_ms": (t1 - t0) * 1000.0,
            "t_derope_ms": t_derope_ms,
            "t_rerope_ms": t_rerope_ms,
            "t_rope_fused_wrapper_ms": t_rope_fused_wrapper_ms,
            "t_rope_fused_direct_ms": t_rope_fused_direct_ms,
            "t_rope_fused_ms": t_rope_fused_ms,
            "t_rope_total_ms": t_rope_total_ms,
            "t_page_write_ms": (t3 - t2) * 1000.0,
            "t_total_inject_ms": (t3 - t0) * 1000.0,
            "kv_dtype_runtime": str(key.dtype),
            "kv_cache_dtype": kv_cache_dtype,
            "device": str(key.device),
            "ts_unix": time.time(),
        }
        self._append_metrics(metrics)
        logger.info(
            "External KV injected at %s (layer=%d, tokens=%d): "
            "rope_impl=%s cos_prepare=%.3f ms cos_gen=%.3f ms cos_hit=%s "
            "h2d=%.3f ms derope=%.3f ms rerope=%.3f ms "
            "rope_fused_wrapper=%.3f ms rope_fused_direct=%.3f ms "
            "rope_fused=%.3f ms rope_total=%.3f ms "
            "page_write=%.3f ms total=%.3f ms",
            layer_name,
            layer_id,
            inject_tokens,
            rope_impl,
            metrics["t_cos_sin_prepare_ms"],
            metrics["t_cos_sin_gen_ms"],
            metrics["cos_sin_cache_hit"],
            metrics["t_h2d_copy_ms"],
            metrics["t_derope_ms"],
            metrics["t_rerope_ms"],
            metrics["t_rope_fused_wrapper_ms"],
            metrics["t_rope_fused_direct_ms"],
            metrics["t_rope_fused_ms"],
            metrics["t_rope_total_ms"],
            metrics["t_page_write_ms"],
            metrics["t_total_inject_ms"],
        )

        if self.config.layer_id is None:
            self._injected_layers.add(layer_id)
            if self.config.inject_once and len(self._injected_layers) >= len(
                    self._src_layer_kv):
                self._done = True
        elif self.config.inject_once:
            self._done = True

        return True


def get_external_kv_injector() -> Optional[ExternalKVInjector]:
    global _INJECTOR_SINGLETON, _INJECTOR_INITED

    if _INJECTOR_INITED:
        return _INJECTOR_SINGLETON

    _INJECTOR_SINGLETON = ExternalKVInjector.build_from_env()
    _INJECTOR_INITED = True
    return _INJECTOR_SINGLETON


def configure_external_kv_injector(
    *,
    pt_path: str,
    layer_id: Optional[int] = None,
    max_tokens: int = 0,
    metrics_path: Optional[str] = None,
    inject_once: bool = True,
    prefill_only: bool = True,
    rope_impl: str = _ROPE_IMPL_TORCH_TWO_STEP,
) -> None:
    """Programmatically configure the singleton injector for benchmarking."""
    global _INJECTOR_SINGLETON, _INJECTOR_INITED

    pt = Path(pt_path)
    if not pt.exists():
        raise FileNotFoundError(f"External KV pt_path not found: {pt}")

    cfg = ExternalKVInjectorConfig(
        enabled=True,
        pt_path=pt,
        layer_id=layer_id,
        max_tokens=max_tokens,
        metrics_path=Path(metrics_path) if metrics_path else None,
        inject_once=inject_once,
        prefill_only=prefill_only,
        rope_impl=_normalize_rope_impl(rope_impl),
    )

    if _INJECTOR_SINGLETON is None:
        _INJECTOR_SINGLETON = ExternalKVInjector(cfg)
    else:
        _INJECTOR_SINGLETON.reconfigure(cfg)
    _INJECTOR_INITED = True
