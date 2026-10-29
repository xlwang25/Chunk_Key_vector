from __future__ import annotations

from typing import Any, Iterable

from .types import PromptPlan, ChunkCapture, StagedReuse


class StagingArena:
    """One reusable pinned allocation for the prepared context."""

    def __init__(self, capture: ChunkCapture, prefix_tokens: int):
        import torch

        self.capture_index = capture.context_id
        max_tokens = int(prefix_tokens) + sum(
            capture.body_token_count(index) for index in range(capture.num_chunks))
        self.layer_k = tuple(torch.empty(
            (max_tokens, *source.shape[1:]), dtype=source.dtype,
            device="cpu", pin_memory=True) for source in capture.layer_k)
        self.layer_v = tuple(torch.empty(
            (max_tokens, *source.shape[1:]), dtype=source.dtype,
            device="cpu", pin_memory=True) for source in capture.layer_v)
        if not all(value.is_pinned() for value in (*self.layer_k, *self.layer_v)):
            raise RuntimeError("staging arena is not pinned")
        self.current_signature: tuple[str, tuple[int, ...]] | None = None


def positions_for_orig_chunks(prompt: PromptPlan,
                              orig_indices: Iterable[int]) -> tuple[int, ...]:
    selected = {int(index) for index in orig_indices}
    return tuple(position for span in prompt.chunks if span.orig_idx in selected
                 for position in range(span.body_start, span.body_end))


def _body_source_rows(prompt: PromptPlan, capture: ChunkCapture) -> dict[int, int]:
    rows: dict[int, int] = {}
    for span in prompt.chunks:
        source_start, source_end = capture.body_span(span.orig_idx)
        if source_end - source_start != span.body_tokens:
            raise RuntimeError("prompt/capture body length mismatch")
        for offset, prompt_position in enumerate(range(span.body_start,
                                                       span.body_end)):
            rows[prompt_position] = source_start + offset
    return rows


def _old_rope_positions(prompt: PromptPlan) -> dict[int, int]:
    old: dict[int, int] = {}
    for position in range(prompt.prefix_end):
        old[position] = position
    for span in prompt.chunks:
        for offset, prompt_position in enumerate(range(span.body_start,
                                                       span.body_end)):
            # Each chunk was captured independently with BOS at position 0.
            old[prompt_position] = offset + 1
    return old


def stage_reuse(
    *, prompt: PromptPlan, capture: ChunkCapture,
    prefix_layer_kv: dict[int, dict[str, Any]],
    selected_body_positions: Iterable[int],
    arena: StagingArena | None = None,
    prefix_reuse_tokens: int | None = None,
) -> StagedReuse:
    """Build one pre-pinned dense source before scheduling the request."""
    import torch

    selected = tuple(sorted(set(int(x) for x in selected_body_positions)))
    body_rows = _body_source_rows(prompt, capture)
    prefix_tokens = (prompt.prefix_end if prefix_reuse_tokens is None else
                     int(prefix_reuse_tokens))
    if not 0 <= prefix_tokens <= prompt.prefix_end:
        raise ValueError("prefix reuse count is outside the fixed prefix")
    reusable_prefix = tuple(range(prefix_tokens))
    reusable = reusable_prefix + tuple(
        position for position in sorted(body_rows) if position not in set(selected))
    recompute = tuple(position for position in range(len(prompt.token_ids))
                      if position not in set(reusable))
    if not recompute:
        raise RuntimeError("selective request has no recompute positions")
    old_map = _old_rope_positions(prompt)
    old_positions = tuple(old_map[position] for position in reusable)
    layers = sorted(prefix_layer_kv)
    if layers != list(range(capture.num_layers)):
        raise RuntimeError("prefix/capture layer mismatch")
    layer_kv: dict[int, dict[str, Any]] = {}
    if arena is not None and arena.capture_index != capture.context_id:
        raise RuntimeError("staging arena belongs to another context")
    signature = (prompt.prompt_hash, selected)
    cache_hit = arena is not None and arena.current_signature == signature
    for layer in layers:
        source_k = capture.layer_k[layer]
        source_v = capture.layer_v[layer]
        shape = (len(reusable), *source_k.shape[1:])
        if arena is None:
            dense_k = torch.empty(shape, dtype=source_k.dtype, device="cpu",
                                  pin_memory=True)
            dense_v = torch.empty(shape, dtype=source_v.dtype, device="cpu",
                                  pin_memory=True)
        else:
            dense_k = arena.layer_k[layer][:len(reusable)]
            dense_v = arena.layer_v[layer][:len(reusable)]
        write = len(reusable) if cache_hit else 0
        if not cache_hit:
            if prefix_tokens:
                dense_k[:prefix_tokens].copy_(
                    prefix_layer_kv[layer]["k"][:prefix_tokens])
                dense_v[:prefix_tokens].copy_(
                    prefix_layer_kv[layer]["v"][:prefix_tokens])
                write = prefix_tokens
            source_rows = [body_rows[position] for position in
                           reusable[prefix_tokens:]]
            if source_rows:
                index = torch.tensor(source_rows, dtype=torch.long)
                count = len(source_rows)
                dense_k[write:write + count].copy_(source_k.index_select(0, index))
                dense_v[write:write + count].copy_(source_v.index_select(0, index))
                write += count
        if write != len(reusable) or not dense_k.is_pinned() or not dense_v.is_pinned():
            raise RuntimeError("staging did not produce complete pinned KV")
        layer_kv[layer] = {"k": dense_k, "v": dense_v}
    if arena is not None:
        arena.current_signature = signature
    return StagedReuse(
        layer_kv=layer_kv,
        reuse_positions=reusable,
        reuse_old_positions=old_positions,
        recompute_positions=recompute,
        selected_body_positions=selected,
    )
