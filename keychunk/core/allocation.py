from __future__ import annotations

from typing import Any, Iterable


PREFIX_BODY_TOKENS_PER_CHUNK = 2


def keychunk_prefix_union_positions(
    prompt: Any,
    selected_orig_indices: Iterable[int],
    prefix_tokens: int = PREFIX_BODY_TOKENS_PER_CHUNK,
) -> tuple[int, ...]:
    """Return selected chunks union the fixed body prefix of every chunk."""
    if prefix_tokens < 0:
        raise ValueError("prefix token count must be non-negative")
    selected = {int(index) for index in selected_orig_indices}
    positions: set[int] = set()
    for span in prompt.chunks:
        if span.orig_idx in selected:
            positions.update(range(span.body_start, span.body_end))
        else:
            positions.update(range(
                span.body_start,
                min(span.body_start + prefix_tokens, span.body_end),
            ))
    return tuple(sorted(positions))
