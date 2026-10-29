# SPDX-License-Identifier: Apache-2.0
"""Metrics helpers for external KV injector."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional


def append_metrics_jsonl(metrics_path: Optional[Path], metrics: dict[str, Any]) -> None:
    if metrics_path is None:
        return
    with metrics_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(metrics, ensure_ascii=False) + "\n")
