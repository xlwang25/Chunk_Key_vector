from __future__ import annotations

import re
import string
from collections import Counter
from typing import Iterable
import math


def _normalize(text: str) -> list[str]:
    value = text.lower()
    value = "".join(character for character in value
                    if character not in string.punctuation)
    value = re.sub(r"\b(a|an|the)\b", " ", value)
    return re.sub(r"\s+", " ", value).strip().split()


def token_f1(prediction: str, golds: str | Iterable[str]) -> float:
    if isinstance(golds,str):golds=(golds,)
    pred = _normalize(prediction)
    best = 0.0
    for gold in golds:
        target = _normalize(gold)
        common = sum((Counter(pred) & Counter(target)).values())
        if not pred or not target:
            score = float(pred == target)
        elif common == 0:
            score = 0.0
        else:
            precision = common / len(pred)
            recall = common / len(target)
            score = 2 * precision * recall / (precision + recall)
        best = max(best, score)
    return best



def ttft(output) -> float:
    """Seconds between vLLM first scheduling and first generated token."""
    metrics=getattr(output,'metrics',None)
    first=getattr(metrics,'first_token_time',None)
    scheduled=getattr(metrics,'first_scheduled_time',None)
    if first is None or scheduled is None:raise RuntimeError('first-token timestamps are unavailable')
    elapsed=float(first)-float(scheduled)
    if not math.isfinite(elapsed) or elapsed<0:raise RuntimeError('invalid first-token timestamps')
    return elapsed
