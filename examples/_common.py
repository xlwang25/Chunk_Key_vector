"""Shared input and output helpers for the single-GPU examples."""
import argparse
from contextlib import ExitStack, contextmanager
import json
import math
import os
from pathlib import Path
from statistics import fmean

from keychunk.budget import add_recompute_argument, validate_recompute_chunks
from keychunk.cli import input_fields, read_inputs


def positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError('must be a positive integer')
    return number


def parser_for(description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    add_recompute_argument(parser)
    parser.add_argument('--skip-clustering', action='store_true',
                        help='skip redundancy clustering for chunks already deduplicated upstream')
    parser.add_argument('--max-new-tokens', type=positive_int, default=32)
    parser.add_argument('--max-model-len', type=positive_int, default=8192)
    parser.add_argument('--dtype', choices=('bfloat16', 'float16'), default='bfloat16')
    parser.add_argument('--quantization')
    return parser


def engine_options(args):
    if int(os.environ.get('WORLD_SIZE', '1')) != 1:
        raise ValueError('these examples use one GPU; run without torchrun')
    return dict(recompute_chunks=args.recompute_chunks,
                max_new_tokens=args.max_new_tokens, max_model_len=args.max_model_len,
                dtype=args.dtype, quantization=args.quantization,
                skip_clustering=args.skip_clustering)


def request_fields(row, count, *, require_answers=False):
    chunks, question, answers = input_fields(row)
    if not chunks or any(not text.strip() for text in chunks):
        raise ValueError('each request must contain nonempty chunks')
    if not isinstance(question, str) or not question.strip():
        raise ValueError('each request must contain a nonempty question')
    references = [answers] if isinstance(answers, str) else answers
    if references is not None and (not isinstance(references, list)
                                  or any(not isinstance(a, str) for a in references)):
        raise ValueError('reference answers must be strings')
    if require_answers and (not references or not any(a.strip() for a in references)):
        raise ValueError('quality evaluation requires reference answers for every request')
    validate_recompute_chunks(count, len(chunks))
    return chunks, question, references


@contextmanager
def output_files(directory, names):
    directory = Path(directory)
    for name in names:
        if (directory / name).exists():
            raise ValueError(f'output already exists: {directory / name}')
    directory.mkdir(parents=True, exist_ok=True)
    with ExitStack() as stack:
        yield {name: stack.enter_context((directory / name).open('x', encoding='utf-8'))
               for name in names}


def write_json(handle, value):
    handle.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + '\n')
    handle.flush()


def summarize(rows):
    if not rows:
        raise ValueError('no requests were evaluated')
    times = [float(row['ttft_s']) for row in rows]
    scores = [float(row['f1']) for row in rows if row['f1'] is not None]
    if any(not math.isfinite(t) or t < 0 for t in times):
        raise ValueError('invalid TTFT in results')
    if any(not math.isfinite(s) or not 0 <= s <= 1 for s in scores):
        raise ValueError('invalid F1 in results')
    return dict(samples=len(rows), scored_samples=len(scores),
                mean_f1=fmean(scores) if scores else None,
                mean_ttft_s=fmean(times))
