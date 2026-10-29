"""Main GPU-page performance entrypoint: run caller-provided requests sequentially."""
from contextlib import ExitStack
import json
from itertools import islice
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

from keychunk import KeyChunk
from _common import (engine_options, output_files, parser_for, positive_int,
                     read_inputs, request_fields, summarize, write_json)
from gpu_reuse import verified_reuse


def build_parser():
    parser = parser_for(__doc__)
    parser.add_argument('--input', type=Path, required=True,
                        help='ordered JSON array or JSONL of requests with chunks or ctxs')
    parser.add_argument('--kv-cache-dir', type=Path, required=True,
                        help='parent directory for temporary prepared CPU KV files')
    parser.add_argument('--limit', type=positive_int)
    parser.add_argument('--max-cache-chunks', type=positive_int, default=128)
    parser.add_argument('--gpu-pages', type=positive_int,
                        help='physical GPU KV pages (16 tokens each); default: automatic allocation')
    parser.add_argument('--verify-reuse', action='store_true',
                        help='collect actual per-layer GPU page reads; adds diagnostic overhead')
    return parser


def input_rows(path):
    if path.suffix.lower() == '.jsonl':
        with path.open(encoding='utf-8') as handle:
            first = handle.read(1)
            while first and first.isspace():
                first = handle.read(1)
        if first != '[':
            yield from read_inputs(path)
            return
    with path.open(encoding='utf-8') as handle:
        rows = json.load(handle)
    if isinstance(rows, dict):
        rows = [rows]
    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
        raise ValueError('JSON input must contain a request object or an array of request objects')
    yield from rows


def fields(row, count):
    # Official retrieved passages contain separate title and text fields.
    if 'chunks' not in row and isinstance(row.get('ctxs'), list):
        passages = []
        for context in row['ctxs']:
            if isinstance(context, dict) and not context.get('contents'):
                title, text = context.get('title'), context.get('text')
                if isinstance(title, str) and title and isinstance(text, str):
                    context = dict(context, contents=title + '\n' + text)
            passages.append(context)
        row = dict(row, ctxs=passages)
    chunks, question, answers = request_fields(row, count)
    if count == len(chunks):
        raise ValueError('GPU page reuse needs at least one chunk outside the complete recompute set')
    return chunks, question, answers


def run(args):
    options = engine_options(args)
    rows = list(islice(input_rows(args.input), args.limit))
    if not rows:
        raise ValueError('input contains no requests')
    requests = [fields(row, args.recompute_chunks) for row in rows]
    args.kv_cache_dir.mkdir(parents=True, exist_ok=True)
    results = []
    with output_files(args.output_dir, ['requests.jsonl', 'summary.json']) as files:
        with KeyChunk(args.model, **options, gpu_page_cache=True,
                      max_cache_chunks=args.max_cache_chunks,
                      gpu_reuse_diagnostics=args.verify_reuse, gpu_pages=args.gpu_pages) as engine:
            with TemporaryDirectory(prefix='gpu-page-', dir=args.kv_cache_dir) as cache_dir:
                with ExitStack() as stack:
                    contexts, remaining = {}, {}
                    for chunks, _, _ in requests:
                        key = tuple(chunks)
                        remaining[key] = remaining.get(key, 0) + 1
                    prepare_start = perf_counter()
                    # Finish all model captures before requests can populate the GPU page index.
                    for key in remaining:
                        context = stack.enter_context(engine.prepare(key))
                        path = Path(cache_dir) / (str(len(contexts)) + '.pt')
                        context.spill(path)
                        contexts[key] = context, path
                    preparation_s = perf_counter() - prepare_start
                    for index, (row, (chunks, question, answers)) in enumerate(zip(rows, requests)):
                        key = tuple(chunks)
                        context, path = contexts[key]
                        start = perf_counter()
                        result = engine.generate(context, question, answers=answers).to_dict()
                        result.update(request_index=index, id=row.get('id', index),
                                      elapsed_s=perf_counter() - start)
                        write_json(files['requests.jsonl'], result)
                        results.append(result)
                        context.spill(path)
                        remaining[key] -= 1
                        if remaining[key] == 0:
                            context.close()
                            path.unlink()
                            del contexts[key]
        verified = sum(verified_reuse(result) for result in results) if args.verify_reuse else None
        summary = dict(summarize(results), input=str(args.input), model=str(args.model),
                       recompute_chunks=args.recompute_chunks, skip_clustering=args.skip_clustering,
                       max_new_tokens=args.max_new_tokens, max_model_len=args.max_model_len,
                       dtype=args.dtype, quantization=args.quantization,
                       max_cache_chunks=args.max_cache_chunks, preparation_s=preparation_s,
                       gpu_pages=args.gpu_pages,
                       gpu_reuse_diagnostics=args.verify_reuse,
                       requests_with_gpu_reuse=verified,
                       gpu_reuse_verified=verified > 0 if verified is not None else None)
        write_json(files['summary.json'], summary)
    return summary


def main():
    parser = build_parser()
    args = parser.parse_args()
    try:
        summary = run(args)
    except (ValueError, OSError, RuntimeError) as exc:
        parser.error(str(exc))
    print(json.dumps(summary, ensure_ascii=False, allow_nan=False))


if __name__ == '__main__':
    main()
