"""Check GPU chunk-page reuse by repeating one request on a prepared context."""
import json
from itertools import islice
from pathlib import Path

from keychunk import KeyChunk
from _common import (engine_options, output_files, parser_for, positive_int,
                     read_inputs, request_fields, summarize, write_json)


def build_parser():
    parser = parser_for(__doc__)
    parser.add_argument('--input', type=Path, required=True,
                        help='caller-provided JSON or JSONL with chunks and a question')
    parser.add_argument('--sample-index', type=int, default=0,
                        help='zero-based record index in a JSONL file')
    parser.add_argument('--repeats', type=positive_int, default=2,
                        help='number of sequential calls, including the initial call (default: 2)')
    parser.add_argument('--max-cache-chunks', type=positive_int, default=128)
    return parser


def verified_reuse(row):
    stats = row.get('gpu_reuse') or {}
    expected = stats.get('expected_layers', 0)
    return (expected > 0 and stats.get('observed_layers') == expected
            and stats.get('gpu_reused_tokens_per_layer_min', 0) > 0)


def run(args):
    options = engine_options(args)
    if args.sample_index < 0:
        raise ValueError('--sample-index must be nonnegative')
    if args.repeats < 2:
        raise ValueError('--repeats must include an initial call and at least one repeated call')
    row = next(islice(read_inputs(args.input), args.sample_index, args.sample_index + 1), None)
    if row is None:
        raise ValueError('the requested input record does not exist')
    chunks, question, answers = request_fields(row, args.recompute_chunks)
    if args.recompute_chunks == len(chunks):
        raise ValueError('GPU reuse needs at least one chunk outside the complete recompute set')
    results = []
    with output_files(args.output_dir, ['requests.jsonl', 'summary.json']) as files:
        with KeyChunk(args.model, **options, gpu_page_cache=True,
                      max_cache_chunks=args.max_cache_chunks, gpu_reuse_diagnostics=True) as engine:
            with engine.prepare(chunks) as context:
                for index in range(args.repeats):
                    result = engine.generate(context, question, answers=answers).to_dict()
                    result.update(call_index=index, phase='initial' if index == 0 else 'repeat')
                    write_json(files['requests.jsonl'], result)
                    results.append(result)
        repeats = results[1:]
        observed = sum(verified_reuse(result) for result in repeats)
        summary = dict(input=str(args.input), model=str(args.model), sample_index=args.sample_index,
                       recompute_chunks=args.recompute_chunks, max_new_tokens=args.max_new_tokens,
                       skip_clustering=args.skip_clustering,
                       max_model_len=args.max_model_len, dtype=args.dtype, quantization=args.quantization,
                       max_cache_chunks=args.max_cache_chunks, initial=results[0],
                       repeated_calls=summarize(repeats), repeated_calls_with_gpu_reuse=observed,
                       gpu_reuse_verified=observed > 0,
                       repeated_text_matches_initial=all(r['text'] == results[0]['text'] for r in repeats))
        write_json(files['summary.json'], summary)
    return summary


def main():
    parser = build_parser()
    args = parser.parse_args()
    try:
        summary = run(args)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    print(json.dumps(summary, ensure_ascii=False, allow_nan=False))
    if not summary['gpu_reuse_verified']:
        parser.exit(2, 'No repeated call confirmed GPU page reads on every model layer; inspect requests.jsonl.\n')


if __name__ == '__main__':
    main()
