"""Generate answers and report mean F1 and TTFT for retrieved QA inputs."""
import json
from itertools import islice
from pathlib import Path

from keychunk import KeyChunk
from _common import (engine_options, output_files, parser_for, positive_int,
                     read_inputs, request_fields, summarize, write_json)


def build_parser():
    parser = parser_for(__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--limit', type=positive_int,
                        help='process at most this many input records; default: all records')
    return parser


def run(args):
    options = engine_options(args)
    rows = list(islice(read_inputs(args.input), args.limit))
    if not rows:
        raise ValueError('input contains no requests')
    requests = [request_fields(row, args.recompute_chunks, require_answers=True) for row in rows]
    results = []
    with output_files(args.output_dir, ['predictions.jsonl', 'summary.json']) as files:
        with KeyChunk(args.model, **options) as engine:
            for index, (row, (chunks, question, answers)) in enumerate(zip(rows, requests)):
                result = engine.generate(chunks, question, answers=answers).to_dict()
                result.update(sample_index=index, id=row.get('id', index))
                write_json(files['predictions.jsonl'], result)
                results.append(result)
        summary = dict(summarize(results), input=str(args.input), model=str(args.model),
                       recompute_chunks=args.recompute_chunks,
                       skip_clustering=args.skip_clustering,
                       max_new_tokens=args.max_new_tokens, max_model_len=args.max_model_len,
                       dtype=args.dtype, quantization=args.quantization)
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


if __name__ == '__main__':
    main()
