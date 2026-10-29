"""User-controlled number of complete chunks to recompute."""
import argparse
import operator

DEFAULT_RECOMPUTE_CHUNKS = 6


def validate_recompute_chunks(value, num_chunks=None):
    try:
        if isinstance(value, bool):raise TypeError
        count=operator.index(value)
    except TypeError as exc:
        raise ValueError('recompute chunk count must be an integer') from exc
    if count < 0 or (num_chunks is not None and count > num_chunks):
        raise ValueError('recompute chunk count must be between zero and the number of chunks')
    return count


def parse_recompute_chunks(value):
    try:return validate_recompute_chunks(int(value))
    except (TypeError, ValueError) as exc:raise argparse.ArgumentTypeError(str(exc)) from exc


def add_recompute_argument(parser):
    parser.add_argument('--recompute-chunks', type=parse_recompute_chunks,
        default=DEFAULT_RECOMPUTE_CHUNKS, metavar='COUNT',
        help='number of complete chunks to recompute (default: %(default)s)')
