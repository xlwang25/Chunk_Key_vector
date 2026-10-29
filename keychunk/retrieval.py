"""Download QA data and build E5 + BM25 + RRF inputs for KeyChunk."""
import argparse
from contextlib import contextmanager
import hashlib
from importlib.metadata import version
from itertools import islice
import json
import os
from pathlib import Path
import shutil
import tempfile
from zipfile import ZipFile

DATA_REPO = 'RUC-NLPIR/FlashRAG_datasets'
DATA_REVISION = 'bcafb8dd07d453be3cbeeeb3f78be1841bddf92c'
BACKEND_VERSION = '0.3.0.dev0'
BACKEND_COMMIT = '7926ac3c3c5ce0781f53d60ebac48fc511abed12'
RETRIEVAL_METHOD = 'e5+bm25+rrf'
RRF_K = 60
DENSE_INDEX = 'dense/e5_Flat.index'
SPARSE_INDEX = 'sparse/bm25'
SPLITS = {
    'nq': ('train', 'dev', 'test'),
    'hotpotqa': ('train', 'dev'),
    'musique': ('train', 'dev'),
    'triviaqa': ('train', 'dev', 'test'),
}
DEFAULT_SPLITS = {'nq': 'test', 'hotpotqa': 'dev', 'musique': 'dev', 'triviaqa': 'test'}


def positive_int(value):
    value = int(value)
    if value < 1:
        raise argparse.ArgumentTypeError('must be a positive integer')
    return value


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def require_new(path):
    if Path(path).exists():
        raise ValueError(f'output already exists: {path}')


@contextmanager
def atomic_output(path, *, binary=False):
    """Publish a completed file without replacing existing user data."""
    path = Path(path)
    require_new(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    options = {} if binary else {'encoding': 'utf-8'}
    handle = tempfile.NamedTemporaryFile(mode='wb' if binary else 'w',
                                         dir=path.parent, prefix='.' + path.name + '.',
                                         delete=False, **options)
    temporary = Path(handle.name)
    try:
        with handle:
            yield handle
        os.link(temporary, path)
    finally:
        handle.close()
        temporary.unlink(missing_ok=True)


def write_json(path, value):
    with atomic_output(path) as handle:
        json.dump(value, handle, ensure_ascii=False, allow_nan=False, indent=2)
        handle.write('\n')


def read_qa(path):
    with Path(path).open(encoding='utf-8') as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f'QA line {line_number} must be an object')
            question = row.get('question')
            answers = row.get('golden_answers', row.get('answers'))
            if isinstance(answers, str):
                answers = [answers]
            if not isinstance(question, str) or not question.strip():
                raise ValueError(f'QA line {line_number} has no question')
            if (not isinstance(answers, list) or not answers
                    or any(not isinstance(a, str) for a in answers)
                    or not any(a.strip() for a in answers)):
                raise ValueError(f'QA line {line_number} has no valid reference answers')
            yield dict(id=row.get('id', line_number - 1), question=question, answers=answers)


def hub_file(filename, revision):
    from huggingface_hub import hf_hub_download
    return Path(hf_hub_download(repo_id=DATA_REPO, repo_type='dataset',
                               filename=filename, revision=revision))


def download_qa(args):
    names = list(SPLITS) if args.dataset == 'all' else [args.dataset]
    targets = []
    for name in names:
        split = args.split or DEFAULT_SPLITS[name]
        if split not in SPLITS[name]:
            raise ValueError(f'{name} has splits {SPLITS[name]}, not {split}')
        target = args.output_dir / name / (split + '.jsonl')
        require_new(target)
        targets.append((name, split, target))
    for name, split, target in targets:
        source = hub_file(f'{name}/{split}.jsonl', args.revision)
        count = sum(1 for _ in read_qa(source))
        if not count:
            raise ValueError(f'{name}/{split} contains no QA records')
        with atomic_output(target, binary=True) as output, source.open('rb') as handle:
            shutil.copyfileobj(handle, output)
        print(json.dumps(dict(dataset=name, split=split, samples=count,
                              revision=args.revision, output=str(target))))


def download_corpus(args):
    require_new(args.output)
    if args.output.suffix != '.jsonl':
        raise ValueError('--output must be a .jsonl file')
    source = hub_file('retrieval-corpus/wiki18_100w.zip', args.revision)
    with ZipFile(source) as archive:
        members = [m for m in archive.infolist() if not m.is_dir()
                   and m.filename.endswith('.jsonl') and not m.filename.startswith('__MACOSX/')]
        preferred = [m for m in members if Path(m.filename).name == 'wiki18_100w.jsonl']
        members = preferred or members
        if len(members) != 1:
            raise ValueError('expected one Wikipedia JSONL corpus in the official archive')
        # Stream the selected file only; archive paths are never extracted.
        with atomic_output(args.output, binary=True) as output, archive.open(members[0]) as handle:
            shutil.copyfileobj(handle, output)
    print(json.dumps(dict(corpus='wiki18_100w', revision=args.revision, output=str(args.output))))


def load_retrieval_backend():
    if version('flashrag-dev') != BACKEND_VERSION:
        raise ValueError(f'install the retrieval extra for backend version {BACKEND_VERSION}')
    from flashrag.retriever.index_builder import Index_Builder
    from flashrag.retriever.retriever import BM25Retriever, DenseRetriever
    return Index_Builder, DenseRetriever, BM25Retriever


def build_index(args):
    require_new(args.output_dir)
    corpus = args.corpus.expanduser().resolve()
    model = args.retriever_model.expanduser().resolve()
    if not corpus.is_file() or corpus.suffix != '.jsonl':
        raise ValueError('--corpus must be a local JSONL file with id and contents fields')
    if not (model / 'config.json').is_file():
        raise ValueError('--retriever-model must be a local E5 checkpoint directory')
    builder_class, _, _ = load_retrieval_backend()
    before = corpus.stat()
    args.output_dir.mkdir(parents=True)
    builder = builder_class(retrieval_method='e5', model_path=str(model), corpus_path=str(corpus),
                            save_dir=str(args.output_dir / 'dense'), max_length=args.max_length,
                            batch_size=args.batch_size, use_fp16=args.fp16,
                            pooling_method='mean', instruction='passage:', faiss_type='Flat',
                            faiss_gpu=False, use_sentence_transformer=False)
    builder.build_index()
    del builder
    index = args.output_dir / DENSE_INDEX
    if not index.is_file():
        raise ValueError('retrieval backend did not produce e5_Flat.index')
    builder = builder_class(retrieval_method='bm25', model_path=None, corpus_path=str(corpus),
                            save_dir=str(args.output_dir / 'sparse'), max_length=args.max_length,
                            batch_size=args.batch_size, use_fp16=False, pooling_method='mean',
                            bm25_backend='bm25s')
    builder.build_index()
    del builder
    sparse_index = args.output_dir / SPARSE_INDEX
    if not (sparse_index / 'params.index.json').is_file():
        raise ValueError('retrieval backend did not produce a BM25s index')
    after = corpus.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError('corpus changed during index construction')
    metadata = dict(format_version=2, flashrag_version=BACKEND_VERSION, flashrag_commit=BACKEND_COMMIT,
                    corpus=str(corpus), corpus_sha256=sha256(corpus),
                    retriever_model=str(model), model_config_sha256=sha256(model / 'config.json'),
                    index=DENSE_INDEX, sparse_index=SPARSE_INDEX, bm25_backend='bm25s',
                    pooling='mean', method=RETRIEVAL_METHOD, rrf_k=RRF_K,
                    document_max_length=args.max_length, document_fp16=args.fp16)
    write_json(args.output_dir / 'retrieval.json', metadata)
    print(json.dumps(dict(index=str(index), sparse_index=str(sparse_index),
                          metadata=str(args.output_dir / 'retrieval.json'))))


def rrf_fusion(dense_documents, sparse_documents, top_k):
    """Fuse document ranks, retaining dense-first ordering for equal scores."""
    scores, documents = {}, {}
    for ranked in (dense_documents, sparse_documents):
        if len(ranked) != top_k:
            raise ValueError('retriever did not return the requested number of documents')
        seen = set()
        for rank, document in enumerate(ranked):
            doc_id = document.get('id')
            contents = document.get('contents')
            if doc_id is None or not str(doc_id).strip():
                raise ValueError('retrieved documents must have an id')
            if not isinstance(contents, str) or not contents.strip():
                raise ValueError('retrieved documents must have nonempty contents')
            doc_id = str(doc_id)
            if doc_id in seen:
                raise ValueError('retriever returned duplicate document ids')
            seen.add(doc_id)
            if doc_id in documents and documents[doc_id]['contents'] != contents:
                raise ValueError('retrievers returned different contents for the same document id')
            documents.setdefault(doc_id, document)
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (RRF_K + rank + 1)
    ordered = sorted(scores, key=scores.__getitem__, reverse=True)[:top_k]
    return [documents[doc_id] for doc_id in ordered], [scores[doc_id] for doc_id in ordered]


def retrieve(args):
    require_new(args.output)
    if args.output.suffix != '.jsonl':
        raise ValueError('--output must be a .jsonl file')
    metadata = json.loads((args.index_dir / 'retrieval.json').read_text())
    if (metadata.get('format_version') != 2 or metadata.get('method') != RETRIEVAL_METHOD
            or metadata.get('flashrag_version') != BACKEND_VERSION
            or metadata.get('flashrag_commit') != BACKEND_COMMIT
            or metadata.get('index') != DENSE_INDEX
            or metadata.get('sparse_index') != SPARSE_INDEX
            or metadata.get('bm25_backend') != 'bm25s' or metadata.get('rrf_k') != RRF_K):
        raise ValueError('unsupported retrieval index metadata; rebuild E5 + BM25 indexes in a new directory')
    corpus = args.corpus or Path(metadata['corpus'])
    model = args.retriever_model or Path(metadata['retriever_model'])
    _, retriever_class, sparse_class = load_retrieval_backend()
    if sha256(corpus) != metadata['corpus_sha256']:
        raise ValueError('corpus differs from the one used to build this index')
    if sha256(model / 'config.json') != metadata['model_config_sha256']:
        raise ValueError('retriever model configuration differs from the index metadata')
    # Load the retrieval components independently of generation.
    config = dict(retrieval_method='e5', retrieval_model_path=str(model),
                  retrieval_pooling_method='mean', retrieval_topk=args.top_k,
                  index_path=str(args.index_dir / DENSE_INDEX), corpus_path=str(corpus),
                  retrieval_batch_size=args.batch_size, retrieval_query_max_length=args.query_max_length,
                  retrieval_use_fp16=args.fp16, instruction='query:', faiss_gpu=False,
                  use_sentence_transformer=False, use_reranker=False, silent_retrieval=True,
                  save_retrieval_cache=False, use_retrieval_cache=False, retrieval_cache_path=None)
    retriever = retriever_class(config)
    if retriever.index.ntotal != len(retriever.corpus) or not 0 < args.top_k <= retriever.index.ntotal:
        raise ValueError('index/corpus size mismatch or top-k exceeds corpus size')
    sparse_path = args.index_dir / SPARSE_INDEX
    sparse_params = json.loads((sparse_path / 'params.index.json').read_text())
    if sparse_params.get('num_docs') != retriever.index.ntotal:
        raise ValueError('BM25 index/corpus size mismatch')
    sparse = sparse_class(dict(config, retrieval_method='bm25', index_path=str(sparse_path),
                               bm25_backend='bm25s'))
    rows = iter(islice(read_qa(args.input), args.limit))
    count = 0
    with atomic_output(args.output) as output:
        while batch := list(islice(rows, args.batch_size)):
            questions = [row['question'] for row in batch]
            dense_documents = retriever.batch_search(questions, num=args.top_k)
            sparse_documents = sparse.batch_search(questions, num=args.top_k)
            if len(dense_documents) != len(batch) or len(sparse_documents) != len(batch):
                raise ValueError('retriever returned an incomplete query batch')
            for row, dense_docs, sparse_docs in zip(batch, dense_documents, sparse_documents):
                docs, values = rrf_fusion(dense_docs, sparse_docs, args.top_k)
                chunks = [doc.get('contents') for doc in docs]
                row.update(chunks=chunks, retrieved_doc_ids=[doc.get('id') for doc in docs],
                           retrieval_scores=values, retrieval_method=RETRIEVAL_METHOD, rrf_k=RRF_K)
                output.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
                count += 1
        if not count:
            raise ValueError('input contains no QA records')
    print(json.dumps(dict(samples=count, retrieved_chunks=args.top_k, output=str(args.output))))


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    download = commands.add_parser('download', help='download the four supported QA datasets')
    download.add_argument('--dataset', choices=[*SPLITS, 'all'], required=True)
    download.add_argument('--split', choices=['train', 'dev', 'test'])
    download.add_argument('--output-dir', type=Path, required=True)
    download.add_argument('--revision', default=DATA_REVISION)
    download.set_defaults(run=download_qa)
    corpus = commands.add_parser('download-corpus', help='download and unpack the official Wikipedia corpus')
    corpus.add_argument('--output', type=Path, required=True)
    corpus.add_argument('--revision', default=DATA_REVISION)
    corpus.set_defaults(run=download_corpus)
    index = commands.add_parser('build-index', help='build E5/FAISS Flat and BM25s indexes')
    index.add_argument('--corpus', type=Path, required=True)
    index.add_argument('--retriever-model', type=Path, required=True)
    index.add_argument('--output-dir', type=Path, required=True)
    index.add_argument('--max-length', type=positive_int, default=512)
    index.add_argument('--batch-size', type=positive_int, default=64)
    index.add_argument('--fp16', action='store_true')
    index.set_defaults(run=build_index)
    query = commands.add_parser('retrieve', help='fuse E5 and BM25 ranks with RRF and export QA inputs')
    query.add_argument('--input', type=Path, required=True)
    query.add_argument('--index-dir', type=Path, required=True)
    query.add_argument('--corpus', type=Path, help='override the corpus location recorded in index metadata')
    query.add_argument('--retriever-model', type=Path, help='override the E5 checkpoint location')
    query.add_argument('--output', type=Path, required=True)
    query.add_argument('--top-k', type=positive_int, required=True,
                       help='documents per retriever and final chunks after RRF fusion')
    query.add_argument('--batch-size', type=positive_int, default=32)
    query.add_argument('--query-max-length', type=positive_int, default=128)
    query.add_argument('--limit', type=positive_int)
    query.add_argument('--fp16', action='store_true')
    query.set_defaults(run=retrieve)
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    try:
        args.run(args)
    except (ValueError, OSError, ImportError) as exc:
        parser.error(str(exc))


if __name__ == '__main__':
    main()
