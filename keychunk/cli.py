"""Generate answers for caller-provided JSON or JSONL inputs."""
import argparse
from contextlib import nullcontext
import json
from pathlib import Path
import sys

from .budget import add_recompute_argument
from .engine import KeyChunk


def read_inputs(path):
    with Path(path).open(encoding='utf-8') as handle:
        if Path(path).suffix.lower()=='.jsonl':
            for number,line in enumerate(handle,1):
                if line.strip():
                    row=json.loads(line)
                    if not isinstance(row,dict):raise ValueError(f'input line {number} must be an object')
                    yield row
        else:
            row=json.load(handle)
            if not isinstance(row,dict):raise ValueError('JSON input must be an object')
            yield row


def input_fields(row):
    nested=row.get('output') if isinstance(row.get('output'),dict) else {}
    chunks=row.get('chunks',row.get('ctxs',row.get('retrieval_result',nested.get('retrieval_result'))))
    if not isinstance(chunks,list):raise ValueError('input must supply a chunks list')
    texts=[]
    for chunk in chunks:
        if isinstance(chunk,dict):chunk=chunk.get('contents') or chunk.get('text') or chunk.get('passage')
        if not isinstance(chunk,str):raise ValueError('each chunk must contain text')
        texts.append(chunk)
    question=row.get('question') or row.get('query')
    answers=row.get('answers',row.get('golden_answers',row.get('answer',nested.get('answers',nested.get('golden_answers')))))
    if isinstance(answers,list):answers=[value for item in answers for value in (item if isinstance(item,list) else [item])]
    return texts,question,answers


def build_parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',type=Path,required=True)
    p.add_argument('--input',type=Path,required=True)
    p.add_argument('--output',type=Path)
    add_recompute_argument(p)
    p.add_argument('--skip-clustering',action='store_true',
        help='skip redundancy clustering when input chunks are already deduplicated')
    p.add_argument('--max-new-tokens',type=int,default=32)
    p.add_argument('--max-model-len',type=int,default=8192)
    p.add_argument('--tensor-parallel-size',type=int,default=1)
    p.add_argument('--dtype',choices=('bfloat16','float16'),default='bfloat16')
    p.add_argument('--quantization')
    return p


def main():
    p=build_parser();args=p.parse_args()
    if not args.input.is_file():p.error('--input must be a JSON or JSONL file')
    if args.output and args.output.resolve()==args.input.resolve():p.error('--output must differ from --input')
    if args.output and args.output.exists():p.error('--output already exists')
    if args.output:args.output.parent.mkdir(parents=True,exist_ok=True)
    import os
    rank=int(os.environ.get('RANK','0'))
    stream=(args.output.open('x',encoding='utf-8') if args.output and rank==0 else nullcontext(sys.stdout))
    with stream as output:
        with KeyChunk(args.model,recompute_chunks=args.recompute_chunks,max_new_tokens=args.max_new_tokens,
            max_model_len=args.max_model_len,tensor_parallel_size=args.tensor_parallel_size,
            dtype=args.dtype,quantization=args.quantization,skip_clustering=args.skip_clustering) as engine:
            for row in read_inputs(args.input):
                chunks,question,answers=input_fields(row)
                result=engine.generate(chunks,question,answers=answers).to_dict()
                if 'id' in row:result['id']=row['id']
                if rank==0:output.write(json.dumps(result,ensure_ascii=False)+'\n');output.flush()
