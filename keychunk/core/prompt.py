"""Assemble contexts without dropping any user-supplied chunk."""
import hashlib
import json
from .types import ChunkSpan,PromptPlan


def build_prompt(tokenizer,chunks,question,selected,capture=None):
    original=tuple(range(len(chunks)));chosen=tuple(selected)
    if len(set(chosen))!=len(chosen) or not set(chosen)<=set(original):
        raise ValueError('invalid selected chunk indices')
    order=tuple(i for i in original if i not in chosen)+tuple(reversed(chosen))
    if tokenizer.bos_token_id is None:raise ValueError('a BOS tokenizer is required')
    ids=[int(tokenizer.bos_token_id)];spans=[]
    separator=tokenizer.encode('\n\n',add_special_tokens=False)
    for index in order:
        body=tuple(map(int,tokenizer.encode(chunks[index],add_special_tokens=False)))
        if not body:raise ValueError('chunk encodes to an empty body')
        if capture is not None:
            start,end=capture.body_span(index)
            if body!=tuple(map(int,capture.token_ids[start:end].tolist())):
                raise ValueError('prepared KV does not match chunk tokenization')
        start=len(ids);ids.extend(body);spans.append(ChunkSpan(index,start,len(ids)))
        ids.extend(map(int,separator))
    query_start=len(ids)
    ids.extend(map(int,tokenizer.encode('Question: '+question+'\nAnswer:',add_special_tokens=False)))
    payload=json.dumps([ids,order],separators=(',',':')).encode()
    return PromptPlan(order,tuple(ids),tuple(spans),1,query_start,len(ids),hashlib.sha256(payload).hexdigest())
