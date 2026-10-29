"""Capture reusable chunk KV and semantic features from an in-process model."""
from keychunk.core.types import ChunkCapture


def capture_chunks(llm,model,layers,tokenizer,chunks,params,context_id,max_model_len):
    import torch
    try:
        records=[];encoded=[]
        for text in chunks:
            ids=[int(tokenizer.bos_token_id),*map(int,tokenizer.encode(text,add_special_tokens=False))]
            if len(ids)<2:raise ValueError('chunk encodes to an empty body')
            if len(ids)>max_model_len:raise ValueError('chunk exceeds max_model_len')
            model.reset_hack_kv();model.reset_semvec()
            model.set_collect_kv(True);model.set_collect_semvec(True)
            try:llm.generate([{'prompt_token_ids':ids}],params,use_tqdm=False)
            finally:
                model.set_collect_kv(False);model.set_collect_semvec(False)
            keys=[];values=[];means=[]
            for index,layer in enumerate(layers):
                attn=layer.self_attn;k,v=attn.finalize_hack_kv()
                if len(k)!=len(ids) or k.shape!=v.shape:raise RuntimeError('captured KV/token alignment changed')
                keys.append(k.detach().cpu().contiguous());values.append(v.detach().cpu().contiguous())
                if index>=max(0,len(layers)-8):
                    mean=attn.finalize_sem_k_mean()
                    if mean is None:raise RuntimeError('semantic feature capture is unavailable')
                    means.append(mean.detach().cpu().float().contiguous())
            if not keys or not means:raise RuntimeError('capture returned no KV or semantic features')
            records.append((keys,values,torch.stack(means).mean(dim=0)));encoded.append(ids)
        offsets=[0]
        for ids in encoded:offsets.append(offsets[-1]+len(ids))
        return ChunkCapture(context_id,torch.tensor(offsets,dtype=torch.int64),
            torch.arange(len(chunks),dtype=torch.int32),torch.tensor([i for ids in encoded for i in ids],dtype=torch.int32),
            torch.stack([record[2] for record in records]).contiguous(),
            tuple(torch.cat([record[0][layer] for record in records]).contiguous() for layer in range(len(layers))),
            tuple(torch.cat([record[1][layer] for record in records]).contiguous() for layer in range(len(layers))))
    finally:
        model.reset_hack_kv();model.reset_semvec()


def capture_prefix(llm,model,layers,tokenizer,params):
    try:
        model.reset_hack_kv();model.set_collect_kv(True)
        try:llm.generate([{'prompt_token_ids':[int(tokenizer.bos_token_id)]}],params,use_tqdm=False)
        finally:model.set_collect_kv(False)
        result={}
        for index,layer in enumerate(layers):
            k,v=layer.self_attn.finalize_hack_kv()
            if len(k)!=1 or k.shape!=v.shape:raise RuntimeError('prefix capture must contain one BOS token')
            result[index]=dict(k=k.detach().cpu().contiguous(),v=v.detach().cpu().contiguous())
        if not result:raise RuntimeError('prefix KV capture is unavailable')
        return result
    finally:
        model.reset_hack_kv()
