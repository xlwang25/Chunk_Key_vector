"""Public API for reusable chunk contexts and GPU text generation."""
from dataclasses import dataclass,field
import itertools
import math
from pathlib import Path
import threading

from .budget import DEFAULT_RECOMPUTE_CHUNKS,validate_recompute_chunks
from .core.prompt import build_prompt
from .core.selection import build_cluster_plan,select_from_plan
from .core.types import ChunkCapture
from .metrics import token_f1

_LOCK=threading.RLock()


def _positive_integer(value,name):
    if isinstance(value,bool) or not isinstance(value,int) or value<1:
        raise ValueError(name+' must be a positive integer')
    return value


def _chunks(values):
    if isinstance(values,(str,bytes)):raise ValueError('chunks must be a sequence of text strings')
    result=tuple(values)
    if not result or any(not isinstance(s,str) or not s.strip() for s in result):
        raise ValueError('chunks must contain nonempty text strings')
    return result


@dataclass
class PreparedChunks:
    chunks:tuple[str,...]
    _capture:object=field(repr=False)
    _plan:object=field(repr=False)
    _owner:object=field(repr=False)
    _arena:object=field(default=None,repr=False)
    _closed:bool=field(default=False,repr=False)
    _kv_file:Path|None=field(default=None,repr=False)

    def spill(self,path):
        """Store CPU KV once, then release its in-memory copy and staging arena."""
        with _LOCK:
            if self._closed:raise ValueError('prepared chunks are closed')
            target=Path(path).expanduser().resolve()
            if self._kv_file is not None and target!=self._kv_file:
                raise ValueError('prepared chunks already have a different KV file')
            if self._kv_file is None:
                import torch
                capture=self._capture
                payload={name:getattr(capture,name) for name in (
                    'context_id','offsets','orig_indices','token_ids','semvec','layer_k','layer_v')}
                with target.open('xb') as handle:torch.save(payload,handle)
                self._kv_file=target
            self._capture=None;self._arena=None

    def _load_capture(self):
        if self._capture is None:
            if self._kv_file is None:raise ValueError('prepared KV is unavailable')
            import torch
            payload=torch.load(self._kv_file,map_location='cpu',weights_only=True,mmap=True)
            self._capture=ChunkCapture(**payload)

    def close(self):
        with _LOCK:
            self._capture=None;self._plan=None;self._arena=None;self._kv_file=None;self._closed=True

    def __enter__(self):return self
    def __exit__(self,*args):self.close()


@dataclass(frozen=True)
class GenerationResult:
    text:str
    ttft_s:float
    f1:float|None
    selected_chunks:tuple[int,...]
    gpu_reuse:dict|None=None

    def to_dict(self):
        result=dict(text=self.text,ttft_s=self.ttft_s,f1=self.f1,
            selected_chunks=list(self.selected_chunks))
        if self.gpu_reuse is not None:result['gpu_reuse']=dict(self.gpu_reuse)
        return result


class KeyChunk:
    """Prepare chunk KV once and answer questions with selective recomputation."""
    def __init__(self,model,*,recompute_chunks=DEFAULT_RECOMPUTE_CHUNKS,max_new_tokens=32,
                 max_model_len=8192,tensor_parallel_size=1,dtype='bfloat16',quantization=None,
                 gpu_memory_utilization=.9,gpu_page_cache=True,max_cache_chunks=128,
                 gpu_reuse_diagnostics=False,skip_clustering=False,gpu_pages=None):
        model=Path(model).expanduser().resolve()
        if not model.is_dir():raise ValueError('model must be a local checkpoint directory')
        self.recompute_chunks=validate_recompute_chunks(recompute_chunks)
        # Skip redundancy clustering only when input chunks have already been deduplicated.
        if not isinstance(skip_clustering,bool):raise ValueError('skip_clustering must be a boolean')
        self.skip_clustering=skip_clustering
        self.max_new_tokens=_positive_integer(max_new_tokens,'max_new_tokens')
        self.max_model_len=_positive_integer(max_model_len,'max_model_len')
        if self.max_new_tokens>=self.max_model_len:raise ValueError('max_new_tokens must be less than max_model_len')
        if dtype not in ('bfloat16','float16'):raise ValueError('dtype must be bfloat16 or float16')
        if not isinstance(gpu_page_cache,bool):raise ValueError('gpu_page_cache must be a boolean')
        if not isinstance(gpu_reuse_diagnostics,bool):raise ValueError('gpu_reuse_diagnostics must be a boolean')
        if gpu_reuse_diagnostics and not gpu_page_cache:raise ValueError('gpu_reuse_diagnostics requires gpu_page_cache')
        _positive_integer(tensor_parallel_size,'tensor_parallel_size')
        _positive_integer(max_cache_chunks,'max_cache_chunks')
        if gpu_pages is not None:_positive_integer(gpu_pages,'gpu_pages')
        if not math.isfinite(gpu_memory_utilization) or not 0<gpu_memory_utilization<=1:
            raise ValueError('gpu_memory_utilization must be in (0,1]')
        from .runtime.vllm import ModelConfig,VLLMBackend
        self._owner=object();self._ids=itertools.count();self._closed=False
        with _LOCK:
            self._backend=VLLMBackend(ModelConfig(model,tensor_parallel_size,dtype,quantization,
                self.max_model_len,gpu_memory_utilization,gpu_page_cache,max_cache_chunks,
                gpu_reuse_diagnostics,gpu_pages))

    def _check_open(self):
        if self._closed:raise RuntimeError('KeyChunk engine is closed')

    def prepare(self,chunks):
        """Capture input chunks once; the returned context can answer many questions."""
        texts=_chunks(chunks)
        with _LOCK:
            self._check_open()
            capture=self._backend.capture(texts,next(self._ids))
            vectors=capture.semvec
            if hasattr(vectors,'numpy'):vectors=vectors.numpy()
            return PreparedChunks(texts,capture,
                build_cluster_plan(vectors,skip_clustering=self.skip_clustering),self._owner)

    def generate(self,chunks,question,*,answers=None,recompute_chunks=None,max_new_tokens=None):
        """Return generated text, TTFT, selected chunk indices and optional answer F1."""
        if not isinstance(question,str) or not question.strip():raise ValueError('question must be nonempty text')
        references=() if answers is None else (answers,) if isinstance(answers,str) else tuple(answers)
        if any(not isinstance(s,str) for s in references):raise ValueError('answers must contain strings')
        count=self.recompute_chunks if recompute_chunks is None else recompute_chunks
        tokens=self.max_new_tokens if max_new_tokens is None else _positive_integer(max_new_tokens,'max_new_tokens')
        temporary=not isinstance(chunks,PreparedChunks)
        context=None
        with _LOCK:
            self._check_open()
            if temporary:
                texts=_chunks(chunks);validate_recompute_chunks(count,len(texts))
                context=self.prepare(texts)
            else:context=chunks
            try:
                if context._closed or context._owner is not self._owner:raise ValueError('prepared chunks belong to another engine or are closed')
                count=validate_recompute_chunks(count,len(context.chunks))
                context._load_capture()
                selected=select_from_plan(context._plan,count)
                prompt=build_prompt(self._backend.tokenizer,context.chunks,question,selected,context._capture)
                if len(prompt.token_ids)+tokens>self.max_model_len:raise ValueError('prompt plus max_new_tokens exceeds max_model_len')
                if context._arena is None:context._arena=self._backend.staging(context._capture)
                row=self._backend.generate(context._capture,prompt,selected,context._arena,tokens)
                return GenerationResult(row['text'],row['ttft_s'],
                    token_f1(row['text'],references) if references else None,selected,row.get('gpu_reuse'))
            finally:
                if temporary and context is not None:context.close()

    def close(self):
        with _LOCK:
            if not self._closed:self._backend.close();self._closed=True

    def __enter__(self):return self
    def __exit__(self,*args):self.close()
