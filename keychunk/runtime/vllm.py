"""In-process vLLM integration for chunk capture and selective GPU prefill."""
from dataclasses import dataclass
from pathlib import Path
import os
import time
import uuid

from keychunk.core.staging import StagingArena,stage_reuse
from keychunk.core.allocation import keychunk_prefix_union_positions
from keychunk.metrics import ttft
from .access import resolve_capture_model
from .shared import resolve_layers
from .capture import capture_chunks,capture_prefix
from .diagnostics import GpuReuseTrace


@dataclass(frozen=True)
class ModelConfig:
    model:Path
    tensor_parallel_size:int=1
    dtype:str='bfloat16'
    quantization:str|None=None
    max_model_len:int=8192
    gpu_memory_utilization:float=.9
    gpu_page_cache:bool=True
    max_cache_chunks:int=128
    gpu_reuse_diagnostics:bool=False
    gpu_pages:int|None=None


def bind_page_cache(layers,key,prompt_len):
    import torch
    from vllm.v1.bench.external_kv_common import (
        _KEEPOLD_PAGE_BRIDGES,_KeepOldPageBridge,_KeepOldLayerSnapshot)
    snapshots={}
    for index,layer in enumerate(layers):
        cache=layer.self_attn.attn.kv_cache[0]
        if not isinstance(cache,torch.Tensor) or cache.shape[0]!=2:
            raise RuntimeError('paged KV cache is unavailable')
        k,v=cache.unbind(0)
        snapshots[index]=_KeepOldLayerSnapshot(k,v,
            torch.empty(0,dtype=torch.long,device=k.device),prompt_len)
    _KEEPOLD_PAGE_BRIDGES[key]=_KeepOldPageBridge(prompt_len,None,snapshots,time.time())


class VLLMBackend:
    def __init__(self,config):
        os.environ.update(VLLM_USE_V1='1',VLLM_ENABLE_V1_MULTIPROCESSING='0',TOKENIZERS_PARALLELISM='false')
        import torch
        if not torch.cuda.is_available():raise RuntimeError('KeyChunk inference requires CUDA')
        if config.tensor_parallel_size>1:
            if int(os.environ.get('WORLD_SIZE','1'))!=config.tensor_parallel_size:
                raise ValueError('tensor_parallel_size requires a matching torchrun process group')
            torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
        from vllm import LLM,SamplingParams
        from vllm.v1.bench.external_kv_injector import clear_selective_recompute_runtime
        clear_selective_recompute_runtime()
        # vLLM only populates first-token timestamps when request stats are enabled.
        options=dict(model=str(config.model),enforce_eager=True,enable_prefix_caching=False,
            enable_chunked_prefill=False,disable_log_stats=False,
            tensor_parallel_size=config.tensor_parallel_size,dtype=config.dtype,quantization=config.quantization,
            max_model_len=config.max_model_len,max_num_batched_tokens=config.max_model_len,
            max_num_seqs=1,block_size=16,gpu_memory_utilization=config.gpu_memory_utilization)
        if config.gpu_pages is not None:options['num_gpu_blocks_override']=config.gpu_pages
        if config.tensor_parallel_size>1:options['distributed_executor_backend']='external_launcher'
        self.config=config;self.identity=uuid.uuid4().hex
        self.llm=LLM(**options);self.tokenizer=self.llm.get_tokenizer()
        if self.tokenizer.bos_token_id is None:raise ValueError('KeyChunk requires a BOS tokenizer')
        self.model=resolve_capture_model(self.llm);self.layers=list(resolve_layers(self.model))
        self.capture_params=SamplingParams(max_tokens=1,temperature=0.)
        self.prefix=capture_prefix(self.llm,self.model,self.layers,self.tokenizer,self.capture_params)

    def capture(self,chunks,context_id):
        from vllm.v1.bench.external_kv_injector import clear_selective_recompute_runtime
        clear_selective_recompute_runtime()
        capture=capture_chunks(self.llm,self.model,self.layers,self.tokenizer,chunks,
            self.capture_params,context_id,self.config.max_model_len)
        if self.config.tensor_parallel_size>1:
            import torch
            import torch.distributed as dist
            local=capture.semvec.contiguous().cuda()
            peers=[torch.empty_like(local) for _ in range(self.config.tensor_parallel_size)]
            dist.all_gather(peers,local);capture.semvec=torch.cat(peers,dim=-1).cpu()
        return capture

    def staging(self,capture):return StagingArena(capture,1)

    def generate(self,capture,prompt,selected,arena,max_new_tokens):
        from vllm import SamplingParams
        from vllm.v1.bench.external_kv_injector import (
            clear_selective_recompute_runtime,configure_selective_recompute_runtime,
            get_selective_recompute_runtime,clear_keepold_page_bridge)
        positions=keychunk_prefix_union_positions(prompt,selected)
        staged=stage_reuse(prompt=prompt,capture=capture,prefix_layer_kv=self.prefix,
            selected_body_positions=positions,arena=arena)
        bridge_key=self.identity+':'+uuid.uuid4().hex
        options={}
        if self.config.gpu_page_cache:
            bind_page_cache(self.layers,bridge_key,len(prompt.token_ids))
            options=dict(keepold_bridge_key=bridge_key,keepold_phase='reuse',
                keepold_index_mode='chunk_page',keepold_switch_prompt_positions=[
                    i for span in prompt.chunks for i in range(span.body_start,span.body_end)],
                keepold_chunk_infos=[dict(orig_idx=span.orig_idx,start=span.body_start,end=span.body_end)
                    for span in prompt.chunks],keepold_prompt_token_ids=list(prompt.token_ids),
                keepold_model_id=self.identity,keepold_chunk_page_max_chunks=self.config.max_cache_chunks,
                keepold_chunk_page_max_entries_per_hash=1,keepold_chunk_page_clear_each_sample=False,
                keepold_chunk_page_global_lookup=True,keepold_chunk_page_block_invalidation=True,
                keepold_chunk_page_defer_register=True,keepold_chunk_page_execution_enabled=True,
                keepold_chunk_page_exec_granularity='span',
                keepold_layer_metrics_mode='full' if self.config.gpu_reuse_diagnostics else 'minimal')
        clear_selective_recompute_runtime()
        try:
            configure_selective_recompute_runtime(layer_kv_data=dict(staged.layer_kv),
                prompt_len=len(prompt.token_ids),recompute_positions=staged.recompute_positions,
                reuse_old_positions=staged.reuse_old_positions,reuse_rope_impl='lmcache_cops_direct',
                lmcache_root=None,prefill_only=True,metrics_path=None,
                # The bundled GPU page-gather path requires this source-layout mode.
                dynamic_update_backend='cpu_rebuild',eager_load_source=True,**options)
            runtime=get_selective_recompute_runtime()
            if runtime is None or not runtime._source_loaded:raise RuntimeError('reusable KV was not loaded')
            trace=GpuReuseTrace() if self.config.gpu_reuse_diagnostics else None
            if trace is not None:trace.attach(runtime)
            params=SamplingParams(max_tokens=max_new_tokens,temperature=0.)
            output=self.llm.generate([{'prompt_token_ids':list(prompt.token_ids)}],params,use_tqdm=False)[0]
            elapsed=ttft(output)
            registration=runtime.commit_deferred_chunk_page_entries() if self.config.gpu_page_cache else {}
            result=dict(text=output.outputs[0].text.strip(),ttft_s=elapsed)
            if trace is not None:
                result['gpu_reuse']=trace.snapshot()
                result['gpu_reuse'].update(
                    expected_layers=len(self.layers),
                    registered_chunks=int((registration or {}).get('post_register_success',0)))
            return result
        finally:
            clear_selective_recompute_runtime()
            clear_keepold_page_bridge(bridge_key)

    def close(self):
        from vllm.v1.bench.external_kv_injector import (
            clear_selective_recompute_runtime,clear_keepold_chunk_page_index)
        clear_selective_recompute_runtime()
        clear_keepold_chunk_page_index()
        self.prefix={};self.layers=[];self.model=None
        llm,self.llm=self.llm,None
        del llm
