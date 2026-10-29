"""Check local imports and pre-integrated model hooks without loading weights."""
import json
from pathlib import Path
from verify_runtime import verify

ROOT=Path(__file__).resolve().parents[1]


def main():
    runtime=verify(ROOT)
    from keychunk import KeyChunk,PreparedChunks,GenerationResult,select_chunks,token_f1,ttft
    from keychunk.runtime.vllm import VLLMBackend
    from vllm.v1.bench.external_kv_selective import SelectiveRecomputeRuntime
    assert all(callable(value) for value in (KeyChunk,PreparedChunks,GenerationResult,
        select_chunks,token_f1,ttft,VLLMBackend,SelectiveRecomputeRuntime))
    hook=ROOT/'vendor/vllm/vllm/model_executor/models/llama.py'
    compile(hook.read_text(),str(hook),'exec')
    print(json.dumps(dict(passed=True,version=runtime['version'],
        public_api_imported=True,selective_runtime_imported=True,
        model_hook_source_compiled=True,model_loaded=False),indent=2))


if __name__=='__main__':main()
