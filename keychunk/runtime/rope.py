"""Standalone fused position kernel for KeyChunk position rebasing."""
from functools import lru_cache
from pathlib import Path

@lru_cache(maxsize=1)
def load():
    from torch.utils.cpp_extension import load as compile_extension
    root = Path(__file__).with_name('native')
    return compile_extension(name='keychunk_position', sources=[str(root/'binding.cpp'), str(root/'position.cu')], extra_cflags=['-O3'], extra_cuda_cflags=['-O3'])

if __name__ == '__main__':
    load()
