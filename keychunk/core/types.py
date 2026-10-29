from __future__ import annotations
from dataclasses import dataclass,field
from typing import Any,Mapping


@dataclass
class ChunkCapture:
    context_id:int
    offsets:Any
    orig_indices:Any
    token_ids:Any
    semvec:Any
    layer_k:tuple[Any,...]
    layer_v:tuple[Any,...]

    @property
    def num_chunks(self):return int(self.orig_indices.numel())

    @property
    def num_layers(self):return len(self.layer_k)

    def body_span(self,index):
        start,end=int(self.offsets[index]),int(self.offsets[index+1])
        if end<=start:raise ValueError('empty captured chunk')
        return start+1,end

    def body_token_count(self,index):
        start,end=self.body_span(index)
        return end-start


@dataclass(frozen=True)
class ChunkSpan:
    orig_idx:int
    body_start:int
    body_end:int

    @property
    def body_tokens(self):return self.body_end-self.body_start


@dataclass(frozen=True)
class PromptPlan:
    order:tuple[int,...]
    token_ids:tuple[int,...]
    chunks:tuple[ChunkSpan,...]
    prefix_end:int
    query_start:int
    query_end:int
    prompt_hash:str

    @property
    def total_body_tokens(self):return sum(span.body_tokens for span in self.chunks)


@dataclass(frozen=True)
class ClusterPlan:
    representatives:tuple[int,...]
    clusters:tuple[tuple[int,...],...]
    similarity:Any=field(repr=False,compare=False)


@dataclass
class StagedReuse:
    layer_kv:Mapping[int,Mapping[str,Any]]
    reuse_positions:tuple[int,...]
    reuse_old_positions:tuple[int,...]
    recompute_positions:tuple[int,...]
    selected_body_positions:tuple[int,...]
