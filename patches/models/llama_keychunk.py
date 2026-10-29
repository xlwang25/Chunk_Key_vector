# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Adapted from
# https://github.com/huggingface/transformers/blob/v4.28.0/src/transformers/models/llama/modeling_llama.py
# Copyright 2023 The vLLM team.
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Inference-only LLaMA model compatible with HuggingFace weights."""
from collections.abc import Iterable
from typing import Any, Optional, Union
import time

import torch
import torch.nn.functional as F
from torch import nn
from transformers import LlamaConfig

from vllm.attention import Attention, AttentionType
from vllm.compilation.decorators import support_torch_compile
from vllm.config import CacheConfig, VllmConfig
from vllm.distributed import get_pp_group, get_tensor_model_parallel_world_size
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (MergedColumnParallelLinear,
                                               QKVParallelLinear,
                                               RowParallelLinear)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.vocab_parallel_embedding import (
    DEFAULT_VOCAB_PADDING_SIZE, ParallelLMHead, VocabParallelEmbedding)
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader, maybe_remap_kv_scale_name)
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.sequence import IntermediateTensors
from vllm.v1.bench.external_kv_injector import get_selective_recompute_runtime

from .interfaces import SupportsEagle3, SupportsLoRA, SupportsPP
from .utils import (AutoWeightsLoader, PPMissingLayer, extract_layer_index,
                    is_pp_missing_parameter,
                    make_empty_intermediate_tensors_factory, make_layers,
                    maybe_prefix)


class LlamaMLP(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        quant_config: Optional[QuantizationConfig] = None,
        bias: bool = False,
        prefix: str = "",
        reduce_results: bool = True,
    ) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            input_size=hidden_size,
            output_sizes=[intermediate_size] * 2,
            bias=bias,
            quant_config=quant_config,
            prefix=f"{prefix}.gate_up_proj",
        )
        self.down_proj = RowParallelLinear(
            input_size=intermediate_size,
            output_size=hidden_size,
            bias=bias,
            quant_config=quant_config,
            reduce_results=reduce_results,
            prefix=f"{prefix}.down_proj",
        )
        if hidden_act != "silu":
            raise ValueError(f"Unsupported activation: {hidden_act}. "
                             "Only silu is supported for now.")
        self.act_fn = SiluAndMul()

    def forward(self, x):
        x, _ = self.gate_up_proj(x)
        x = self.act_fn(x)
        x, _ = self.down_proj(x)
        return x


class LlamaAttention(nn.Module):

    def __init__(
        self,
        config: LlamaConfig,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        rope_theta: float = 10000,
        rope_scaling: Optional[dict[str, Any]] = None,
        max_position_embeddings: int = 8192,
        quant_config: Optional[QuantizationConfig] = None,
        bias: bool = False,
        bias_o_proj: bool = False,
        cache_config: Optional[CacheConfig] = None,
        prefix: str = "",
        attn_type: str = AttentionType.DECODER,
    ) -> None:
        super().__init__()
        layer_idx = extract_layer_index(prefix)
        self.hidden_size = hidden_size
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        if self.total_num_kv_heads >= tp_size:
            # Number of KV heads is greater than TP size, so we partition
            # the KV heads across multiple tensor parallel GPUs.
            assert self.total_num_kv_heads % tp_size == 0
        else:
            # Number of KV heads is less than TP size, so we replicate
            # the KV heads across multiple tensor parallel GPUs.
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        # MistralConfig has an optional head_dim introduced by Mistral-Nemo
        head_dim = getattr(config, "head_dim", None)
        if head_dim is None:
            head_dim = self.hidden_size // self.total_num_heads
        self.head_dim = head_dim
        # Phi models introduced a partial_rotary_factor parameter in the config
        self.partial_rotary_factor = getattr(config, "partial_rotary_factor",
                                             1)
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.rope_theta = rope_theta
        self.max_position_embeddings = max_position_embeddings

        self.qkv_proj = QKVParallelLinear(
            hidden_size=hidden_size,
            head_size=self.head_dim,
            total_num_heads=self.total_num_heads,
            total_num_kv_heads=self.total_num_kv_heads,
            bias=bias,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )

        self.o_proj = RowParallelLinear(
            input_size=self.total_num_heads * self.head_dim,
            output_size=hidden_size,
            bias=bias_o_proj,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )

        self._init_rotary_emb(config,
                              rope_scaling=rope_scaling,
                              quant_config=quant_config)

        sliding_window = None
        if layer_types := getattr(config, "layer_types", None):
            is_sliding = layer_types[layer_idx] == "sliding_attention"
            if is_sliding:
                sliding_window = config.sliding_window

        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            cache_config=cache_config,
            quant_config=quant_config,
            per_layer_sliding_window=sliding_window,
            attn_type=attn_type,
            prefix=f"{prefix}.attn",
        )

        # KeyChunk test hook: capture post-RoPE K/V for prompt(prefill) tokens.
        self.hack_kv = None
        self._collect_kv = False
        self._hack_k_segments = []
        self._hack_v_segments = []

        # KeyChunk test hook: capture pre-RoPE K semantic mean online.
        self._collect_semvec = False
        self._sem_k_sum = None
        self._sem_token_count = 0
        self.layer_idx = int(layer_idx) if layer_idx is not None else -1

    def _expand_kv_heads_to_attention_heads(self,
                                            tensor: torch.Tensor) -> torch.Tensor:
        """Expand KV heads to Q heads for GQA/MQA in selective mode."""
        if self.num_heads == self.num_kv_heads:
            return tensor
        if self.num_heads % self.num_kv_heads != 0:
            raise ValueError(
                "num_heads must be divisible by num_kv_heads in selective mode, "
                f"got num_heads={self.num_heads}, num_kv_heads={self.num_kv_heads}")
        repeat_factor = self.num_heads // self.num_kv_heads
        return tensor.repeat_interleave(repeat_factor, dim=1)

    def _selective_attention(
        self,
        q: torch.Tensor,
        full_k: torch.Tensor,
        full_v: torch.Tensor,
        query_positions: torch.Tensor,
        full_positions: torch.Tensor,
    ) -> torch.Tensor:
        """Attention over full K/V with compressed active Q tokens."""
        # q: [A, num_heads, head_dim]
        # full_k/full_v: [T, num_kv_heads, head_dim]
        q_heads = q.transpose(0, 1).unsqueeze(0)
        k_heads = self._expand_kv_heads_to_attention_heads(
            full_k).transpose(0, 1).unsqueeze(0)
        v_heads = self._expand_kv_heads_to_attention_heads(
            full_v).transpose(0, 1).unsqueeze(0)

        # Allow keys with position <= query position (causal).
        allow = full_positions.unsqueeze(0) <= query_positions.unsqueeze(1)
        attn_mask = torch.full(
            allow.shape,
            float("-inf"),
            dtype=q.dtype,
            device=q.device,
        )
        attn_mask.masked_fill_(allow, 0.0)

        out = F.scaled_dot_product_attention(
            q_heads,
            k_heads,
            v_heads,
            attn_mask=attn_mask,
            dropout_p=0.0,
            is_causal=False,
        )
        out = out.squeeze(0).transpose(0, 1).contiguous()
        return out

    def _forward_standard(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

        if self._collect_semvec:
            # Pre-RoPE semantic feature capture.
            sem_k = None
            if k.shape[0] > 1:
                sem_k = k
            elif self._sem_token_count == 0:
                sem_k = k
            if sem_k is not None:
                sem_sum = sem_k.to(torch.float32).sum(dim=0)
                if self._sem_k_sum is None:
                    self._sem_k_sum = sem_sum
                else:
                    self._sem_k_sum = self._sem_k_sum + sem_sum
                self._sem_token_count += int(sem_k.shape[0])

        q, k = self.rotary_emb(positions, q, k)

        if self._collect_kv:
            # We only need prompt/prefill tokens. Decode step is usually length=1.
            if k.shape[0] > 1:
                self._hack_k_segments.append(k.detach().clone().contiguous())
                self._hack_v_segments.append(v.detach().clone().contiguous())
            elif not self._hack_k_segments:
                # Fallback for very short prompts.
                self._hack_k_segments.append(k.detach().clone().contiguous())
                self._hack_v_segments.append(v.detach().clone().contiguous())

        attn_output = self.attn(q, k, v)
        output, _ = self.o_proj(attn_output)
        return output

    def _forward_selective(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        selective_runtime: Any,
    ) -> torch.Tensor:
        # hidden_states/positions here are already compressed to recompute tokens.
        qkv, _ = self.qkv_proj(hidden_states)
        q, k_new, v_new = qkv.split([self.q_size, self.kv_size, self.kv_size],
                                    dim=-1)
        if q.device.type == "cuda":
            torch.cuda.synchronize(q.device)
        t_rope0 = time.perf_counter()
        q, k_new = self.rotary_emb(positions, q, k_new)
        if q.device.type == "cuda":
            torch.cuda.synchronize(q.device)
        t_rope1 = time.perf_counter()
        selective_runtime.append_metric_event({
            "event": "selective_recompute_rope",
            "layer_name": self.attn.layer_name,
            "layer_id": self.layer_idx,
            "recompute_tokens": int(q.shape[0]),
            "t_recompute_rope_ms": (t_rope1 - t_rope0) * 1000.0,
            "ts_unix": time.time(),
        })
        q = q.view(-1, self.num_heads, self.head_dim)
        k_new = k_new.view(-1, self.num_kv_heads, self.head_dim)
        v_new = v_new.view(-1, self.num_kv_heads, self.head_dim)

        full_k, full_v = selective_runtime.compose_full_kv(
            layer_id=self.layer_idx,
            new_k=k_new,
            new_v=v_new,
            layer_name=self.attn.layer_name,
            reuse_rope_cos_sin_cache=self.rotary_emb.cos_sin_cache,
            reuse_rope_is_neox_style=self.rotary_emb.is_neox_style,
        )
        selective_runtime.write_layer_kv_to_paged_cache(
            attn_layer=self.attn,
            layer_name=self.attn.layer_name,
            full_k=full_k,
            full_v=full_v,
            layer_id=self.layer_idx,
        )
        full_positions = selective_runtime.get_full_positions(positions.device)
        attn_output = self._selective_attention(
            q,
            full_k,
            full_v,
            positions,
            full_positions,
        )
        attn_output = attn_output.view(attn_output.shape[0], -1)
        output, _ = self.o_proj(attn_output)
        return output

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        selective_runtime = get_selective_recompute_runtime()
        if selective_runtime is not None and selective_runtime.is_active():
            return self._forward_selective(positions, hidden_states,
                                           selective_runtime)
        return self._forward_standard(positions, hidden_states)

    def reset_hack_kv(self) -> None:
        self.hack_kv = None
        self._hack_k_segments = []
        self._hack_v_segments = []

    def finalize_hack_kv(self):
        if self.hack_kv is not None:
            return self.hack_kv
        if not self._hack_k_segments:
            return None
        if len(self._hack_k_segments) == 1:
            self.hack_kv = [self._hack_k_segments[0], self._hack_v_segments[0]]
            return self.hack_kv
        self.hack_kv = [
            torch.cat(self._hack_k_segments, dim=0).contiguous(),
            torch.cat(self._hack_v_segments, dim=0).contiguous(),
        ]
        return self.hack_kv

    def reset_sem_k_mean(self) -> None:
        self._sem_k_sum = None
        self._sem_token_count = 0

    def finalize_sem_k_mean(self):
        if self._sem_k_sum is None or self._sem_token_count <= 0:
            return None
        return (self._sem_k_sum / float(self._sem_token_count)).contiguous()

    def _init_rotary_emb(self, config: LlamaConfig,
                         rope_scaling: Optional[dict[str, Any]],
                         quant_config: Optional[QuantizationConfig]) -> None:
        is_neox_style = True
        is_gguf = quant_config and quant_config.get_name() == "gguf"
        if is_gguf and config.model_type == "llama":
            is_neox_style = False

        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=self.max_position_embeddings,
            base=self.rope_theta,
            rope_scaling=rope_scaling,
            is_neox_style=is_neox_style,
            partial_rotary_factor=self.partial_rotary_factor,
        )


class LlamaDecoderLayer(nn.Module):

    def __init__(
        self,
        config: LlamaConfig,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        rope_theta = getattr(config, "rope_theta", 10000)
        rope_scaling = getattr(config, "rope_scaling", None)
        if rope_scaling is not None and getattr(
                config, "original_max_position_embeddings", None):
            rope_scaling["original_max_position_embeddings"] = (
                config.original_max_position_embeddings)
        max_position_embeddings = getattr(config, "max_position_embeddings",
                                          8192)
        # Support abacusai/Smaug-72B-v0.1 with attention_bias
        # Support internlm/internlm-7b with bias
        attention_bias = getattr(config, "attention_bias", False) or getattr(
            config, "bias", False)
        bias_o_proj = attention_bias
        # support internlm/internlm3-8b with qkv_bias
        if hasattr(config, 'qkv_bias'):
            attention_bias = config.qkv_bias

        # By default, Llama uses causal attention as it is a decoder-only model.
        # You can override the HF config with `is_causal=False` to enable
        # bidirectional attention, which is used in some embedding models
        # (e.g. parasail-ai/GritLM-7B-vllm)
        if getattr(config, "is_causal", True):
            attn_type = AttentionType.DECODER
        else:
            attn_type = AttentionType.ENCODER_ONLY

        self.self_attn = LlamaAttention(
            config=config,
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=getattr(config, "num_key_value_heads",
                                 config.num_attention_heads),
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            max_position_embeddings=max_position_embeddings,
            quant_config=quant_config,
            bias=attention_bias,
            bias_o_proj=bias_o_proj,
            cache_config=cache_config,
            prefix=f"{prefix}.self_attn",
            attn_type=attn_type,
        )
        self.mlp = LlamaMLP(
            hidden_size=self.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            quant_config=quant_config,
            bias=getattr(config, "mlp_bias", False),
            prefix=f"{prefix}.mlp",
        )
        self.input_layernorm = RMSNorm(config.hidden_size,
                                       eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size,
                                                eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Self Attention
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(
                hidden_states, residual)
        hidden_states = self.self_attn(positions=positions,
                                       hidden_states=hidden_states)

        # Fully Connected
        hidden_states, residual = self.post_attention_layernorm(
            hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


@support_torch_compile
class LlamaModel(nn.Module):

    def __init__(self,
                 *,
                 vllm_config: VllmConfig,
                 prefix: str = "",
                 layer_type: type[nn.Module] = LlamaDecoderLayer):
        super().__init__()

        config = vllm_config.model_config.hf_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config
        lora_config = vllm_config.lora_config

        self.config = config
        self.quant_config = quant_config
        lora_vocab = (lora_config.lora_extra_vocab_size *
                      (lora_config.max_loras or 1)) if lora_config else 0
        self.vocab_size = config.vocab_size + lora_vocab
        self.org_vocab_size = config.vocab_size
        if get_pp_group().is_first_rank or (config.tie_word_embeddings
                                            and get_pp_group().is_last_rank):
            self.embed_tokens = VocabParallelEmbedding(
                self.vocab_size,
                config.hidden_size,
                org_num_embeddings=config.vocab_size,
                quant_config=quant_config,
            )
        else:
            self.embed_tokens = PPMissingLayer()
        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: layer_type(config=config,
                                      cache_config=cache_config,
                                      quant_config=quant_config,
                                      prefix=prefix),
            prefix=f"{prefix}.layers",
        )
        if get_pp_group().is_last_rank:
            self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        else:
            self.norm = PPMissingLayer()

        self.aux_hidden_state_layers: tuple[int] = tuple()

        self.make_empty_intermediate_tensors = (
            make_empty_intermediate_tensors_factory(
                ["hidden_states", "residual"], config.hidden_size))

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: Optional[torch.Tensor],
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors],
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, IntermediateTensors, tuple[torch.Tensor,
                                                        list[torch.Tensor]]]:
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = self.get_input_embeddings(input_ids)
            residual = None
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]

        selective_runtime = get_selective_recompute_runtime()
        selective_active = False
        if selective_runtime is not None:
            selective_active = selective_runtime.begin_forward(positions)
            if selective_active:
                if not (get_pp_group().is_first_rank
                        and get_pp_group().is_last_rank):
                    raise RuntimeError(
                        "Selective recompute currently supports single PP rank only."
                    )
                recompute_pos = selective_runtime.get_recompute_positions(
                    hidden_states.device)
                hidden_states = hidden_states.index_select(0, recompute_pos)
                positions = positions.index_select(0, recompute_pos)
                if residual is not None:
                    residual = residual.index_select(0, recompute_pos)

        aux_hidden_states = []
        try:
            for idx, layer in enumerate(
                    self.layers[self.start_layer:self.end_layer]):
                if idx in self.aux_hidden_state_layers:
                    aux_hidden_states.append(hidden_states + residual)
                hidden_states, residual = layer(positions, hidden_states,
                                                residual)

            if not get_pp_group().is_last_rank:
                return IntermediateTensors({
                    "hidden_states": hidden_states,
                    "residual": residual
                })

            hidden_states, _ = self.norm(hidden_states, residual)
            if selective_active:
                hidden_states = selective_runtime.expand_hidden_states(
                    hidden_states)
                selective_runtime.end_forward()

            if len(aux_hidden_states) > 0:
                return hidden_states, aux_hidden_states
            return hidden_states
        finally:
            if (selective_active and selective_runtime is not None
                    and selective_runtime.is_active()):
                selective_runtime.end_forward()

    def set_collect_kv(self, flag: bool) -> None:
        for layer in self.layers:
            if hasattr(layer, "self_attn"):
                attn = layer.self_attn
                attn._collect_kv = flag
                if flag:
                    attn.reset_hack_kv()

    def reset_hack_kv(self) -> None:
        for layer in self.layers:
            if hasattr(layer, "self_attn"):
                layer.self_attn.reset_hack_kv()

    def set_collect_semvec(self, flag: bool) -> None:
        for layer in self.layers:
            if hasattr(layer, "self_attn"):
                attn = layer.self_attn
                attn._collect_semvec = flag
                if flag:
                    attn.reset_sem_k_mean()

    def reset_semvec(self) -> None:
        for layer in self.layers:
            if hasattr(layer, "self_attn"):
                layer.self_attn.reset_sem_k_mean()

    def load_weights(self, weights: Iterable[tuple[str,
                                                   torch.Tensor]]) -> set[str]:
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            (".qkv_proj", ".q_proj", "q"),
            (".qkv_proj", ".k_proj", "k"),
            (".qkv_proj", ".v_proj", "v"),
            (".gate_up_proj", ".gate_proj", 0),
            (".gate_up_proj", ".up_proj", 1),
        ]
        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()
        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue
            if ("rotary_emb.cos_cached" in name
                    or "rotary_emb.sin_cached" in name):
                # Models trained using ColossalAI may include these tensors in
                # the checkpoint. Skip them.
                continue
            if (self.quant_config is not None and
                (scale_name := self.quant_config.get_cache_scale(name))):
                # Loading kv cache quantization scales
                param = params_dict[scale_name]
                weight_loader = getattr(param, "weight_loader",
                                        default_weight_loader)
                loaded_weight = (loaded_weight if loaded_weight.dim() == 0 else
                                 loaded_weight[0])
                weight_loader(param, loaded_weight)
                loaded_params.add(scale_name)
                continue
            if "scale" in name:
                # Remapping the name of FP8 kv-scale.
                name = maybe_remap_kv_scale_name(name, params_dict)
                if name is None:
                    continue
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue

                if is_pp_missing_parameter(name, self):
                    continue

                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue

                if is_pp_missing_parameter(name, self):
                    continue

                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader",
                                        default_weight_loader)
                weight_loader(param, loaded_weight)
            loaded_params.add(name)
        return loaded_params


class LlamaForCausalLM(nn.Module, SupportsLoRA, SupportsPP, SupportsEagle3):
    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"]
    }

    # LoRA specific attributes
    embedding_modules = {
        "embed_tokens": "input_embeddings",
        "lm_head": "output_embeddings"
    }
    embedding_padding_modules = ["lm_head"]

    # Mistral/Llama models can also be loaded with --load-format mistral
    # from consolidated.safetensors checkpoints
    mistral_mapping = {
        "layers": "model.layers",
        "attention": "self_attn",
        "qscale_act": "input_scale",
        "qscale_weight": "weight_scale",
        "kv_fake_quantizer.qscale_act": "kv_scale",
        "q_fake_quantizer.qscale_act": "attn.q_scale",
        "k_fake_quantizer.qscale_act": "k_scale",
        "v_fake_quantizer.qscale_act": "v_scale",
        "wq": "q_proj",
        "wk": "k_proj",
        "wv": "v_proj",
        "wo": "o_proj",
        "attention_norm": "input_layernorm",
        "feed_forward": "mlp",
        "w1": "gate_proj",
        "w2": "down_proj",
        "w3": "up_proj",
        "ffn_norm": "post_attention_layernorm",
        "tok_embeddings": "model.embed_tokens",
        "output": "lm_head",
        "norm": "model.norm",
    }

    def __init__(self,
                 *,
                 vllm_config: VllmConfig,
                 prefix: str = "",
                 layer_type: type[nn.Module] = LlamaDecoderLayer):
        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        lora_config = vllm_config.lora_config
        self.config = config
        self.lora_config = lora_config

        self.model = self._init_model(vllm_config=vllm_config,
                                      prefix=maybe_prefix(prefix, "model"),
                                      layer_type=layer_type)

        if get_pp_group().is_last_rank:
            self.unpadded_vocab_size = config.vocab_size
            if lora_config:
                self.unpadded_vocab_size += lora_config.lora_extra_vocab_size
            self.lm_head = ParallelLMHead(
                self.unpadded_vocab_size,
                config.hidden_size,
                org_num_embeddings=config.vocab_size,
                padding_size=(
                    DEFAULT_VOCAB_PADDING_SIZE
                    # We need bigger padding if using lora for kernel
                    # compatibility
                    if not lora_config else
                    lora_config.lora_vocab_padding_size),
                quant_config=quant_config,
                prefix=maybe_prefix(prefix, "lm_head"),
            )
            if config.tie_word_embeddings:
                self.lm_head = self.lm_head.tie_weights(
                    self.model.embed_tokens)

            logit_scale = getattr(config, "logit_scale", 1.0)
            self.logits_processor = LogitsProcessor(self.unpadded_vocab_size,
                                                    config.vocab_size,
                                                    logit_scale)
        else:
            self.lm_head = PPMissingLayer()

        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors)

    def set_aux_hidden_state_layers(self, layers: tuple[int]) -> None:
        self.model.aux_hidden_state_layers = layers

    def get_eagle3_aux_hidden_state_layers(self) -> tuple[int]:
        num_layers = len(self.model.layers)
        return (2, num_layers // 2, num_layers - 3)

    def _init_model(self,
                    vllm_config: VllmConfig,
                    prefix: str = "",
                    layer_type: type[nn.Module] = LlamaDecoderLayer):
        return LlamaModel(vllm_config=vllm_config,
                          prefix=prefix,
                          layer_type=layer_type)

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.get_input_embeddings(input_ids)

    def set_collect_kv(self, flag: bool) -> None:
        self.model.set_collect_kv(flag)

    def reset_hack_kv(self) -> None:
        self.model.reset_hack_kv()

    def set_collect_semvec(self, flag: bool) -> None:
        self.model.set_collect_semvec(flag)

    def reset_semvec(self) -> None:
        self.model.reset_semvec()

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        model_output = self.model(input_ids, positions, intermediate_tensors,
                                  inputs_embeds)
        return model_output

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[torch.Tensor]:
        logits = self.logits_processor(self.lm_head, hidden_states,
                                       sampling_metadata)
        return logits

    def load_weights(self, weights: Iterable[tuple[str,
                                                   torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(
            self,
            skip_prefixes=(["lm_head."]
                           if self.config.tie_word_embeddings else None),
        )
        return loader.load_weights(
            self.maybe_remap_mistral(name, loaded_weight)
            for name, loaded_weight in weights)

    # This function is used to remap the mistral format as
    # used by Mistral and Llama <=2
    def maybe_remap_mistral(
        self,
        name: str,
        loaded_weight: torch.Tensor,
    ) -> tuple[str, torch.Tensor]:

        def permute(w: torch.Tensor, n_heads: int):
            attn_in = self.config.head_dim * n_heads
            attn_out = self.config.hidden_size

            return w.view(n_heads, attn_in // n_heads // 2, 2,
                          attn_out).transpose(1, 2).reshape(attn_in, attn_out)

        mapping = self.mistral_mapping
        modules = name.split(".")

        # rotary embeds should be sliced
        if "wk" in modules and modules[-1] == "weight":
            loaded_weight = permute(loaded_weight,
                                    self.config.num_key_value_heads)
        elif "wq" in modules and modules[-1] == "weight":
            loaded_weight = permute(loaded_weight,
                                    self.config.num_attention_heads)

        num_modules = len(modules)
        for i in range(num_modules):
            item = modules[i]
            next_item = modules[i + 1] if i < num_modules - 1 else None

            combined_item = (f"{item}.{next_item}"
                             if next_item is not None else None)

            if combined_item in mapping:
                name = name.replace(combined_item, mapping[combined_item])
            elif item in mapping and mapping[item] not in name:
                name = name.replace(item, mapping[item])

        return name, loaded_weight
