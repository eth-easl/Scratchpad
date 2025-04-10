from typing import Any, Dict, Iterable, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from transformers import LlamaConfig
from scratchpad.config.cache_config import CacheConfig
from scratchpad.distributed import get_tensor_model_parallel_world_size
from scratchpad.nn.layers.rotary_embedding import get_rope
from scratchpad.nn.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from scratchpad.model_executor.model_loader import default_weight_loader

from scratchpad.nn.layers.activation import SiluAndMul
from scratchpad.nn.layers.layernorm import RMSNorm
from scratchpad.nn.layers.linear import (
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from scratchpad.nn.layers.logits_processor import LogitsProcessor, LogitsProcessorOutput
from scratchpad.nn.quantization.base_config import QuantizationConfig
from scratchpad.nn.attention.radix_attention import RadixAttention
from scratchpad.nn.utils import apply_torchao_config_
from scratchpad.scheduler.schedule_batch import global_args
from scratchpad.model_executor.forward_info import ForwardBatch
from .llama import LlamaAttention


class LlamaMLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.gate_up_proj",
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.down_proj",
        )
        if hidden_act != "silu":
            raise ValueError(
                f"Unsupported activation: {hidden_act}. "
                "Only silu is supported for now."
            )
        self.act_fn = SiluAndMul()

    def forward(self, x):
        x, _ = self.gate_up_proj(x)
        x = self.act_fn(x)
        x, _ = self.down_proj(x)
        return x


class LlamaMoE(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        num_experts: int,
        experts_per_token: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.experts_per_token = experts_per_token
        self.num_experts = num_experts
        self.mlp = nn.ModuleList(
            [
                LlamaMLP(
                    hidden_size=hidden_size,
                    intermediate_size=intermediate_size,
                    hidden_act=hidden_act,
                    quant_config=quant_config,
                    prefix=f"{prefix}.mlp.{i}",
                )
                for i in range(num_experts)
            ]
        )
        self.gate = nn.Linear(hidden_size, num_experts, bias=False)

    def forward(self, x):
        original_shape = x.shape
        x = x.view(1, *x.shape) if x.dim() == 2 else x
        batch_size, sequence_length, hidden_dim = x.shape
        router_logits = self.gate(x)
        weights, selected_experts = torch.topk(router_logits, self.experts_per_token)
        weights = F.softmax(weights, dim=2, dtype=torch.float).to(x.dtype)
        results = torch.zeros(
            (batch_size, sequence_length, hidden_dim),
            device=x.device,
            dtype=x.dtype,
        )
        selected_experts = selected_experts.to(dtype=torch.int32)
        for ix, expert in enumerate(self.mlp):
            ix = torch.tensor(ix, device=selected_experts.device)
            mask = selected_experts == ix
            batch_idx, tok_idx, expert_idx = torch.where(mask)
            if batch_idx.numel() != 0 and tok_idx.numel() != 0:
                results[batch_idx, tok_idx] += expert(x[batch_idx, tok_idx]) * weights[
                    batch_idx, tok_idx, expert_idx
                ].unsqueeze(-1)
        results = results.view(original_shape)
        return results


class LlamaDecoderLayer(nn.Module):
    def __init__(
        self,
        config: LlamaConfig,
        layer_id: int = 0,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        rope_theta = getattr(config, "rope_theta", 10000)
        rope_scaling = getattr(config, "rope_scaling", None)
        if rope_scaling is not None and getattr(
            config, "original_max_position_embeddings", None
        ):
            rope_scaling[
                "original_max_position_embeddings"
            ] = config.original_max_position_embeddings
        rope_is_neox_style = getattr(config, "rope_is_neox_style", True)
        max_position_embeddings = getattr(config, "max_position_embeddings", 8192)
        self.self_attn = LlamaAttention(
            config=config,
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            layer_id=layer_id,
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            rope_is_neox_style=rope_is_neox_style,
            max_position_embeddings=max_position_embeddings,
            quant_config=quant_config,
            prefix=f"{prefix}.self_attn",
        )
        self.moe = LlamaMoE(
            hidden_size=self.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            quant_config=quant_config,
            num_experts=config.num_experts,
            experts_per_token=config.experts_per_token,
            prefix=f"{prefix}.moe",
        )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        input_metadata: ForwardBatch,
        residual: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Self Attention
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
            input_metadata=input_metadata,
        )

        # Fully Connected
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.moe(hidden_states)
        return hidden_states, residual


class LlamaModel(nn.Module):
    def __init__(
        self,
        config: LlamaConfig,
        quant_config: Optional[QuantizationConfig] = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
        )
        self.layers = nn.ModuleList(
            [
                LlamaDecoderLayer(
                    config, i, quant_config=quant_config, prefix=f"model.layers.{i}"
                )
                for i in range(config.num_hidden_layers)
            ]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        input_metadata: ForwardBatch,
        input_embeds: torch.Tensor = None,
    ) -> torch.Tensor:
        if input_embeds is None:
            hidden_states = self.embed_tokens(input_ids)
        else:
            hidden_states = input_embeds
        residual = None
        for i in range(len(self.layers)):
            layer = self.layers[i]
            hidden_states, residual = layer(
                positions,
                hidden_states,
                input_metadata,
                residual,
            )
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


class LlamaMoEForCausalLM(nn.Module):
    def __init__(
        self,
        config: LlamaConfig,
        quant_config: Optional[QuantizationConfig] = None,
        cache_config: Optional[CacheConfig] = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.quant_config = quant_config
        self.torchao_config = global_args.torchao_config
        self.model = LlamaModel(config, quant_config=quant_config)
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)
        self.logits_processor = LogitsProcessor(config)

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        input_metadata: ForwardBatch,
        input_embeds: torch.Tensor = None,
    ) -> LogitsProcessorOutput:
        hidden_states = self.model(input_ids, positions, input_metadata, input_embeds)
        return self.logits_processor(
            input_ids, hidden_states, self.lm_head, input_metadata
        )

    def get_hidden_dim(self, module_name):
        if module_name in ["q_proj", "o_proj", "qkv_proj"]:
            return self.config.hidden_size, self.config.hidden_size
        elif module_name in ["kv_proj"]:
            return self.config.hidden_size, self.config.hidden_size // (
                self.config.num_attention_heads // self.config.num_key_value_heads
            )
        elif module_name == "gate_up_proj":
            return self.config.hidden_size, self.config.intermediate_size
        elif module_name == "down_proj":
            return self.config.intermediate_size, self.config.hidden_size
        else:
            raise NotImplementedError()

    def get_module_name(self, name):
        params_mapping = {
            "q_proj": "qkv_proj",
            "k_proj": "qkv_proj",
            "v_proj": "qkv_proj",
            "gate_proj": "gate_up_proj",
            "up_proj": "gate_up_proj",
        }
        return params_mapping.get(name, name)

    def get_module_name_from_weight_name(self, name):
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id, num_shard)
            ("qkv_proj", "q_proj", "q", 3),
            ("qkv_proj", "k_proj", "k", 3),
            ("qkv_proj", "v_proj", "v", 3),
            ("gate_up_proj", "gate_proj", 0, 2),
            ("gate_up_proj", "up_proj", 1, 2),
        ]
        for param_name, weight_name, shard_id, num_shard in stacked_params_mapping:
            if weight_name in name:
                return (
                    name.replace(weight_name, param_name)[: -len(".weight")],
                    num_shard,
                )
        return name[: -len(".weight")], 1

    def get_num_params(self):
        params_dict = dict(self.named_parameters())
        return len(params_dict)

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            (".qkv_proj", ".q_proj", "q"),
            (".qkv_proj", ".k_proj", "k"),
            (".qkv_proj", ".v_proj", "v"),
            (".gate_up_proj", ".gate_proj", 0),
            (".gate_up_proj", ".up_proj", 1),
        ]
        params_dict = dict(self.named_parameters())

        for name, loaded_weight in weights:
            # print(name)
            # assert not loaded_weight.isnan().any()
            # continue
            if "rotary_emb.inv_freq" in name or "projector" in name:
                continue
            if "rotary_emb.cos_cached" in name or "rotary_emb.sin_cached" in name:
                # Models trained using ColossalAI may include these tensors in
                # the checkpoint. Skip them.
                continue
            if name.startswith("model.vision_tower") and name not in params_dict:
                continue

            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                # print(name, name.replace(weight_name, param_name), shard_id)
                name = name.replace(weight_name, param_name)
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)

        if (
            hasattr(self.config, "tie_word_embeddings")
            and self.config.tie_word_embeddings
        ):
            # Tie output embedding layer to input embedding layer, to solve issues where lm_head.weight is missing
            param = self.lm_head.weight
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, self.model.embed_tokens.weight)
        apply_torchao_config_(self, params_dict, set(["proj.weight"]))


EntryClass = LlamaMoEForCausalLM
