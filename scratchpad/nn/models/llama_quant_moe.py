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
from triteia.python.nn.linear import sparse_low_precision_linear
from triteia.python.ops.matmul.sbmm import (
    sbmm_4bit_2_4_native,
    sbmm_4bit_2_4_multilaunch,
    sbmm_4bit_2_4_forloop,
)

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
        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x, _ = self.down_proj(x)
        return x


class LLamaSBmm(nn.Module):
    def __init__(
        self, num_experts, infeatures, outfeatures, sbmm_type="naive", groupsize=-1
    ):
        super().__init__()
        if groupsize == -1:
            groupsize = infeatures
        self.infeatures = infeatures
        self.outfeatures = outfeatures
        self.groupsize = groupsize
        self.qweight = nn.Parameter(
            torch.empty(
                (num_experts, self.infeatures // 32, self.outfeatures * 16 // 8),
                dtype=torch.int32,
            ),
            False,
        )
        self.meta = nn.Parameter(
            torch.empty(
                (num_experts, self.outfeatures, self.infeatures // 16),
                dtype=torch.int16,
            ),
            False,
        )
        self.scales = nn.Parameter(
            torch.empty(
                (num_experts, self.infeatures // groupsize, self.outfeatures),
                dtype=torch.float16,
            ),
            False,
        )
        self.workspace = nn.Parameter(
            torch.zeros(num_experts, self.outfeatures // 128 * 16, dtype=torch.int32),
            False,
        )
        if sbmm_type == "naive":
            self.sbmm_func = sbmm_4bit_2_4_native
        elif sbmm_type == "multilaunch":
            self.sbmm_func = sbmm_4bit_2_4_multilaunch
        elif sbmm_type == "forloop":
            self.sbmm_func = sbmm_4bit_2_4_forloop
        else:
            raise NotImplementedError

    def forward(self, x, indices):
        return self.sbmm_func(
            qweights=self.qweight.data,
            xs=x,
            metas=self.meta.data,
            ss=self.scales.data,
            indices=indices,
        )


class LlamaCompressedMLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        num_experts: int,
        sbmm_type: str,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.intermediate_size = intermediate_size
        self.hidden_size = hidden_size
        self.gate_up_proj = LLamaSBmm(
            num_experts=num_experts,
            infeatures=hidden_size,
            outfeatures=intermediate_size * 2,
            sbmm_type=sbmm_type,
        )
        self.down_proj = LLamaSBmm(
            num_experts=num_experts,
            infeatures=intermediate_size,
            outfeatures=hidden_size,
            sbmm_type=sbmm_type,
        )

    def forward(self, x, indices):
        # assert not x.isnan().any()
        x = self.gate_up_proj(x, indices)
        # assert not gate_up.isnan().any()
        d = x.shape[-1] // 2
        x = F.silu(x[..., :d]) * x[..., d:]
        # assert not x.isnan().any()
        x = self.down_proj(x, indices)
        # assert not x.isnan().any()
        return x


class LlamaMoE(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        num_experts: int,
        experts_per_token: int,
        sbmm_type: str,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.experts_per_token = experts_per_token
        self.num_experts = num_experts
        self.base_mlp = LlamaMLP(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            hidden_act=hidden_act,
            quant_config=quant_config,
            prefix=f"{prefix}.mlp.EXPERT_ID",
        )

        self.mlp = LlamaCompressedMLP(
            num_experts=num_experts,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            hidden_act=hidden_act,
            quant_config=quant_config,
            sbmm_type=sbmm_type,
            prefix=f"{prefix}.mlp.",
        )
        self.gate = nn.Linear(hidden_size, num_experts, bias=False)

    def forward(self, x):
        base_y = self.base_mlp(x)
        original_shape = x.shape
        x = x.view(1, *x.shape) if x.dim() == 2 else x
        batch_size, sequence_length, hidden_dim = x.shape

        x = x.view(-1, hidden_dim)
        router_logits = self.gate(x)
        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(
            routing_weights, self.experts_per_token, dim=-1
        )
        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        routing_weights = routing_weights.to(x.dtype).T
        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, hidden_dim), dtype=x.dtype, device=x.device
        )
        sort_selected_experts, argsort_selected_experts = torch.sort(
            selected_experts.T, dim=-1
        )
        for k in range(self.experts_per_token):
            current_selected_experts = sort_selected_experts[k]
            current_routing_weights = routing_weights[k].view(-1, 1)
            current_argsort_selected_experts = argsort_selected_experts[k]
            sort_x = x[current_argsort_selected_experts]
            current_hidden_states = (
                self.mlp(sort_x, current_selected_experts)[
                    current_argsort_selected_experts
                ]
                * current_routing_weights
            )
            final_hidden_states += current_hidden_states

        final_hidden_states = final_hidden_states.reshape(
            batch_size, sequence_length, hidden_dim
        )
        final_hidden_states = final_hidden_states.view(original_shape)

        final_hidden_states = final_hidden_states.contiguous()
        base_y = base_y.contiguous()

        # For debugging
        # assert final_hidden_states.is_contiguous(), "final_hidden_states is not contiguous"
        # assert base_y.is_contiguous(), "base_y is not contiguous"
        # assert final_hidden_states.device == base_y.device, "Tensors are on different devices"
        # assert final_hidden_states.dtype == base_y.dtype, "Tensors have different dtypes"
        # assert not torch.isnan(final_hidden_states).any(), "NaN found in final_hidden_states"
        # assert not torch.isnan(base_y).any(), "NaN found in base_y"
        # assert not torch.isinf(final_hidden_states).any(), "Inf found in final_hidden_states"
        # assert not torch.isinf(base_y).any(), "Inf found in base_y"
        # assert final_hidden_states.shape == base_y.shape, "Tensors have different shapes"
        # torch.cuda.synchronize()
        result = final_hidden_states + base_y
        return result


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
            sbmm_type=config.sbmm_type,
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


class LlamaQuantisedMoEForCausalLM(nn.Module):
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

        name_transformations = [
            ("down_proj.0", "down_proj"),
            ("gate_up_proj.0", "gate_up_proj"),
            ("mlp.EXPERT_ID", "base_mlp"),
        ]
        for name, loaded_weight in weights:
            assert not loaded_weight.isnan().any()
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
                name = name.replace(weight_name, param_name)
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                for transformation in name_transformations:
                    if transformation[0] in name:
                        name = name.replace(transformation[0], transformation[1])
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                if name == "lm_head.0.weight":
                    continue
                if name == "model.embed_tokens.0.weight":
                    continue
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                for transformation in name_transformations:
                    if transformation[0] in name:
                        name = name.replace(transformation[0], transformation[1])
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


EntryClass = [LlamaQuantisedMoEForCausalLM]
