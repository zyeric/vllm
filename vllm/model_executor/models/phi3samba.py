from typing import List, Optional, Tuple, Union, Iterable, Dict
import math
import copy

import torch
import torch.nn as nn

from einops import rearrange, repeat
from transformers.activations import ACT2FN

from vllm.attention import Attention, AttentionMetadata
from vllm.config import CacheConfig, LoRAConfig
from vllm.distributed import (get_pp_group, get_tensor_model_parallel_rank,
                              get_tensor_model_parallel_world_size)
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (MergedColumnParallelLinear,
                                               QKVParallelLinear,
                                               RowParallelLinear,
                                               ColumnParallelLinear)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.quantization.compressed_tensors.utils import (
    get_compressed_tensors_cache_scale)
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.sampler import Sampler, SamplerOutput
from vllm.model_executor.layers.vocab_parallel_embedding import (
    DEFAULT_VOCAB_PADDING_SIZE, ParallelLMHead, VocabParallelEmbedding)
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader, kv_cache_scales_loader, maybe_remap_kv_scale_name)
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.sequence import IntermediateTensors
from vllm.utils import is_hip
from vllm.worker.model_runner import (_BATCH_SIZES_TO_CAPTURE,
                                      _get_graph_batch_size)
from vllm.model_executor.models.mamba_cache import (MambaCacheManager,
                                                    MambaCacheParams)
from vllm.model_executor.models.interfaces import (HasInnerState,
                                                   IsAttentionFree)
from vllm.model_executor.layers.mamba.ops.causal_conv1d import (
    causal_conv1d_fn, causal_conv1d_update)
from vllm.model_executor.layers.mamba.ops.mamba_ssm import (
    selective_scan_fn, selective_state_update)


class SambaRotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Build here to make `torch.jit.trace` work.
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings, device=self.inv_freq.device, dtype=torch.get_default_dtype()
        )

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=torch.int64).type_as(self.inv_freq)

        freqs = torch.outer(t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    def forward(self):
        # TODO
        pass


class SambaMLP(nn.Module):
    """Gated Linear Unit.

    Reference:
        Language Modeling with Gated Convolutional Networks.
        https://arxiv.org/pdf/1612.08083v3.pdf.

    """

    def __init__(self, config):
        super().__init__()

        self.config = config
        self.fc1 = nn.Linear(config.hidden_size, 2 * config.intermediate_size, bias=False)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

        self.activation_fn = ACT2FN[config.hidden_act]

    def forward(self, hidden_states):
        y = self.fc1(hidden_states)
        gate, y = y.chunk(2, dim=-1)
        y = y * self.activation_fn(gate)
        return self.fc2(y)


class SambaAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config, layer_idx: Optional[int] = None, yoco_cross: bool = False):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        if layer_idx is None:
            logger.warning_once(
                f"Instantiating {self.__class__.__name__} without passing a `layer_idx` is not recommended and will "
                "lead to errors during the forward call if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )
        
        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True
        self.yoco_cross = yoco_cross
        
        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )

        op_size = self.num_heads * self.head_dim + 2 * (self.num_key_value_heads * self.head_dim)
        self.out_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        if yoco_cross:
            self.Wqkv =  nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        else:
            self.Wqkv = nn.Linear(self.hidden_size, op_size, bias=False)
        
        self.rotary_emb = SambaRotaryEmbedding(
            self.head_dim,
            max_position_embeddings=self.max_position_embeddings,
            base=self.rope_theta,
        )

    def forward(self):
        # TODO
        assert False


class Phi3Mamba(nn.Module):
    def __init__(
        self,
        d_model,
        d_state=16,
        d_conv=4,
        expand=2,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        conv_bias=True,
        bias=False,
        use_fast_path=True,  # Fused kernel options
        layer_idx=None,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.use_fast_path = use_fast_path
        self.layer_idx = layer_idx

        # self.conv1d = nn.Conv1d(
        #     in_channels=self.d_inner,
        #     out_channels=self.d_inner,
        #     bias=conv_bias,
        #     kernel_size=d_conv,
        #     groups=self.d_inner,
        #     padding=d_conv - 1,
        #     **factory_kwargs,
        # )

        self.conv1d = ColumnParallelLinear(
            input_size=d_conv,
            output_size=self.d_inner,
            bias=conv_bias,
            params_dtype=dtype,
        )
        # unsqueeze to fit conv1d weights shape into the linear weights shape.
        # Can't do this in `weight_loader` since it already exists in
        # `ColumnParallelLinear` and `set_weight_attrs`
        # doesn't allow to override it
        self.conv1d.weight.data = self.conv1d.weight.data.unsqueeze(1)

        # self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.in_proj = MergedColumnParallelLinear(self.d_model,
                                                  [self.d_inner] * 2,
                                                  bias=bias,
                                                  params_dtype=dtype)

        # self.x_proj = nn.Linear(
        #     self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
        # )
        # selective projection used to make dt, B and C input dependent
        self.x_proj = RowParallelLinear(
            self.d_inner,
            self.dt_rank + self.d_state * 2,
            bias=False,
            params_dtype=dtype,
        )

        # self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)
        # time step projection (discretization) -
        # In the forward we need to apply dt_proj without the bias,
        # as the bias is added in the selective scan kernel.
        self.dt_proj = ColumnParallelLinear(self.dt_rank,
                                            self.d_inner,
                                            bias=True,
                                            skip_bias_add=True,
                                            params_dtype=dtype)

        # S4D real initialization
        A = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32),
            "n -> d n",
            d=self.d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        self.A_log = nn.Parameter(A_log)

        # D "skip" parameter
        self.D = nn.Parameter(torch.ones(self.d_inner))  # Keep in fp32

        # self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.out_proj = RowParallelLinear(
            self.d_inner,
            self.d_model,
            bias=bias,
            input_is_parallel=True,
            params_dtype=dtype,
        )

        self.activation = "silu"

    def forward(
            self,
            hidden_states: torch.Tensor,
            attn_metadata: AttentionMetadata,
            mamba_cache_params: MambaCacheParams,
        ) -> torch.Tensor:
        input_dtype = hidden_states.dtype

        # 1. Gated MLP's linear projection
        # projected_states = self.in_proj(hidden_states)[0].transpose(-2, -1)
        projected_states = self.in_proj(hidden_states.to(self.in_proj.weight.dtype))[0].transpose(-2, -1)
        hidden_states, gate = projected_states.chunk(2, dim=-2)

        # 2. Convolution sequence transformation
        conv_weights = self.conv1d.weight.view(self.conv1d.weight.size(0),
                                               self.conv1d.weight.size(2))

        if attn_metadata.query_start_loc is not None \
            and attn_metadata.context_lens_tensor is not None:
            # |---------- N-1 iteration --------|
            # |---------------- N iteration ---------------------|
            # |- tokenA -|......................|-- newTokens ---|
            # |---------- context_len ----------|
            # |-------------------- seq_len ---------------------|
            #                                   |-- query_len ---|
            hidden_states = causal_conv1d_fn(
                hidden_states,
                conv_weights,
                self.conv1d.bias,
                activation=self.activation,
                conv_states=mamba_cache_params.conv_state,
                has_initial_state=attn_metadata.context_lens_tensor > 0,
                cache_indices=mamba_cache_params.state_indices_tensor,
                query_start_loc=attn_metadata.query_start_loc)
        else:
            hidden_states = causal_conv1d_update(
                hidden_states.transpose(0, 1),
                mamba_cache_params.conv_state,
                conv_weights,
                self.conv1d.bias,
                self.activation,
                conv_state_indices=mamba_cache_params.state_indices_tensor)
            hidden_states = hidden_states.transpose(0, 1)

        # 3. State Space Model sequence transformation
        # 3.a. input varying initialization of time_step, B and C
        ssm_parameters = self.x_proj(hidden_states.transpose(-2, -1))[0]

        time_step, B, C = torch.split(
            ssm_parameters,
            [self.dt_rank, self.d_state, self.d_state],
            dim=-1,
        )

        # Note that Jamba normalizes B, C, and time_step here but Mamba doesn't.

        discrete_time_step = self.dt_proj(time_step)[0].transpose(-2, -1)
        # 3.c perform the recurrence y ← SSM(A, B, C)(x)
        time_proj_bias = (self.dt_proj.bias.float() if hasattr(
            self.dt_proj, "bias") else None)

        if attn_metadata.query_start_loc is not None \
            and attn_metadata.context_lens_tensor is not None:
            scan_outputs = selective_scan_fn(
                hidden_states,
                mamba_cache_params.ssm_state,
                discrete_time_step,
                self.A_log,
                B.transpose(-2, -1),
                C.transpose(-2, -1),
                self.D.float(),
                gate,
                time_proj_bias,
                delta_softplus=True,
                cache_indices=mamba_cache_params.state_indices_tensor,
                has_initial_state=attn_metadata.context_lens_tensor > 0,
                query_start_loc=attn_metadata.query_start_loc)
        else:
            scan_outputs = selective_state_update(
                mamba_cache_params.ssm_state,
                hidden_states.transpose(0, 1),
                discrete_time_step.transpose(0, 1),
                self.A_log,
                B,
                C,
                self.D,
                gate.transpose(0, 1),
                time_proj_bias,
                dt_softplus=True,
                state_batch_indices=mamba_cache_params.state_indices_tensor)
            scan_outputs = scan_outputs.transpose(0, 1)

        # 4. Final linear projection
        contextualized_states = self.out_proj(scan_outputs.transpose(-2,
                                                                     -1))[0]

        contextualized_states = contextualized_states.to(input_dtype)
        return contextualized_states


class SambaDecoderLayer(nn.Module):
    
    def __init__(self, config, layer_idx) -> None:
        super().__init__()
    
        self.config = config
        self.layer_idx = layer_idx

        self.mlp = SambaMLP(config)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.layer_norm_eps)
        
        self.yoco_cross = False
        assert config.num_hidden_layers % 4 == 0, 'n_layer should be divisible by 4 for samba + yoco'
        if layer_idx< config.num_hidden_layers//2:
            self.use_mamba = config.mb_per_layer > 0 and layer_idx % config.mb_per_layer == 0
        else:
            self.yoco_cross = (layer_idx >= (config.num_hidden_layers//2 +2))
            self.use_mamba = False
            config = copy.deepcopy(config)
            config.sliding_window = None
        if self.use_mamba:
            factory_kwargs = {"dtype": torch.float32}
            self.attn = Phi3Mamba(config.hidden_size, layer_idx=layer_idx, **factory_kwargs)
        else:
            self.attn = SambaAttention(config, layer_idx=layer_idx, yoco_cross=self.yoco_cross)

        self.resid_attn_dropout = nn.Dropout(config.resid_pdrop)
        self.resid_mlp_dropout = nn.Dropout(config.resid_pdrop)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: AttentionMetadata,
        mamba_cache_params: MambaCacheParams,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        if self.use_mamba:
            assert kv_cache is None and mamba_cache_params is not None
        else:
            assert kv_cache is not None and mamba_cache_params is None

        residual = hidden_states
        # why need to cast here?
        hidden_states = self.input_layernorm(hidden_states.to(dtype=self.input_layernorm.weight.dtype))
        if self.use_mamba:
            attn_outputs = self.attn(
                hidden_states,
                attn_metadata,
                mamba_cache_params,
            )
        else:
            attn_outputs = self.attn(
                hidden_states,
                positions,
                kv_cache,
                attn_metadata,
            )

        hidden_states = residual + self.resid_attn_dropout(attn_outputs)

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states.to(dtype=self.post_attention_layernorm.weight.dtype))
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + self.resid_mlp_dropout(hidden_states)

        return hidden_states


class SambaModel(nn.Module):

    def __init__(
        self,
        config,
        cache_config = None,
        quant_config = None,
        lora_config = None,
    ) -> None:
        super().__init__()

        self.config = config

        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.embed_dropout = nn.Dropout(config.embd_pdrop)
        self.layers = nn.ModuleList(
            [SambaDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.final_layernorm = RMSNorm(config.hidden_size, eps=config.layer_norm_eps)
        # TODO

    def forward(
        self,
        input_ids: Optional[torch.Tensor],
        positions: torch.Tensor,
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
        mamba_cache_params: MambaCacheParams,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        print('>>>', input_ids.shape, input_ids)
        print('>>>', positions.shape, positions, positions[:100])
        hidden_states = self.embed_tokens(input_ids)
        print('>>>', hidden_states.shape)
        residual = None

        is_prefill = attn_metadata.prefill_metadata.block_tables.numel() == 0
        print('>>>', is_prefill)

        kv_cache_idx = 0
        mamba_state_idx = 0
        for i, layer in enumerate(self.layers):
            # no need to compute the last half of the layers in prefill for YOCO arch
            if is_prefill and i >= self.config.num_hidden_layers // 2:
                break

            print('>>>', i, layer.use_mamba, layer.yoco_cross)
            if layer.use_mamba:
                hidden_states = layer(
                    hidden_states,
                    positions,
                    None,
                    attn_metadata,
                    mamba_cache_params.at_layer_idx(mamba_state_idx),
                )
                mamba_state_idx += 1
            else:
                if i < self.config.num_hidden_layers // 2:
                    # sliding window attention
                    kv_cache = kv_caches[kv_cache_idx]
                    kv_cache_idx += 1
                elif not layer.yoco_cross:
                    # full attention that generates kv cache
                    kv_cache = kv_caches[kv_cache_idx]
                    kv_cache_idx += 1
                else:
                    # full attention that reuses kv cache
                    kv_cache = kv_caches[kv_cache_idx - 1]

                hidden_states = layer(
                    hidden_states,
                    positions,
                    kv_cache,
                    attn_metadata,
                    None,
                )

        hidden_states, _ = self.final_layernorm(hidden_states)
        assert False
        return hidden_states


class SambaForCausalLM(nn.Module, HasInnerState):

    def __init__(
        self,
        config,
        cache_config = None,
        quant_config = None,
        lora_config = None,
        scheduler_config = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.scheduler_config = scheduler_config
        self.model = SambaModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=True)

        self.mamba_cache: Optional[MambaCacheManager] = None
        self.logits_processor = LogitsProcessor(config.vocab_size,
                                                config.vocab_size)
        self.sampler = Sampler()
        # TODO

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        **kwargs,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        # print(attn_metadata)
        if self.mamba_cache is None:
            max_batch_size = _get_graph_batch_size(
                self.scheduler_config.max_num_seqs) if self.scheduler_config else \
                max(_BATCH_SIZES_TO_CAPTURE) + 2
            self.mamba_cache = MambaCacheManager(
                torch.float32, self.config.num_hidden_layers // 2 // self.config.mb_per_layer,
                max_batch_size, *self._get_mamba_cache_shape()
            )

        (
            mamba_cache_tensors,
            state_indices_tensor,
        ) = self.mamba_cache.current_run_tensors(input_ids, attn_metadata,
                                                 **kwargs)

        mamba_cache_params = MambaCacheParams(mamba_cache_tensors[0],
                                              mamba_cache_tensors[1],
                                              state_indices_tensor)

        # print(mamba_cache[0].shape, mamba_cache[1].shape)
        hidden_states = self.model(input_ids, positions, kv_caches,
                                   attn_metadata, mamba_cache_params)
        return hidden_states

    def copy_inputs_before_cuda_graphs(self, input_buffers, **kwargs):
        return self.mamba_cache.copy_inputs_before_cuda_graphs(
            input_buffers, **kwargs)

    def get_seqlen_agnostic_capture_inputs(self, batch_size: int):
        return self.mamba_cache.get_seqlen_agnostic_capture_inputs(batch_size)

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[torch.Tensor]:
        # TODO
        pass

    def sample(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[SamplerOutput]:
        # TODO
        pass

    def _swap_mamba_cache(self, from_index: int, to_index: int):
        assert len(self.mamba_cache) > 0
        for cache_t in self.mamba_cache:
            cache_t[:, [to_index,from_index]] = \
             cache_t[:, [from_index,to_index]]

    def _copy_mamba_cache(self, from_index: int, to_index: int):
        assert len(self.mamba_cache) > 0
        for cache_t in self.mamba_cache:
            cache_t[:, to_index].copy_(cache_t[:, from_index],
                                       non_blocking=True)

    def _move_out_if_already_occupied(self, index: int,
                                      all_occupied_indices: List[int]):
        if index in all_occupied_indices:
            first_free_index = self._first_free_index_in_mamba_cache()
            # In case occupied, move the occupied to a new empty block
            self._move_cache_index_and_mappings(from_index=index,
                                                to_index=first_free_index)

    def _assign_seq_id_to_mamba_cache_in_specific_dest(self, cur_rid: str,
                                                       seq_id: int,
                                                       destination_index: int):
        """
        Assign (req_id,seq_id) pair to a `destination_index` index, if
        already occupied, move the occupying index to a free index.
        """
        all_occupied_indices = self._get_all_occupied_indices()
        if cur_rid not in self.mamba_cache_indices_mapping:
            self._move_out_if_already_occupied(
                index=destination_index,
                all_occupied_indices=all_occupied_indices)
            self.mamba_cache_indices_mapping[cur_rid] = {
                seq_id: destination_index
            }
        elif seq_id not in (seq_ids2indices :=
                            self.mamba_cache_indices_mapping[cur_rid]):
            # parallel sampling , where n > 1, assume prefill have
            # already happened now we only need to copy the already
            # existing cache into the siblings seq_ids caches
            self._move_out_if_already_occupied(
                index=destination_index,
                all_occupied_indices=all_occupied_indices)
            index_exists = list(seq_ids2indices.values())[0]
            # case of decoding n>1, copy prefill cache to decoding indices
            self._copy_mamba_cache(from_index=index_exists,
                                   to_index=destination_index)
            self.mamba_cache_indices_mapping[cur_rid][
                seq_id] = destination_index
        else:
            # already exists
            cache_index_already_exists = self.mamba_cache_indices_mapping[
                cur_rid][seq_id]
            if cache_index_already_exists != destination_index:
                # In case the seq id already exists but not in
                # the right destination, swap it with what's occupying it
                self._swap_pair_indices_and_mappings(
                    from_index=cache_index_already_exists,
                    to_index=destination_index)

    def _prepare_current_run_mamba_cache(
            self, request_ids_to_seq_ids: Dict[str, list[int]],
            finished_requests_ids: List[str]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        running_indices = []
        request_ids_to_seq_ids_flatten = [
            (req_id, seq_id)
            for req_id, seq_ids in request_ids_to_seq_ids.items()
            for seq_id in seq_ids
        ]
        batch_size = len(request_ids_to_seq_ids_flatten)
        for dest_index, (request_id,
                         seq_id) in enumerate(request_ids_to_seq_ids_flatten):
            if request_id in finished_requests_ids:
                # Do not allocate cache index for requests that run
                # and finish right after
                continue
            self._assign_seq_id_to_mamba_cache_in_specific_dest(
                request_id, seq_id, dest_index)
            running_indices.append(dest_index)

        self._clean_up_first_bs_blocks(batch_size, running_indices)
        conv_state = self.mamba_cache[0][:, :batch_size]
        temporal_state = self.mamba_cache[1][:, :batch_size]

        return (conv_state, temporal_state)

    def _get_all_occupied_indices(self):
        return [
            cache_idx
            for seq_ids2indices in self.mamba_cache_indices_mapping.values()
            for cache_idx in seq_ids2indices.values()
        ]

    def _clean_up_first_bs_blocks(self, batch_size: int,
                                  indices_for_current_run: List[int]):
        # move out all of the occupied but currently not running blocks
        # outside of the first n blocks
        destination_indices = range(batch_size)
        max_possible_batch_size = self.mamba_cache[0].shape[1]
        for destination_index in destination_indices:
            if destination_index in self._get_all_occupied_indices() and  \
               destination_index not in indices_for_current_run:
                # move not running indices outside of the batch
                all_other_indices = list(
                    range(batch_size, max_possible_batch_size))
                first_avail_index = self._first_free_index_in_mamba_cache(
                    all_other_indices)
                self._swap_indices(from_index=destination_index,
                                   to_index=first_avail_index)

    def _move_cache_index_and_mappings(self, from_index: int, to_index: int):
        self._copy_mamba_cache(from_index=from_index, to_index=to_index)
        self._update_mapping_index(from_index=from_index, to_index=to_index)

    def _swap_pair_indices_and_mappings(self, from_index: int, to_index: int):
        self._swap_mamba_cache(from_index=from_index, to_index=to_index)
        self._swap_mapping_index(from_index=from_index, to_index=to_index)

    def _swap_mapping_index(self, from_index: int, to_index: int):
        for seq_ids2index in self.mamba_cache_indices_mapping.values():
            for seq_id, index in seq_ids2index.items():
                if from_index == index:
                    seq_ids2index.update({seq_id: to_index})
                elif to_index == index:
                    seq_ids2index.update({seq_id: from_index})

    def _update_mapping_index(self, from_index: int, to_index: int):
        for seq_ids2index in self.mamba_cache_indices_mapping.values():
            for seq_id, index in seq_ids2index.items():
                if from_index == index:
                    seq_ids2index.update({seq_id: to_index})
                    return

    # def copy_inputs_before_cuda_graphs(self, input_buffers, **kwargs):
    #     """
    #     Copy the relevant Mamba cache into the CUDA graph input buffer 
    #     that was provided during the capture runs 
    #     (JambaForCausalLM.mamba_gc_cache_buffer). 
    #     """
    #     self._release_finished_and_prepare_mamba_cache(
    #         kwargs["finished_requests_ids"], kwargs["request_ids_to_seq_ids"])

    # def get_seqlen_agnostic_capture_inputs(self, batch_size: int):
    #     """
    #     Provide the CUDA graph capture runs with a buffer in adjusted size.
    #     The buffer is used to maintain the Mamba Cache during the CUDA graph 
    #     replay runs.
    #     """
    #     return tuple(buffer[:, :batch_size] for buffer in self.mamba_cache)

    def _release_finished_and_prepare_mamba_cache(
            self, finished_requests_ids,
            request_ids_to_seq_ids) -> Tuple[torch.Tensor, torch.Tensor]:
        self._release_mamba_cache(finished_requests_ids)
        return self._prepare_current_run_mamba_cache(request_ids_to_seq_ids,
                                                     finished_requests_ids)

    def _release_mamba_cache(self, finished_seq_groups_req_ids: List[str]):
        for req_id in finished_seq_groups_req_ids:
            if req_id in self.mamba_cache_indices_mapping:
                self.mamba_cache_indices_mapping.pop(req_id)

    def _get_mamba_cache_shape(
            self
    ) -> Tuple[Optional[Tuple[int, int]], Optional[Tuple[int, int]]]:
        world_size = get_tensor_model_parallel_world_size()
        hidden_size = self.config.hidden_size
        # TODO: use hard-coded values for now
        mamba_expand = 2
        mamba_d_conv = 4
        mamba_d_state = 16
        conv_state_shape = (
            mamba_expand * hidden_size // world_size,
            mamba_d_conv - 1,
        )
        temporal_state_shape = (
            mamba_expand * hidden_size // world_size,
            mamba_d_state,
        )
        return conv_state_shape, temporal_state_shape

    def _prepare_mamba_cache(self):
        # dtype = self.lm_head.weight.dtype
        # TODO: hard-code
        dtype = torch.float32
        layers_type = self.config.layers_block_type
        mamba_layers = self.config.num_hidden_layers // 2 // self.config.mb_per_layer
        max_batch_size = (_get_graph_batch_size(
            self.scheduler_config.max_num_seqs) if self.scheduler_config else
                          max(_BATCH_SIZES_TO_CAPTURE) + 2)
        conv_state_shape, temporal_state_shape = self._get_mamba_cache_shape()
        assert conv_state_shape is not None and temporal_state_shape is not None

        self.mamba_cache = (torch.empty(size=(mamba_layers, max_batch_size) +
                                        conv_state_shape,
                                        dtype=dtype,
                                        device="cuda"),
                            torch.empty(size=(mamba_layers, max_batch_size) +
                                        temporal_state_shape,
                                        dtype=dtype,
                                        device="cuda"))

    def load_weights(
        self,
        weights: Iterable[Tuple[str, torch.Tensor]],
    ):
        weights = {name: weight for name, weight in weights}
        # for name, loaded_weight in weights.items():
        #     print(name, loaded_weight.shape)

        # for name, param in self.named_parameters():
        #     print(name, param.shape)
        missing_keys, unexpected_keys = self.load_state_dict(weights, strict=False)
        assert missing_keys == ['lm_head.weight'], f"Missing keys: {missing_keys}"
        assert not unexpected_keys, f"Unexpected keys: {unexpected_keys}"
        self.lm_head.weight.data.copy_(weights['model.embed_tokens.weight'])
