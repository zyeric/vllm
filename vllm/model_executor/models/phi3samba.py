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
from vllm.attention.backends.abstract import (AttentionBackend, AttentionImpl,
                                              AttentionMetadata, AttentionType)
from vllm.attention.ops.paged_attn import (PagedAttention,
                                           PagedAttentionMetadata)
from vllm.attention.backends.xformers import _get_seq_len_block_table_args


PREFILL_BLOCK_TABLES = []


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

    def __init__(self, config, layer_idx: Optional[int] = None, yoco_cross: bool = False, cache_config: Optional[CacheConfig] = None):
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

        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=self.max_position_embeddings,
            base=self.rope_theta,
        )
        assert self.config.attention_dropout == 0.0, 'Attention dropout is not supported for now'
        # TODO: double check sliding window
        new_cache_config = copy.deepcopy(cache_config)
        # disable sliding window for the second half of the model
        if layer_idx >= config.num_hidden_layers // 2:
            new_cache_config.sliding_window = None
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.head_dim**-0.5,
            num_kv_heads=self.num_key_value_heads,
            cache_config=new_cache_config,
        )

    def forward(
            self,
            hidden_states: torch.Tensor,
            positions: torch.Tensor,
            kv_cache: torch.Tensor,
            attn_metadata: AttentionMetadata,
        ):
        if not self.yoco_cross:
            qkv = self.Wqkv(hidden_states)
            q, k, v = qkv.split([self.hidden_size, self.num_key_value_heads * self.head_dim, self.num_key_value_heads * self.head_dim], dim=-1)
            q, k = self.rotary_emb(positions, q, k)
            attn_output = self.attn(q, k, v, kv_cache, attn_metadata)
        else:
            q = self.Wqkv(hidden_states)
            mock_k = torch.zeros_like(q)
            q, _ = self.rotary_emb(positions, q, mock_k)
            key_cache, value_cache = PagedAttention.split_kv_cache(kv_cache, self.num_key_value_heads, self.head_dim)
            # print('>>>', q.shape, key_cache.shape, value_cache.shape)
            # print('>>> prefill_metadata', attn_metadata.prefill_metadata)
            # print('>>> decode_metadata', attn_metadata.decode_metadata)

            # use randn_like to make it e2e runnable temporarily
            # should use empty_like instead
            output = torch.empty_like(q)
            # means prefill?
            if not attn_metadata.decode_metadata:
                # print('>>>', PREFILL_BLOCK_TABLES)
                block_tables_arg = torch.Tensor(PREFILL_BLOCK_TABLES).to(q.device).to(torch.int32)
                seq_lens_arg = attn_metadata.seq_lens_tensor
                # hard code for now
                max_seq_len_arg = 4096
            else:
                block_tables_arg = attn_metadata.block_tables
                seq_lens_arg = attn_metadata.seq_lens_tensor
                max_seq_len_arg = attn_metadata.max_decode_seq_len
            # print('>>> block_tables_arg', block_tables_arg)
            # print('>>> seq_lens_arg', seq_lens_arg)
            # print('>>> max_seq_len_arg', max_seq_len_arg)

            query = q.view(-1, self.num_heads, self.head_dim)
            # seems we cannot use the wrapped flash-attn backend here
            output = PagedAttention.forward_decode(
                query,
                key_cache,
                value_cache,
                block_tables_arg,
                seq_lens_arg,
                max_seq_len_arg,
                self.attn.kv_cache_dtype,
                self.num_key_value_heads,
                self.head_dim**-0.5,
                alibi_slopes=None,
                k_scale=1.0,
                v_scale=1.0,
            )
            attn_output = output.view(-1, q.size(-1))
        return self.out_proj(attn_output)


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

        # # S4D real initialization
        # A = repeat(
        #     torch.arange(1, self.d_state + 1, dtype=torch.float32),
        #     "n -> d n",
        #     d=self.d_inner,
        # ).contiguous()
        # A_log = torch.log(A)  # Keep A_log in fp32
        # self.A_log = nn.Parameter(A_log)

        # # D "skip" parameter
        # self.D = nn.Parameter(torch.ones(self.d_inner))  # Keep in fp32
        self.A = nn.Parameter(
            torch.empty(
                self.d_inner,
                self.d_state,
                dtype=torch.float32,
            ))
        self.D = nn.Parameter(torch.ones(self.d_inner))

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
                self.A,
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
                self.A,
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

        return contextualized_states


class SambaDecoderLayer(nn.Module):
    
    def __init__(self, config, layer_idx, cache_config) -> None:
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
            self.attn = SambaAttention(config, layer_idx=layer_idx, yoco_cross=self.yoco_cross, cache_config=cache_config)

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
        hidden_states = self.input_layernorm(hidden_states.to(dtype=self.input_layernorm.weight.dtype))

        if self.use_mamba:
            attn_outputs = self.attn(
                hidden_states,
                attn_metadata,
                mamba_cache_params,
            )
            residual = residual.to(torch.float32)
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
            [SambaDecoderLayer(config, layer_idx, cache_config) for layer_idx in range(config.num_hidden_layers)]
        )
        self.final_layernorm = RMSNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(
        self,
        input_ids: Optional[torch.Tensor],
        positions: torch.Tensor,
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
        mamba_cache_params: MambaCacheParams,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        hidden_states = self.embed_tokens(input_ids)
        residual = None

        # print('>>> one step')
        # print('>>>', attn_metadata)

        kv_cache_idx = 0
        mamba_state_idx = 0
        for i, layer in enumerate(self.layers):
            if i == self.config.num_hidden_layers // 2 + 2:
                # seems pre-run for cuda graph will not store kv_cache
                if kv_caches[kv_cache_idx - 1].shape == torch.Size([0]):
                    break
                # print('>>>', kv_caches[kv_cache_idx - 1].shape)
                # print('>>>', hidden_states.shape)

                # if not equal, means we are in the prefill phase
                if hidden_states.size(0) != attn_metadata.seq_lens_tensor.size(0):
                    # seq_start_loc is for flash attn
                    # need to compute it for xformers backend
                    selected_token_indices = torch.cumsum(attn_metadata.seq_lens_tensor, dim=0) - 1
                    # print('>>>', selected_token_indices)
                    hidden_states = hidden_states.index_select(0, selected_token_indices)
                    # print('>>>', hidden_states.shape)

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

        hidden_states = self.final_layernorm(hidden_states.to(dtype=self.final_layernorm.weight.dtype))
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
        self.model = SambaModel(config, cache_config=cache_config)
        self.vocab_size = config.vocab_size
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
        )
        self.embedding_bias = nn.Parameter(torch.zeros(config.vocab_size))

        self.mamba_cache: Optional[MambaCacheManager] = None
        self.logits_processor = LogitsProcessor(config.vocab_size,
                                                config.vocab_size,
                                                logits_as_input=False)
        self.sampler = Sampler()

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        **kwargs,
    ) -> Union[torch.Tensor, IntermediateTensors]:
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

        hidden_states = self.model(input_ids, positions, kv_caches,
                                   attn_metadata, mamba_cache_params)
        return hidden_states

    def _get_mamba_cache_shape(self) -> Tuple[Optional[Tuple[int, int]], Optional[Tuple[int, int]]]:
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
        if hidden_states.size(0) == sampling_metadata.selected_token_indices.size(0):
            # print('>>>', hidden_states.shape)
            # print('>>>', hidden_states)
            # we have manually selected tokens for YOCO at SambaModel's forward function
            self.logits_processor.logits_as_input = True
            # import traceback
            # traceback.print_stack()
            # print('>>>', hidden_states.shape)
            logits = self.logits_processor._get_logits(hidden_states, self.lm_head, self.embedding_bias)
            processed_logits = self.logits_processor(self.lm_head, logits,
                                           sampling_metadata)
            # print('*' * 50)
            self.logits_processor.logits_as_input = False
        else:
            # the 1st profile run
            processed_logits = self.logits_processor(self.lm_head, hidden_states, sampling_metadata, self.embedding_bias)
        return processed_logits

    def sample(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[SamplerOutput]:
        next_tokens = self.sampler(logits, sampling_metadata)
        return next_tokens

    def load_weights(
        self,
        weights: Iterable[Tuple[str, torch.Tensor]],
    ):
        weights = {name: weight for name, weight in weights}
        adjusted_weights = {}
        for name, weight in weights.items():
            if "A_log" in name:
                name = name.replace("A_log", "A")
                weight = -torch.exp(weight.float())
            adjusted_weights[name] = weight
        # for name, loaded_weight in weights.items():
        #     print(name, loaded_weight.shape)

        # for name, param in self.named_parameters():
        #     print(name, param.shape)
        missing_keys, unexpected_keys = self.load_state_dict(adjusted_weights, strict=False)
        assert missing_keys == ['embedding_bias', 'lm_head.weight',], f"Missing keys: {missing_keys}"
        assert unexpected_keys == ['lm_head.bias',], f"Unexpected keys: {unexpected_keys}"
        self.lm_head.weight.data.copy_(adjusted_weights['model.embed_tokens.weight'])
        self.embedding_bias.data.copy_(adjusted_weights['lm_head.bias'])
