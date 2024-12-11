# coding=utf-8
# Copyright 2018 Mesh TensorFlow authors, T5 Authors and HuggingFace Inc. team.
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
""" PyTorch T5 model."""


import copy


import math
from typing import Iterable, List, Optional, Tuple

import torch
from torch import nn
import torch.nn.functional as F
from transformers import T5Config
from transformers.utils import logging

from vllm.attention import Attention, AttentionMetadata, AttentionType
from vllm.config import CacheConfig, LoRAConfig
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.model_executor.layers.activation import get_act_fn
from vllm.model_executor.layers.linear import (ColumnParallelLinear,
                                               QKVParallelLinear,
                                               RowParallelLinear)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig)
from vllm.model_executor.layers.sampler import Sampler, SamplerOutput
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead, VocabParallelEmbedding)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.sequence import IntermediateTensors
# from flash_attn import flash_attn_func

logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "T5Config"
_CHECKPOINT_FOR_DOC = "t5-small"

####################################################
# This dict contains ids and associated url
# for the pretrained weights provided with the models
####################################################
T5_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "t5-small",
    "t5-base",
    "t5-large",
    "t5-3b",
    "t5-11b",
    # See all T5 models at https://huggingface.co/models?filter=t5
]


####################################################
# This is a conversion method from TF 1.0 to PyTorch
# More details: https://medium.com/huggingface/from-tensorflow-to-pytorch-265f40ef2a28
####################################################
def load_tf_weights_in_t5(model, config, tf_checkpoint_path):
    """Load tf checkpoints in a pytorch model."""
    try:
        import re

        import numpy as np
        import tensorflow as tf
    except ImportError:
        logger.error(
            "Loading a TensorFlow model in PyTorch, requires TensorFlow to be installed. Please see "
            "https://www.tensorflow.org/install/ for installation instructions."
        )
        raise
    tf_path = os.path.abspath(tf_checkpoint_path)
    logger.info(f"Converting TensorFlow checkpoint from {tf_path}")
    # Load weights from TF model
    init_vars = tf.train.list_variables(tf_path)
    names = []
    tf_weights = {}
    for name, shape in init_vars:
        logger.info(f"Loading TF weight {name} with shape {shape}")
        array = tf.train.load_variable(tf_path, name)
        names.append(name)
        tf_weights[name] = array

    for txt_name in names:
        name = txt_name.split("/")
        # adam_v and adam_m are variables used in AdamWeightDecayOptimizer to calculated m and v
        # which are not required for using pretrained model
        if any(
            n in ["adam_v", "adam_m", "AdamWeightDecayOptimizer", "AdamWeightDecayOptimizer_1", "global_step"]
            for n in name
        ):
            logger.info(f"Skipping {'/'.join(name)}")
            tf_weights.pop(txt_name, None)
            continue
        if "_slot_" in name[-1]:
            logger.info(f"Skipping {'/'.join(name)}")
            tf_weights.pop(txt_name, None)
            continue
        pointer = model
        array = tf_weights[txt_name]

        for m_name in name:
            if re.fullmatch(r"[A-Za-z]+_\d+", m_name):
                scope_names = re.split(r"_(\d+)", m_name)
            else:
                scope_names = [m_name]
            if scope_names[0] in ["kernel", "scale", "embedding"]:
                pointer = getattr(pointer, "weight")
            elif scope_names[0] == "self_attention":
                pointer = getattr(pointer, "layer")
                pointer = pointer[0]
            elif scope_names[0] == "enc_dec_attention":
                pointer = getattr(pointer, "layer")
                pointer = pointer[1]
            elif scope_names[0] == "dense_relu_dense":
                pointer = getattr(pointer, "layer")
                pointer = pointer[2]
            elif scope_names[0] == "rms_norm":
                if hasattr(pointer, "layer_norm"):
                    pointer = getattr(pointer, "layer_norm")
                elif hasattr(pointer, "final_layer_norm"):
                    pointer = getattr(pointer, "final_layer_norm")
            elif scope_names[0] == "scale":
                pointer = getattr(pointer, "weight")
            elif scope_names[0] == "output_bias" or scope_names[0] == "beta":
                pointer = getattr(pointer, "bias")
            elif scope_names[0] == "squad":
                pointer = getattr(pointer, "classifier")
            elif scope_names[0] == "decoder" and name[1] == "logits":
                continue
            elif scope_names[0] == "logits":
                pointer = getattr(pointer, "lm_head")
            elif scope_names[0] == "wi" and len(scope_names) > 1 and scope_names[1].isdigit():
                pointer = getattr(pointer, f"wi_{scope_names[1]}")
                continue
            else:
                try:
                    pointer = getattr(pointer, scope_names[0])
                except AttributeError:
                    logger.info(f"Skipping {'/'.join(name)}")
                    continue
            if len(scope_names) >= 2:
                num = int(scope_names[1])
                pointer = pointer[num]
        if scope_names[0] not in ["kernel", "scale", "embedding"]:
            pointer = getattr(pointer, "weight")
        if scope_names[0] != "embedding":
            logger.info(f"Transposing numpy weight of shape {array.shape} for {name}")
            array = np.transpose(array)
        try:
            if pointer.shape != array.shape:
                raise ValueError(f"Pointer shape {pointer.shape} and array shape {array.shape} mismatched")
        except AssertionError as e:
            e.args += (pointer.shape, array.shape)
            raise
        logger.info(f"Initialize PyTorch weight {name}")
        pointer.data = torch.from_numpy(array.astype(np.float32))
        tf_weights.pop(txt_name, None)

    logger.info(f"Weights not copied to PyTorch model: {', '.join(tf_weights.keys())}.")
    return model


####################################################
# PyTorch Models are constructed by sub-classing
# - torch.nn.Module for the layers and
# - PreTrainedModel for the models (it-self a sub-class of nn.Module)
####################################################
PARALLELIZE_DOCSTRING = r"""
    This is an experimental feature and is a subject to change at a moment's notice.

    Uses a device map to distribute attention modules of the model across several devices. If no device map is given,
    it will evenly distribute blocks across all devices.

    Args:
        device_map (`Dict[int, list]`, optional, defaults to None):
            A dictionary that maps attention modules to devices. Note that the embedding module and LMHead are always
            automatically mapped to the first device (for esoteric reasons). That means that the first device should
            have fewer attention modules mapped to it than other devices. For reference, the t5 models have the
            following number of attention modules:

                - t5-small: 6
                - t5-base: 12
                - t5-large: 24
                - t5-3b: 24
                - t5-11b: 24

    Example:

    ```python
    # Here is an example of a device map on a machine with 4 GPUs using t5-3b, which has a total of 24 attention modules:
    model = T5ForConditionalGeneration.from_pretrained("t5-3b")
    device_map = {
        0: [0, 1, 2],
        1: [3, 4, 5, 6, 7, 8, 9],
        2: [10, 11, 12, 13, 14, 15, 16],
        3: [17, 18, 19, 20, 21, 22, 23],
    }
    model.parallelize(device_map)
    ```
"""
DEPARALLELIZE_DOCSTRING = r"""
    Moves the model to cpu from a model parallel state.

    Example:

    ```python
    # On a 4 GPU machine with t5-3b:
    model = T5ForConditionalGeneration.from_pretrained("t5-3b")
    device_map = {
        0: [0, 1, 2],
        1: [3, 4, 5, 6, 7, 8, 9],
        2: [10, 11, 12, 13, 14, 15, 16],
        3: [17, 18, 19, 20, 21, 22, 23],
    }
    model.parallelize(device_map)  # Splits the model across several devices
    model.deparallelize()  # Put the model back on cpu and cleans memory by calling torch.cuda.empty_cache()
    ```
"""


class T5LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        Construct a layernorm module in the T5 style. No bias and no subtraction of mean.
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        # T5 uses a layer_norm which only scales and doesn't shift, which is also known as Root Mean
        # Square Layer Normalization https://arxiv.org/abs/1910.07467 thus varience is calculated
        # w/o mean and there is no bias. Additionally we want to make sure that the accumulation for
        # half-precision inputs is done in fp32

        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)

        # convert into half-precision if necessary
        if self.weight.dtype in [torch.float16, torch.bfloat16]:
            hidden_states = hidden_states.to(self.weight.dtype)

        return self.weight * hidden_states


class T5DenseActDense(nn.Module):
    def __init__(self, config: T5Config, quant_config: Optional[QuantizationConfig] = None):
        super().__init__()
        self.wi = ColumnParallelLinear(config.d_model, config.d_ff, bias=False, quant_config=quant_config)
        self.wo = RowParallelLinear(config.d_ff, config.d_model, bias=False, quant_config=quant_config)
        self.dropout = nn.Dropout(config.dropout_rate)
        self.act = get_act_fn(config.dense_act_fn, quant_config)

    def forward(self, hidden_states):
        
        hidden_states = self.wi(hidden_states)[0]
        hidden_states = self.act(hidden_states)
        hidden_states = self.dropout(hidden_states)
        if (
            isinstance(self.wo.weight, torch.Tensor)
            and hidden_states.dtype != self.wo.weight.dtype
            and self.wo.weight.dtype != torch.int8
        ):
            hidden_states = hidden_states.to(self.wo.weight.dtype)
        hidden_states = self.wo(hidden_states)[0]
        return hidden_states


class T5DenseGatedActDense(nn.Module):
    def __init__(self, config: T5Config, quant_config: Optional[QuantizationConfig] = None):
        super().__init__()
        self.wi_0 = ColumnParallelLinear(config.d_model, config.d_ff, bias=False, quant_config=quant_config)
        self.wi_1 = ColumnParallelLinear(config.d_model, config.d_ff, bias=False, quant_config=quant_config)
        self.wo = RowParallelLinear(config.d_ff, config.d_model, bias=False, quant_config=quant_config)
        self.dropout = nn.Dropout(config.dropout_rate)
        self.act = get_act_fn(config.dense_act_fn, quant_config)

    def forward(self, hidden_states):
        hidden_gelu = self.act(self.wi_0(hidden_states))
        hidden_linear = self.wi_1(hidden_states)
        hidden_states = hidden_gelu * hidden_linear
        hidden_states = self.dropout(hidden_states)

        # To make 8bit quantization work for google/flan-t5-xxl, self.wo is kept in float32.
        # See https://github.com/huggingface/transformers/issues/20287
        # we also make sure the weights are not in `int8` in case users will force `_keep_in_fp32_modules` to be `None``
        if (
            isinstance(self.wo.weight, torch.Tensor)
            and hidden_states.dtype != self.wo.weight.dtype
            and self.wo.weight.dtype != torch.int8
        ):
            hidden_states = hidden_states.to(self.wo.weight.dtype)

        hidden_states = self.wo(hidden_states)
        return hidden_states


class T5LayerFF(nn.Module):
    def __init__(self, config: T5Config, quant_config: Optional[QuantizationConfig] = None):
        super().__init__()
        if config.is_gated_act:
            self.DenseReluDense = T5DenseGatedActDense(config, quant_config)
        else:
            self.DenseReluDense = T5DenseActDense(config, quant_config)

        self.layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(self, hidden_states):
        forwarded_states = self.layer_norm(hidden_states)
        forwarded_states = self.DenseReluDense(forwarded_states)
        hidden_states = hidden_states + self.dropout(forwarded_states)
        return hidden_states

class T5Attention(nn.Module):
    def __init__(self, config: T5Config, cache_config: Optional[CacheConfig] = None, quant_config: Optional[QuantizationConfig] = None, has_relative_attention_bias=False):
        super().__init__()
        self.is_decoder = config.is_decoder
        self.has_relative_attention_bias = has_relative_attention_bias
        self.relative_attention_num_buckets = config.relative_attention_num_buckets
        self.relative_attention_max_distance = config.relative_attention_max_distance
        self.d_model = config.d_model
        self.key_value_proj_dim = config.d_kv
        self.n_heads = config.num_heads
        self.dropout = config.dropout_rate
        self.inner_dim = self.n_heads * self.key_value_proj_dim


        self.qkv_proj = QKVParallelLinear(
            self.d_model,
            self.inner_dim // self.n_heads,
            self.n_heads,
            bias=False,
            quant_config=quant_config,
        )
        self.out_proj = RowParallelLinear(
            self.inner_dim,
            self.d_model,
            bias=False,
            quant_config=quant_config,
        )
        self.attn = Attention(self.n_heads,
                              self.inner_dim // self.n_heads,
                              scale = 1,
                              cache_config=cache_config,
                              quant_config=quant_config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: AttentionMetadata,
        encoder_hidden_states: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        """
        Self-attention (if key_value_states is None) or attention over source sentence (provided by key_value_states).
        """
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([2048, 2048, 2048], dim=-1)
        if encoder_hidden_states is None:
            attn_output = F.scaled_dot_product_attention(q,
                                                     k,
                                                     v,
                                                     dropout_p=0.0)
        else:
            qkv_enc, _ = self.qkv_proj(encoder_hidden_states)
            _, k, v = qkv.split([2048, 2048, 2048], dim=-1)
            attn_output = F.scaled_dot_product_attention(q,
                                                     k,
                                                     v,
                                                     dropout_p=0.0)
        output, _ = self.out_proj(attn_output)
        present_key_value_state = (k, v) if self.is_decoder else None
        return output, present_key_value_state

class T5LayerSelfAttention(nn.Module):
    def __init__(self, config, cache_config: Optional[CacheConfig] = None, quant_config: Optional[QuantizationConfig] = None, has_relative_attention_bias=False):
        super().__init__()
        self.SelfAttention = T5Attention(config, cache_config, quant_config, has_relative_attention_bias=has_relative_attention_bias)
        self.layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(self, hidden_states: torch.Tensor, kv_cache: torch.Tensor,
                attn_metadata: AttentionMetadata) -> torch.Tensor:
        hidden_states = self.layer_norm(hidden_states)
        attention_output = self.SelfAttention(
            hidden_states,
            kv_cache,
            attn_metadata)
        hidden_states = hidden_states + self.dropout(attention_output[0])
        outputs = (hidden_states,) + attention_output[1:]  # add attentions if we output them
        return outputs


class T5LayerCrossAttention(nn.Module):
    def __init__(self, config, cache_config: Optional[CacheConfig] = None, quant_config: Optional[QuantizationConfig] = None):
        super().__init__()
        self.EncDecAttention = T5Attention(config, cache_config, quant_config, has_relative_attention_bias=False)
        self.layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(self, 
                hidden_states: torch.Tensor,
                kv_cache: torch.Tensor,
        attn_metadata: AttentionMetadata,
        encoder_hidden_states: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        hidden_states = self.layer_norm(hidden_states)
        attention_output = self.EncDecAttention(
            hidden_states,
            kv_cache,
            attn_metadata,
            encoder_hidden_states,
        )
        layer_output = hidden_states + self.dropout(attention_output[0])
        outputs = (layer_output,) + attention_output[1:]  # add attentions if we output them
        return outputs


class T5Block(nn.Module):
    def __init__(self, config: T5Config, cache_config: Optional[CacheConfig] = None, quant_config: Optional[QuantizationConfig] = None, has_relative_attention_bias=False):
        super().__init__()
        self.is_decoder = config.is_decoder
        self.self_attn = T5LayerSelfAttention(config, cache_config, quant_config, has_relative_attention_bias=has_relative_attention_bias)
        if self.is_decoder:
            self.cross_attn = T5LayerCrossAttention(config, cache_config, quant_config)
        self.fc = T5LayerFF(config, quant_config)
        

    def forward(
        self,
        hidden_states: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: AttentionMetadata,
        encoder_hidden_states: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        self_attention_outputs = self.self_attn(hidden_states, kv_cache, attn_metadata)
        hidden, _ = self_attention_outputs[:2]
        attention_outputs = self_attention_outputs[2:]  # Keep self-attention outputs and relative position weights

        # clamp inf values to enable fp16 training
        if hidden.dtype == torch.float16:
            clamp_value = torch.where(
                torch.isinf(hidden).any(),
                torch.finfo(hidden.dtype).max - 1000,
                torch.finfo(hidden.dtype).max,
            )
            hidden = torch.clamp(hidden, min=-clamp_value, max=clamp_value)

        do_cross_attention = self.is_decoder and encoder_hidden_states is not None
        if do_cross_attention:
            cross_attention_outputs = self.cross_attn(hidden, kv_cache, attn_metadata, encoder_hidden_states)
            hidden = cross_attention_outputs[0]

            # clamp inf values to enable fp16 training
            if hidden.dtype == torch.float16:
                clamp_value = torch.where(
                    torch.isinf(hidden).any(),
                    torch.finfo(hidden.dtype).max - 1000,
                    torch.finfo(hidden.dtype).max,
                )
                hidden = torch.clamp(hidden, min=-clamp_value, max=clamp_value)

            attention_outputs = attention_outputs + cross_attention_outputs[2:]

        # Apply Feed Forward layer
        hidden = self.fc(hidden)

        # clamp inf values to enable fp16 training
        if hidden.dtype == torch.float16:
            clamp_value = torch.where(
                torch.isinf(hidden).any(),
                torch.finfo(hidden.dtype).max - 1000,
                torch.finfo(hidden.dtype).max,
            )
            hidden = torch.clamp(hidden, min=-clamp_value, max=clamp_value)

        outputs = (hidden,) + attention_outputs
        return outputs


class T5ClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config: T5Config):
        super().__init__()
        self.dense = nn.Linear(config.d_model, config.d_model)
        self.dropout = nn.Dropout(p=config.classifier_dropout)
        self.out_proj = nn.Linear(config.d_model, config.num_labels)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.dense(hidden_states)
        hidden_states = torch.tanh(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.out_proj(hidden_states)
        return hidden_states

class T5Stack(nn.Module):
    def __init__(self, 
                config: T5Config,
                cache_config: Optional[CacheConfig] = None,
                quant_config: Optional[QuantizationConfig] = None,
                embed_tokens=None):
        super().__init__()
        self.cache_config = cache_config
        self.embed_tokens = embed_tokens
        self.is_decoder = config.is_decoder

        self.block = nn.ModuleList(
            [T5Block(config, cache_config, quant_config, has_relative_attention_bias=bool(i == 0)) for i in range(config.num_layers)]
        )
        self.final_layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(self, input_ids: torch.Tensor,
                positions: torch.Tensor,
                kv_caches: List[torch.Tensor],
                attn_metadata: AttentionMetadata,
                encoder_hidden_states: Optional[torch.Tensor]=None) -> torch.Tensor:
        inputs_embeds = self.embed_tokens(input_ids)
        hidden_states = self.dropout(inputs_embeds)
        # print('t5 stack', type(hidden_states))
        for i, layer in enumerate(self.block):
            layer_outputs = layer(hidden_states,
                                kv_caches[i], 
                                attn_metadata,
                                encoder_hidden_states)
            hidden_states = layer_outputs[0]

        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states


T5_START_DOCSTRING = r"""

    The T5 model was proposed in [Exploring the Limits of Transfer Learning with a Unified Text-to-Text
    Transformer](https://arxiv.org/abs/1910.10683) by Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan
    Narang, Michael Matena, Yanqi Zhou, Wei Li, Peter J. Liu. It's an encoder decoder transformer pre-trained in a
    text-to-text denoising generative setting.

    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`T5Config`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

T5_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. T5 is a model with relative position embeddings so you
            should be able to pad the inputs on both the right and the left.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for detail.

            [What are input IDs?](../glossary#input-ids)

            To know more on how to prepare `input_ids` for pretraining take a look a [T5 Training](./t5#training).
        attention_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)
        decoder_input_ids (`torch.LongTensor` of shape `(batch_size, target_sequence_length)`, *optional*):
            Indices of decoder input sequence tokens in the vocabulary.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are decoder input IDs?](../glossary#decoder-input-ids)

            T5 uses the `pad_token_id` as the starting token for `decoder_input_ids` generation. If `past_key_values`
            is used, optionally only the last `decoder_input_ids` have to be input (see `past_key_values`).

            To know more on how to prepare `decoder_input_ids` for pretraining take a look at [T5
            Training](./t5#training).
        decoder_attention_mask (`torch.BoolTensor` of shape `(batch_size, target_sequence_length)`, *optional*):
            Default behavior: generate a tensor that ignores pad tokens in `decoder_input_ids`. Causal mask will also
            be used by default.
        head_mask (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*):
            Mask to nullify selected heads of the self-attention modules in the encoder. Mask values selected in `[0,
            1]`:

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.

        decoder_head_mask (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*):
            Mask to nullify selected heads of the self-attention modules in the decoder. Mask values selected in `[0,
            1]`:

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.

        cross_attn_head_mask (`torch.Tensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*):
                Mask to nullify selected heads of the cross-attention modules in the decoder. Mask values selected in
                `[0, 1]`:

                - 1 indicates the head is **not masked**,
                - 0 indicates the head is **masked**.

        encoder_outputs (`tuple(tuple(torch.FloatTensor)`, *optional*):
            Tuple consists of (`last_hidden_state`, `optional`: *hidden_states*, `optional`: *attentions*)
            `last_hidden_state` of shape `(batch_size, sequence_length, hidden_size)` is a sequence of hidden states at
            the output of the last layer of the encoder. Used in the cross-attention of the decoder.
        past_key_values (`tuple(tuple(torch.FloatTensor))` of length `config.n_layers` with each tuple having 4 tensors of shape `(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`):
            Contains precomputed key and value hidden states of the attention blocks. Can be used to speed up decoding.

            If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those that
            don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of all
            `decoder_input_ids` of shape `(batch_size, sequence_length)`.
        inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
            model's internal embedding lookup matrix.
        decoder_inputs_embeds (`torch.FloatTensor` of shape `(batch_size, target_sequence_length, hidden_size)`, *optional*):
            Optionally, instead of passing `decoder_input_ids` you can choose to directly pass an embedded
            representation. If `past_key_values` is used, optionally only the last `decoder_inputs_embeds` have to be
            input (see `past_key_values`). This is useful if you want more control over how to convert
            `decoder_input_ids` indices into associated vectors than the model's internal embedding lookup matrix.

            If `decoder_input_ids` and `decoder_inputs_embeds` are both unset, `decoder_inputs_embeds` takes the value
            of `inputs_embeds`.

        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).

        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""

T5_ENCODER_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. T5 is a model with relative position embeddings so you
            should be able to pad the inputs on both the right and the left.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for detail.

            To know more on how to prepare `input_ids` for pretraining take a look a [T5 Training](./t5#training).
        attention_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)
        head_mask (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*):
            Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.

        inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
            model's internal embedding lookup matrix.
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""

# Warning message for FutureWarning: head_mask was separated into two input args - head_mask, decoder_head_mask
__HEAD_MASK_WARNING_MSG = """
The input argument `head_mask` was split into two arguments `head_mask` and `decoder_head_mask`. Currently,
`decoder_head_mask` is set to copy `head_mask`, but this feature is deprecated and will be removed in future versions.
If you do not want to use any `decoder_head_mask` now, please set `decoder_head_mask = torch.ones(num_layers,
num_heads)`.
"""

class T5Model(nn.Module):
    _keys_to_ignore_on_load_unexpected = [
        "decoder.block.0.layer.1.EncDecAttention.relative_attention_bias.weight",
    ]
    _tied_weights_keys = ["encoder.embed_tokens.weight", "decoder.embed_tokens.weight"]

    def __init__(self, 
                config: T5Config,
                cache_config: Optional[CacheConfig] = None,
                quant_config: Optional[QuantizationConfig] = None,
                lora_config: Optional[LoRAConfig] = None):
        super().__init__()
        # self.shared = nn.Embedding(config.vocab_size, config.d_model)
        self.padding_idx = config.pad_token_id
        lora_vocab = (lora_config.lora_extra_vocab_size *
                      (lora_config.max_loras or 1)) if lora_config else 0
        self.vocab_size = config.vocab_size + lora_vocab
        self.shared = VocabParallelEmbedding(
            self.vocab_size,
            config.d_model,
            org_num_embeddings=config.vocab_size,
        )

        encoder_config = copy.deepcopy(config)
        encoder_config.is_decoder = False
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        self.encoder = T5Stack(encoder_config, cache_config, quant_config, self.shared)

        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
        decoder_config.is_encoder_decoder = False
        decoder_config.num_layers = config.num_decoder_layers
        self.decoder = T5Stack(decoder_config, cache_config, quant_config, self.shared)

    def forward(
        self,
    input_ids: torch.Tensor, positions: torch.Tensor,
    encoder_input_ids: torch.Tensor,
    encoder_positions: torch.Tensor, kv_caches: List[torch.Tensor],
    attn_metadata: AttentionMetadata) -> torch.Tensor:
        r"""
        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, T5Model

        >>> tokenizer = AutoTokenizer.from_pretrained("t5-small")
        >>> model = T5Model.from_pretrained("t5-small")

        >>> input_ids = tokenizer(
        ...     "Studies have been shown that owning a dog is good for you", return_tensors="pt"
        ... ).input_ids  # Batch size 1
        >>> decoder_input_ids = tokenizer("Studies show that", return_tensors="pt").input_ids  # Batch size 1

        >>> # preprocess: Prepend decoder_input_ids with start token which is pad token for T5Model.
        >>> # This is not needed for torch's T5ForConditionalGeneration as it does this internally using labels arg.
        >>> decoder_input_ids = model._shift_right(decoder_input_ids)

        >>> # forward pass
        >>> outputs = model(input_ids=input_ids, decoder_input_ids=decoder_input_ids)
        >>> last_hidden_states = outputs.last_hidden_state
        ```"""
        encoder_hidden_states = None

        if encoder_input_ids.numel() > 0:
            # Run encoder attention if a non-zero number of encoder tokens
            # are provided as input
            encoder_hidden_states = self.encoder(input_ids=encoder_input_ids,
                                                 positions=encoder_positions,
                                                 kv_caches=kv_caches,
                                                 attn_metadata=attn_metadata)
        decoder_outputs = self.decoder(
            input_ids=input_ids,
            positions=positions,
            encoder_hidden_states=encoder_hidden_states,
            kv_caches=kv_caches,
            attn_metadata=attn_metadata)

        return decoder_outputs

class T5ForConditionalGeneration(nn.Module):
    def __init__(self,
                 config: T5Config,
                 cache_config: Optional[CacheConfig] = None,
                 quant_config: Optional[QuantizationConfig] = None,
                 lora_config: Optional[LoRAConfig] = None):
        super().__init__()
        self.config = config
        self.model_dim = config.d_model
        self.model = T5Model(config,
                            cache_config,
                            quant_config,
                            lora_config=lora_config)
        print('lora_config', lora_config)
        self.unpadded_vocab_size = config.vocab_size
        if lora_config:
            self.unpadded_vocab_size += lora_config.lora_extra_vocab_size
        self.lm_head = ParallelLMHead(num_embeddings= self.unpadded_vocab_size,
                                    embedding_dim=config.d_model,
                                    org_num_embeddings=config.vocab_size,
                                    bias=False)

        
        self.logits_processor = LogitsProcessor(self.unpadded_vocab_size,
                                                config.vocab_size)
        self.sampler = Sampler()

    def forward(
        self,
    input_ids: torch.Tensor,
    positions: torch.Tensor,
    kv_caches: List[torch.Tensor],
    attn_metadata: AttentionMetadata,
    encoder_input_ids: torch.Tensor,
    encoder_positions: torch.Tensor,
    **kwargs,
    ) -> torch.Tensor:
        r"""
        Args:
            input_ids
                torch.Tensor of *decoder* input token ids.
            positions
                torch.Tensor of *decoder* position indices.
            encoder_input_ids
                torch.Tensor of *encoder* input token ids.
            encoder_positions
                torch.Tensor of *encoder* position indices
            kv_caches:
                Layer-wise list of KV cache tensors
            attn_metadata:
                vLLM Attention metadata structure
        Returns:
            Output torch.Tensor
        """
        return self.model(input_ids, positions, encoder_input_ids,
                          encoder_positions, kv_caches, attn_metadata)

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[torch.Tensor]:
        logits = self.logits_processor(self.lm_head, hidden_states,
                                       sampling_metadata)
        return logits

    def sample(
        self,
        logits: Optional[torch.Tensor],
        sampling_metadata: SamplingMetadata,
    ) -> Optional[SamplerOutput]:
        next_tokens = self.sampler(logits, sampling_metadata)
        return next_tokens

    stacked_params_mapping = {
        "q.weight": {
            "param_name": "qkv_proj.weight",
            "shard_id": "q",
        },
        "k.weight": {
            "param_name": "qkv_proj.weight",
            "shard_id": "k",
        },
        "v.weight": {
            "param_name": "qkv_proj.weight",
            "shard_id": "v",
        },
        "o.weight": {
            "param_name": "out_proj.weight",
            "shard_id": None,
        }
    }

    

    params_mapping = {
        "beta": "bias",
        "gamma": "weight",
        "LayerNorm": "layernorm",
    }

    def _rename_key(self, key: str):
        prefix = f"{self.base_model_prefix}."
        key = key[len(prefix):] if key.startswith(prefix) else key

        for src, dst in self.params_mapping.items():
            key = key.replace(src, dst)

        return key
    
    layer_type_mapping = {
        "encoder": {
            "layer.0": "self_attn",
            "layer.1": "fc",
        },
        "decoder": {
            "layer.0": "self_attn",
            "layer.1": "cross_attn",
            "layer.2": "fc",
        }
    }
    
    def _rename_layer_types(
        self,
        name: str, 
    ) -> str:
        for enc_dec, mapping in self.layer_type_mapping.items():
            if enc_dec in name:
                for layer_num in mapping.keys():
                    if layer_num in name:
                        name = name.replace(layer_num, mapping[layer_num])
        return name        

    def _rename_stacked_param(
        self,
        name: str,
    ) -> Tuple[str, Optional[str]]:
        for key, mapping in self.stacked_params_mapping.items():
            if key in name and '.wo.' not in name:
                name = name.replace(key, mapping["param_name"])
                return name, mapping["shard_id"]
        return name, None
    
    # def get_set(self, model_params_dict):
    #     out = set()
    #     for key in model_params_dict.keys():
    #         if "bias" in key:
    #             print('BBBIIIAAASSSSS..................')
    #         if 'decoder' not in key and 'encoder' not in key:
    #             print(key)
    #         # print(key.split('.'))
    #         lst = key.split('.') 
    #         if len(lst)>=4:
    #             out.add(lst[3])
    #     return out
    
    def match_weight_name(self, weights_tuple_list):
        out = set()
        for name, _ in weights_tuple_list:
            # print(name)
            if 'decoder' in name and 'layer_norm' not in name:
                if 'layer.0' in name and 'SelfAttention' not in name:
                    print(name)
                    out.add(False)
                elif 'layer.1' in name and 'EncDecAttention' not in name:
                    print(name)
                    out.add(False)
                elif 'layer.2' in name and 'DenseReluDense' not in name:
                    print(name)
                    out.add(False)
                else:
                    out.add(True)
            elif 'encoder' in name and 'layer_norm' not in name:
                if 'layer.0' in name and 'SelfAttention' not in name:
                    print(name)
                    out.add(False)
                elif 'layer.1' in name and 'DenseReluDense' not in name:
                    print(name)
                    out.add(False)
                else:
                    out.add(True)
            elif 'decoder' not in name and 'encoder' not in name:
                print(name)
        return out

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):

        model_params_dict = dict(self.model.named_parameters())
        # types = self.get_set(model_params_dict)
        top_params_dict = dict(self.named_parameters())

        weights_tuple_list = list(weights)

        shared_embedding_weight = None
        shared_embedding_shard_id = None

        for name, loaded_weight in weights_tuple_list:
            name = self._rename_layer_types(name)
            name, shard_id = self._rename_stacked_param(name)
            if ('encoder.embed_tokens.weight' in name
                    or 'decoder.embed_tokens.weight' in name
                    or 'lm_head.weight' in name):
                assert shared_embedding_weight is None, (
                    "Conflicting embedding weights.")
                shared_embedding_weight = loaded_weight
                shared_embedding_shard_id = shard_id
            else:
                # Skip the specific downstream task weight.
                if name.startswith('cls.'):
                    continue
                # use Pooler instead.
                if name.startswith('pooler.'):
                    continue
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in model_params_dict:
                    continue
                if "bias.weight" in name and name not in model_params_dict:
                    continue

                param = model_params_dict[name]
                weight_loader = getattr(param, "weight_loader",
                                        default_weight_loader)
                if shard_id:
                    weight_loader(param, loaded_weight, shard_id)
                else:
                    weight_loader(param, loaded_weight)
