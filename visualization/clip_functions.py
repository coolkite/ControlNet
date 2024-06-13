import torch
from transformers.modeling_attn_mask_utils import _create_4d_causal_attention_mask, _prepare_4d_attention_mask

#from transformers.models.modeling_attn_mask_utils import _create_4d_causal_attention_mask, _prepare_4d_attention_mask

from transformers.modeling_outputs import BaseModelOutputWithPooling
from transformers import AutoTokenizer
import torch
from transformers import CLIPTextModel, CLIPTokenizer
from tqdm.auto import tqdm
from torch import autocast
from PIL import Image
from itertools import product
import numpy as np
from typing import Any, Optional, Tuple, Union

def get_clip_embeddings(
        text_transformer,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        inputs_embeds=None
) -> Union[Tuple, BaseModelOutputWithPooling]:
    # if position_ids is None:
    #         position_ids = self.position_ids[:, :seq_length]

    # if inputs_embeds is None:
    #     inputs_embeds = self.token_embedding(input_ids)

    # position_embeddings = self.position_embedding(position_ids)
    # embeddings = inputs_embeds + position_embeddings
    
    output_attentions = output_attentions if output_attentions is not None else text_transformer.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else text_transformer.config.output_hidden_states
    )
    return_dict = return_dict if return_dict is not None else text_transformer.config.use_return_dict

    if input_ids is None:
        raise ValueError("You have to specify input_ids")

    input_shape = input_ids.size()
    input_ids = input_ids.view(-1, input_shape[-1])

    hidden_states = text_transformer.embeddings(input_ids=input_ids, position_ids=position_ids,
                                                inputs_embeds=inputs_embeds)

    # CLIP's text model uses causal mask, prepare it here.
    # https://github.com/openai/CLIP/blob/cfcffb90e69f37bf2ff1e988237a0fbe41f33c04/clip/model.py#L324
    causal_attention_mask = _create_4d_causal_attention_mask(
            input_shape, hidden_states.dtype, device=hidden_states.device
        )
    # expand attention_mask
    if attention_mask is not None:
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        attention_mask = _prepare_4d_attention_mask(attention_mask, hidden_states.dtype)

    encoder_outputs = text_transformer.encoder(
        inputs_embeds=hidden_states,
        attention_mask=attention_mask,
        causal_attention_mask=causal_attention_mask,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
    )
    last_hidden_state = encoder_outputs[0]
    last_hidden_state = text_transformer.final_layer_norm(last_hidden_state)

    # text_embeds.shape = [batch_size, sequence_length, transformer.width]
    # take features from the eot embedding (eot_token is the highest number in each sequence)
    # casting to torch.int for onnx compatibility: argmax doesn't support int64 inputs with opset 14
    pooled_output = last_hidden_state[
        torch.arange(last_hidden_state.shape[0], device=last_hidden_state.device),
        input_ids.to(dtype=torch.int, device=last_hidden_state.device).argmax(dim=-1),
    ]

    if not return_dict:
        return (last_hidden_state, pooled_output) + encoder_outputs[1:]

    return BaseModelOutputWithPooling(
        last_hidden_state=last_hidden_state,
        pooler_output=pooled_output,
        hidden_states=encoder_outputs.hidden_states,
        attentions=encoder_outputs.attentions,
    )








