import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from transformers import T5Tokenizer, T5EncoderModel, CLIPTokenizer, CLIPTextModel

import open_clip
from ldm.util import default, count_params
from .clip_functions import get_clip_embeddings
import numpy as np

import os
import sys
from typing import Dict, List, Optional, Union


SCRIPT_DIR = os.path.dirname(os.path.abspath("/project/pi_ekalogerakis_umass_edu/dshivashok/ControlNet/evaluations/baseline_learned_control_k.py"))
print(SCRIPT_DIR)
sys.path.append(os.path.dirname(SCRIPT_DIR))

from diffusers.loaders.textual_inversion import *


class TILM(TextualInversionLoaderMixin):
    def __init__(self, tokenizer, text_encoder):
        super().__init__()
        self.tokenizer = tokenizer
        self.text_encoder = text_encoder
        self.inverted_token = None
        self.inverted_token_path = None
    
    def load_textual_inversion(
        self,
        pretrained_model_name_or_path: Union[str, List[str], Dict[str, torch.Tensor], List[Dict[str, torch.Tensor]]],
        token: Optional[Union[str, List[str]]] = None,
        tokenizer: Optional["PreTrainedTokenizer"] = None,  # noqa: F821
        text_encoder: Optional["PreTrainedModel"] = None,  # noqa: F821
        **kwargs,
    ):
        # 1. Set correct tokenizer and text encoder
        tokenizer = tokenizer or getattr(self, "tokenizer", None)
        text_encoder = text_encoder or getattr(self, "text_encoder", None)

        # 2. Normalize inputs
        pretrained_model_name_or_paths = (
            [pretrained_model_name_or_path]
            if not isinstance(pretrained_model_name_or_path, list)
            else pretrained_model_name_or_path
        )
        tokens = [token] if not isinstance(token, list) else token
        if tokens[0] is None:
            tokens = tokens * len(pretrained_model_name_or_paths)

        # 3. Check inputs
        self._check_text_inv_inputs(tokenizer, text_encoder, pretrained_model_name_or_paths, tokens)

        # 4. Load state dicts of textual embeddings
        state_dicts = load_textual_inversion_state_dicts(pretrained_model_name_or_paths, **kwargs)

        # 4.1 Handle the special case when state_dict is a tensor that contains n embeddings for n tokens
        if len(tokens) > 1 and len(state_dicts) == 1:
            if isinstance(state_dicts[0], torch.Tensor):
                state_dicts = list(state_dicts[0])
                if len(tokens) != len(state_dicts):
                    raise ValueError(
                        f"You have passed a state_dict contains {len(state_dicts)} embeddings, and list of tokens of length {len(tokens)} "
                        f"Make sure both have the same length."
                    )

        # 4. Retrieve tokens and embeddings
        tokens, embeddings = self._retrieve_tokens_and_embeddings(tokens, state_dicts, tokenizer)

        # 5. Extend tokens and embeddings for multi vector
        tokens, embeddings = self._extend_tokens_and_embeddings(tokens, embeddings, tokenizer)

        # 6. Make sure all embeddings have the correct size
        expected_emb_dim = text_encoder.get_input_embeddings().weight.shape[-1]
        if any(expected_emb_dim != emb.shape[-1] for emb in embeddings):
            print([(expected_emb_dim, emb.shape[-1]) for emb in embeddings])
            raise ValueError(
                "Loaded embeddings are of incorrect shape. Expected each textual inversion embedding "
                "to be of shape {input_embeddings.shape[-1]}, but are {embeddings.shape[-1]} "
            )

        # 7. Now we can be sure that loading the embedding matrix works
        # < Unsafe code:

        # 7.2 save expected device and dtype
        device = text_encoder.device
        dtype = text_encoder.dtype

        # 7.3 Increase token embedding matrix
        text_encoder.resize_token_embeddings(len(tokenizer) + len(tokens))
        input_embeddings = text_encoder.get_input_embeddings().weight

        # 7.4 Load token and embedding
        for token, embedding in zip(tokens, embeddings):
            # add tokens and get ids
            tokenizer.add_tokens(token)
            token_id = tokenizer.convert_tokens_to_ids(token)
            input_embeddings.data[token_id] = embedding
            logger.info(f"Loaded textual inversion embedding for {token}.")

        input_embeddings.to(dtype=dtype, device=device)
        self.inverted_token = tokens[0]
        self.inverted_token_path = pretrained_model_name_or_path

class AbstractEncoder(nn.Module):
    def __init__(self):
        super().__init__()

    def encode(self, *args, **kwargs):
        raise NotImplementedError


class IdentityEncoder(AbstractEncoder):

    def encode(self, x):
        return x


class ClassEmbedder(nn.Module):
    def __init__(self, embed_dim, n_classes=1000, key='class', ucg_rate=0.1):
        super().__init__()
        self.key = key
        self.embedding = nn.Embedding(n_classes, embed_dim)
        self.n_classes = n_classes
        self.ucg_rate = ucg_rate

    def forward(self, batch, key=None, disable_dropout=False):
        if key is None:
            key = self.key
        # this is for use in crossattn
        c = batch[key][:, None]
        if self.ucg_rate > 0. and not disable_dropout:
            mask = 1. - torch.bernoulli(torch.ones_like(c) * self.ucg_rate)
            c = mask * c + (1-mask) * torch.ones_like(c)*(self.n_classes-1)
            c = c.long()
        c = self.embedding(c)
        return c

    def get_unconditional_conditioning(self, bs, device="cuda"):
        uc_class = self.n_classes - 1  # 1000 classes --> 0 ... 999, one extra class for ucg (class 1000)
        uc = torch.ones((bs,), device=device) * uc_class
        uc = {self.key: uc}
        return uc


def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self


class FrozenT5Embedder(AbstractEncoder):
    """Uses the T5 transformer encoder for text"""
    def __init__(self, version="google/t5-v1_1-large", device="cuda", max_length=77, freeze=True):  # others are google/t5-v1_1-xl and google/t5-v1_1-xxl
        super().__init__()
        self.tokenizer = T5Tokenizer.from_pretrained(version)
        self.transformer = T5EncoderModel.from_pretrained(version)
        self.device = device
        self.max_length = max_length   # TODO: typical value?
        if freeze:
            self.freeze()

    def freeze(self):
        self.transformer = self.transformer.eval()
        #self.train = disabled_train
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, text):
        batch_encoding = self.tokenizer(text, truncation=True, max_length=self.max_length, return_length=True,
                                        return_overflowing_tokens=False, padding="max_length", return_tensors="pt")
        tokens = batch_encoding["input_ids"].to(self.device)
        outputs = self.transformer(input_ids=tokens)

        z = outputs.last_hidden_state
        return z

    def encode(self, text):
        return self(text)


class FrozenCLIPEmbedder_1_5(AbstractEncoder):
    """Uses the CLIP transformer encoder for text (from huggingface)"""
    LAYERS = [
        "last",
        "penultimate"
    ]
    #openai/clip-vit-large-patch14    stabilityai/stable-diffusion-2-1
    def __init__(self, version="runwayml/stable-diffusion-v1-5", device="cuda", max_length=77,
                 freeze=True, layer="last", layer_idx=None):  # clip-vit-base-patch32
        super().__init__()
        #assert layer in 
        self.tokenizer = CLIPTokenizer.from_pretrained("/project/pi_ekalogerakis_umass_edu/dshivashok/ControlNet/evaluations/stable-diffusion-v1-5/tokenizer")
        self.transformer = CLIPTextModel.from_pretrained("/project/pi_ekalogerakis_umass_edu/dshivashok/ControlNet/evaluations/stable-diffusion-v1-5/text_encoder")
        
        self.tilm = TILM(self.tokenizer, self.transformer)
        self.inverted_token_path = None
        self.token = None
        if self.inverted_token_path is not None:
            self.tilm.load_textual_inversion(pretrained_model_name_or_path=self.inverted_token_path, token=self.token)

        self.device = device
        self.max_length = max_length
        if freeze:
            self.freeze()
        self.layer = layer
        self.layer_idx = layer_idx
        if self.layer == "last":
            self.layer_idx = 0
        elif self.layer == "penultimate":
            self.layer_idx = 1
        else:
            raise NotImplementedError()

    def freeze(self):
        self.transformer = self.transformer.eval()
        #self.train = disabled_train
        for param in self.parameters():
            param.requires_grad = False


    def forward(self, text, learned_embedding=None):
        print("Forwarding", learned_embedding)
        if learned_embedding is not None:
            return self.forward_learned_token(text, learned_embedding)
        else:
            return self.forward_normal(text)


    def forward_normal(self, text):
        batch_encoding = self.tokenizer(text, truncation=True, max_length=self.max_length, return_length=True,
                                        return_overflowing_tokens=False, padding="max_length", return_tensors="pt")
        tokens = batch_encoding["input_ids"].to(self.device)
        outputs = self.transformer(input_ids=tokens, output_hidden_states=self.layer=="hidden")
        if self.layer == "last" or self.layer == "penultimate":
            z = outputs.last_hidden_state
        elif self.layer == "pooled":
            z = outputs.pooler_output[:, None, :]
        else:
            z = outputs.hidden_states[self.layer_idx]
        return z

    def forward_learned_token(self, text, learned_embedding):
        token, inverted_token_path = learned_embedding
        print("learning token", token, inverted_token_path)

        if self.inverted_token_path is None and self.token is None:
            self.inverted_token_path = inverted_token_path
            self.token = token
            self.tilm.load_textual_inversion(pretrained_model_name_or_path=inverted_token_path, token=self.token)
            print("Loaded textual inversion")
        
        print("Using learned_token")
        batch_encoding = self.tokenizer(text, truncation=True, max_length=self.max_length, return_length=True,
                                        return_overflowing_tokens=False, padding="max_length", return_tensors="pt")
        tokens = batch_encoding["input_ids"].to(self.device)
        inputs_embeds = self.transformer.text_model.embeddings.token_embedding(tokens)
        decoded_tokens = self.tokenizer.decode(tokens[0])
        print("Decoded Tokens:", decoded_tokens)

        # loaded_tokens = torch.from_numpy(learned_token)
        # print(loaded_tokens.shape)

        # inputs_embeds = operate_on_embed(inputs_embeds, loaded_tokens, idx=object_idx+1, type="replace")

        #outputs = self.transformer(input_ids=tokens, output_hidden_states=self.layer=="hidden")
        #print("Outputs:", outputs[0].shape)
        outputs = get_clip_embeddings(self.transformer.text_model, tokens, inputs_embeds=inputs_embeds)
        if self.layer == "last" or self.layer == "penultimate":
            z = outputs.last_hidden_state
        elif self.layer == "pooled":
            z = outputs.pooler_output[:, None, :]
        else:
            z = outputs.hidden_states[self.layer_idx]
        return z

    def encode(self, text, learned_embedding=None):
        return self(text, learned_embedding=learned_embedding)

def operate_on_embed(inputs_embeds, learned_token, idx=0, type="add"):
    if type == "identity":
        inputs_embeds[0,idx,:] = inputs_embeds[0,idx,:]
    if type == "add":
        inputs_embeds[0,idx,:] = inputs_embeds[0,idx, :] + learned_token
    if type == "replace":
        print("learned token shape: ", learned_token.shape)
        inputs_embeds[0,idx,:] = learned_token
    return inputs_embeds

class FrozenCLIPEmbedder_2_1(AbstractEncoder):
    """Uses the CLIP transformer encoder for text (from huggingface)"""
    LAYERS = [
        "last",
        "penultimate"
    ]
    #openai/clip-vit-large-patch14    stabilityai/stable-diffusion-2-1
    def __init__(self, version="runwayml/stable-diffusion-v1-5", device="cuda", max_length=77,
                 freeze=True, layer="last", layer_idx=None):  # clip-vit-base-patch32
        super().__init__()
        #assert layer in 
        self.tokenizer = CLIPTokenizer.from_pretrained("/project/pi_ekalogerakis_umass_edu/dshivashok/ControlNet/evaluations/stable-diffusion-2-1/tokenizer")
        self.transformer = CLIPTextModel.from_pretrained("/project/pi_ekalogerakis_umass_edu/dshivashok/ControlNet/evaluations/stable-diffusion-2-1/text_encoder")
        
        self.tilm = TILM(self.tokenizer, self.transformer)
        self.inverted_token_path = None
        self.token = None
        if self.inverted_token_path is not None:
            self.tilm.load_textual_inversion(pretrained_model_name_or_path=self.inverted_token_path, token=self.token)

        self.device = device
        self.max_length = max_length
        if freeze:
            self.freeze()
        self.layer = layer
        self.layer_idx = layer_idx
        if self.layer == "last":
            self.layer_idx = 0
        elif self.layer == "penultimate":
            self.layer_idx = 1
        else:
            raise NotImplementedError()

    def freeze(self):
        self.transformer = self.transformer.eval()
        #self.train = disabled_train
        for param in self.parameters():
            param.requires_grad = False


    def forward(self, text, learned_embedding=None):
        print("Forwarding", learned_embedding)
        if learned_embedding is not None:
            return self.forward_learned_token(text, learned_embedding)
        else:
            return self.forward_normal(text)


    def forward_normal(self, text):
        batch_encoding = self.tokenizer(text, truncation=True, max_length=self.max_length, return_length=True,
                                        return_overflowing_tokens=False, padding="max_length", return_tensors="pt")
        tokens = batch_encoding["input_ids"].to(self.device)
        outputs = self.transformer(input_ids=tokens, output_hidden_states=self.layer=="hidden")
        if self.layer == "last" or self.layer == "penultimate":
            z = outputs.last_hidden_state
        elif self.layer == "pooled":
            z = outputs.pooler_output[:, None, :]
        else:
            z = outputs.hidden_states[self.layer_idx]
        return z

    def forward_learned_token(self, text, learned_embedding):
        token, inverted_token_path = learned_embedding
        print("learning token", token, inverted_token_path)

        if self.inverted_token_path is None and self.token is None:
            self.inverted_token_path = inverted_token_path
            self.token = token
            self.tilm.load_textual_inversion(pretrained_model_name_or_path=inverted_token_path, token=self.token)
            print("Loaded textual inversion")
        
        print("Using learned_token")
        batch_encoding = self.tokenizer(text, truncation=True, max_length=self.max_length, return_length=True,
                                        return_overflowing_tokens=False, padding="max_length", return_tensors="pt")
        tokens = batch_encoding["input_ids"].to(self.device)
        inputs_embeds = self.transformer.text_model.embeddings.token_embedding(tokens)
        decoded_tokens = self.tokenizer.decode(tokens[0])
        print("Decoded Tokens:", decoded_tokens)

        # loaded_tokens = torch.from_numpy(learned_token)
        # print(loaded_tokens.shape)

        # inputs_embeds = operate_on_embed(inputs_embeds, loaded_tokens, idx=object_idx+1, type="replace")

        #outputs = self.transformer(input_ids=tokens, output_hidden_states=self.layer=="hidden")
        #print("Outputs:", outputs[0].shape)
        outputs = get_clip_embeddings(self.transformer.text_model, tokens, inputs_embeds=inputs_embeds)
        if self.layer == "last" or self.layer == "penultimate":
            z = outputs.last_hidden_state
        elif self.layer == "pooled":
            z = outputs.pooler_output[:, None, :]
        else:
            z = outputs.hidden_states[self.layer_idx]
        return z

    def encode(self, text, learned_embedding=None):
        return self(text, learned_embedding=learned_embedding)

def operate_on_embed(inputs_embeds, learned_token, idx=0, type="add"):
    if type == "identity":
        inputs_embeds[0,idx,:] = inputs_embeds[0,idx,:]
    if type == "add":
        inputs_embeds[0,idx,:] = inputs_embeds[0,idx, :] + learned_token
    if type == "replace":
        print("learned token shape: ", learned_token.shape)
        inputs_embeds[0,idx,:] = learned_token
    return inputs_embeds

class FrozenCLIPEmbedderControl(AbstractEncoder):
    """Uses the CLIP transformer encoder for text (from huggingface)"""
    LAYERS = [
        "last",
        "pooled",
        "hidden"
    ]
    #openai/clip-vit-large-patch14
    def __init__(self, version="openai/clip-vit-large-patch14", device="cuda", max_length=77,
                 freeze=True, layer="last", layer_idx=None):  # clip-vit-base-patch32
        super().__init__()
        assert layer in self.LAYERS
        self.tokenizer = CLIPTokenizer.from_pretrained(version)
        self.transformer = CLIPTextModel.from_pretrained(version)

        self.tilm = TILM(self.tokenizer, self.transformer)
        self.inverted_token_path = None
        self.token = None
        if self.inverted_token_path is not None:
            self.tilm.load_textual_inversion(pretrained_model_name_or_path=self.inverted_token_path, token=self.token)


        self.device = device
        self.max_length = max_length
        if freeze:
            self.freeze()
        self.layer = layer
        self.layer_idx = layer_idx
        if layer == "hidden":
            assert layer_idx is not None
            assert 0 <= abs(layer_idx) <= 12

    def freeze(self):
        self.transformer = self.transformer.eval()
        #self.train = disabled_train
        for param in self.parameters():
            param.requires_grad = False


    def forward(self, text, learned_embedding=None):
        print("Control Forwarding")
        if learned_embedding is not None:
            return self.forward_learned_token(text, learned_embedding)
        else:
            return self.forward_normal(text)


    def forward_normal(self, text):
        batch_encoding = self.tokenizer(text, truncation=True, max_length=self.max_length, return_length=True,
                                        return_overflowing_tokens=False, padding="max_length", return_tensors="pt")
        print(text)
        tokens = batch_encoding["input_ids"].to(self.device)
        print(tokens)
        outputs = self.transformer(input_ids=tokens, output_hidden_states=self.layer=="hidden")
        if self.layer == "last":
            z = outputs.last_hidden_state
        elif self.layer == "pooled":
            z = outputs.pooler_output[:, None, :]
        else:
            z = outputs.hidden_states[self.layer_idx]
        return z

    def forward_learned_token(self, text, learned_embedding):
        token, inverted_token_path = learned_embedding
        print("learning token", token, inverted_token_path)

        if self.inverted_token_path is None and self.token is None:
            self.inverted_token_path = inverted_token_path
            self.token = token
            self.tilm.load_textual_inversion(pretrained_model_name_or_path=inverted_token_path, token=self.token)
            print("Loaded textual inversion")
        
        print("Using learned_token")
        batch_encoding = self.tokenizer(text, truncation=True, max_length=self.max_length, return_length=True,
                                        return_overflowing_tokens=False, padding="max_length", return_tensors="pt")
        tokens = batch_encoding["input_ids"].to(self.device)
        inputs_embeds = self.transformer.text_model.embeddings.token_embedding(tokens)
        decoded_tokens = self.tokenizer.decode(tokens[0])
        print("Decoded Tokens:", decoded_tokens)

        # loaded_tokens = torch.from_numpy(learned_token)
        # print(loaded_tokens.shape)

        # inputs_embeds = operate_on_embed(inputs_embeds, loaded_tokens, idx=object_idx+1, type="replace")

        #outputs = self.transformer(input_ids=tokens, output_hidden_states=self.layer=="hidden")
        #print("Outputs:", outputs[0].shape)
        outputs = get_clip_embeddings(self.transformer.text_model, tokens, inputs_embeds=inputs_embeds)
        if self.layer == "last" or self.layer == "penultimate":
            z = outputs.last_hidden_state
        elif self.layer == "pooled":
            z = outputs.pooler_output[:, None, :]
        else:
            z = outputs.hidden_states[self.layer_idx]
        return z


    # def forward_learned_token(self, text, learned_embedding):
    #     learned_token, object_idx = learned_embedding
    #     print("learning token", text)
    #     batch_encoding = self.tokenizer(text, truncation=True, max_length=self.max_length, return_length=True,
    #                                     return_overflowing_tokens=False, padding="max_length", return_tensors="pt")
    #     tokens = batch_encoding["input_ids"].to(self.device)
    #     print(tokens)
    #     inputs_embeds = self.transformer.text_model.embeddings.token_embedding(tokens)
    #     decoded_tokens = self.tokenizer.decode(tokens[0])

    #     loaded_tokens = torch.from_numpy(learned_token)

    #     inputs_embeds = operate_on_embed(inputs_embeds, loaded_tokens, idx=object_idx+1, type="replace")

    #     #outputs = self.transformer(input_ids=tokens, output_hidden_states=self.layer=="hidden")
    #     #print("Outputs:", outputs[0].shape)
    #     outputs = get_clip_embeddings(self.transformer.text_model, tokens, inputs_embeds=inputs_embeds)
    #     if self.layer == "last":
    #         z = outputs.last_hidden_state
    #     elif self.layer == "pooled":
    #         z = outputs.pooler_output[:, None, :]
    #     else:
    #         z = outputs.hidden_states[self.layer_idx]
    #     return z

    def encode(self, text, learned_embedding=None):
        return self(text, learned_embedding=learned_embedding)

def operate_on_embed(inputs_embeds, learned_token, idx=0, type="add"):
    if type == "identity":
        inputs_embeds[0,idx,:] = inputs_embeds[0,idx,:]
    if type == "add":
        inputs_embeds[0,idx,:] = inputs_embeds[0,idx, :] + learned_token
    if type == "replace":
        print("learned token shape: ", learned_token.shape)
        inputs_embeds[0,idx,:] = learned_token
    return inputs_embeds

class FrozenOpenCLIPEmbedder(AbstractEncoder):
    """
    Uses the OpenCLIP transformer encoder for text
    """
    LAYERS = [
        #"pooled",
        "last",
        "penultimate"
    ]
    def __init__(self, arch="ViT-H-14", version="laion2b_s32b_b79k", device="cuda", max_length=77,
                 freeze=True, layer="last"):
        super().__init__()
        assert layer in self.LAYERS
        model, _, _ = open_clip.create_model_and_transforms(arch, device=torch.device('cpu'), pretrained=version)
        del model.visual
        self.model = model

        self.device = device
        self.max_length = max_length
        if freeze:
            self.freeze()
        self.layer = layer
        if self.layer == "last":
            self.layer_idx = 0
        elif self.layer == "penultimate":
            self.layer_idx = 1
        else:
            raise NotImplementedError()
    
    def get_vocab(self):
        return dict(self.encoder, **self.added_tokens_encoder)

    def freeze(self):
        self.model = self.model.eval()
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, text):
        tokens = open_clip.tokenize(text)
        z = self.encode_with_transformer(tokens.to(self.device))
        return z

    def encode_with_transformer(self, text):
        x = self.model.token_embedding(text)  # [batch_size, n_ctx, d_model]
        x = x + self.model.positional_embedding
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.text_transformer_forward(x, attn_mask=self.model.attn_mask)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.model.ln_final(x)
        return x

    def text_transformer_forward(self, x: torch.Tensor, attn_mask = None):
        for i, r in enumerate(self.model.transformer.resblocks):
            if i == len(self.model.transformer.resblocks) - self.layer_idx:
                break
            if self.model.transformer.grad_checkpointing and not torch.jit.is_scripting():
                x = checkpoint(r, x, attn_mask)
            else:
                x = r(x, attn_mask=attn_mask)
        return x

    def encode(self, text):
        return self(text)


class FrozenCLIPT5Encoder(AbstractEncoder):
    def __init__(self, clip_version="openai/clip-vit-large-patch14", t5_version="google/t5-v1_1-xl", device="cuda",
                 clip_max_length=77, t5_max_length=77):
        super().__init__()
        self.clip_encoder = FrozenCLIPEmbedder(clip_version, device, max_length=clip_max_length)
        self.t5_encoder = FrozenT5Embedder(t5_version, device, max_length=t5_max_length)
        print(f"{self.clip_encoder.__class__.__name__} has {count_params(self.clip_encoder)*1.e-6:.2f} M parameters, "
              f"{self.t5_encoder.__class__.__name__} comes with {count_params(self.t5_encoder)*1.e-6:.2f} M params.")

    def encode(self, text):
        return self(text)

    def forward(self, text):
        clip_z = self.clip_encoder.encode(text)
        t5_z = self.t5_encoder.encode(text)
        return [clip_z, t5_z]




