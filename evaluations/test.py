import os
import sys
from typing import Dict, List, Optional, Union


SCRIPT_DIR = os.path.dirname(os.path.abspath("/project/pi_ekalogerakis_umass_edu/dshivashok/ControlNet/evaluations/baseline_learned_control_k.py"))
print(SCRIPT_DIR)
sys.path.append(os.path.dirname(SCRIPT_DIR))

from diffusers.loaders.textual_inversion import *
from visualization.modules import FrozenCLIPEmbedder

inverted_token_path = "/project/pi_ekalogerakis_umass_edu/dmpetrov/data/textual_inversion_results/03001627_72bc27a22e5f516e8aee1b6cfa0c3234/learned_embeds-steps-4001.safetensors"
token = "<03001627-chair>"


class TILM(TextualInversionLoaderMixin):
    def __init__(self):
        super().__init__()
        self.tokenizer = None
        self.text_encoder = None
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
        tilm.inverted_token = tokens[0]
        tilm.inverted_token_path = embeddings






tilm = TILM()

tilm.tokenizer = FrozenCLIPEmbedder().tokenizer
tilm.text_encoder = FrozenCLIPEmbedder().transformer
tilm.load_textual_inversion(inverted_token_path, token)
print(tilm.inverted_token)
print(tilm.inverted_token_path[0].shape)



