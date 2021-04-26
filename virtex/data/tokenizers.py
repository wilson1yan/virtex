import csv
import os
from typing import Any, Dict, List

import sentencepiece as sp


class SentencePieceBPETokenizer(object):
    r"""
    A tokenizer based on `SentencePiece <https://github.com/google/sentencepiece>`_
    with BPE sub-routine. It encodes caption strings into list of tokens.

    Parameters
    ----------
    model_path: str
        Path to the ``.model`` file trained by SentencePiece.
    """
    SP_SPACE = u"▁"

    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = self._load_model(model_path)

    def _load_model(self, model_path):
        if not os.path.exists(model_path):
            if model_path == 'clip':
                from clip.clip import _tokenizer
                model = _tokenizer
            elif model_path == 'gpt2':
                from transformers import GPT2Tokenizer
                model = GPT2Tokenizer.from_pretrained('gpt2')
            else:
                raise Exception(f"Invalid model_path = {model_path}")
        else:
            model = sp.SentencePieceProcessor()
            model.Load(model_path)
        return model

    def __getstate__(self):
        r"""
        This magic method, along with ``__setstate__`` makes an object of this
        class picklable (and usable while data loading with multiple workers).
        """
        state_dict = self.__dict__.copy()
        state_dict["model"] = None
        return state_dict

    def __setstate__(self, state_dict: Dict[str, Any]):
        self.__dict__ = state_dict
        self.model = self._load_model(self.model_path)

    @property
    def bos_id(self):
        if self.model_path == 'clip':
            return self.model.encoder["<|startoftext|>"]
        elif self.model_path == 'gpt2':
            return self.model.bos_token_id
        else:
            return self.model.bos_id()
    
    @property
    def eos_id(self):
        if self.model_path == 'clip':
            return self.model.encoder["<|endoftext|>"]
        elif self.model_path == 'gpt2':
            return self.model.eos_token_id
        else:
            return self.model.eos_id()

    @property
    def pad_id(self):
        if self.model_path in ['clip', 'gpt2']:
            return 0
        else:
            return self.model.pad_id()

    def get_vocab_size(self) -> int:
        r"""Return number of tokens in vocabulary (including special tokens)."""
        if self.model_path == 'clip':
            return len(self.model.encoder)
        else:
            return len(self.model)

    def token_to_id(self, token: str) -> int:
        r"""Get integer ID of a string token (``<unk>`` if does not exist)."""
        # Since tokenizer uses subword regularization, one token may break down to multiple IDs.
        # Keep trying till we get a single ID.
        return self.model.piece_to_id(token)

    def id_to_token(self, token_id: int) -> str:
        r"""Get string token of an integer ID (``<unk>`` if does not exist)."""
        return self.model.id_to_piece(token_id)

    def encode(self, text: str) -> List[int]:
        r"""Convert a text string to a list of integer token ids."""
        if self.model_path in ['clip', 'gpt2']:
            return self.model.encode(text)
        else:
            return self.model.EncodeAsIds(text)

    def decode(self, token_ids: List[int]) -> str:
        r"""Convert a sequence of token IDs to a text string."""
        if self.model_path in ['clip', 'gpt2']:
            token_ids = token_ids[:token_ids.index(self.eos_id)]
            return self.model.decode(token_ids)
        else:
            return self.model.DecodeIds(token_ids)
