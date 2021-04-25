from typing import Optional

import torch
from torch import nn


# code ref: https://github.com/alexmt-scale/causal-transformer-decoder/blob/master/causal_transformer_decoder/model.py
class CausalTransformerDecoder(nn.TransformerDecoder):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sampling = False

    def sample_on(self):
        self.sampling = True

    def sample_off(self):
        self.sampling = False
    
    def forward(
        self,
        tgt: torch.Tensor,
        memory: Optional[torch.Tensor] = None,
        cache: Optional[torch.Tensor] = None,
        tgt_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None,
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
        memory_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        output = tgt

        if not self.sampling:
            if cache is not None:
                raise ValueError("cache parameter should be None in trainig mode")
            for mod in self.layers:
                output = mod(
                    output,
                    memory,
                    tgt_mask=tgt_mask,
                    memory_mask=memory_mask,
                    tgt_key_padding_mask=tgt_key_padding_mask,
                    memory_key_padding_mask=memory_key_padding_mask,
                    sampling=False
                )
            return output, None

        new_token_cache = []
        for i, mod in enumerate(self.layers):
            output = mod(output, memory, sampling=True)
            new_token_cache.append(output)
            if cache is not None:
                output = torch.cat([cache[i], output], dim=0)

        if cache is not None:
            new_cache = torch.cat([cache, torch.stack(new_token_cache, dim=0)], dim=1)
        else:
            new_cache = torch.stack(new_token_cache, dim=0)

        return output, new_cache


class PreNormTransformerDecoderLayer(nn.TransformerDecoderLayer):
    r"""
    A variant of :class:`torch.nn.TransformerDecoderLayer` where layer
    normalization is included inside the residual branch, and performed before
    self-attention and feedforward layers.

    Refer documentation of :class:`torch.nn.TransformerDecoderLayer` for more
    details on the API.
    """

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None,
                sampling=False):
        if not sampling:
            # We use the members (modules) from super-class, just the order of
            # operations is changed here. First layernorm, then attention.
            tgt2 = self.norm1(tgt)
            tgt2, _ = self.self_attn(
                tgt2, tgt2, tgt2, attn_mask=tgt_mask,
                key_padding_mask=tgt_key_padding_mask
            )
            tgt = tgt + self.dropout1(tgt2)

            # Layernorm first, then decoder attention.
            tgt2 = self.norm2(tgt)
            tgt2, _ = self.multihead_attn(
                tgt2, memory, memory, attn_mask=memory_mask,
                key_padding_mask=memory_key_padding_mask
            )
            tgt = tgt + self.dropout2(tgt2)

            # Layernorm first, then transformation through feedforward network.
            tgt2 = self.norm3(tgt)
            tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
            tgt = tgt + self.dropout3(tgt2)
            return tgt
        else:
            tgt_last_tok = tgt[-1:, :, :]

            # self-attention
            tgt2 = self.norm1(tgt)
            tgt2, _ = self.self_attn(
                tgt2[-1:, :, :], tgt2, tgt2,
                attn_mask=None, key_padding_mask=tgt_key_padding_mask
            )
            tgt_last_tok = tgt_last_tok + self.dropout1(tgt2)

            # encoder-decoder cross attention
            if memory is not None:
                tgt2 = self.norm2(tgt_last_tok)
                tgt2, _ = self.multihead_attn(
                    tgt2, memory, memory, attn_mask=memory_mask,
                    key_padding_mask=memory_key_padding_mask
                )
                tgt_last_tok = tgt_last_tok + self.dropout2(tgt2)
            
            # feed forward
            tgt2 = self.norm3(tgt_last_tok)
            tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
            tgt_last_tok = tgt_last_tok + self.dropout3(tgt2)
            return tgt_last_tok

