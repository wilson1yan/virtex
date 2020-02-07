import functools
from typing import Optional

import torch
from torch import nn

from viswsl.modules.embedding import WordAndPositionalEmbedding
from viswsl.modules.transformer import (
    PreNormTransformerEncoderLayer,
    PreNormTransformerDecoderLayer,
)


class TextualStream(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        is_bidirectional: bool = True,
        padding_idx: int = 0,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.is_bidirectional = is_bidirectional
        self.padding_idx = padding_idx

    @property
    def textual_feature_size(self):
        return self.hidden_size

    @staticmethod
    def _init_weights(module):
        r"""Initialize weights like BERT - N(0.0, 0.02), bias = 0."""

        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.MultiheadAttention):
            module.in_proj_weight.data.normal_(mean=0.0, std=0.02)
            module.out_proj.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    def forward(
        self,
        caption_tokens: torch.Tensor,
        caption_lengths: torch.Tensor,
        visual_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        raise NotImplementedError

    @functools.lru_cache(maxsize=10)
    def _generate_square_subsequent_mask(
        self, size: int, device: torch.device
    ) -> torch.Tensor:
        r"""
        Generate a mask for "future" positions, useful when using this module
        for language modeling.

        Parameters
        ----------
        size: int
        """
        # Default mask is for forward direction. Flip for backward direction.
        mask = torch.triu(torch.ones(size, size, device=device), diagonal=1)
        mask = mask.masked_fill(mask == 1, float("-inf"))
        return mask


class EmbeddingTextualStream(TextualStream):
    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        dropout: float = 0.1,
        is_bidirectional: bool = True,
        padding_idx: int = 0,
    ):
        super().__init__(
            vocab_size,
            hidden_size,
            is_bidirectional=is_bidirectional,
            padding_idx=padding_idx,
        )
        self.embedding = WordAndPositionalEmbedding(
            self.vocab_size, self.textual_feature_size, dropout=dropout
        )
        self.apply(self._init_weights)

    def forward(
        self,
        caption_tokens: torch.Tensor,
        caption_lengths: torch.Tensor,
        visual_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # shape: (batch_size, max_caption_length, textual_feature_size)
        textual_features = self.embedding(caption_tokens)
        return textual_features


class AllLayersFusionTextualStream(TextualStream):
    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        feedforward_size: int,
        attention_heads: int,
        num_layers: int,
        dropout: float = 0.1,
        is_bidirectional: bool = True,
        norm_type: str = "pre",
        padding_idx: int = 0,
    ):
        super().__init__(
            vocab_size,
            hidden_size,
            is_bidirectional=is_bidirectional,
            padding_idx=padding_idx,
        )
        self.feedforward_size = feedforward_size
        self.attention_heads = attention_heads
        self.num_layers = num_layers

        self.embedding = WordAndPositionalEmbedding(
            self.vocab_size, self.textual_feature_size, dropout=dropout
        )
        # Make encoder layer depending on whether it's a Pre-Norm or Post-Norm.
        LayerClass = (
            nn.TransformerDecoderLayer
            if norm_type == "post"
            else PreNormTransformerDecoderLayer
        )
        _layer = LayerClass(
            self.textual_feature_size,
            self.attention_heads,
            dim_feedforward=self.feedforward_size,
            dropout=dropout,
            activation="gelu",
        )
        # We call this member as "encoder" for consistent naming, and because
        # it still "encodes" the caption for us.
        self.encoder = nn.TransformerDecoder(_layer, self.num_layers)
        self.apply(self._init_weights)

    def forward(
        self,
        caption_tokens: torch.Tensor,
        caption_lengths: torch.Tensor,
        visual_features: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, max_caption_length = caption_tokens.size()

        # Create a mask based on caption lengths, shape: (batch_size, )
        # Form a binary mask: it is True for padding positions.
        # These positions will be ignored for multi-headed attention.
        ones = torch.ones_like(caption_tokens)
        caption_mask = caption_lengths.unsqueeze(1) < ones.cumsum(dim=1)

        # shape: (batch_size, max_caption_length, textual_feature_size)
        caption_embeddings = self.embedding(caption_tokens)

        # An additive mask for masking the future (one direction) if textual
        # stream is unidirectional. For bidirectional stream, it is `None`.
        unidirectional_mask = (
            None
            if self.is_bidirectional
            else self._generate_square_subsequent_mask(
                max_caption_length, caption_embeddings.device
            )
        )
        # We transpose the first two dimensions of tokens embeddings and visual
        # features, as required by encoder.
        caption_embeddings = caption_embeddings.transpose(0, 1)
        visual_features = visual_features.transpose(0, 1)

        # shape: (max_caption_length, batch_size, hidden_size)
        textual_features = self.encoder(
            caption_embeddings,
            visual_features,
            tgt_mask=unidirectional_mask,
            tgt_key_padding_mask=caption_mask,
        )
        # Undo the transpose and bring batch to dim 0.
        # shape: (batch_size, sequence_length, hidden_size)
        textual_features = textual_features.transpose(0, 1)
        return textual_features


class LastLayerFusionTextualStream(TextualStream):
    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        feedforward_size: int,
        attention_heads: int,
        num_layers: int,
        dropout: float = 0.1,
        is_bidirectional: bool = True,
        norm_type: str = "pre",
        padding_idx: int = 0,
    ):
        super().__init__(
            vocab_size,
            hidden_size,
            is_bidirectional=is_bidirectional,
            padding_idx=padding_idx,
        )
        self.feedforward_size = feedforward_size
        self.attention_heads = attention_heads
        self.num_layers = num_layers

        self.embedding = WordAndPositionalEmbedding(
            self.vocab_size, self.textual_feature_size, dropout=dropout
        )
        # Make encoder layer depending on whether it's a Pre-Norm or Post-Norm.
        NonLastLayerClass = (
            nn.TransformerEncoderLayer
            if norm_type == "post"
            else PreNormTransformerEncoderLayer
        )
        _layer = NonLastLayerClass(
            self.textual_feature_size,
            self.attention_heads,
            dim_feedforward=self.feedforward_size,
            dropout=dropout,
            activation="gelu",
        )
        # We still call this member as "encoder" for consistent API, and
        # because it still "encodes" the caption for us.
        self.encoder = nn.TransformerEncoder(_layer, self.num_layers - 1)

        # Make encoder layer depending on whether it's a Pre-Norm or Post-Norm.
        LastLayerClass = (
            nn.TransformerDecoderLayer
            if norm_type == "post"
            else PreNormTransformerDecoderLayer
        )
        self.last_layer = LastLayerClass(
            self.textual_feature_size,
            self.attention_heads,
            dim_feedforward=self.feedforward_size,
            dropout=dropout,
            activation="gelu",
        )
        self.apply(self._init_weights)

    def forward(
        self,
        caption_tokens: torch.Tensor,
        caption_lengths: torch.Tensor,
        visual_features: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, max_caption_length = caption_tokens.size()

        # All of the method body here is same as `AllLayersFusionTextualStream`,
        # except the call to `self.encoder`.
        ones = torch.ones_like(caption_tokens)
        caption_mask = caption_lengths.unsqueeze(1) < ones.cumsum(dim=1)

        # shape: (batch_size, max_caption_length, textual_feature_size)
        caption_embeddings = self.embedding(caption_tokens)

        # An additive mask for masking the future (one direction) if textual
        # stream is unidirectional. For bidirectional stream, it is `None`.
        unidirectional_mask = (
            None
            if self.is_bidirectional
            else self._generate_square_subsequent_mask(
                max_caption_length, caption_embeddings.device
            )
        )
        # We transpose the first two dimensions of tokens embeddings and visual
        # features, as required by encoder.
        caption_embeddings = caption_embeddings.transpose(0, 1)
        visual_features = visual_features.transpose(0, 1)

        # Get textual features from all layers except the last one.
        textual_features = self.encoder(
            caption_embeddings,
            mask=unidirectional_mask,
            src_key_padding_mask=caption_mask,
        )
        # Finally fuse visual features in the last layer.
        # shape: (max_caption_length, batch_size, hidden_size)
        textual_features = self.last_layer(
            textual_features,
            visual_features,
            tgt_mask=unidirectional_mask,
            tgt_key_padding_mask=caption_mask,
        )
        # shape: (batch_size, max_caption_length, hidden_size)
        textual_features = textual_features.transpose(0, 1)
        return textual_features
