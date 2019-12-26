import copy
from typing import Dict, List

import torch
from torch import nn

# TODO (kd): have attention/fusion technique as a dependency injection.
from viswsl.modules.attention import ScaledDotProductAttention


class WordMaskingModel(nn.Module):
    def __init__(self, visual, textual, fused_normalize: bool = False):
        super().__init__()
        self.visual = visual
        self.textual = textual
        self._fused_projection_size = 2048 + textual.hidden_size

        self._attention = ScaledDotProductAttention(2048, textual.hidden_size)
        self._layer_norm = (
            nn.Identity()
            if not fused_normalize
            else nn.LayerNorm(self._fused_projection_size, eps=1e-8)
        )
        self._linear = nn.Linear(self._fused_projection_size, textual.vocab_size)
        self._loss = nn.CrossEntropyLoss(ignore_index=textual.padding_idx)

    def forward(
        self,
        image: torch.Tensor,
        caption_tokens: torch.Tensor,
        masked_labels: torch.Tensor,
    ):
        # shape: (batch_size, 2048, 7, 7)
        image_features = self.visual(image)

        # shape: (batch_size, 49, 2048)
        image_features = image_features.view(-1, 2048, 49).permute(0, 2, 1)

        # shape: (batch_size, max_caption_length, hidden_size)
        output_hidden = self.textual(caption_tokens)

        # shape: (batch_size, max_caption_length, 2048)
        attended_features = self._attention(image_features, output_hidden)

        # shape: (batch_size, max_caption_length, fused_projection_size)
        concatenated = torch.cat((attended_features, output_hidden), dim=-1)
        concatenated = self._layer_norm(concatenated)

        # shape: (batch_size, max_caption_length, vocab_size)
        output_logits = self._linear(concatenated)

        # Get predictions from logits, only the predictions at [MASK]ed
        # positions would be useful.
        predictions = torch.argmax(output_logits, dim=-1)
        output_dict = {"predictions": predictions}

        # Collapse dimensions: convert logits to (N, C), targets to (N,).
        output_dict["loss"] = self._loss(
            output_logits.view(-1, output_logits.size(-1)), masked_labels.view(-1)
        )
        return output_dict


class MomentumContrastModel(nn.Module):
    def __init__(
        self,
        visual,
        textual,
        momentum: float = 0.999,
        queue_size: int = 4096,
        temperature: float = 0.07,
        fused_normalize: bool = False,
    ):
        super().__init__()
        self.visual = visual
        self.textual = textual

        self._momentum = momentum
        self._max_queue_size = queue_size
        self._temperature = temperature

        self._visual_projection = nn.Linear(2048, textual.hidden_size)
        self._layer_norm = (
            nn.Identity()
            if not fused_normalize
            else nn.LayerNorm(textual.hidden_size, eps=1e-8)
        )

        # Initialize empty queues for visual and textual features.
        self._visual_queue = torch.zeros((0, textual.hidden_size))
        self._textual_queue = torch.zeros((0, textual.hidden_size))

        # Instantiate key encoders for visual and textual streams.
        self._visual_momentum_encoder = copy.deepcopy(visual)
        self._visual_momentum_encoder.requires_grad = False

        self._textual_momentum_encoder = copy.deepcopy(textual)
        self._textual_momentum_encoder.requires_grad = False

        self._loss = nn.CrossEntropyLoss()

    def forward(self, image: torch.Tensor, caption_tokens: torch.Tensor):

        batch_size = image.size(0)

        # ====================================================================
        # Prepare query vectors for both images, and captions.
        # --------------------------------------------------------------------
        # shape: (batch_size, 2048, 7, 7)
        image_features = self.visual(image)

        # shape: (batch_size, 49, 2048)
        image_features = image_features.view(-1, 2048, 49).permute(0, 2, 1)

        # Perform global average pooling and projection.
        image_features = image_features.mean(dim=1)

        # shape: (batch_size, textual.hidden_size)
        image_query = self._visual_projection(image_features)

        # Collect features for each caption corresponding to [CLS] token.
        # shape: (batch_size, textual.hidden_size)
        caption_query = self.textual(caption_tokens)[:, 0]
        # ====================================================================

        # Fill queues completely before starting momentum contrast.
        if self._visual_queue.size(0) < self._max_queue_size:
            self._update_queues(image_query, caption_query)
            return

        # Momentum update: update key encoders from prior iteration.
        self._momentum_update(self.visual, self._visual_momentum_encoder)
        self._momentum_update(self.textual, self._textual_momentum_encoder)

        with torch.no_grad():
            # Compute keys like queries, just using the momentum encoder.
            image_keys = self._visual_momentum_encoder(image)
            image_keys = image_keys.view(-1, 2048, 49).permute(0, 2, 1)
            image_keys = image_keys.mean(dim=1)

            # shape: (batch_size, textual.hidden_size)
            image_keys = self._visual_projection(image_keys)

            # shape: (batch_size, textual.hidden_size)
            caption_keys = self._textual_momentum_encoder(caption_tokens)[:, 0]

        # Compute dot product similarity between image query and caption keys.
        # shape: (batch_size, 1)
        positive_logits = (image_query * caption_keys).sum(dim=1).unsqueeze(-1)

        # shape: (batch_size, queue_size)
        negative_logits = torch.mm(image_query, self._textual_queue.T)

        # shape: (batch_size, 1 + queue_size)
        logits = torch.cat((positive_logits, negative_logits), dim=1)

        # Matching key (aligned caption) is always at index 0, prepare labels.
        labels = torch.zeros(batch_size, device=logits.device).long()
        image_query_loss = self._loss(logits / self._temperature, labels)

        # Compute dot product similarity between caption query and image keys.
       	# shape: (batch_size, 1)
        positive_logits = (caption_query * image_keys).sum(dim=1).unsqueeze(-1)

       	# shape: (batch_size, queue_size)
        negative_logits = torch.mm(caption_query, self._visual_queue.T)

       	# shape: (batch_size, 1	+ queue_size)
        logits = torch.cat((positive_logits, negative_logits), dim=1)

        # Matching key (aligned image) is always at index 0, prepare labels.
        labels = torch.zeros(batch_size, device=logits.device).long()
        caption_query_loss = self._loss(logits / self._temperature, labels)

        # Add current batch (keys) to queues.
        self._update_queues(image_keys, caption_keys)
        # ====================================================================

        return {
            "loss": image_query_loss + caption_query_loss,
            # Extra two keys only for logging purposes.
            "image_query_loss": image_query_loss.detach().mean(),
            "caption_query_loss": caption_query_loss.detach().mean(),
        }

    def _momentum_update(self, query_encoder, key_encoder):
        r"""
        Perform momentum update step on key encoders. We accept arguments
        instead of accessing directly by `self.` to reduce memory usage.
        """
        key_parameters = key_encoder.state_dict()
        for name, param in query_encoder.named_parameters():
            if name in key_parameters:
                key_parameters[name].data.copy_(
                    self._momentum * key_parameters[name].data
                    + (1 - self._momentum) * param.data
                )
        key_encoder.load_state_dict(key_parameters)

    def _update_queues(self, image_keys: torch.Tensor, caption_keys: torch.Tensor):
        # Detach features to avoid gradient/memory leaks.
        image_keys = image_keys.detach()
        caption_keys = caption_keys.detach()

        # Both of these values will be same for both queues.
        batch_size = image_keys.size(0)
        current_queue_size = self._visual_queue.size(0)

        # Enqueue: insert current batch in the front.
        self._visual_queue = torch.cat((image_keys, self._visual_queue), dim=0)
        self._textual_queue = torch.cat((caption_keys, self._textual_queue), dim=0)

        # Dequeue: trim the oldest batch from the end.
        self._visual_queue = self._visual_queue[: self._max_queue_size, :]
        self._textual_queue = self._textual_queue[: self._max_queue_size, :]

    def to(self, *args, **kwargs):
        r"""Override super class method to move queue to specified device."""
        self._visual_queue = self._visual_queue.to(*args, **kwargs)
        self._textual_queue = self._textual_queue.to(*args, **kwargs)
        return super().to(*args, **kwargs)


class VOC07ClassificationFeatureExtractor(nn.Module):
    r"""
    Pool intermediate layer outputs for ResNet-like visual streams such that
    their feature size is approximately 9000. We train linear SVMs using these
    features for one vs. all classification on Pascal VOC dataset. This is
    consistent with FAIR Self Supervision Benchmark (Goyal et al, 2019).

    References
    ----------
    Scaling and Benchmarking Self-Supervised Visual Representation Learning.
    Priya Goyal, Dhruv Mahajan, Abhinav Gupta, Ishan Misra
    https://arxiv.org/abs/1905.01235
    """

    def __init__(self, pretrained_model, mode: str = "avg", normalize: bool = True):
        super().__init__()
        self._cnn = pretrained_model.visual

        layer = nn.AdaptiveAvgPool2d if mode == "avg" else nn.AdaptiveMaxPool2d
        self._normalize = normalize

        # This spatial size will downsample features from ResNet-like models
        # so their size is ~9000 when flattened
        self._layer1_pool = layer(6)
        self._layer2_pool = layer(4)
        self._layer3_pool = layer(3)
        self._layer4_pool = layer(2)

        # fmt: off
        # A dict of references to layers for convenience.
        self._pool = {
            "layer1": self._layer1_pool, "layer2": self._layer2_pool,
            "layer3": self._layer3_pool, "layer4": self._layer4_pool,
        }
        # fmt: on

    def forward(
        self, image: torch.Tensor, layer_names: List[str] = None
    ) -> Dict[str, torch.Tensor]:

        layer_names = layer_names or list(self._pool.keys())
        features = self._cnn(image, return_intermediate_outputs=True)

        # keys: {"layer1", "layer2", "layer3", "layer4"}
        for layer_name in features:
            if layer_name in layer_names:
                pooled = self._pool[layer_name](features[layer_name])
                pooled = pooled.view(pooled.size(0), -1)
                if self._normalize:
                    pooled = pooled / torch.norm(pooled, dim=-1).unsqueeze(-1)
                features[layer_name] = pooled

        return features
