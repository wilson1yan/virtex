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
        output_hidden = self.textual(caption_tokens, masked_labels)

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


class VisualMoCoModel(nn.Module):
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

        self._visual_projection = nn.Linear(2048, textual.hidden_size)
        self._layer_norm = (
            nn.Identity()
            if not fused_normalize
            else nn.LayerNorm(textual.hidden_size, eps=1e-8)
        )
        # Hold a copy for encoding keys.
        self._momentum_encoder = visual
        self._max_queue_size = queue_size

        # Initialize an empty queue.
        self._visual_queue = torch.zeros((0, textual.hidden_size))

    def forward(self, image: torch.Tensor, caption_tokens: torch.Tensor):
        # shape: (batch_size, 2048, 7, 7)
        image_features = self.visual(image)

        # shape: (batch_size, 49, 2048)
        image_features = image_features.view(-1, 2048, 49).permute(0, 2, 1)

        # Perform global average pooling and projection.
        # shape: (batch_size, projected_image_feature_size)
        # `projected_image_feature_size` is `textual.hidden_size`
        image_features = image_features.mean(dim=1)
        projected_image_features = self._visual_projection(image_features)

        # If queue is not large enough, just add batch to queue and finish.
        if self._visual_queue.size(0) < self._max_queue_size:
            self._update_queue(projected_image_features)
            return

    def _update_queue(self, projected_image_features: torch.Tensor):
        # Detach features to avoid gradient/memory leaks.
        # shape: (batch_size, projected_image_feature_size)
        projected_image_features = projected_image_features.detach()

        batch_size = projected_image_features.size(0)
        current_queue_size = self._visual_queue.size(0)

        # Enqueue: insert current batch in the front.
        # shape: (current_queue_size + batch_size, projected_image_feature_size)
        self._visual_queue = torch.cat(
            (projected_image_features, self._visual_queue), dim=0
        )
        # Dequeue: trim the oldest batch from the end.
        self._visual_queue = self._visual_queue[: self._max_queue_size, :]

    def to(self, *args, **kwargs):
        r"""Override super class method to move queue to specified device."""
        self._visual_queue = self._visual_queue.to(*args, **kwargs)
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
