import copy

import torch
from torch import nn
from torch.nn import functional as F


class MomentumContrastModel(nn.Module):
    def __init__(
        self,
        visual,
        textual,
        feature_size: int,
        momentum: float,
        queue_size: int,
        temperature: float,
    ):
        super().__init__()
        self.visual = visual
        self.textual = textual

        self._momentum = momentum
        self._max_queue_size = queue_size
        self._temperature = temperature

        self._visual_projection = nn.Linear(2048, feature_size)
        self._textual_projection = nn.Linear(textual.hidden_size, feature_size)

        # Initialize empty queues for visual and textual features.
        self._visual_queue = torch.zeros((0, feature_size))
        self._textual_queue = torch.zeros((0, feature_size))

        # Instantiate key encoders for visual and textual streams.
        self._visual_key_encoder = copy.deepcopy(visual)
        self._textual_key_encoder = copy.deepcopy(textual)

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

        # shape: (batch_size, feature_size)
        image_query = self._visual_projection(image_features)
        image_query = F.normalize(image_query, dim=1, p=2)

        # Collect features for each caption corresponding to [CLS] token.
        # shape: (batch_size, feature_size)
        caption_query = self.textual(caption_tokens)[:, 0]
        caption_query = self._textual_projection(caption_query)
        caption_query = F.normalize(caption_query, dim=1, p=2)
        # ====================================================================

        # Momentum update: update key encoders from prior iteration.
        self._momentum_update(self.visual, self._visual_key_encoder)
        self._momentum_update(self.textual, self._textual_key_encoder)

        with torch.no_grad():
            # Compute keys like queries, just using the momentum encoder.
            image_keys = self._visual_key_encoder(image)
            image_keys = image_keys.view(-1, 2048, 49).permute(0, 2, 1)
            image_keys = image_keys.mean(dim=1)

            # shape: (batch_size, feature_size)
            image_keys = self._visual_projection(image_keys)
            image_keys = F.normalize(image_keys, dim=1, p=2)

            # shape: (batch_size, feature_size)
            caption_keys = self._textual_key_encoder(caption_tokens)[:, 0]
            caption_keys = self._textual_projection(image_keys)
            caption_keys = F.normalize(caption_keys, dim=1, p=2)

        # Compute dot product similarity between image query and caption keys.
        # shape: (batch_size, 1)
        positive_logits = (image_query * caption_keys).sum(dim=1).unsqueeze(-1)

        # shape: (batch_size, queue_size)
        negative_logits = torch.mm(image_query, self._textual_queue.T)

        # shape: (batch_size, 1 + queue_size)
        logits = torch.cat((positive_logits, negative_logits), dim=1)

        # Matching key (aligned caption) is always at index 0, prepare labels.
        labels = torch.zeros(batch_size, device=logits.device).long()
        visual_moco_loss = self._loss(logits / self._temperature, labels)

        # Compute dot product similarity between caption query and image keys.
        # shape: (batch_size, 1)
        positive_logits = (caption_query * image_keys).sum(dim=1).unsqueeze(-1)

        # shape: (batch_size, queue_size)
        negative_logits = torch.mm(caption_query, self._visual_queue.T)

        # shape: (batch_size, 1	+ queue_size)
        logits = torch.cat((positive_logits, negative_logits), dim=1)

        # Matching key (aligned image) is always at index 0, prepare labels.
        labels = torch.zeros(batch_size, device=logits.device).long()
        textual_moco_loss = self._loss(logits / self._temperature, labels)

        # Add current batch (keys) to queues.
        self._update_queues(image_keys, caption_keys)
        # ====================================================================

        return {
            "loss": visual_moco_loss + textual_moco_loss,
            "loss_components": {
                "total_moco": (visual_moco_loss + textual_moco_loss).detach(),
                "visual_moco": visual_moco_loss.detach(),
                "textual_moco": textual_moco_loss.detach(),
            },
        }

    def fill_queues(self, image: torch.Tensor, caption_tokens: torch.Tensor):
        r"""
        Fill both queues until maximum size before starting training. This
        method returns `True` once queues are filled as a signal to stop.
        """
        with torch.no_grad():
            image_features = self.visual(image)
            image_features = image_features.view(-1, 2048, 49).permute(0, 2, 1)
            image_features = image_features.mean(dim=1)

            # shape: (batch_size, feature_size)
            image_keys = self._visual_projection(image_features)
            image_keys = F.normalize(image_keys, dim=1, p=2)

            # Collect features for each caption corresponding to [CLS] token.
            # shape: (batch_size, feature_size)
            caption_keys = self.textual(caption_tokens)[:, 0]
            caption_keys = self._textual_projection(caption_keys)
            caption_keys = F.normalize(caption_keys, dim=1, p=2)

        # Both queues will always be of same size, so check just one.
        if self._visual_queue.size(0) < self._max_queue_size:
            self._update_queues(image_keys, caption_keys)
            return False
        return True

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
