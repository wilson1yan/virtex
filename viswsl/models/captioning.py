from typing import Any, Dict

import torch
from torch import nn

from viswsl.modules.fusion import Fusion


class CaptioningModel(nn.Module):
    def __init__(self, visual, textual, fusion: Fusion):
        super().__init__()
        self.visual = visual
        self.textual = textual
        self.fusion = fusion

        # Tie input and output word embeddings to reduce parameters.
        # Output embedding layer will also learn a bias.
        if textual.textual_feature_size == fusion.fused_feature_size:
            self.output: nn.Module = nn.Linear(
                fusion.fused_feature_size, textual.vocab_size
            )
            self.output.weight = self.textual.embedding.word_embedding.weight
        else:
            # Add an intermediate projection layer to `textual_feature_size`
            # if fused features have different size than textual features.
            self.output = nn.Sequential(
                nn.Linear(
                    fusion.fused_feature_size,
                    textual.textual_feature_size,
                    bias=False,
                ),
                nn.Linear(textual.textual_feature_size, textual.vocab_size),
            )
            self.output[0].weight.data.normal_(mean=0.0, std=0.02)
            self.output[-1].weight = self.textual.embedding.word_embedding.weight

        self.loss = nn.CrossEntropyLoss(ignore_index=textual.padding_idx)

    def forward(self, image: torch.Tensor, caption_tokens: torch.Tensor):
        batch_size = image.size(0)

        # shape: (batch_size, visual_feature_size, ...)
        visual_features = self.visual(image)

        # shape: (batch_size, ..., visual_feature_size)
        visual_features = visual_features.view(
            batch_size, self.visual.visual_feature_size, -1
        ).permute(0, 2, 1)

        # Trim some token positions from the end if all captions are smaller
        # than max length.
        caption_lengths = (caption_tokens != self.textual.padding_idx).sum(dim=1)
        max_caption_length = caption_lengths.max()
        caption_tokens = caption_tokens[:, :max_caption_length]

        # shape: (batch_size, max_caption_length, textual_feature_size)
        textual_features = self.textual(caption_tokens)

        # shape: (batch_size, max_caption_length, fused_feature_size)
        fused_features = self.fusion(visual_features, textual_features)

        # shape: (batch_size, max_caption_length, vocab_size)
        output_logits = self.output(fused_features)

        # Get predictions from logits, these will be shifted right by one
        # time-step (using forward transformer encoder).
        predictions = torch.argmax(output_logits, dim=-1)

        output_dict: Dict[str, Any] = {
            "predictions": predictions,
            "loss": self.loss(
                output_logits[:, :-1].contiguous().view(-1, self.textual.vocab_size),
                caption_tokens[:, 1:].contiguous().view(-1),
            ),
        }
        # Single scalar per batch for logging in training script.
        output_dict["loss_components"] = {"captioning": output_dict["loss"].detach()}
        return output_dict
