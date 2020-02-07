import copy
from typing import Any, Dict

import tokenizers as tkz
import torch
from torch import nn

from viswsl.data.structures import CaptioningBatch
from viswsl.modules.textual_stream import TextualStream
from viswsl.modules.visual_stream import VisualStream


class CaptioningModel(nn.Module):
    def __init__(
        self,
        visual: VisualStream,
        textual: TextualStream,
        is_bidirectional: bool = False,
    ):
        super().__init__()
        self.visual = visual
        self.textual = textual

        # Linear layer to project image features to `textual_feature_size` to
        # facilitate decoder multi-head attention etc.
        self.visual_projection = nn.Linear(
            self.visual.visual_feature_size, self.textual.textual_feature_size
        )
        self.output = nn.Linear(
            self.textual.textual_feature_size, self.textual.vocab_size
        )
        self.is_bidirectional = is_bidirectional
        self.padding_idx = self.textual.padding_idx

        # Clone the textual module for backward direction if doing captioning
        # in both directions (separately).
        if self.is_bidirectional:
            self.backward_textual = copy.deepcopy(self.textual)
            # Tie word and position embeddings for both directions.
            self.backward_textual.embedding = self.textual.embedding

        self.loss = nn.CrossEntropyLoss(ignore_index=self.padding_idx)

        # Tie input and output word embeddings to reduce parameters.
        # However, output embedding layer will learn its own bias.
        self.output.weight = self.textual.embedding.words.weight

    def forward(self, batch: CaptioningBatch):

        # shape: (batch_size, visual_feature_size, ...)
        visual_features = self.visual(batch["image"])

        # shape: (batch_size, ..., visual_feature_size)
        visual_features = visual_features.view(
            batch["image"].size(0), self.visual.visual_feature_size, -1
        ).permute(0, 2, 1)

        # Now visual and textual features are of same size.
        # shape: (batch_size, ..., textual_feature_size)
        projected_visual_features = self.visual_projection(visual_features)

        caption_tokens = batch["caption_tokens"]
        caption_lengths = batch["caption_lengths"]

        # shape: (batch_size, max_caption_length, textual_feature_size)
        textual_features = self.textual(
            caption_tokens, caption_lengths, projected_visual_features
        )
        # shape: (batch_size, max_caption_length, vocab_size)
        output_logits = self.output(textual_features)

        loss = self.loss(
            output_logits[:, :-1].contiguous().view(-1, self.textual.vocab_size),
            caption_tokens[:, 1:].contiguous().view(-1),
        )
        output_dict: Dict[str, Any] = {
            "loss": loss,
            # Single scalar per batch for logging in training script.
            "loss_components": {"captioning_forward": loss.clone().detach()},
        }
        # Do captioning in backward direction.
        if self.is_bidirectional:
            backward_caption_tokens = batch["noitpac_tokens"]

            backward_textual_features = self.backward_textual(
                backward_caption_tokens, caption_lengths, projected_visual_features
            )
            backward_output_logits = self.output(backward_textual_features)

            backward_loss = self.loss(
                backward_output_logits[:, :-1]
                .contiguous()
                .view(-1, self.textual.vocab_size),
                backward_caption_tokens[:, 1:].contiguous().view(-1),
            )
            output_dict["loss"] += backward_loss

            # Single scalar per batch for logging in training script.
            output_dict["loss_components"].update(
                captioning_backward=backward_loss.clone().detach()
            )

        # During evaluation, get predictions from logits. Useful for logging.
        # Predictions from forward transformer will be shifted right by one
        # time-step, and vice-versa.
        if not self.training:
            predictions = torch.argmax(output_logits, dim=-1)[:, :-1]
            redundant_positions = caption_tokens[:, 1:] == self.padding_idx
            predictions[redundant_positions] = self.padding_idx
            output_dict["predictions"] = {"forward": predictions}

            if self.is_bidirectional:
                backward_predictions = backward_predictions = torch.argmax(
                    backward_output_logits, dim=-1
                )[:, :-1]
                backward_predictions[redundant_positions] = self.padding_idx
                output_dict["predictions"]["backward"] = backward_predictions

        return output_dict

    def log_predictions(
        self, batch: CaptioningBatch, tokenizer: tkz.implementations.BaseTokenizer
    ) -> str:

        self.eval()
        with torch.no_grad():
            predictions = self.forward(batch)["predictions"]
        self.train()

        predictions_str = ""
        for tokens, preds in zip(batch["caption_tokens"], predictions["forward"]):
            predictions_str += f"""
                Caption tokens : {tokenizer.decode(tokens.tolist())}
                Predictions (f): {tokenizer.decode(preds.tolist())}

                """

        if self.is_bidirectional:
            for tokens, preds in zip(
                batch["noitpac_tokens"], predictions["backward"]
            ):
                predictions_str += f"""
                Noitpac tokens : {tokenizer.decode(tokens.tolist())}
                Predictions (b): {tokenizer.decode(preds.tolist())}

                    """
        return predictions_str
