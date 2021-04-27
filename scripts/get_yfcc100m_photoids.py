import argparse
from tqdm import tqdm
import json
import os
from typing import Any, Dict, List

from loguru import logger
import torch
from torch.utils.data import DataLoader, DistributedSampler

from virtex.config import Config
from virtex.factories import TokenizerFactory, PretrainingModelFactory, PretrainingDatasetFactory
from virtex.utils.checkpointing import CheckpointManager
from virtex.utils.common import common_parser


# fmt: off
parser = common_parser(
    description="""Run image captioning inference on a pretrained model, and/or
    evaluate pretrained model on COCO Captions val2017 split."""
)
parser.add_argument(
    "--output", default=None, required=True,
    help="Path to save predictions as a JSON file."
)
# fmt: on

def main(_A: argparse.Namespace):

    if _A.num_gpus_per_machine == 0:
        # Set device as CPU if num_gpus_per_machine = 0.
        device = torch.device("cpu")
    else:
        # Get the current device (this will be zero here by default).
        device = torch.cuda.current_device()

    _C = Config(_A.config, _A.config_override)

    tokenizer = TokenizerFactory.from_config(_C)

    dataset = PretrainingDatasetFactory.from_config(_C, split='train')

    dataloader = DataLoader(
        dataset,
        batch_size=_C.OPTIM.BATCH_SIZE,
        shuffle=False,
        num_workers=_A.cpu_workers,
        drop_last=False,
        collate_fn=dataset.collate_fn
    )
    print('dataloader size:', len(dataloader))

    photoids = dict()

    pbar = tqdm(total=len(dataloader))
    for batch in dataloader:
        for image_id, photo_id in zip(
            batch["image_id"], batch["photo_id"]
        ):
            photoids[image_id.item()] = photo_id.item()
        pbar.update(1)
    pbar.close()

    # Save predictions as a JSON file if specified.
    os.makedirs(os.path.dirname(_A.output), exist_ok=True)
    json.dump(photoids, open(_A.output, "w"))
    logger.info(f"Saved photoids to {_A.output}")


if __name__ == "__main__":
    _A = parser.parse_args()

    main(_A)
