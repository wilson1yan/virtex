import argparse
import json
import os
from typing import Any, Dict, List

from loguru import logger
import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from virtex.config import Config
from virtex.data import ImageDirectoryDataset
from virtex.data.transforms import IMAGENET_COLOR_MEAN, IMAGENET_COLOR_STD
from virtex.factories import TokenizerFactory, PretrainingModelFactory
from virtex.utils.checkpointing import CheckpointManager
from virtex.utils.common import common_parser
from virtex.utils.metrics import CocoCaptionsEvaluator


# fmt: off
parser = common_parser(
    description="""Run image captioning inference on a pretrained model, and/or
    evaluate pretrained model on COCO Captions val2017 split."""
)
parser.add_argument(
    "--data-root", default=None,
    help="""Path to a directory containing image files to generate captions for.
    Default: COCO val2017 image directory as expected relative to project root."""
)
parser.add_argument(
    "--checkpoint-path", required=True,
    help="Path to load checkpoint and run captioning evaluation."
)
parser.add_argument(
    "--output", default=None,
    help="Path to save predictions as a JSON file."
)
parser.add_argument(
    "--calc-metrics", action="store_true",
    help="""Calculate CIDEr and SPICE metrics using ground truth COCO Captions.
    This flag should not be set when running inference on arbitrary images."""
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

    if _A.data_root is None:
        _A.data_root = os.path.join(_C.DATA.ROOT, "val2017")

    val_dataloader = DataLoader(
        ImageDirectoryDataset(_A.data_root),
        batch_size=16,
        num_workers=_A.cpu_workers,
        pin_memory=True,
        shuffle=True,
    )
    # Initialize model from a checkpoint.
    model = PretrainingModelFactory.from_config(_C).to(device)
    ITERATION = CheckpointManager(model=model).load(_A.checkpoint_path)
    model.eval()
    model.sample_on()
    
    val_batch = next(iter(val_dataloader))
    val_batch['image'] = val_batch['image'].to(device)
    with torch.no_grad():
        output_dict = model(val_batch, sample_mode='greedy')

    for caption in output_dict["predictions"][:, 1:]:
        print(tokenizer.decode(caption.tolist()))

    mean = torch.tensor(IMAGENET_COLOR_MEAN, dtype=torch.float).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_COLOR_STD, dtype=torch.float).view(1, 3, 1, 1)
    images = val_batch['image'].cpu() * std + mean

    save_image(images, 'images.png', nrow=4)



if __name__ == "__main__":
    _A = parser.parse_args()
    if _A.num_gpus_per_machine > 1:
        raise ValueError("Using multiple GPUs is not supported for this script.")

    # No distributed training here, just a single process.
    main(_A)
