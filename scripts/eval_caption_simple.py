import os
import argparse

import torch
import torch.utils.data as data
from torchvision.utils import save_image

from virtex.config import Config
from virtex.data import ImageDirectoryDataset
from virtex.factories import TokenizerFactory, PretrainingModelFactory
from virtex.utils.checkpointing import CheckpointManager
from virtex.utils.common import common_parser


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

def main(_A: argparse.Namespace):
    batch_size = 64

    if _A.num_gpus_per_machine == 0:
        device = torch.device('cuda')
    else:
        device = torch.cuda.current_device()

    _C = Config(_A.config, _A.config_override)
    tokenizer = TokenizerFactory.from_config(_C)

    if _A.data_root is None:
        _A.data_root = os.path.join(_C.DATA.ROOT, 'val2017')

    val_dataloader = data.DataLoader(
        ImageDirectoryDataset(_A.data_root),
        batch_size=batch_size,
        num_workers=_A.cpu_workers,
        pin_memory=True,
    )

    model = PretrainingModelFactory.from_config(_C).to(device)
    ITERATION = CheckpointManager(model=model).load(_A.checkpoint_path)
    model.eval()

    batch = next(iter(val_dataloader))
    batch = {
        'image': batch['image'].to(device)
    }

    with torch.no_grad():
        predictions = model(batch)['predictions']
    for pred in predictions:
        text = tokenizer.decode(pred.cpu().tolist())
        print(text)

    mean = torch.tensor((0.485, 0.456, 0.406), dtype=torch.float32).view(1, 3, 1, 1)
    std = torch.tensor((0.229, 0.224, 0.225), dtype=torch.float32).view(1, 3, 1, 1)
    images = batch['image'].cpu()
    images = images * std + mean

    save_image(images, 'images.png', nrow=8)


if __name__ == '__main__':
    _A = parser.parse_args()
    if _A.num_gpus_per_machine > 1:
        raise ValueError("Using multiple GPUs is not supported for this script.")
    main(_A)


