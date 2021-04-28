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
import virtex.utils.distributed as dist


# fmt: off
parser = common_parser(
    description="""Run image captioning inference on a pretrained model, and/or
    evaluate pretrained model on COCO Captions val2017 split."""
)
parser.add_argument(
    "--checkpoint-path", required=True,
    help="Path to load checkpoint and run captioning evaluation."
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

    sampler = (
        DistributedSampler(dataset, shuffle=False)
        if _A.num_gpus_per_machine > 0
        else None
    )
    val_dataloader = DataLoader(
        dataset,
        batch_size=_C.OPTIM.BATCH_SIZE // dist.get_world_size(),
        sampler=sampler,
        shuffle=False,
        num_workers=_A.cpu_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=dataset.collate_fn
    )

    # Initialize model from a checkpoint.
    model = PretrainingModelFactory.from_config(_C).to(device)
    ITERATION = CheckpointManager(model=model).load(_A.checkpoint_path)
    model.eval()
    torch.set_grad_enabled(False)
    model.sample_on()

    captions = dict()
    
    if dist.is_master_process():
        pbar = tqdm(total=len(val_dataloader))
    for val_iteration, val_batch in enumerate(val_dataloader, start=1):
        val_batch = {'image_id': val_batch['image_id'].to(device),
                     'image': val_batch['image'].to(device)}
        predictions = []
        for k in [1]:
            predictions.append(model(val_batch, sample_mode='beam', n_samples_per_image=k)['predictions'][:, 1:])
        max_length = max([p.shape[1] for p in predictions])
        predictions = [torch.cat((p, torch.zeros(p.shape[0], max_length - p.shape[1], device=device)), dim=1) for p in predictions]
        predictions = torch.stack(predictions, dim=1)

        # Make a dictionary of predictions in COCO format.
        for image_id, caption in zip(
            val_batch["image_id"], predictions
        ):
            captions[image_id.item()] = [tokenizer.decode(c.tolist()).strip() for c in caption]
        if dist.is_master_process():
            pbar.update(1)
    if dist.is_master_process():
        pbar.close()

    # Save predictions as a JSON file if specified.
    folder = os.path.dirname(_A.output)
    os.makedirs(folder, exist_ok=True)

    filename = os.path.basename(_A.output)
    filepath = os.path.join(folder, f'{dist.get_rank()}_{filename}')
    
    json.dump(captions, open(filepath, "w"))
    logger.info(f"Saved predictions to {filepath}")


if __name__ == "__main__":
    _A = parser.parse_args()

    if _A.num_gpus_per_machine == 0:
        main(_A)
    else:
        dist.launch(
            main,
            num_machines=_A.num_machines,
            num_gpus_per_machine=_A.num_gpus_per_machine,
            machine_rank=_A.machine_rank,
            dist_url=_A.dist_url,
            args=(_A,),
        )
