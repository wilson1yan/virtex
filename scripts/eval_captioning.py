import argparse
from tqdm import tqdm
import json
import os
from typing import Any, Dict, List

from loguru import logger
import torch
from torch.utils.data import DataLoader, DistributedSampler

from virtex.config import Config
from virtex.data import ImageDirectoryDataset
from virtex.factories import TokenizerFactory, PretrainingModelFactory
from virtex.utils.checkpointing import CheckpointManager
from virtex.utils.common import common_parser
from virtex.utils.metrics import CocoCaptionsEvaluator
import virtex.utils.distributed as dist


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

    val_dataset = ImageDirectoryDataset(_A.data_root)

    val_sampler = (
        DistributedSampler(val_dataset, shuffle=False)
        if _A.num_gpus_per_machine > 0
        else None
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=_C.OPTIM.BATCH_SIZE // dist.get_world_size(),
        sampler=val_sampler,
        shuffle=False,
        num_workers=_A.cpu_workers,
        pin_memory=True,
        drop_last=False
    )
    # Initialize model from a checkpoint.
    model = PretrainingModelFactory.from_config(_C).to(device)
    ITERATION = CheckpointManager(model=model).load(_A.checkpoint_path)
    model.eval()
#    model.sample_on()

    # Make a list of predictions to evaluate.
    predictions: List[Dict[str, Any]] = []

    if dist.is_master_process():
        pbar = tqdm(total=len(val_dataloader))
    for val_iteration, val_batch in enumerate(val_dataloader, start=1):
        val_batch["image"] = val_batch["image"].to(device)
        with torch.no_grad():
            output_dict = model(val_batch)

        # Make a dictionary of predictions in COCO format.
        for image_id, caption in zip(
            val_batch["image_id"], output_dict["predictions"]
        ):
            predictions.append(
                {
                    # Convert image id to int if possible (mainly for COCO eval).
                    "image_id": int(image_id) if image_id.isdigit() else image_id,
                    "caption": tokenizer.decode(caption.tolist()),
                }
            )
        if dist.is_master_process():
            pbar.update(1)
    if dist.is_master_process():
        pbar.close()

    # Save predictions as a JSON file if specified.
    if _A.output is not None:
        os.makedirs(os.path.dirname(_A.output), exist_ok=True)
        json.dump(predictions, open(_A.output, "w"))
        logger.info(f"Saved predictions to {_A.output}")

    # Calculate CIDEr and SPICE metrics using ground truth COCO Captions. This
    # should be skipped when running inference on arbitrary images.
    if _A.calc_metrics:
        valid_image_ids = [p['image_id'] for p in predictions]
        # Assume ground truth (COCO val2017 annotations) exist.
        gt = os.path.join(_C.DATA.ROOT, "annotations", "captions_val2017.json")
        gt = json.load(open(gt, 'r'))['annotations']
        gt = [g for g in gt if g['image_id'] in valid_image_ids]

        metrics = CocoCaptionsEvaluator(gt).evaluate(predictions)
        metrics = {k: torch.tensor(v, dtype=torch.float, device=device)
                   for k, v in metrics.items()}
        dist.average_across_processes(metrics)
        metrics = {k: v.item() for k, v in metrics.items()}
        if dist.is_master_process():
            logger.info(f"Iter: {ITERATION} | Metrics: {metrics}")


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
