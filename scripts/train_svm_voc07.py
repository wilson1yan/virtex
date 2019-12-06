import argparse
from collections import defaultdict
import os
import pickle
import random
import sys
from typing import Dict, List

from loguru import logger
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score
import torch
from torch.utils.data import DataLoader

from viswsl.config import Config
from viswsl.data.datasets import VOC07ClassificationDataset
from viswsl.factories import VisualStreamFactory
from viswsl.model import ViswslModel
from viswsl.modules.linguistic_stream import LinguisticStream
from viswsl.modules.visual_pooler import VisualIntermediateOutputPooler


# fmt: off
parser = argparse.ArgumentParser(
    description="""Train SVMs on intermediate features of pre-trained
    ResNet-like models for Pascal VOC2007 classification."""
)
parser.add_argument(
    "--config", required=True,
    help="""Path to a config file used to train the model whose checkpoint will
    be loaded."""
)
parser.add_argument(
    "--voc-root", required=True, help="Path to directory containing VOC07 dataset."
)
parser.add_argument(
    "--checkpoint-path", required=True,
    help="""Path to load checkpoint and run downstream task evaluation. The
    name of checkpoint file is required to be `checkpoint_*.pth`, where * is
    iteration number from which the checkpoint was serialized."""
)
parser.add_argument(
    "--serialization-dir", default=None,
    help="""Path to a directory to save results log as a Tensorboard event
    file. If not provided, this will be the parent directory of checkpoint."""
)

parser.add_argument(
    "--costs", type=float, nargs="+", default=[0.01, 0.1, 1.0, 10.0],
    help="List of costs to train SVM on.",
)

parser.add_argument_group("Compute resource management arguments.")
parser.add_argument(
    "--cpu-workers", type=int, default=2,
    help="Number of CPU workers per GPU to use for data loading.",
)
parser.add_argument(
    "--gpu-id", type=int, default=0, help="ID of GPU to use (-1 for CPU)."
)

# parser.add_argument_group("Checkpointing and Logging")
# fmt: on


if __name__ == "__main__":

    # -------------------------------------------------------------------------
    #   INPUT ARGUMENTS AND CONFIG
    # -------------------------------------------------------------------------
    _A = parser.parse_args()
    _C = Config(_A.config)

    # Set random seeds for reproucibility.
    random.seed(_C.RANDOM_SEED)
    np.random.seed(_C.RANDOM_SEED)
    torch.manual_seed(_C.RANDOM_SEED)

    device = torch.device(f"cuda:{_A.gpu_id}" if _A.gpu_id != -1 else "cpu")

    # Configure our custom logger.
    logger.remove(0)
    logger.add(
        sys.stdout, format="<g>{time}</g>: <lvl>{message}</lvl>", colorize=True
    )
    os.makedirs(_A.serialization_dir, exist_ok=True)
    # Print config and args.
    for arg in vars(_A):
        logger.info("{:<20}: {}".format(arg, getattr(_A, arg)))

    # -------------------------------------------------------------------------
    #   EXTRACT FEATURES FOR TRAINING SVMs
    # -------------------------------------------------------------------------

    train_dataset = VOC07ClassificationDataset(_A.voc_root, split="train")
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=_C.OPTIM.BATCH_SIZE,
        num_workers=_A.cpu_workers,
        pin_memory=True,
    )
    test_dataset = VOC07ClassificationDataset(_A.voc_root, split="val")
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=_C.OPTIM.BATCH_SIZE,
        num_workers=_A.cpu_workers,
        pin_memory=True,
    )
    # Initialize from a checkpoint, but only keep the visual module.
    visual_module = VisualStreamFactory.from_config(_C)
    linguistic_module = LinguisticStream.from_config(_C)
    model = ViswslModel(visual_module, linguistic_module).to(device)
    model.load_state_dict(torch.load(_A.checkpoint_path)["model"])

    # Now that we loaded weights, wrap with a feature pooler to get features
    # like we need to train SVMs.
    visual_module = model._visual
    feature_extractor = VisualIntermediateOutputPooler(mode="avg")
    del linguistic_module, model

    # keys: {"layer1", "layer2", "layer3", "layer4"}
    # Each key holds a list of numpy arrays, one per example.
    features_train: Dict[str, List[np.ndarray]] = defaultdict(list)
    features_test: Dict[str, List[np.ndarray]] = defaultdict(list)

    targets_train: List[np.ndarray] = []
    targets_test: List[np.ndarray] = []

    # VOC07 is small, extract all features together.
    with torch.no_grad():
        for batch in train_dataloader:
            targets_train.append(batch["label"].numpy())

            # keys: {"layer1", "layer2", "layer3", "layer4"}
            features = visual_module(
                batch["image"].to(device), return_intermediate_outputs=True
            )
            pooled_features = feature_extractor(features)

            for key in pooled_features:
                features_train[key].append(
                    pooled_features[key].detach().cpu().numpy()
                )

        for batch in test_dataloader:
            targets_test.append(batch["label"].numpy())

            # keys: {"layer1", "layer2", "layer3", "layer4"}
            features = visual_module(
                batch["image"].to(device), return_intermediate_outputs=True
            )
            pooled_features = feature_extractor(features)

            for key in pooled_features:
                features_test[key].append(
                    pooled_features[key].detach().cpu().numpy()
                )

    # Convert batches of features/targets to one large tensor.
    features_train = {
        k: np.concatenate(v, axis=0) for k, v in features_train.items()
    }
    features_test = {k: np.concatenate(v, axis=0) for k, v in features_test.items()}

    targets_train = np.concatenate(targets_train, axis=0)
    targets_test = np.concatenate(targets_test, axis=0)

    # -------------------------------------------------------------------------
    #   TRAIN SVMs WITH EXTRACTED FEATURES
    # -------------------------------------------------------------------------

    # Iterate over all VOC classes and train one-vs-all linear SVMs.
    for cls_idx in range(targets_train.shape[1]):
        # keys: {"layer1", "layer2", "layer3", "layer4"}
        for layer_name in features_train:
            for cost_idx in range(len(_A.costs)):
                cost = _A.costs[cost_idx]
                logger.info(f"Training SVM for class {cls_idx}, cost {cost}")

                clf = LinearSVC(
                    C=cost,
                    class_weight={1: 2, -1: 1},
                    intercept_scaling=1.0,
                    verbose=1,
                    penalty="l2",
                    loss="squared_hinge",
                    tol=0.0001,
                    dual=True,
                    max_iter=2000,
                )
                cls_labels = targets_train[:, cls_idx].astype(
                    dtype=np.int32, copy=True
                )
                # meaning of labels in VOC/COCO original loaded target files:
                # label 0 = not present, set it to -1 as svm train target
                # label 1 = present. Make the svm train target labels as -1, 1.
                cls_labels[np.where(cls_labels == 0)] = -1
                num_positives = len(np.where(cls_labels == 1)[0])
                num_negatives = len(cls_labels) - num_positives
                logger.info(
                    "cls: {} has +ve: {} -ve: {} ratio: {}".format(
                        cls_idx,
                        num_positives,
                        num_negatives,
                        float(num_positives) / num_negatives,
                    )
                )
                logger.info(
                    "features: {} cls_labels: {}".format(
                        features_train[layer_name].shape, cls_labels.shape
                    )
                )
                ap_scores = cross_val_score(
                    clf,
                    features_train[layer_name],
                    cls_labels,
                    cv=3,
                    scoring="average_precision",
                )
                clf.fit(features_train[layer_name], cls_labels)
                logger.info(
                    "cls: {} cost: {} AP: {} mean:{}".format(
                        cls_idx, cost, ap_scores, ap_scores.mean()
                    )
                )

                # -----------------------------------------------------------------
                #   SAVE MODELS TO DISK
                # -----------------------------------------------------------------
                out_file = os.path.join(
                    _A.serialization_dir, f"SVM_cls_{cls_idx}_cost_{cost}.pickle"
                )
                ap_out_file = os.path.join(
                    _A.serialization_dir, f"AP_cls_{cls_idx}_cost_{cost}.npy"
                )

                logger.info(f"Saving cls cost AP to: {ap_out_file}")
                np.save(ap_out_file, np.array([ap_scores.mean()]))

                logger.info(f"Saving SVM model to: {out_file}")
                with open(out_file, "wb") as fwrite:
                    pickle.dump(clf, fwrite)
