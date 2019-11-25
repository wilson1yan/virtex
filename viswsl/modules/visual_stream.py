import os
import pkg_resources
import re
from typing import Dict

import torch
from torch import nn


class TorchvisionVisualStream(nn.Module):
    def __init__(
        self,
        name: str,
        pretrained: bool = False,
        norm_layer: str = "groupnorm",
        num_groups: int = 32,
        **kwargs,
    ):
        super().__init__()

        # Protect import because it should be an optional dependency, one may
        # only use `D2BackboneVisualStream`.
        from torchvision import models as tv_models
        try:
            model_creation_method = getattr(tv_models, name)
        except AttributeError as err:
            raise RuntimeError(f"{name} if not a torchvision model.")

        # Initialize with BatchNorm, convert all Batchnorm to GroupNorm if
        # needed, later.
        self._cnn = model_creation_method(
            pretrained, zero_init_residual=True, **kwargs
        )
        # Do nothing after the final res stage.
        self._cnn.avgpool = nn.Identity()
        self._cnn.fc = nn.Identity()

        if norm_layer == "groupnorm":
            self._cnn = self._batchnorm_to_groupnorm(self._cnn, num_groups)

    def forward(self, image: torch.Tensor):
        # Get a flat feature vector, view it as spatial features.
        # TODO (kd): Hardcoded values now, deal with them later.
        flat_spatial_features = self._cnn(image)

        # shape: (batch_size, 7, 7, 2048)
        spatial_features = flat_spatial_features.view(-1, 49, 2048)
        return spatial_features

    def _batchnorm_to_groupnorm(self, module, num_groups: int):
        mod = module
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            mod = nn.GroupNorm(
                num_groups, module.num_features, affine=module.affine
            )
        for name, child in module.named_children():
            mod.add_module(
                name, self._batchnorm_to_groupnorm(child, num_groups)
            )
        return mod


class D2BackboneVisualStream(nn.Module):

    # TODO: (add more later). Think about FPNs.
    MODEL_NAME_TO_CFG_PATH: Dict[str, str] = {
        "coco_faster_rcnn_R_50_C4": "COCO-Detection/faster_rcnn_R_50_C4_3x.yaml",
        "coco_faster_rcnn_R_50_DC5": "COCO-Detection/faster_rcnn_R_50_DC5_3x.yaml",
        "coco_faster_rcnn_R_101_C4": "COCO-Detection/faster_rcnn_R_101_C4_3x.yaml",
        "coco_faster_rcnn_R_101_DC5": "COCO-Detection/faster_rcnn_R_101_DC5_3x.yaml",

        "coco_mask_rcnn_R_50_C4": "COCO-InstanceSegmentation/mask_rcnn_R_50_C4_3x.yaml",
        "coco_mask_rcnn_R_50_DC5": "COCO-InstanceSegmentation/mask_rcnn_R_50_DC5_3x.yaml",
        "coco_mask_rcnn_R_101_C4": "COCO-InstanceSegmentation/mask_rcnn_R_101_C4_3x.yaml",
        "coco_mask_rcnn_R_101_DC5": "COCO-InstanceSegmentation/mask_rcnn_R_101_DC5_3x.yaml",
    }

    def __init__(
        self, name: str, pretrained: bool = False, **kwargs,
    ):
        super().__init__()

        # Protect import because it should be an optional dependency, one may
        # only use `TorchvisionVisualStream`.
        from detectron2 import model_zoo as d2mz

        # If not pretrained, set MODEL.WEIGHTS key in config as empty string to
        # avoid downloading backbone weights.
        try:
            if not pretrained:
                d2config_path = pkg_resources.resource_filename(
                    "detectron2.model_zoo",
                    os.path.join("configs", self.MODEL_NAME_TO_CFG_PATH[name])
                )
                with open(d2config_path, "r") as d2config:
                    contents = d2config.read()
                    weights_cfg = re.search("WEIGHTS: \"(.*)\"\n", contents)[1]
                    contents = contents.replace(weights_cfg, "")

                with open(d2config_path, "w") as d2config:
                    d2config.write(contents)
        except FileNotFoundError as err:
            raise RuntimeError(f"{name} is no available from D2 model zoo!")

        try:
            # trained=False is not the same as our ``pretrained`` argument.
            d2_model = d2mz.get(
                self.MODEL_NAME_TO_CFG_PATH[name], trained=False
            )
            self._cnn = d2_model.backbone
        except Exception as e:
            self._cnn = None
            exception = e

        # Revert back the changing of config file in case of any exception.
        if not pretrained:
            with open(d2config_path, "r") as d2config:
                contents = d2config.read().replace(
                    "WEIGHTS: \"\"", f"WEIGHTS: \"{weights_cfg}\""
                )

            with open(d2config_path, "w") as d2config:
                d2config.write(contents)

        if self._cnn is None:
            raise(exception)

    def forward(self, image: torch.Tensor):
        spatial_features = self._cnn(image)
        return spatial_features


class BlindVisualStream(nn.Module):
    r"""A visual stream which cannot see the image."""

    def __init__(self, bias: torch.Tensor = torch.ones(49, 2048)):
        super().__init__()

        # We never update the bias because a blind model cannot learn anything
        # about the image.
        self._bias = nn.Parameter(bias, requires_grad=False)

    def forward(self, image: torch.Tensor):
        batch_size = image.size(0)
        return self._bias.unsqueeze(0).repeat(batch_size, 1, 1)
