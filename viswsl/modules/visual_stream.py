import torch
from torch import nn
from torchvision import models as tv_models


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
        try:
            model_creation_method = getattr(tv_models, name)
        except AttributeError as err:
            raise AttributeError(f"{name} if not a torchvision model.")

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
