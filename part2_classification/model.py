"""
Binary acne classifier.

Default backbone: ResNet-50 pretrained on ImageNet (torchvision).
Optionally swap in a face-pretrained backbone (VGGFace2 via facenet-pytorch)
for the "pretrained on faces" bonus.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torchvision


def build_resnet50(num_classes: int = 2, pretrained: bool = True) -> nn.Module:
    weights = torchvision.models.ResNet50_Weights.DEFAULT if pretrained else None
    model = torchvision.models.resnet50(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def build_face_resnet(num_classes: int = 2) -> nn.Module:
    """Inception-ResNet-V1 pretrained on VGGFace2 (via facenet-pytorch).

    Useful when ACNE04 (selfies) -> DermNet (clinical skin) and we want
    the backbone to start with face/skin priors rather than imagenet objects.
    """
    try:
        from facenet_pytorch import InceptionResnetV1
    except Exception as e:
        raise RuntimeError(
            "facenet-pytorch not installed. pip install facenet-pytorch"
        ) from e

    backbone = InceptionResnetV1(pretrained="vggface2", classify=False)
    # backbone outputs a 512-d embedding
    head = nn.Linear(512, num_classes)
    return nn.Sequential(backbone, head)


def get_target_layers_for_gradcam(model: nn.Module):
    """Return the layer(s) Grad-CAM should hook into for the default ResNet-50."""
    if isinstance(model, torchvision.models.ResNet):
        return [model.layer4[-1]]
    # Sequential(backbone, head) where backbone is InceptionResnetV1
    if isinstance(model, nn.Sequential):
        bb = model[0]
        # last block before pooling
        if hasattr(bb, "block8"):
            return [bb.block8]
    raise ValueError("Don't know how to find Grad-CAM target layers for this model")
