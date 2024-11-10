import torch
from torchvision import models
from torchvision.models.segmentation import DeepLabV3_ResNet50_Weights
from torchvision.models.segmentation.deeplabv3 import DeepLabHead


# Adapted DeepLabV3 with a ResNet-50 backbone to accept 12 input channels
def create_deeplabv3(in_channels=12, out_channels=1):
    model = models.segmentation.deeplabv3_resnet50(
        progress=True, weights=DeepLabV3_ResNet50_Weights.DEFAULT
    )
    # Change the first convolution to take 12 input channels
    old_conv1 = model.backbone.conv1
    model.backbone.conv1 = torch.nn.Conv2d(
        in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
    )
    # copy over weights
    with torch.no_grad():
        model.backbone.conv1.weight[:, :3] = old_conv1.weight
    # Change the classifier to 1 class
    model.classifier = DeepLabHead(2048, out_channels)
    return model