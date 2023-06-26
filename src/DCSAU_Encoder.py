import torch
from src.DCSAU_Resnet import ResNet, Bottleneck

# ResNet = DCSAU_Resnet.ResNet
# Bottleneck = DCSAU_Resnet.Bottleneck


def CSA(pretrained=False, root='~/.encoding/models', **kwargs):
    model = ResNet(Bottleneck, [2, 2, 2, 2],
                   radix=2, groups=1, bottleneck_width=64,
                   deep_stem=True, stem_width=32, avg_down=True,
                   avd=True, avd_first=False, **kwargs)

    return model