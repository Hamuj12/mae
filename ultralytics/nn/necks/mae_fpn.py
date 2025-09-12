import torch.nn as nn
import torch.nn.functional as F


class MaeSimpleFPN(nn.Module):
    """Simple FPN producing P3, P4, P5 from ViT stride-16 features."""

    def __init__(self, in_channels: int = 768, out_channels: int = 256) -> None:
        super().__init__()
        self.conv_p4 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.conv_p3 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.conv_p5 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1)

    def forward(self, feats: dict):
        p4 = feats["p4"]
        p4 = self.conv_p4(p4)

        p3 = F.interpolate(p4, scale_factor=2, mode="nearest")
        p3 = self.conv_p3(p3)

        p5 = self.conv_p5(p4)
        return [p3, p4, p5]
