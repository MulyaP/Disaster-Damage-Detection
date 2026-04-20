import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50


class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels=256):
        super().__init__()
        # Global average pool branch (use_bias=True on 1x1 conv, matching Keras)
        # GroupNorm instead of BatchNorm to handle 1x1 spatial size during training
        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=True),
            nn.GroupNorm(32, out_channels), nn.ReLU()
        )
        # 1x1 and dilated 3x3 branches
        self.branches = nn.ModuleList([
            nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=False),
                          nn.BatchNorm2d(out_channels), nn.ReLU()),
            *[nn.Sequential(nn.Conv2d(in_channels, out_channels, 3,
                                      padding=r, dilation=r, bias=False),
                            nn.BatchNorm2d(out_channels), nn.ReLU())
              for r in (6, 12, 18)],
        ])
        self.proj = nn.Sequential(
            nn.Conv2d(out_channels * 5, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(), nn.Dropout(0.5)
        )

    def forward(self, x):
        size = x.shape[2:]
        # Concat order: [pool, 1x1, rate6, rate12, rate18] — matches Keras
        pool = F.interpolate(self.global_pool(x), size, mode="bilinear",
                             align_corners=False)
        feats = [pool] + [b(x) for b in self.branches]
        return self.proj(torch.cat(feats, dim=1))


class DeepLabV3Plus(nn.Module):
    def __init__(self, num_classes=1):
        super().__init__()
        backbone = resnet50(weights="IMAGENET1K_V1")

        # conv2_block3_2_relu equivalent: output of layer1 (stride 4, 256 ch)
        self.low_level = nn.Sequential(backbone.conv1, backbone.bn1,
                                       backbone.relu, backbone.maxpool,
                                       backbone.layer1)

        # conv4_block6_2_relu equivalent: output of layer3 (stride 16, 1024 ch)
        # Make layer3 dilated so spatial resolution stays at stride 16
        layer3 = backbone.layer3
        for m in layer3.modules():
            if isinstance(m, nn.Conv2d) and m.stride == (2, 2):
                m.stride = (1, 1)
            if isinstance(m, nn.Conv2d) and m.kernel_size == (3, 3):
                m.dilation = (2, 2)
                m.padding = (2, 2)
        self.high_level = nn.Sequential(backbone.layer2, layer3)

        self.aspp = ASPP(1024)
        self.low_proj = nn.Sequential(
            nn.Conv2d(256, 48, 1, bias=False),
            nn.BatchNorm2d(48), nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(256 + 48, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256), nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256), nn.ReLU(),
            nn.Conv2d(256, num_classes, 1)
        )

    def forward(self, x):
        low = self.low_level(x)           # stride 4
        high = self.high_level(low)       # stride 16 (dilated)
        aspp = self.aspp(high)            # stride 16
        aspp = F.interpolate(aspp, low.shape[2:], mode="bilinear",
                             align_corners=False)  # upsample to stride 4
        x = torch.cat([aspp, self.low_proj(low)], dim=1)
        x = self.decoder(x)
        return F.interpolate(x, scale_factor=4, mode="bilinear",
                             align_corners=False)  # upsample to full resolution
