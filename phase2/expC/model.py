import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50


class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels=256):
        super().__init__()
        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=True),
            nn.GroupNorm(32, out_channels), nn.ReLU()
        )
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
        pool = F.interpolate(self.global_pool(x), size, mode="bilinear", align_corners=False)
        feats = [pool] + [b(x) for b in self.branches]
        return self.proj(torch.cat(feats, dim=1))


class DecoderHead(nn.Module):
    def __init__(self, num_classes, high_in, low_in, dropout=0.3):
        super().__init__()
        self.aspp     = ASPP(high_in)
        self.low_proj = nn.Sequential(
            nn.Conv2d(low_in, 48, 1, bias=False),
            nn.BatchNorm2d(48), nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(256 + 48, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256), nn.ReLU(), nn.Dropout(dropout),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256), nn.ReLU(), nn.Dropout(dropout),
            nn.Conv2d(256, num_classes, 1)
        )

    def forward(self, low, high):
        aspp = self.aspp(high)
        aspp = F.interpolate(aspp, low.shape[2:], mode="bilinear", align_corners=False)
        x    = torch.cat([aspp, self.low_proj(low)], dim=1)
        x    = self.decoder(x)
        return F.interpolate(x, scale_factor=4, mode="bilinear", align_corners=False)


class JointDamageNet(nn.Module):
    def __init__(self, loc_classes=1, dmg_classes=5, dropout=0.3):
        super().__init__()
        backbone = resnet50(weights=None)

        self.low_level = nn.Sequential(
            backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool,
            backbone.layer1
        )  # -> (B, 256, H/4, W/4)

        layer3 = backbone.layer3
        for m in layer3.modules():
            if isinstance(m, nn.Conv2d) and m.stride == (2, 2):
                m.stride = (1, 1)
            if isinstance(m, nn.Conv2d) and m.kernel_size == (3, 3):
                m.dilation = (2, 2)
                m.padding  = (2, 2)

        self.high_level = nn.Sequential(backbone.layer2, layer3)
        # -> (B, 1024, H/16, W/16)

        # Localization head: cat(pre, post) — same as expB
        self.loc_head = DecoderHead(
            num_classes=loc_classes,
            high_in=1024 * 2, low_in=256 * 2,
            dropout=dropout,
        )

        # Damage head: cat(pre, post, post-pre) — includes explicit change signal
        self.dmg_head = DecoderHead(
            num_classes=dmg_classes,
            high_in=1024 * 3, low_in=256 * 3,
            dropout=dropout,
        )

        # Projection head for pixel-level contrastive loss on difference features
        self.contrast_proj = nn.Sequential(
            nn.Conv2d(1024, 256, 1, bias=False),
            nn.BatchNorm2d(256), nn.ReLU(),
            nn.Conv2d(256, 128, 1, bias=False),
        )

    def _encode(self, x):
        low  = self.low_level(x)
        high = self.high_level(low)
        return low, high

    def forward(self, pre, post):
        pre_low,  pre_high  = self._encode(pre)
        post_low, post_high = self._encode(post)

        # ── Localization: cat(pre, post) — same as expB ──────────────────────
        loc_low  = torch.cat([pre_low,  post_low],  dim=1)  # (B, 512,  H/4,  W/4)
        loc_high = torch.cat([pre_high, post_high], dim=1)   # (B, 2048, H/16, W/16)
        loc_out  = self.loc_head(loc_low, loc_high)

        # ── Damage: cat(pre, post, post-pre) — explicit change signal ────────
        low_diff  = post_low  - pre_low                      # (B, 256,  H/4,  W/4)
        high_diff = post_high - pre_high                     # (B, 1024, H/16, W/16)

        dmg_low  = torch.cat([pre_low,  post_low,  low_diff],  dim=1)  # (B, 768,  H/4,  W/4)
        dmg_high = torch.cat([pre_high, post_high, high_diff], dim=1)   # (B, 3072, H/16, W/16)
        dmg_out  = self.dmg_head(dmg_low, dmg_high)

        # ── Contrastive embeddings from change features ──────────────────────
        contrast_embed = self.contrast_proj(high_diff)       # (B, 128, H/16, W/16)

        return loc_out, dmg_out, contrast_embed

    def load_phase1_weights(self, ckpt_path, device):
        ckpt  = torch.load(ckpt_path, map_location=device, weights_only=False)
        state = ckpt["model_state_dict"]

        new_state = {k: v for k, v in state.items()
                     if k.startswith("low_level.") or k.startswith("high_level.")}

        missing, unexpected = self.load_state_dict(new_state, strict=False)
        print(f"Loaded Phase 1 weights | missing: {len(missing)} | unexpected: {len(unexpected)}")
