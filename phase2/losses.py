import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, weight=None, ignore_index=0, label_smoothing=0.0):
        super().__init__()
        self.gamma           = gamma
        self.weight          = weight
        self.ignore_index    = ignore_index
        self.label_smoothing = label_smoothing

    def forward(self, logits, targets):
        ce   = F.cross_entropy(logits, targets, weight=self.weight,
                               ignore_index=self.ignore_index, reduction="none",
                               label_smoothing=self.label_smoothing)
        pt   = torch.exp(-ce)
        loss = ((1 - pt) ** self.gamma) * ce
        # ignore_index positions have ce=0, safe to mean over all
        return loss.mean()


class BinaryFocalLoss(nn.Module):
    def __init__(self, gamma=2.0):
        super().__init__()
        self.gamma = gamma

    def forward(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        pt  = torch.exp(-bce)
        return (((1 - pt) ** self.gamma) * bce).mean()


class EMDLoss(nn.Module):
    """Earth Mover's Distance loss for ordinal segmentation.

    Penalises predictions proportional to how far they are from the true class
    on the ordinal scale, so predicting class 1 when truth is 2 is a smaller
    error than predicting class 4.
    """
    def __init__(self, num_classes: int = 5, ignore_index: int = 0):
        super().__init__()
        self.num_classes  = num_classes
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # logits: (B, C, H, W)  targets: (B, H, W)
        C    = self.num_classes
        mask = targets != self.ignore_index

        probs       = F.softmax(logits, dim=1)
        t           = targets.clamp(0, C - 1)
        target_dist = F.one_hot(t, C).permute(0, 3, 1, 2).float()

        pred_cdf   = probs.cumsum(dim=1)
        target_cdf = target_dist.cumsum(dim=1)

        emd = ((pred_cdf - target_cdf) ** 2).mean(dim=1)  # (B, H, W)

        if mask.sum() == 0:
            return emd.sum() * 0.0
        return emd[mask].mean()
