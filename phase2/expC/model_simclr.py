import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights


class SimCLREncoder(nn.Module):
    """
    ResNet50 backbone (ImageNet init) + MLP projection head for SimCLR.
    Backbone components are kept as named attributes so weights can be
    transferred into JointDamageNet via load_simclr_weights().
    """
    def __init__(self, proj_dim=128, proj_hidden=2048):
        super().__init__()
        backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)

        # Keep as named attrs — state dict keys will be "conv1.*", "layer1.*", etc.
        self.conv1   = backbone.conv1
        self.bn1     = backbone.bn1
        self.relu    = backbone.relu
        self.maxpool = backbone.maxpool
        self.layer1  = backbone.layer1
        self.layer2  = backbone.layer2
        self.layer3  = backbone.layer3
        self.layer4  = backbone.layer4
        self.avgpool = backbone.avgpool  # AdaptiveAvgPool2d(1) → (B, 2048, 1, 1)

        # Projection head: 2048 → proj_hidden → proj_dim
        # BN after each linear, no affine on the final BN (SimCLR v2 convention)
        self.projector = nn.Sequential(
            nn.Linear(2048, proj_hidden, bias=False),
            nn.BatchNorm1d(proj_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(proj_hidden, proj_dim, bias=False),
            nn.BatchNorm1d(proj_dim, affine=False),
        )

    def encode(self, x):
        """Backbone only — used at transfer time, not during pre-training."""
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        return torch.flatten(x, 1)  # (B, 2048)

    def forward(self, x):
        h = self.encode(x)
        z = self.projector(h)
        return F.normalize(z, dim=1)  # L2-normalised


def nt_xent_loss(z1, z2, temperature=0.07):
    """
    NT-Xent (Normalised Temperature-scaled Cross-Entropy) loss.
    z1, z2: (N, D) L2-normalised embeddings of N positive pairs.
    Each z1[i] has z2[i] as its positive; all other 2N-2 embeddings are negatives.
    """
    N = z1.size(0)
    z = torch.cat([z1, z2], dim=0)          # (2N, D)
    sim = torch.mm(z, z.T) / temperature     # (2N, 2N) cosine similarities

    # Mask out self-similarities (diagonal)
    mask = torch.eye(2 * N, dtype=torch.bool, device=z.device)
    sim  = sim.masked_fill(mask, float('-inf'))

    # For z1[i] the positive is z2[i] = z[N+i]; for z2[i] the positive is z1[i] = z[i]
    labels = torch.cat([
        torch.arange(N, 2 * N, device=z.device),
        torch.arange(0,     N, device=z.device),
    ])

    return F.cross_entropy(sim, labels)
