import torch
from torch import nn
from torch.nn import functional as F

class SelectiveConv(nn.Module):
    """
    Implementation of the Selective Convolution Layer used in Selective Kernel Networks
    https://arxiv.org/pdf/1903.06586.pdf
    """

    def __init__(self, in_channels: int, out_channels: int, groups: int, ratio: int = 16, paths: int = 2):
        super(SelectiveConv, self).__init__()

        self.reduced_dim = max(out_channels // ratio, 32)
        self.paths = paths
        self.convs = []

        for i in range(paths):
            self.convs.append(nn.Sequential(*[
                nn.Conv2d(in_channels, out_channels, groups=groups, kernel_size=3, dilation=1 + i, padding=1 + i),
                nn.BatchNorm2d(out_channels),
                nn.ReLU()
            ]))

        self.convs = nn.ModuleList(self.convs)

        self.flatten = nn.Sequential(*[
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            Flatten(),
        ])

        self.compress = nn.Sequential(*[
            nn.Linear(out_channels, self.reduced_dim),
            nn.BatchNorm1d(self.reduced_dim),
            nn.ReLU(),
        ])

        # using a 1x1 conv for Z reprojection
        self.projections = nn.ModuleList([nn.Conv2d(self.reduced_dim, out_channels, 1) for _ in range(paths)])

    def forward(self, x: torch.Tensor):
        b, _, h, w = x.shape

        # pass through all convolutions
        features = torch.cat([conv(x) for conv in self.convs], dim=1)

        # sum features across path dimension
        features = features.view(features.shape[0], self.paths, -1, h, w)
        fused_x = features.sum(dim=1)

        #print(fused_x.shape)
        _, c, _, _ = fused_x.shape

        # flatten and compress fused features
        fused_flat = self.flatten(fused_x)
        fused_compress = self.compress(fused_flat)

        # reproject the compression in order to compute channel-wise attention scores
        attention_scores = torch.cat([projection(fused_compress[:, :, None, None]) for projection in self.projections], dim=1)

        # compute path-wise softmax for each channel
        attention_scores = attention_scores.view(b, self.paths, c, 1, 1)
        attention_scores = F.softmax(attention_scores, dim=1)

        # scale features with attention and fuse them back
        features = features * attention_scores
        features = features.sum(dim=1)
        return features


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x: torch.Tensor):
        return x.view(x.shape[0], -1)
