"""
ACON helper modules – Adversarial Consistency for Cross-Domain Time Series Adaptation.
Moved out of algorithms.py for cleaner organisation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FrequencyEncoder(nn.Module):
    """Learnable frequency-domain encoder via complex multiplication (from ACON paper)."""

    def __init__(self, in_channels, out_channels, mode, normalize=False):
        super(FrequencyEncoder, self).__init__()
        self.normalize = normalize
        self.mode = mode
        self.out_channels = out_channels
        self.scale = 1 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, mode, dtype=torch.cfloat)
        )

    def compl_mul1d(self, input, weights):
        dim_num = input.dim()
        if dim_num == 3:
            return torch.einsum("bix,iox->box", input, weights)
        elif dim_num == 4:
            return torch.einsum("bixy,ioy->boxy", input, weights)

    def forward(self, x):
        batchsize = x.size(0)
        x_ft = torch.fft.rfft(x, norm='ortho', dim=-1)
        if self.normalize:
            x_ft = F.normalize(x_ft, dim=-1)
        dim_num = x_ft.dim()
        if dim_num == 3:
            out_ft = torch.zeros(batchsize, self.out_channels, self.mode,
                                 device=x.device, dtype=torch.cfloat)
            out_ft[:, :, :] = self.compl_mul1d(x_ft[:, :, :self.mode], self.weights1)
        elif dim_num == 4:
            out_ft = torch.zeros(batchsize, self.out_channels, x_ft.size(2), self.mode,
                                 device=x.device, dtype=torch.cfloat)
            out_ft[:, :, :, :] = self.compl_mul1d(x_ft[:, :, :, :self.mode], self.weights1)
        return out_ft


class FrequencyClassifierHead(nn.Module):
    """Two-layer classifier for frequency-domain features (from ACON paper)."""

    def __init__(self, in_dim, num_classes):
        super(FrequencyClassifierHead, self).__init__()
        self.linear1 = nn.Linear(in_dim, in_dim)
        self.linear2 = nn.Linear(in_dim, num_classes)

    def forward(self, x, get_feat=False):
        x = self.linear1(x)
        predictions = self.linear2(x)
        if get_feat:
            return predictions, x
        else:
            return predictions


class TemporalClassifierHead(nn.Module):
    """Simple linear classifier head for temporal features (from ACON paper)."""

    def __init__(self, in_dim, num_classes):
        super(TemporalClassifierHead, self).__init__()
        self.head = nn.Linear(in_dim, num_classes)

    def forward(self, x):
        return self.head(x)


class ACON_Discriminator(nn.Module):
    """Discriminator for ACON (from ACON paper)."""

    def __init__(self, in_dim, disc_hid_dim):
        super(ACON_Discriminator, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(in_dim, disc_hid_dim),
            nn.ReLU(),
            nn.Linear(disc_hid_dim, disc_hid_dim),
            nn.ReLU(),
            nn.Linear(disc_hid_dim, 2)
        )

    def forward(self, input):
        return self.layer(input)
