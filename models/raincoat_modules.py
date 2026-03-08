"""
RAINCOAT helper modules – Domain Adaptation for Time Series Under Feature and Label Shifts.
Moved out of algorithms.py for cleaner organisation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectralConv1d(nn.Module):
    """1D Fourier layer: FFT, linear transform, Inverse FFT (from RAINCOAT)."""
    def __init__(self, in_channels, out_channels, modes1, fl=128):
        super(SpectralConv1d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.scale = 1 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, dtype=torch.cfloat)
        )

    def compl_mul1d(self, input, weights):
        return torch.einsum("bix,iox->box", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        x = torch.cos(x)
        x_ft = torch.fft.rfft(x, norm='ortho')
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-1) // 2 + 1,
                             device=x.device, dtype=torch.cfloat)
        out_ft[:, :, :self.modes1] = self.compl_mul1d(x_ft[:, :, :self.modes1], self.weights1)
        r = out_ft[:, :, :self.modes1].abs()
        p = out_ft[:, :, :self.modes1].angle()
        return torch.concat([r, p], -1), out_ft


class Raincoat_CNN(nn.Module):
    """Temporal CNN backbone used inside RAINCOAT's tf_encoder."""
    def __init__(self, configs):
        super(Raincoat_CNN, self).__init__()
        self.conv_block1 = nn.Sequential(
            nn.Conv1d(configs.input_channels, configs.mid_channels, kernel_size=configs.kernel_size,
                      stride=configs.stride, bias=False, padding=(configs.kernel_size // 2)),
            nn.BatchNorm1d(configs.mid_channels),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
            nn.Dropout(configs.dropout)
        )
        self.conv_block3 = nn.Sequential(
            nn.Conv1d(configs.mid_channels, configs.final_out_channels, kernel_size=8, stride=1, bias=False,
                      padding=4),
            nn.BatchNorm1d(configs.final_out_channels),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
        )
        self.adaptive_pool = nn.AdaptiveAvgPool1d(configs.features_len)

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block3(x)
        x = self.adaptive_pool(x)
        x_flat = x.reshape(x.shape[0], -1)
        return x_flat


class tf_encoder(nn.Module):
    """RAINCOAT time-frequency encoder: combines spectral and temporal features."""
    def __init__(self, configs):
        super(tf_encoder, self).__init__()
        self.modes1 = configs.fourier_modes
        self.width = configs.input_channels
        self.length = configs.sequence_len
        self.freq_feature = SpectralConv1d(self.width, self.width, self.modes1, self.length)
        self.bn_freq = nn.BatchNorm1d(configs.fourier_modes * 2)
        self.cnn = Raincoat_CNN(configs)
        self.avg = nn.Conv1d(self.width, 1, kernel_size=3,
                             stride=configs.stride, bias=False, padding=(3 // 2))

    def forward(self, x):
        ef, out_ft = self.freq_feature(x)
        ef = F.relu(self.bn_freq(self.avg(ef).squeeze(1)))
        et = self.cnn(x)
        f = torch.concat([ef, et], -1)
        return F.normalize(f), out_ft


class tf_decoder(nn.Module):
    """RAINCOAT time-frequency decoder: reconstructs time series from features."""
    def __init__(self, configs):
        super(tf_decoder, self).__init__()
        self.input_channels = configs.input_channels
        self.sequence_len = configs.sequence_len
        self.bn1 = nn.BatchNorm1d(self.input_channels, self.sequence_len)
        self.bn2 = nn.BatchNorm1d(self.input_channels, self.sequence_len)
        self.convT = nn.ConvTranspose1d(
            configs.final_out_channels, self.sequence_len, self.input_channels, stride=1
        )
        self.modes = configs.fourier_modes

    def forward(self, f, out_ft):
        x_low = self.bn1(torch.fft.irfft(out_ft, n=self.sequence_len))
        et = f[:, self.modes * 2:]
        x_high = F.relu(self.bn2(self.convT(et.unsqueeze(2)).permute(0, 2, 1)))
        return x_low + x_high


class Raincoat_classifier(nn.Module):
    """RAINCOAT classifier with temperature-scaled logits."""
    def __init__(self, configs):
        super(Raincoat_classifier, self).__init__()
        self.logits = nn.Linear(configs.out_dim, configs.num_classes, bias=False)
        self.tmp = 0.1

    def forward(self, x):
        return self.logits(x) / self.tmp
