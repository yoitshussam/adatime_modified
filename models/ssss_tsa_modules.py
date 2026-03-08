"""
SSSS_TSA helper modules – Sensor-Specific Subspace learning with channel Selection
for Time Series domain Adaptation.
Moved out of algorithms.py for cleaner organisation.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SSSS_ScaledDotProductAttention(nn.Module):
    """Scaled dot-product attention with diagonal masking and temperature.
    Instead of full attention, extracts diagonal self-similarity scores per channel
    and applies softmax with temperature to produce a channel weighting."""

    def forward(self, query, key, value, temp, mask=None):
        dk = query.size()[-1]
        scores = query.matmul(key.transpose(-2, -1)) / math.sqrt(dk)
        # Diagonal = self-similarity score for each channel
        diag = torch.diagonal(scores, 0, 2)
        # Softmax with temperature → channel weights
        diag = F.softmax(diag / temp, dim=-1)
        attention = torch.diag_embed(diag)
        return attention.matmul(value), attention


class SSSS_MultiHeadAttention(nn.Module):
    """Multi-head attention for channel selection.
    Projects Q and K but passes V through unchanged."""

    def __init__(self, in_features, head_num, temp, bias=True, activation=None):
        super().__init__()
        if in_features % head_num != 0:
            raise ValueError('in_features({}) should be divisible by head_num({})'.format(in_features, head_num))
        self.in_features = in_features
        self.head_num = head_num
        self.activation = activation
        self.bias = bias
        self.temp = temp
        self.linear_q = nn.Linear(in_features, in_features, bias)
        self.linear_k = nn.Linear(in_features, in_features, bias)
        # Note: no linear_v — value is passed through unchanged

    def forward(self, q, k, v, mask=None):
        q, k, v = self.linear_q(q), self.linear_k(k), v  # V not projected
        if self.activation is not None:
            q = self.activation(q)
            k = self.activation(k)

        q = self._reshape_to_batches(q)
        k = self._reshape_to_batches(k)
        v = self._reshape_to_batches(v)
        if mask is not None:
            mask = mask.repeat(self.head_num, 1, 1)
        y, attention = SSSS_ScaledDotProductAttention()(q, k, v, self.temp, mask)
        y = self._reshape_from_batches(y)
        if self.activation is not None:
            y = self.activation(y)
        return y, attention

    def _reshape_to_batches(self, x):
        batch_size, seq_len, in_feature = x.size()
        sub_dim = in_feature // self.head_num
        return x.reshape(batch_size, seq_len, self.head_num, sub_dim)\
                .permute(0, 2, 1, 3)\
                .reshape(batch_size * self.head_num, seq_len, sub_dim)

    def _reshape_from_batches(self, x):
        batch_size, seq_len, in_feature = x.size()
        batch_size //= self.head_num
        out_dim = in_feature * self.head_num
        return x.reshape(batch_size, self.head_num, seq_len, in_feature)\
                .permute(0, 2, 1, 3)\
                .reshape(batch_size, seq_len, out_dim)


class SSSS_SepReps_with_multihead(nn.Module):
    """Separate per-channel backbones with multihead attention channel selection.
    Creates one backbone CNN per input channel (each with input_channels=1),
    then combines channel representations via learned attention weighting."""

    def __init__(self, configs, backbone_class):
        super().__init__()
        self.no_channels = configs.input_channels
        self.backbone_nets = nn.ModuleList([])

        # Create one backbone per channel, each with input_channels=1
        orig_input_channels = configs.input_channels
        configs.input_channels = 1
        for k in range(self.no_channels):
            self.backbone_nets.append(backbone_class(configs))
        configs.input_channels = orig_input_channels  # restore

        self.multihead_attention = SSSS_MultiHeadAttention(
            configs.final_out_channels * configs.features_len, head_num=1, bias=False, temp=configs.temp
        )

    def forward(self, x):
        """Full forward: per-channel backbones → attention → combined flat representation."""
        rep_list = []
        for k in range(self.no_channels):
            x_k = x[:, k, :].unsqueeze(1)
            rep_list.append(F.normalize(self.backbone_nets[k](x_k), dim=1))
        rep_all = torch.stack(rep_list, dim=1)
        rep_comb, _ = self.multihead_attention(rep_all, rep_all, rep_all)
        rep_comb = rep_comb.reshape(rep_comb.shape[0], -1)
        return rep_comb

    def fetch_individual_reps(self, x):
        """Get per-channel representations (also runs attention for Q/K state)."""
        rep_list = []
        for k in range(self.no_channels):
            x_k = x[:, k, :].unsqueeze(1)
            rep_list.append(F.normalize(self.backbone_nets[k](x_k), dim=1))
        # Run attention to keep Q/K projections in the compute graph
        rep_all = torch.stack(rep_list, dim=1)
        rep_comb, rep_attn = self.multihead_attention(rep_all, rep_all, rep_all)
        return rep_list

    def combine_ind_through_attn(self, rep_list):
        """Combine individual channel reps through attention.
        Detaches from backbone graph so only attention + classifier get gradients."""
        rep_all = torch.stack(rep_list, dim=1).detach()
        rep_comb, rep_attn = self.multihead_attention(rep_all, rep_all, rep_all)
        rep_comb = rep_comb.reshape(rep_comb.shape[0], -1)
        return rep_comb, rep_attn
