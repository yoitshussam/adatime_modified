"""
CLUDA helper modules – Contrastive Learning for Unsupervised Domain Adaptation.
Original paper: https://arxiv.org/abs/2301.00149
Moved out of algorithms.py for cleaner organisation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.models import ReverseLayerF


# ---- helper: cosine similarity matrix ----
def cluda_sim_matrix(a, b, eps=1e-8):
    a_n = a.norm(dim=1, keepdim=True)
    b_n = b.norm(dim=1, keepdim=True)
    a_norm = a / torch.clamp(a_n, min=eps)
    b_norm = b / torch.clamp(b_n, min=eps)
    return torch.mm(a_norm, b_norm.transpose(0, 1))


# ---- helper: nearest-neighbour lookup ----
def cluda_NN(key, queue, num_neighbors=1, return_indices=False):
    similarity = cluda_sim_matrix(key, queue)
    indices_top = torch.topk(similarity, k=num_neighbors, dim=1)[1]
    neighbours = [queue[indices_top[:, i], :] for i in range(num_neighbors)]
    if return_indices:
        return torch.stack(neighbours), indices_top
    return torch.stack(neighbours)


# ---- helper: simple MLP (projector / discriminator / predictor) ----
class CLUDA_MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, use_batch_norm=True):
        super().__init__()
        self.input_fc = nn.Linear(input_dim, hidden_dim)
        self.output_fc = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()
        self.batch_norm = nn.BatchNorm1d(hidden_dim) if use_batch_norm else None
        self.output_dim = output_dim

    def forward(self, x):
        h = self.input_fc(x)
        if self.batch_norm is not None:
            h = self.batch_norm(h)
        h = F.relu(h)
        y = self.output_fc(h)
        if self.output_dim == 1:
            y = self.sigmoid(y)
        return y


# ---- helper: Augmenter (time-series augmentations for contrastive views) ----
class CLUDA_Augmenter:
    """Applies history cutout, history crop, Gaussian noise, spatial dropout.
    Operates on tensors of shape (N, L, C) — time-steps in dim-1."""

    def __init__(self, cutout_length=4, cutout_prob=0.5,
                 crop_min_history=0.5, crop_prob=0.5,
                 gaussian_std=0.1, dropout_prob=0.1):
        self.cutout_length = cutout_length
        self.cutout_prob = cutout_prob
        self.crop_min_history = crop_min_history
        self.crop_prob = crop_prob
        self.gaussian_std = gaussian_std
        self.dropout_prob = dropout_prob
        self.augmentations = [
            self.history_cutout, self.history_crop,
            self.gaussian_noise, self.spatial_dropout,
        ]

    def __call__(self, sequence, sequence_mask):
        for f in self.augmentations:
            sequence, sequence_mask = f(sequence, sequence_mask)
        return sequence, sequence_mask

    # -- cutout --
    def history_cutout(self, seq, mask):
        N, L, C = seq.shape
        if L <= self.cutout_length:
            return seq, mask
        start = torch.randint(0, L - self.cutout_length, (N, 1)).expand(-1, L)
        end = start + self.cutout_length
        idx = torch.arange(L, device=seq.device).unsqueeze(0).expand(N, -1)
        keep = ((idx < start.to(seq.device)) | (idx >= end.to(seq.device))).unsqueeze(-1).expand(-1, -1, C).float()
        sel = (torch.rand(N, device=seq.device) < self.cutout_prob).float().view(-1, 1, 1)
        seq = seq * (1 - sel) + seq * sel * keep
        mask = mask * (1 - sel) + mask * sel * keep
        return seq, mask

    # -- crop --
    def history_crop(self, seq, mask):
        N, L, C = seq.shape
        crop_start = (torch.rand(N, 1, device=seq.device) * self.crop_min_history * L).long().expand(-1, L)
        idx = torch.arange(L, device=seq.device).unsqueeze(0).expand(N, -1)
        keep = (idx >= crop_start).unsqueeze(-1).expand(-1, -1, C).float()
        sel = (torch.rand(N, device=seq.device) < self.crop_prob).float().view(-1, 1, 1)
        seq = seq * (1 - sel) + seq * sel * keep
        mask = mask * (1 - sel) + mask * sel * keep
        return seq, mask

    # -- Gaussian noise --
    def gaussian_noise(self, seq, mask):
        pad_mask = (mask != 0).float()
        noise = torch.empty_like(seq).normal_(0, self.gaussian_std)
        return seq + pad_mask * noise, mask

    # -- spatial (channel) dropout --
    def spatial_dropout(self, seq, mask):
        N, L, C = seq.shape
        keep = (torch.rand(N, 1, C, device=seq.device) > self.dropout_prob).float().expand(-1, L, -1)
        return seq * keep, mask * keep


# ---- helper: CLUDA feature-extractor network (momentum encoder + queues) ----
class CLUDA_Network(nn.Module):
    """
    Wraps the AdaTime backbone as query/key encoders, adds a projector,
    a domain discriminator, contrastive queues, and nearest-neighbour logic.
    forward() = inference only (returns normalised query features).
    contrastive_update() = full training forward (returns all contrastive logits).
    """

    def __init__(self, backbone_class, configs):
        super().__init__()
        feat_dim = configs.final_out_channels * configs.features_len
        K = configs.cluda_K
        self.m = configs.cluda_m
        self.T = configs.cluda_T
        self.num_neighbors = configs.cluda_num_neighbors

        # query encoder = backbone
        self.encoder_q = backbone_class(configs)
        # key encoder = momentum copy
        self.encoder_k = backbone_class(configs)

        mlp_hid = configs.cluda_mlp_hidden_dim
        use_bn = configs.cluda_use_batch_norm
        self.projector = CLUDA_MLP(feat_dim, mlp_hid, feat_dim, use_bn)
        self.discriminator = CLUDA_MLP(feat_dim, mlp_hid, 1, use_bn)

        # initialise key encoder from query encoder
        for p_q, p_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            p_k.data.copy_(p_q.data)
            p_k.requires_grad = False

        # queues
        self.register_buffer("queue_s", F.normalize(torch.randn(feat_dim, K), dim=0))
        self.register_buffer("queue_t", F.normalize(torch.randn(feat_dim, K), dim=0))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        self.K = K

    # ---------- momentum helpers ----------
    @torch.no_grad()
    def _momentum_update(self):
        if self.training:
            for p_q, p_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
                p_k.data = p_k.data * self.m + p_q.data * (1.0 - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys_s, keys_t):
        if not self.training:
            return
        bs = keys_s.shape[0]
        ptr = int(self.queue_ptr)
        end = ptr + bs
        if end <= self.K:
            self.queue_s[:, ptr:end] = keys_s.T
            self.queue_t[:, ptr:end] = keys_t.T
        else:
            # Wrap around: split the batch across the queue boundary
            first = self.K - ptr
            self.queue_s[:, ptr:] = keys_s[:first].T
            self.queue_t[:, ptr:] = keys_t[:first].T
            rem = bs - first
            self.queue_s[:, :rem] = keys_s[first:].T
            self.queue_t[:, :rem] = keys_t[first:].T
        self.queue_ptr[0] = (ptr + bs) % self.K

    # ---------- inference (used by AdaTime evaluate) ----------
    def forward(self, x):
        q = self.encoder_q(x)
        return F.normalize(q, dim=1)

    # ---------- full contrastive forward (training) ----------
    def contrastive_update(self, q_src, k_src, q_trg, k_trg, alpha):
        """Returns (logits_s, labels_s, logits_t, labels_t,
                   logits_ts, labels_ts, pred_domain, labels_domain, q_s)"""
        # --- query features ---
        q_s = F.normalize(self.encoder_q(q_src), dim=1)
        p_q_s = F.normalize(self.projector(q_s), dim=1)

        q_t = F.normalize(self.encoder_q(q_trg), dim=1)
        p_q_t = F.normalize(self.projector(q_t), dim=1)

        # --- key features (no grad, momentum update) ---
        with torch.no_grad():
            self._momentum_update()
            k_s = F.normalize(self.encoder_k(k_src), dim=1)
            k_t = F.normalize(self.encoder_k(k_trg), dim=1)

        # --- Source contrastive logits ---
        logits_s = torch.cat([torch.mm(p_q_s, k_s.T),
                              torch.mm(p_q_s, self.queue_s.clone().detach())], dim=1) / self.T
        labels_s = torch.arange(p_q_s.size(0), dtype=torch.long, device=q_s.device)

        # --- Target contrastive logits ---
        logits_t = torch.cat([torch.mm(p_q_t, k_t.T),
                              torch.mm(p_q_t, self.queue_t.clone().detach())], dim=1) / self.T
        labels_t = torch.arange(p_q_t.size(0), dtype=torch.long, device=q_t.device)

        # --- Cross-domain (target→source NN) contrastive logits ---
        _, idx_nn = cluda_NN(k_t, q_s.clone().detach(),
                             num_neighbors=self.num_neighbors, return_indices=True)
        logits_ts = torch.mm(q_t, q_s.T.clone().detach()) / self.T
        labels_ts = idx_nn.squeeze(1).to(q_t.device)

        # --- Domain discrimination ---
        q_s_rev = ReverseLayerF.apply(q_s, alpha)
        q_t_rev = ReverseLayerF.apply(q_t, alpha)
        pred_domain = self.discriminator(torch.cat([q_s_rev, q_t_rev], dim=0))
        labels_domain = torch.cat([torch.ones(len(q_s), 1, device=q_s.device),
                                   torch.zeros(len(q_t), 1, device=q_t.device)], dim=0)

        # enqueue
        self._dequeue_and_enqueue(k_s, k_t)

        return (logits_s, labels_s, logits_t, labels_t,
                logits_ts, labels_ts, pred_domain, labels_domain, q_s)
