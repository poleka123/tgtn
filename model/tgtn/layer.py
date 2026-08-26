"""MTGNN layers and global Tucker-guided extensions.

All graph/temporal factors below are model parameters: no per-batch tensor
decomposition is performed.
"""
from __future__ import annotations

import numbers
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init


class nconv(nn.Module):
    def forward(self, x, A):
        return torch.einsum("ncwl,vw->ncvl", (x, A)).contiguous()


class linear(nn.Module):
    def __init__(self, c_in, c_out, bias=True):
        super().__init__()
        self.mlp = nn.Conv2d(c_in, c_out, kernel_size=(1, 1), bias=bias)

    def forward(self, x):
        return self.mlp(x)


class mixprop(nn.Module):
    """Original MTGNN mix-hop propagation."""
    def __init__(self, c_in, c_out, gdep, dropout, alpha):
        super().__init__()
        self.nconv = nconv()
        self.mlp = linear((gdep + 1) * c_in, c_out)
        self.gdep, self.dropout, self.alpha = gdep, dropout, alpha

    def forward(self, x, adj):
        adj = adj + torch.eye(adj.size(0), device=x.device, dtype=x.dtype)
        d = adj.sum(1).clamp_min(1e-12)
        a = adj / d.view(-1, 1)
        h, states = x, [x]
        for _ in range(self.gdep):
            h = self.alpha * x + (1.0 - self.alpha) * self.nconv(h, a)
            states.append(h)
        return self.mlp(torch.cat(states, dim=1))


class dilated_inception(nn.Module):
    """Original MTGNN multi-scale temporal convolution.

    ``temporal_factor`` optionally supplies global Tucker temporal guidance.
    It has shape [T, r_T] and produces both a temporal gate and four branch
    weights.  With ``temporal_factor=None`` this is exactly the unweighted
    MTGNN inception operation.
    """
    def __init__(self, cin, cout, dilation_factor=2, temporal_rank=None):
        super().__init__()
        self.kernel_set = [2, 3, 6, 7]
        if cout % len(self.kernel_set):
            raise ValueError("conv_channels must be divisible by 4 for dilated_inception")
        branch_channels = cout // len(self.kernel_set)
        self.tconv = nn.ModuleList([
            nn.Conv2d(cin, branch_channels, (1, kernel), dilation=(1, dilation_factor))
            for kernel in self.kernel_set
        ])
        self.temporal_gate = None
        self.scale_selector = None
        if temporal_rank is not None:
            self.temporal_gate = nn.Linear(temporal_rank, 1)
            self.scale_selector = nn.Sequential(
                nn.Linear(temporal_rank, temporal_rank), nn.ReLU(),
                nn.Linear(temporal_rank, len(self.kernel_set)),
            )

    def forward(self, input, temporal_factor=None):
        x = input
        scale = None
        if temporal_factor is not None:
            if self.temporal_gate is None:
                raise RuntimeError("temporal_rank must be set to use temporal_factor")
            # [T,r_T] -> [1,1,1,T]; interpolate because later MTGNN blocks
            # have a shorter temporal axis than the input history.
            gate = torch.sigmoid(self.temporal_gate(temporal_factor).transpose(0, 1))
            gate = F.interpolate(gate[None, None], size=(1, input.size(-1)), mode="bilinear", align_corners=False)
            x = x * gate
            scale = torch.softmax(self.scale_selector(temporal_factor.mean(0)), dim=-1)

        outputs = [conv(x) for conv in self.tconv]
        target_t = outputs[-1].size(-1)
        outputs = [y[..., -target_t:] for y in outputs]
        if scale is not None:
            outputs = [y * scale[i] for i, y in enumerate(outputs)]
        return torch.cat(outputs, dim=1)


class graph_constructor(nn.Module):
    """MTGNN directed sparse graph constructor with optional Tucker prior."""
    def __init__(self, nnodes, k, dim, device, alpha=3, static_feat=None, spatial_rank=None):
        super().__init__()
        self.nnodes, self.k, self.dim, self.alpha = nnodes, min(k, nnodes), dim, alpha
        self.static_feat = static_feat
        if static_feat is not None:
            in_dim = static_feat.shape[1]
            self.lin1, self.lin2 = nn.Linear(in_dim, dim), nn.Linear(in_dim, dim)
        else:
            self.emb1, self.emb2 = nn.Embedding(nnodes, dim), nn.Embedding(nnodes, dim)
            self.lin1, self.lin2 = nn.Linear(dim, dim), nn.Linear(dim, dim)
        self.factor_lin1 = self.factor_lin2 = None
        if spatial_rank is not None:
            self.factor_lin1, self.factor_lin2 = nn.Linear(spatial_rank, dim, bias=False), nn.Linear(spatial_rank, dim, bias=False)

    def _node_vectors(self, idx, spatial_factor=None):
        if self.static_feat is None:
            nodevec1, nodevec2 = self.emb1(idx), self.emb2(idx)
        else:
            static = self.static_feat.to(idx.device)[idx]
            nodevec1 = nodevec2 = static
        nodevec1, nodevec2 = self.lin1(nodevec1), self.lin2(nodevec2)
        if spatial_factor is not None:
            if self.factor_lin1 is None:
                raise RuntimeError("spatial_rank must be set to use spatial_factor")
            factor = spatial_factor[idx]
            nodevec1 = nodevec1 + self.factor_lin1(factor)
            nodevec2 = nodevec2 + self.factor_lin2(factor)
        return torch.tanh(self.alpha * nodevec1), torch.tanh(self.alpha * nodevec2)

    def fullA(self, idx, spatial_factor=None):
        nodevec1, nodevec2 = self._node_vectors(idx, spatial_factor)
        a = nodevec1 @ nodevec2.T - nodevec2 @ nodevec1.T
        return F.relu(torch.tanh(self.alpha * a))

    def forward(self, idx, spatial_factor=None):
        adj = self.fullA(idx, spatial_factor)
        _, topk_index = (adj + torch.rand_like(adj) * 0.01).topk(self.k, dim=1)
        mask = torch.zeros_like(adj).scatter_(1, topk_index, 1.0)
        return adj * mask


class GlobalTuckerRepresentation(nn.Module):
    """A global, learnable Tucker tensor for a fixed N-by-T forecasting task.

    Factors are U_N [N,r_N], U_T [T,r_T], U_C [C,r_C]; the shared core is
    G [r_C,r_N,r_T]. ``decode`` returns a global context [1,C,N,T].
    """
    def __init__(self, num_nodes, seq_length, channels, node_rank, time_rank, channel_rank):
        super().__init__()
        self.U_N = nn.Parameter(torch.empty(num_nodes, node_rank))
        self.U_T = nn.Parameter(torch.empty(seq_length, time_rank))
        self.U_C = nn.Parameter(torch.empty(channels, channel_rank))
        self.G = nn.Parameter(torch.empty(channel_rank, node_rank, time_rank))
        self.reset_parameters()

    def reset_parameters(self):
        for parameter in (self.U_N, self.U_T, self.U_C, self.G):
            nn.init.xavier_uniform_(parameter)

    @property
    def spatial_factor(self):
        return F.normalize(self.U_N, p=2, dim=-1)

    @property
    def temporal_factor(self):
        return F.normalize(self.U_T, p=2, dim=-1)

    def decode(self):
        channels = F.normalize(self.U_C, p=2, dim=-1)
        context = torch.einsum("ca,nb,td,abd->cnt", channels, self.spatial_factor, self.temporal_factor, self.G)
        return context.unsqueeze(0)


class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True):
        super().__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        self.normalized_shape, self.eps = tuple(normalized_shape), eps
        self.elementwise_affine = elementwise_affine
        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(*normalized_shape))
            self.bias = nn.Parameter(torch.zeros(*normalized_shape))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    def forward(self, input, idx):
        if self.elementwise_affine:
            return F.layer_norm(input, tuple(input.shape[1:]), self.weight[:, idx, :], self.bias[:, idx, :], self.eps)
        return F.layer_norm(input, tuple(input.shape[1:]), None, None, self.eps)
