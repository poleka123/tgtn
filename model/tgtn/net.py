"""Tucker-guided MTGNN forecaster for X shaped [B, N, T, F]."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from model.tgtn.layer import (
    GlobalTuckerRepresentation, LayerNorm, dilated_inception,
    graph_constructor, mixprop,
)


class TuckerFactorGraphForecast(nn.Module):
    """MTGNN backbone augmented with global learnable Tucker structure.

    The forward interface remains ``forward(x, adj=None) -> [B,N,T_out]``.
    ``x`` must be [B,N,T_in,F].  The Tucker factors are global parameters,
    shared by all batches, rather than a decomposition recomputed per sample.
    """
    def __init__(
        self, num_nodes: int, input_features: int, output_features: int,
        num_timesteps_input: int, num_timesteps_output: int,
        gcn_true: bool = True, buildA_true: bool = False, gcn_depth: int = 2,
        device: str = "cpu", predefined_A: Tensor = None, static_feat: Tensor = None,
        subgraph_size: int = 20, node_dim: int = 40, dilation_exponential: int = 1,
        conv_channels: int = 16, residual_channels: int = 16, skip_channels: int = 16,
        end_channels: int = 16, layers: int = 3, propalpha: float = 0.05,
        tanhalpha: float = 3, layer_norm_affline: bool = True, dropout: float = 0.3,
        tucker_node_rank: int | None = None, tucker_time_rank: int | None = None,
        tucker_channel_rank: int | None = None, use_tucker: bool = True,
    ) -> None:
        super().__init__()
        if conv_channels % 4:
            raise ValueError("conv_channels must be divisible by 4")
        self.num_nodes, self.seq_length, self.layers = num_nodes, num_timesteps_input, layers
        self.num_timesteps_output, self.in_dim, self.out_dim = num_timesteps_output, input_features, num_timesteps_output
        self.gcn_true, self.buildA_true, self.predefined_A, self.dropout = gcn_true, buildA_true, predefined_A, dropout

        # Rank defaults preserve a compact global representation on any dataset.
        r_n = tucker_node_rank or min(16, num_nodes)
        r_t = tucker_time_rank or min(8, num_timesteps_input)
        r_c = tucker_channel_rank or min(8, residual_channels)
        self.use_tucker = use_tucker
        self.tucker = GlobalTuckerRepresentation(num_nodes, num_timesteps_input, residual_channels, r_n, r_t, r_c) if use_tucker else None

        self.start_conv = nn.Conv2d(input_features, residual_channels, kernel_size=(1, 1))
        if buildA_true:
            self.gc = graph_constructor(num_nodes, subgraph_size, node_dim, device, alpha=tanhalpha, static_feat=static_feat, spatial_rank=r_n if use_tucker else None)

        kernel_size = 7
        self.receptive_field = (int(1 + (kernel_size - 1) * (dilation_exponential ** layers - 1) / (dilation_exponential - 1))
                                if dilation_exponential > 1 else layers * (kernel_size - 1) + 1)
        self.filter_convs, self.gate_convs = nn.ModuleList(), nn.ModuleList()
        self.residual_convs, self.skip_convs = nn.ModuleList(), nn.ModuleList()
        self.gconv1, self.gconv2, self.norm = nn.ModuleList(), nn.ModuleList(), nn.ModuleList()
        # self.context_projs, self.context_gates = nn.ModuleList(), nn.ParameterList()
        self.context_proj = None
        self.context_gate = None
        if use_tucker:
            self.context_proj = nn.Conv2d(
                residual_channels,
                residual_channels,
                kernel_size=(1, 1),
            )
        self.context_gate = nn.Parameter(torch.tensor(-2.0))
        #  屏蔽context_gate
        # self.context_gate = nn.Parameter(torch.tensor(-20.0))


        new_dilation = 1
        for j in range(1, layers + 1):
            rf_size_j = (int(1 + (kernel_size - 1) * (dilation_exponential ** j - 1) / (dilation_exponential - 1))
                         if dilation_exponential > 1 else j * (kernel_size - 1) + 1)
            self.filter_convs.append(dilated_inception(residual_channels, conv_channels, new_dilation, temporal_rank=r_t if use_tucker else None))
            self.gate_convs.append(dilated_inception(residual_channels, conv_channels, new_dilation, temporal_rank=r_t if use_tucker else None))
            self.residual_convs.append(nn.Conv2d(conv_channels, residual_channels, kernel_size=(1, 1)))
            actual_len = self.seq_length - rf_size_j + 1 if self.seq_length > self.receptive_field else self.receptive_field - rf_size_j + 1
            self.skip_convs.append(nn.Conv2d(conv_channels, skip_channels, kernel_size=(1, actual_len)))
            if gcn_true:
                self.gconv1.append(mixprop(conv_channels, residual_channels, gcn_depth, dropout, propalpha))
                self.gconv2.append(mixprop(conv_channels, residual_channels, gcn_depth, dropout, propalpha))
            # self.context_projs.append(nn.Conv2d(residual_channels, residual_channels, kernel_size=(1, 1)))
            # self.context_gates.append(nn.Parameter(torch.tensor(-2.0)))
            self.norm.append(LayerNorm((residual_channels, num_nodes, actual_len), elementwise_affine=layer_norm_affline))
            new_dilation *= dilation_exponential

        initial_length = self.seq_length if self.seq_length > self.receptive_field else self.receptive_field
        self.skip0 = nn.Conv2d(residual_channels, skip_channels, kernel_size=(1, initial_length), bias=True)
        final_length = self.seq_length - self.receptive_field + 1 if self.seq_length > self.receptive_field else 1
        self.skipE = nn.Conv2d(residual_channels, skip_channels, kernel_size=(1, final_length), bias=True)
        self.end_conv_1 = nn.Conv2d(skip_channels, end_channels, kernel_size=(1, 1), bias=True)
        self.end_conv_2 = nn.Conv2d(end_channels, self.out_dim, kernel_size=(1, 1), bias=True)
        self.register_buffer("idx", torch.arange(num_nodes), persistent=False)
        # self._cached_context = None
        # self._cache_timesteps = -1

    # def _global_context(self, target_t: int, dtype: torch.dtype):
    #     context = self.tucker.decode().to(dtype=dtype)
    #     return F.interpolate(context, size=(self.num_nodes, target_t), mode="bilinear", align_corners=False)
        # if self._cached_context is None or self._cache_timesteps != target_t:
        #     context = self.tucker.decode().to(dtype=dtype)
        #     self._cached_context = F.interpolate(
        #         context,
        #         size=(self.num_nodes, target_t),
        #         mode="bilinear",
        #         align_corners=False
        #     )
        #     self._cache_timesteps = target_t
        # return self._cached_context

    def forward(self, x: Tensor, adj: Tensor = None) -> Tensor:
        if x.ndim != 4:
            raise ValueError(f"x must have shape [B,N,T,F], got {tuple(x.shape)}")
        if x.size(1) != self.num_nodes or x.size(2) != self.seq_length or x.size(3) != self.in_dim:
            raise ValueError(f"expected [B,{self.num_nodes},{self.seq_length},{self.in_dim}], got {tuple(x.shape)}")
        x = x.permute(0, 3, 1, 2)
        if self.seq_length < self.receptive_field:
            x = F.pad(x, (self.receptive_field - self.seq_length, 0, 0, 0))

        adp = None
        if self.gcn_true:
            if self.buildA_true:
                factor = self.tucker.spatial_factor if self.use_tucker else None
                adp = self.gc(self.idx, factor)
            else:
                adp = adj if adj is not None else self.predefined_A
                if adp is None:
                    raise ValueError("set buildA_true=True or provide adj/predefined_A when gcn_true=True")
                adp = adp.to(device=x.device, dtype=x.dtype)

        x = self.start_conv(x)
        # skip = self.skip0(F.dropout(x, self.dropout, training=self.training))
        # temporal_factor = self.tucker.temporal_factor if self.use_tucker else None
        # global_context = self.tucker.decode().to(dtype=x.dtype)
        # 全局global_context 仅使用一次
        temporal_factor = None
        if self.use_tucker:
            temporal_factor = self.tucker.temporal_factor

            # 只解码与注入一次全局低秩上下文
            global_context = self.tucker.decode().to(dtype=x.dtype)
            global_context = F.interpolate(
                global_context,
                size=(self.num_nodes, x.size(-1)),
                mode="bilinear",
                align_corners=False,
            )
            x = x + torch.sigmoid(self.context_gate) * self.context_proj(global_context)

        skip = self.skip0(F.dropout(x, self.dropout, training=self.training))

        for i in range(self.layers):
            residual = x
            x = torch.tanh(self.filter_convs[i](x, temporal_factor)) * torch.sigmoid(self.gate_convs[i](x, temporal_factor))
            x = F.dropout(x, self.dropout, training=self.training)
            skip = skip + self.skip_convs[i](x)
            if self.gcn_true and adp is not None:
                x = self.gconv1[i](x, adp) + self.gconv2[i](x, adp.T)
            else:
                x = self.residual_convs[i](x)
            # if self.use_tucker:
            #     # global_context = self._global_context(x.size(-1), x.dtype)
            #     # x = x + torch.sigmoid(self.context_gates[i]) * self.context_projs[i](global_context)
            #     block_context = F.interpolate(
            #         global_context,
            #         size=(self.num_nodes, x.size(-1)),
            #         mode="bilinear",
            #         align_corners=False,
            #     )
            #     x = x + torch.sigmoid(self.context_gates[i]) * self.context_projs[i](block_context)
            x = x + residual[..., -x.size(-1):]
            x = self.norm[i](x, self.idx)

        x = F.relu(self.skipE(x) + skip)
        x = self.end_conv_2(F.relu(self.end_conv_1(x)))
        return x.squeeze(-1).permute(0, 2, 1)
