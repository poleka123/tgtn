"""Tucker-enhanced dynamic Graph Neural ODE for multi-step forecasting.

The model receives ``[batch, nodes, input_steps, features]`` and learns its
dynamic graph internally. It deliberately does not consume an adjacency matrix.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
from torch import Tensor, nn
import torch.nn.functional as F

try:
    from torchdiffeq import odeint
except ImportError:
    odeint = None


class TuckerEncoder(nn.Module):
    """Encode a spatio-temporal tensor into an initial node state H(0)."""

    def __init__(
        self, num_nodes: int, input_steps: int, num_features: int, hidden_dim: int,
        rank_nodes: int, rank_time: int, rank_features: int,
    ) -> None:
        super().__init__()
        self.rank_nodes = min(rank_nodes, num_nodes)
        self.rank_time = min(rank_time, input_steps)
        self.rank_features = min(rank_features, num_features)
        self.node_factor = nn.Parameter(torch.empty(num_nodes, self.rank_nodes))
        self.time_factor = nn.Parameter(torch.empty(input_steps, self.rank_time))
        self.feature_factor = nn.Parameter(torch.empty(num_features, self.rank_features))
        self.state_projection = nn.Linear(self.rank_time * self.rank_features, hidden_dim)
        self.state_norm = nn.LayerNorm(hidden_dim)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.orthogonal_(self.node_factor)
        nn.init.orthogonal_(self.time_factor)
        nn.init.orthogonal_(self.feature_factor)
        self.state_projection.reset_parameters()
        self.state_norm.reset_parameters()

    @staticmethod
    def _orthogonal_factor(factor: Tensor) -> Tensor:
        return torch.linalg.qr(factor, mode="reduced").Q

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        if x.ndim != 4:
            raise ValueError("x must have shape [batch, nodes, input_steps, features].")

        u_n = self._orthogonal_factor(self.node_factor)
        u_t = self._orthogonal_factor(self.time_factor)
        u_c = self._orthogonal_factor(self.feature_factor)
        # G = X x_1 U_N^T x_2 U_T^T x_3 U_C^T
        core = torch.einsum("bntc,nr,ts,cu->brsu", x, u_n, u_t, u_c)
        core_mode1 = core.flatten(start_dim=2)  # [B, r_N, r_T * r_C]
        node_state = torch.einsum("nr,brd->bnd", u_n, core_mode1)
        h0 = self.state_norm(F.gelu(self.state_projection(node_state)))
        return h0, core


class TensorDynamicODEFunc(nn.Module):
    """The first-order ODEFunc with a state-derived dynamic adjacency matrix."""

    def __init__(self, hidden_dim: int, attention_dim: Optional[int] = None) -> None:
        super().__init__()
        self.attention_dim = attention_dim or hidden_dim
        self.query = nn.Linear(hidden_dim, self.attention_dim, bias=False)
        self.key = nn.Linear(hidden_dim, self.attention_dim, bias=False)
        self.message_projection = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.initial_projection = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.gate = nn.Linear(2 * hidden_dim, hidden_dim)
        self.h0: Optional[Tensor] = None

    def set_initial_state(self, h0: Tensor) -> None:
        self.h0 = h0

    def forward(self, t: Tensor, h: Tensor) -> Tensor:
        if self.h0 is None:
            raise RuntimeError("Set the Tucker initial state before integrating the ODE.")
        query, key = self.query(h), self.key(h)
        scores = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(self.attention_dim)
        adjacency = torch.softmax(scores, dim=-1)
        message = torch.matmul(adjacency, h)
        graph_term = F.gelu(self.message_projection(message))
        structural_term = F.gelu(self.initial_projection(self.h0))
        gate = torch.sigmoid(self.gate(torch.cat([h, self.h0], dim=-1)))
        return gate * graph_term + (1.0 - gate) * structural_term


class FirstOrderODEBlock(nn.Module):
    """Integrate an ODEFunc, with a differentiable Euler fallback."""

    def __init__(self, odefunc: TensorDynamicODEFunc, integration_time: float = 1.0,
                 solver: str = "rk4", euler_steps: int = 4) -> None:
        super().__init__()
        self.odefunc = odefunc
        self.integration_time = integration_time
        self.solver = solver
        self.euler_steps = euler_steps

    def forward(self, h: Tensor) -> Tensor:
        time = torch.tensor([0.0, self.integration_time], device=h.device, dtype=h.dtype)
        if odeint is not None:
            return odeint(self.odefunc, h, time, method=self.solver)[-1]
        dt, state = self.integration_time / self.euler_steps, h
        for step in range(self.euler_steps):
            t = torch.tensor(step * dt, device=h.device, dtype=h.dtype)
            state = state + dt * self.odefunc(t, state)
        return state


class TensorGODEBlock(nn.Module):
    """One residual first-order continuous evolution block."""

    def __init__(self, hidden_dim: int, attention_dim: Optional[int], ode_time: float,
                 ode_solver: str, euler_steps: int) -> None:
        super().__init__()
        self.odefunc = TensorDynamicODEFunc(hidden_dim, attention_dim)
        self.odeblock = FirstOrderODEBlock(self.odefunc, ode_time, ode_solver, euler_steps)
        self.norm = nn.LayerNorm(hidden_dim)
        self.residual_scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, h: Tensor, h0: Tensor) -> Tensor:
        self.odefunc.set_initial_state(h0)
        evolved = self.odeblock(h)
        return self.norm(h + self.residual_scale * (evolved - h))


class TensorGODEForecast(nn.Module):
    """Tucker encoder, stacked dynamic first-order GODEs, and forecast decoder."""

    def __init__(
        self, num_nodes: int, num_features: int, num_timesteps_input: int,
        num_timesteps_output: int, hidden_dim: int = 32, rank_nodes: int = 32,
        rank_time: int = 6, rank_features: int = 4, num_ode_layers: int = 2,
        attention_dim: Optional[int] = None, ode_time: float = 1.0,
        ode_solver: str = "rk4", euler_steps: int = 4,
    ) -> None:
        super().__init__()
        if num_ode_layers < 1:
            raise ValueError("num_ode_layers must be at least 1.")
        self.num_nodes = num_nodes
        self.num_features = num_features
        self.num_timesteps_input = num_timesteps_input
        self.num_timesteps_output = num_timesteps_output
        self.encoder = TuckerEncoder(num_nodes, num_timesteps_input, num_features,
                                     hidden_dim, rank_nodes, rank_time, rank_features)
        self.ode_layers = nn.ModuleList(
            TensorGODEBlock(hidden_dim, attention_dim, ode_time, ode_solver, euler_steps)
            for _ in range(num_ode_layers)
        )
        self.decoder = nn.Linear(hidden_dim, num_timesteps_output * num_features)

    def forward(self, x: Tensor) -> Tensor:
        expected = (self.num_nodes, self.num_timesteps_input, self.num_features)
        if x.ndim != 4 or tuple(x.shape[1:]) != expected:
            raise ValueError(f"Expected x as [B, {expected[0]}, {expected[1]}, {expected[2]}], got {tuple(x.shape)}.")
        h0, _ = self.encoder(x)
        h = h0
        for ode_layer in self.ode_layers:
            h = ode_layer(h, h0)
        prediction = self.decoder(h)
        return prediction.view(x.shape[0], self.num_nodes, self.num_timesteps_output, self.num_features)
