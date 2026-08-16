# model/base.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import torch.nn as nn


@dataclass
class DataMeta:
    """从 data_set 提取的、构建模型所需的元信息。"""
    num_nodes: int
    input_features: int
    output_features: int = 1
    input_steps: int = 24  

def extract_data_meta(data_set: dict, output_features: int = 1) -> DataMeta:
    return DataMeta(
        num_nodes=data_set["num_nodes"],
        input_features=data_set["input_features"],
        output_features=output_features,
        input_steps=data_set["train_input"].shape[2],  # ← 从原始数据形状中提取 T_in
    )
class ForecastModel(Protocol):
    """Step 1 约定：所有模型 forward 至少接受 x: [B, N, T_in, F]。"""

    def forward(self, x): ...
    def parameters(self): ...
    def train(self, mode: bool = True): ...
    def eval(self): ...


def extract_data_meta(data_set: dict, output_features: int = 1) -> DataMeta:
    return DataMeta(
        num_nodes=data_set["num_nodes"],
        input_features=data_set["input_features"],
        output_features=output_features,
    )