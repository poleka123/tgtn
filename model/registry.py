# model/registry.py
from __future__ import annotations

from typing import Callable, Dict

import torch.nn as nn

from model.base import DataMeta
from model.tucker_gode import TensorGODEForecast

ModelBuilder = Callable[..., nn.Module]

MODEL_REGISTRY: Dict[str, ModelBuilder] = {}


def register_model(name: str):
    def decorator(fn: ModelBuilder):
        MODEL_REGISTRY[name] = fn
        return fn
    return decorator


@register_model("tuckergode")
def build_tuckergode(args, meta: DataMeta) -> nn.Module:
    # 与 main_sci_odegcn.py 当前硬编码保持一致
    return TensorGODEForecast(
        num_nodes=meta.num_nodes,
        num_features=meta.input_features,
        output_features=meta.output_features,
        num_timesteps_input=args.timesteps_input,
        num_timesteps_output=args.timesteps_output,
        hidden_dim=args.nhid,
        rank_nodes=64,                              # 当前 main 写死值
        rank_time=24,                               # 当前 main 写死值
        rank_features=args.tucker_rank_features,
        num_ode_layers=args.ode_layers,
        ode_time=args.ode_time,
        ode_solver=args.ode_solver,
        euler_steps=args.ode_euler_steps,
    )


def build_model(model_name: str, args, data_set: dict, output_features: int = 1) -> nn.Module:
    if model_name not in MODEL_REGISTRY:
        available = ", ".join(sorted(MODEL_REGISTRY.keys()))
        raise ValueError(f"Unknown model '{model_name}'. Available: {available}")
    from model.base import extract_data_meta
    meta = extract_data_meta(data_set, output_features=output_features)
    model = MODEL_REGISTRY[model_name](args, meta)
    model.requires_graph = MODEL_GRAPH_REQUIREMENTS.get(model_name, False)
    return model


def list_models():
    return sorted(MODEL_REGISTRY.keys())

MODEL_GRAPH_REQUIREMENTS = {
    "tuckergode": False,
    # Step 4 预留
    # "gcn": True,
    # "gat": True,
    # "mtgnn": True,
}