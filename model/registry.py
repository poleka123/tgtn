# model/registry.py
from __future__ import annotations

from typing import Callable, Dict
import torch
import torch.nn as nn
import numpy as np

from model.base import DataMeta
from model.tucker_gode import TensorGODEForecast
from model.tucker_gode_v2 import TuckerGODEForecast
from model.gcn import GCNForecast
from model.gat import GATForecast
from model.dcrnn import DCRNNForecast
from model.mtgnn.net import MTGNNForecast
from model.stgcn import STGCN
from model.tgtn.net import TuckerFactorGraphForecast
from model.stode.stode import ODEGCN
from model.arima import ARIMAForecast
from model.st_tgm import  STTGMTNet
from model.gru import GRUForecast

ModelBuilder = Callable[..., nn.Module]

MODEL_REGISTRY: Dict[str, ModelBuilder] = {}


def register_model(name: str):
    def decorator(fn: ModelBuilder):
        MODEL_REGISTRY[name] = fn
        return fn
    return decorator

@register_model("gru")
def build_gru(args, meta: DataMeta) -> nn.Module:
    return GRUForecast(
        num_nodes=meta.num_nodes,
        input_features=meta.input_features,
        output_features=meta.output_features,
        num_timesteps_input=args.timesteps_input,
        num_timesteps_output=args.timesteps_output,
        hidden_dim=args.nhid,
        num_layers=4,
        dropout=0.2,
    )

@register_model("sttgm")
def build_sttgm(args, meta: DataMeta) -> nn.Module:
    rank_nodes = args.tucker_rank_nodes if getattr(args, 'tucker_rank_nodes', None) else \
                 int(meta.num_nodes * getattr(args, 'tucker_rank_nodes_ratio', 0.5))
    rank_nodes = min(rank_nodes, meta.num_nodes)
    
    rank_time_raw = getattr(args, 'tucker_rank_time', None)
    if rank_time_raw:
        rank_time = min(rank_time_raw, meta.input_steps)
    else:
        rank_time = max(int(meta.input_steps * 0.6), 1)
    
    return STTGMTNet(
        num_nodes=meta.num_nodes,
        input_features=meta.input_features,
        output_features=meta.output_features,
        num_timesteps_input=args.timesteps_input,
        num_timesteps_output=args.timesteps_output,
        spatial_ranks=(rank_nodes, args.tucker_rank_features),
        input_rank=rank_time,
    )


@register_model("arima")
def build_arima(args, meta: DataMeta) -> nn.Module:
    return ARIMAForecast(
        num_nodes=meta.num_nodes,
        num_timesteps_input=args.timesteps_input,
        num_timesteps_output=args.timesteps_output,
        order=(5, 1, 2),
    )
@register_model("stode")
def build_stode(args, meta: DataMeta) -> nn.Module:
    adj = getattr(args, '_adj', None)
    if adj is None:
        # 如果没有邻接矩阵，创建单位矩阵
        adj_np = np.eye(meta.num_nodes, dtype=np.float32)
        adj = torch.from_numpy(adj_np).float()
    return ODEGCN(
        num_nodes=meta.num_nodes,
        num_features=meta.input_features,
        num_timesteps_input=args.timesteps_input,
        num_timesteps_output=args.timesteps_output,
        A_sp_hat=adj,
        A_se_hat=adj,
    )

# # 3. 更新 MODEL_GRAPH_REQUIREMENTS
# MODEL_GRAPH_REQUIREMENTS = {
#     # ... 现有内容 ...
#     "stode": True,
# }

@register_model("tuckergode_v2")
def build_tuckergode_v2(args, meta: DataMeta) -> nn.Module:
    rank_nodes = args.tucker_rank_nodes if getattr(args, 'tucker_rank_nodes', None) else \
                 int(meta.num_nodes * getattr(args, 'tucker_rank_nodes_ratio', 0.8))
    rank_nodes = min(rank_nodes, meta.num_nodes)
    
    rank_time_raw = getattr(args, 'tucker_rank_time', None)
    if rank_time_raw:
        rank_time = min(rank_time_raw, meta.input_steps)
    else:
        rank_time = max(int(meta.input_steps * 0.8), 1)
    return TuckerGODEForecast(
        num_nodes=meta.num_nodes,
        input_features=meta.input_features,
        output_features=meta.output_features,
        num_timesteps_input=args.timesteps_input,
        num_timesteps_output=args.timesteps_output,
        gcn_true=True,
        buildA_true=False,
        gcn_depth=2,
        device="cuda:0" if torch.cuda.is_available() else "cpu",
        layers=2,
        dropout=0.3,
        # Tucker 参数
        rank_nodes=rank_nodes,
        rank_time=rank_time,
        rank_features=args.tucker_rank_features,
        ode_time=args.ode_time,
        use_tucker=True,
    )


@register_model("mtgnn")
def build_mtgnn(args, meta: DataMeta) -> nn.Module:
    adj = None
    # 从 data_set 中取出预计算邻接矩阵
    data_set_adj = None
    try:
        from data.loader import Data_load
        _ = Data_load  # 仅确保模块存在
    except ImportError:
        pass
    return MTGNNForecast(
        num_nodes=meta.num_nodes,
        input_features=meta.input_features,
        output_features=meta.output_features,
        num_timesteps_input=args.timesteps_input,
        num_timesteps_output=args.timesteps_output,
        gcn_true=True,
        buildA_true=False,           # 用外部传入的 adj，不内部构建
        gcn_depth=2,
        device="cuda" if torch.cuda.is_available() else "cpu",
        layers=2,
        dropout=0.3,
    )


@register_model("tuckergode")
def build_tuckergode(args, meta: DataMeta) -> nn.Module:
    # rank_nodes: 优先用 args，兜底为 num_nodes * ratio，上限不超过 num_nodes
    rank_nodes = args.tucker_rank_nodes if getattr(args, 'tucker_rank_nodes', None) else \
                 int(meta.num_nodes * getattr(args, 'tucker_rank_nodes_ratio', 0.8))
    rank_nodes = min(rank_nodes, meta.num_nodes)
    # rank_time: 优先用 args，否则自动设为 input_steps 的一半
    rank_time_raw = getattr(args, 'tucker_rank_time', None)
    if rank_time_raw:
        rank_time = min(rank_time_raw, meta.input_steps)
    else:
        rank_time = max(int(meta.input_steps * 0.8), 1)
    # 与 main_sci_odegcn.py 当前硬编码保持一致
    return TensorGODEForecast(
        num_nodes=meta.num_nodes,
        num_features=meta.input_features,
        output_features=meta.output_features,
        num_timesteps_input=args.timesteps_input,
        num_timesteps_output=args.timesteps_output,
        hidden_dim=args.nhid,
        rank_nodes=rank_nodes,
        rank_time=rank_time,
        rank_features=args.tucker_rank_features,
        num_ode_layers=args.ode_layers,
        ode_time=args.ode_time,
        ode_solver=args.ode_solver,
        euler_steps=args.ode_euler_steps,
    )
@register_model("gcn")
def build_GCNForecast(args, meta: DataMeta) -> nn.Module:
    return GCNForecast(
        num_nodes=meta.num_nodes,
        input_features=meta.input_features,
        output_features=meta.output_features,
        num_timesteps_input=args.timesteps_input,
        num_timesteps_output=args.timesteps_output,
        hidden_dim=args.nhid,
        num_layers=2,
)
@register_model("gat")
def build_gat(args, meta: DataMeta) -> nn.Module:
    return GATForecast(
        num_nodes=meta.num_nodes,
        input_features=meta.input_features,
        output_features=meta.output_features,
        num_timesteps_input=args.timesteps_input,
        num_timesteps_output=args.timesteps_output,
        hidden_dim=args.nhid,
        num_heads=4,
        num_layers=2,
    )
@register_model("dcrnn")
def build_dcrnn(args, meta: DataMeta) -> nn.Module:
    return DCRNNForecast(
        num_nodes=meta.num_nodes,
        input_features=meta.input_features,
        output_features=meta.output_features,
        num_timesteps_input=args.timesteps_input,
        num_timesteps_output=args.timesteps_output,
        hidden_dim=args.nhid,
        num_layers=2,
    )
@register_model("stgcn")
def build_stgcn(args, meta: DataMeta) -> nn.Module:
    return STGCN(
        num_nodes=meta.num_nodes,
        out_channels=args.nhid,
        spatial_channels=args.nhid,
        features=meta.input_features,
        timesteps_input=args.timesteps_input,
        timesteps_output=args.timesteps_output,
    )
@register_model("tgtn")
def build_tgtn(args, meta: DataMeta) -> nn.Module:
    rank_nodes = args.tucker_rank_nodes if getattr(args, 'tucker_rank_nodes', None) else \
                 int(meta.num_nodes * getattr(args, 'tucker_rank_nodes_ratio', 0.5))
    rank_nodes = min(rank_nodes, meta.num_nodes)
    
    rank_time_raw = getattr(args, 'tucker_rank_time', None)
    if rank_time_raw:
        rank_time = min(rank_time_raw, meta.input_steps)
    else:
        rank_time = max(int(meta.input_steps * 0.6), 1)
    # 超参数(4,2,2),(8,4,4),(12,6,6),(16,8,8)
    return TuckerFactorGraphForecast(
        num_nodes=meta.num_nodes,
        input_features=meta.input_features,
        output_features=meta.output_features,
        num_timesteps_input=args.timesteps_input,
        num_timesteps_output=args.timesteps_output,
        gcn_true=True,
        buildA_true=False,
        gcn_depth=2,
        device="cuda:0" if torch.cuda.is_available() else "cpu",
        layers=2,
        dropout=0.3,
        # Tucker 参数
        tucker_node_rank=4,
        tucker_time_rank=2,
        tucker_channel_rank=2,
        use_tucker=True,
    )

MODEL_GRAPH_REQUIREMENTS = {
    "tuckergode": False,
    "gcn":True,
    "gat":True,
    "dcrnn": True,
    "mtgnn": True,
    "stgcn": True,
    "tuckergode_v2": True,
    "tgtn": True,
    "stode": True,
    "sttgm": False,
    "gru": False,
}

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
