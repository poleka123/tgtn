# data/loader.py
import os

import numpy as np
import pandas as pd
import torch

from data.graph_builder import build_graph
from utils.utils import Z_Score, generate_dataset, files

TS_DATASETS = [
    "electricity",
    "traffic",
    "weather",
    "ETTm1",
    "ETTm2",
    "ETTh1",
    "ETTh2",
    "ILI",
    "exchange",
    "solar_AL",
]
# max_rows: int = 1440
# max_rows: int = 8640
# max_rows: int = 2160
def _load_ts_csv(filename: str, max_rows: int) -> np.ndarray:
    filepath = "./data_set/TSdata/"
    # temp_df = pd.read_csv(filepath + filename + ".csv", nrows=0)
    full_path = filepath + filename + ".csv"
    try:
        temp_df = pd.read_csv(full_path, nrows=0, encoding="utf-8")
    except UnicodeDecodeError:
        temp_df = pd.read_csv(full_path, nrows=0, encoding="gbk")
    total_cols = len(temp_df.columns)
    # data = pd.read_csv(
    #     filepath + filename + ".csv",
    #     nrows=max_rows,
    #     usecols=range(total_cols - 1),
    # )

    try:
        data = pd.read_csv(full_path, nrows=max_rows, usecols=range(total_cols - 1), encoding="utf-8")
    except UnicodeDecodeError:
        data = pd.read_csv(full_path, nrows=max_rows, usecols=range(total_cols - 1), encoding="gbk")
    data = data.select_dtypes(include=[np.number]).values.astype(np.float32)

    if filename in ["electricity", "traffic"]:
        data = np.expand_dims(data, axis=-1)
    else:
        data = np.expand_dims(data.T, axis=-1).transpose(1, 0, 2)
    return data


def _load_pems_or_npz(filename: str, max_rows: int) -> np.ndarray:
    filepath = "./data_set/"
    file_entry = files[filename]
    npz_path = filepath + file_entry[0]
    csv_path = filepath + file_entry[1]

    if os.path.exists(npz_path):
        data = np.load(npz_path)["data"][:max_rows].astype(np.float32)
    elif os.path.exists(csv_path):
        data = pd.read_csv(csv_path, header=None, nrows=max_rows).values.astype(np.float32)
    else:
        raise FileNotFoundError(
            f"Cannot find dataset files for '{filename}': {npz_path} or {csv_path}"
        )

    if len(data.shape) == 2:
        data = np.expand_dims(data, axis=-1)
    return data

def _load_solar_al(filename: str, max_rows: int) -> np.ndarray:
    """读取 solar_AL.txt，格式为纯数值无表头，shape=(时间步, 站点数, 1)"""
    filepath = f"./data_set/TSdata/{filename}.txt"
    # shape: (时间步, 站点数)
    data = np.loadtxt(filepath, delimiter=',', dtype=np.float32)
    
    # 只取 max_rows 行
    data = data[:max_rows]
    
    # 扩展为 (时间步, 站点数, 1)，符合模型输入格式
    data = np.expand_dims(data, axis=-1)
    return data

def _load_raw_data(args) -> tuple[np.ndarray, str]:
    filename = args.filename
    max_rows = getattr(args, "max_rows", 8640)

    if filename == 'solar_AL':
        return _load_solar_al(filename, max_rows=max_rows), "ts"
    if filename in TS_DATASETS:
        return _load_ts_csv(filename, max_rows=max_rows), "ts"

    if filename in files:
        return _load_pems_or_npz(filename, max_rows=max_rows), "pems"

    raise ValueError(
        f"Unknown dataset '{filename}'. "
        f"Supported TS: {TS_DATASETS}; PEMS keys: {list(files.keys())}"
    )


def Data_load(args):
    timesteps_input = args.timesteps_input
    timesteps_output = args.timesteps_output

    raw_data, dataset_type = _load_raw_data(args)
    num_nodes = raw_data.shape[1]
    input_features = raw_data.shape[2]

    print("==============================")
    print("Dataset:", args.filename)
    print("Dataset type:", dataset_type)
    print("Tensor shape:", raw_data.shape)
    print("Nodes:", num_nodes)
    print("Features:", input_features)
    print("Graph type:", getattr(args, "graph_type", "none"))
    print("==============================")

    data_length = raw_data.shape[0]
    index_1 = int(data_length * 0.8)
    index_2 = int(data_length * 0.9)

    train_raw = raw_data[:index_1]
    adj_np = build_graph(args, train_raw, dataset_type)
    adj = torch.from_numpy(adj_np).float() if adj_np is not None else None

    if adj is not None:
        print(f"Adjacency shape: {tuple(adj.shape)}, non-zero ratio: {(adj > 0).float().mean():.4f}")
    else:
        print("Adjacency: None")

    data, data_mean, data_std = Z_Score(raw_data.copy())

    train_input, train_target = generate_dataset(data[:index_1], timesteps_input, timesteps_output)
    eval_input, eval_target = generate_dataset(data[index_1:index_2], timesteps_input, timesteps_output)
    test_input, test_target = generate_dataset(data[index_2:], timesteps_input, timesteps_output)

    data_set = {
        "train_input": train_input,
        "train_target": train_target,
        "eval_input": eval_input,
        "eval_target": eval_target,
        "test_input": test_input,
        "test_target": test_target,
        "data_mean": data_mean,
        "data_std": data_std,
        "input_features": input_features,
        "num_nodes": num_nodes,
        "adj": adj,
        "dataset_type": dataset_type,
        "graph_type": getattr(args, "graph_type", "none"),
    }
    return data_set