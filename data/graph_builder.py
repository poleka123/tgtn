# data/graph_builder.py
from __future__ import annotations

import csv
import os
from typing import Optional

import numpy as np

from utils.utils import files, get_normalized_adj


def _get_attr(args, name: str, default):
    return getattr(args, name, default)


def build_identity_adj(num_nodes: int) -> np.ndarray:
    adj = np.eye(num_nodes, dtype=np.float32)
    return get_normalized_adj(adj).astype(np.float32)


def build_correlation_adj(
    train_data: np.ndarray,
    threshold: float = 0.3,
) -> np.ndarray:
    """
    train_data: [T, N, C]，用第 0 个特征算 Pearson 相关图。
    """
    x = train_data[:, :, 0]
    corr = np.corrcoef(x.T)
    corr = np.nan_to_num(corr, nan=0.0)
    adj = np.abs(corr).astype(np.float32)
    np.fill_diagonal(adj, 1.0)
    adj = np.where(adj >= threshold, adj, 0.0).astype(np.float32)
    return get_normalized_adj(adj).astype(np.float32)


def build_distance_adj(args, num_nodes: int) -> Optional[np.ndarray]:
    """
    PEMS 类数据集：从 distance/csv 构建空间图。
    失败时返回 None，由上层 fallback。
    """
    filename = args.filename
    if filename not in files:
        return None

    file_entry = files[filename]
    filepath = "./data_set/"
    dist_csv = filepath + file_entry[1]
    if not os.path.exists(dist_csv):
        return None

    sigma = _get_attr(args, "graph_sigma", 0.1)
    thres = _get_attr(args, "graph_thres", 0.5)

    cache_path = f"./data_set/{filename}_spatial_distance.npy"
    if not os.path.exists(cache_path):
        dist_matrix = np.zeros((num_nodes, num_nodes), dtype=np.float32) + np.inf

        # metr-la 等: from,to,distance
        if len(file_entry) >= 3 and file_entry[2].endswith(".txt"):
            return _build_distance_adj_with_ids(args, num_nodes, sigma, thres)

        with open(dist_csv, "r", encoding="utf-8") as fp:
            reader = csv.reader(fp)
            next(reader, None)  # skip header
            for line in reader:
                if len(line) < 3:
                    continue
                start, end = int(line[0]), int(line[1])
                dist_matrix[start, end] = float(line[2])
                dist_matrix[end, start] = float(line[2])
        np.save(cache_path, dist_matrix)

    dist_matrix = np.load(cache_path)
    finite = dist_matrix[dist_matrix != np.inf]
    if finite.size == 0:
        return None

    dist_matrix = (dist_matrix - np.mean(finite)) / (np.std(finite) + 1e-6)
    sp_matrix = np.exp(-(dist_matrix ** 2) / (sigma ** 2))
    sp_matrix[sp_matrix < thres] = 0.0
    sp_matrix = sp_matrix.astype(np.float32)
    np.fill_diagonal(sp_matrix, 1.0)
    return get_normalized_adj(sp_matrix).astype(np.float32)


def _build_distance_adj_with_ids(args, num_nodes: int, sigma: float, thres: float) -> np.ndarray:
    filename = args.filename
    file_entry = files[filename]
    filepath = "./data_set/"

    with open(filepath + file_entry[2], encoding="utf-8") as f:
        sensor_ids = f.read().strip().split(",")

    sensor_id_to_ind = {sensor_id: i for i, sensor_id in enumerate(sensor_ids)}
    dist_matrix = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    dist_matrix[:] = np.inf

    import pandas as pd
    distance_df = pd.read_csv(filepath + file_entry[1], dtype={"from": "str", "to": "str"})
    for row in distance_df.values:
        if row[0] not in sensor_id_to_ind or row[1] not in sensor_id_to_ind:
            continue
        dist_matrix[sensor_id_to_ind[row[0]], sensor_id_to_ind[row[1]]] = row[2]

    finite = dist_matrix[dist_matrix != np.inf]
    dist_matrix = (dist_matrix - np.mean(finite)) / (np.std(finite) + 1e-6)
    sp_matrix = np.exp(-(dist_matrix ** 2) / (sigma ** 2))
    sp_matrix[sp_matrix < thres] = 0.0
    sp_matrix = sp_matrix.astype(np.float32)
    np.fill_diagonal(sp_matrix, 1.0)
    return get_normalized_adj(sp_matrix).astype(np.float32)


def build_graph(args, train_raw: np.ndarray, dataset_type: str) -> Optional[np.ndarray]:
    """
    根据 args.graph_type 构建邻接矩阵。
    返回 [N, N] numpy；none 时返回 None。
    """
    graph_type = _get_attr(args, "graph_type", "none")
    num_nodes = train_raw.shape[1]

    if graph_type == "none":
        return None

    if graph_type == "identity":
        return build_identity_adj(num_nodes)

    if graph_type == "correlation":
        threshold = _get_attr(args, "graph_threshold", 0.3)
        return build_correlation_adj(train_raw, threshold=threshold)

    if graph_type == "distance":
        adj = build_distance_adj(args, num_nodes)
        if adj is None:
            print("[graph_builder] distance graph failed, fallback to correlation.")
            threshold = _get_attr(args, "graph_threshold", 0.3)
            return build_correlation_adj(train_raw, threshold=threshold)
        return adj

    raise ValueError(f"Unknown graph_type: {graph_type}")