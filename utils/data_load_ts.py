# encoding utf-8
import numpy as np
import pandas as pd
from utils.utils import *
from torch.utils.data import DataLoader

def Data_load(args):
    filename = args.filename
    timesteps_input = args.timesteps_input
    timesteps_output = args.timesteps_output

    filepath = "./data_set/TSdata/"
    if filename in [
        'electricity',
        'traffic',
        'weather',
        'ETTm1',
        'ETTm2',
        'ETTh1',
        'ETTh2',
        'exchange'
    ]:
        data = pd.read_csv(filepath + filename + ".csv", nrows=8640)

        data = data.select_dtypes(
            include=[np.number]
        )

        data = data.values.astype(
            np.float32
        )

        if len(data.shape) == 2:
            if filename in [
                'electricity',
                'traffic'
            ]:
                data = np.expand_dims(
                    data,
                    axis=-1
                )
            else:
                data = np.expand_dims(
                    data.T,
                    axis=-1
                )
                data = data.transpose(
                    1,0,2
                )
    else:
        # =================================================
        # 原始代码:
        # PEMS/METRLA npz读取
        # =================================================
        file = files[filename]
        data = np.load(filepath + file[0])['data'][:8640].astype(np.float32)
        if len(data.shape) == 2:
            data = np.expand_dims(
                data,
                axis=-1
            )
    # 统一获得 N,C
    data_cp = data.copy()
    input_features = data_cp.shape[2]
    num_nodes = data_cp.shape[1]

    print("==============================")
    print("Dataset:",filename)
    print("Tensor shape:",data.shape)
    print("Nodes:",num_nodes)
    print("Features:",input_features)
    print("==============================")
    # 标准化
    data, data_mean, data_std = Z_Score(data)
    data_length = data.shape[0]
    index_1 = int(data_length * 0.8)
    index_2 = int(data_length * 0.9)
    train_original_data = data[:index_1]
    val_original_data = data[index_1: index_2]
    test_original_data = data[index_2:]

    train_input, train_target = generate_dataset(train_original_data, timesteps_input, timesteps_output)
    evaluate_input, evaluate_target = generate_dataset(val_original_data, timesteps_input, timesteps_output)
    test_input, test_target = generate_dataset(test_original_data, timesteps_input, timesteps_output)

    data_set = {}
    data_set['train_input']=train_input
    data_set['train_target']=train_target
    data_set['eval_input']=evaluate_input
    data_set['eval_target']=evaluate_target
    data_set['test_input']=test_input
    data_set['test_target']=test_target
    data_set['data_mean']=data_mean
    data_set['data_std']=data_std
    data_set['input_features']=input_features
    data_set['num_nodes']=num_nodes

    return data_set