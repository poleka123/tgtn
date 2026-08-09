# methods/evaluate.py
import os
import numpy as np
import torch
from utils.utils import RMSE, MAE, SMAPE, Un_Z_Score
from methods.forward import model_forward


def _compute_metrics(pred_index, target_index, criterion, mean, std):
    """对单个时间步计算反标准化后的 loss 与 MAE/RMSE/sMAPE。"""
    pred_index = Un_Z_Score(pred_index, mean, std)
    target_index = Un_Z_Score(target_index, mean, std)
    loss = criterion(pred_index, target_index)
    return {
        'loss': loss,
        'MAE': MAE(target_index, pred_index),
        'RMSE': RMSE(target_index, pred_index),
        'sMAPE': SMAPE(target_index, pred_index),
        'pred': pred_index,
        'target': target_index,
    }


def evaluate(model, data_set, criterion, device, epoch, time_slice,
             results_dir, save_preds=False):
    """
    在验证集上评估（全量 forward，与当前 main 一致）。

    Returns
    -------
    val_loss : list[Tensor]   每个 time step 的 MSE
    val_index : dict          MAE / RMSE / sMAPE 列表
    """
    model.eval()
    std = torch.tensor(data_set['data_std']).to(device)
    mean = torch.tensor(data_set['data_mean']).to(device)

    eval_input = data_set['eval_input'].to(device)
    eval_target = data_set['eval_target'].to(device)

    with torch.no_grad():
        pred = model_forward(model, eval_input, data_set, device)

    val_loss = []
    val_index = {'MAE': [], 'RMSE': [], 'sMAPE': []}

    if save_preds and not os.path.exists(results_dir):
        os.makedirs(results_dir)

    for item in range(1, time_slice + 1):
        pred_step = pred[:, :, item - 1]
        target_step = eval_target[:, :, item - 1]
        metrics = _compute_metrics(pred_step, target_step, criterion, mean, std)

        val_loss.append(metrics['loss'])
        val_index['MAE'].append(metrics['MAE'])
        val_index['RMSE'].append(metrics['RMSE'])
        val_index['sMAPE'].append(metrics['sMAPE'])

        # 与 main 一致：epoch>100 且每 50 epoch 保存一次
        if save_preds:
            np.savetxt(
                os.path.join(results_dir, f"pred_{epoch}.csv"),
                metrics['pred'].cpu().numpy(),
                delimiter=',',
            )
            np.savetxt(
                os.path.join(results_dir, f"true_{epoch}.csv"),
                metrics['target'].cpu().numpy(),
                delimiter=',',
            )

    return val_loss, val_index


def should_save_preds(epoch):
    """是否与 main 中相同的 pred 保存条件。"""
    return (epoch + 1) % 50 == 0 and epoch != 0 and epoch > 100


def log_eval_metrics(epoch, epochs, train_loss, val_loss, val_index,
                     time_slice, elogger=None, time_stride=5):
    """打印并写入日志（从 main 抽出的重复代码）。"""
    sep = "---------------------------------------------------------------------------------------------------"
    print(sep)
    print(f"epoch: {epoch}/{epochs}")
    print(f"Training loss: {train_loss}")

    if elogger is not None:
        elogger.log(f"Epoch:{epoch}")
        elogger.log(f"Training loss: {train_loss}")

    n = time_slice
    for i in range(1, n + 1):
        idx = -(n - i)
        msg = (
            f"time:{i * time_stride}, Evaluation loss:{val_loss[idx]}, "
            f"MAE:{val_index['MAE'][idx]}, RMSE:{val_index['RMSE'][idx]}, "
            f"sMAPE:{val_index['sMAPE'][idx]}"
        )
        print(msg)
        if elogger is not None:
            elogger.log(msg)

    if elogger is not None:
        elogger.log("-----------")
    print(sep)