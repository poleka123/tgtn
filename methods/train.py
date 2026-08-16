# methods/train.py
import torch
from utils.utils import Un_Z_Score
from methods.forward import model_forward


def train_one_epoch(model, train_loader, optimizer, criterion, data_set, device):
    model.train()
    epoch_losses = []
    std = torch.tensor(data_set["data_std"]).to(device)
    mean = torch.tensor(data_set["data_mean"]).to(device)

    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()

        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        pred = model_forward(model, X_batch, data_set, device)
        pred, y_batch = Un_Z_Score(pred, mean, std), Un_Z_Score(y_batch, mean, std)
        loss = criterion(pred, y_batch)

        loss.backward()
        optimizer.step()
        epoch_losses.append(loss.detach().cpu().item())

    return sum(epoch_losses) / len(epoch_losses)
    
# methods/train.py — train_one_epoch 函数

# def train_one_epoch(model, train_loader, optimizer, criterion, data_set, device):
#     model.train()
#     epoch_losses = []

#     for X_batch, y_batch in train_loader:
#         optimizer.zero_grad()

#         X_batch = X_batch.to(device)
#         y_batch = y_batch.to(device)

#         pred = model_forward(model, X_batch, data_set, device)
#         # ---------- 修改：直接用标准化后的数据计算 Loss ----------
#         loss = criterion(pred, y_batch)

#         loss.backward()
#         torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
#         optimizer.step()
#         epoch_losses.append(loss.detach().cpu().item())

#     return sum(epoch_losses) / len(epoch_losses)