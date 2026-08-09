import math
import os
import argparse
import datetime
import torch
import torch.nn as nn
from utils.data_load_ts import Data_load
from utils.utils import *
from methods.train import Train
from methods.evaluate import Evaluate
from torch.utils.data import DataLoader
import logger

from model.tucker_gode import TensorGODEForecast
import numpy as np

if torch.cuda.is_available():
    torch.cuda.current_device()
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

parser = argparse.ArgumentParser()
parser.add_argument('--filename', type=str, default='weather', help='electricity, traffic, weather, ETTm1, ETTm2, ETTh1, ETTh2, ILI exchange')
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--epochs', type=int, default=300)
parser.add_argument('--timesteps_input', type=int, default=24)
parser.add_argument('--timesteps_output', type=int, default=24)
parser.add_argument('--nhid', type=int, default=32, help='number of hidden units per layer (default: 32)')
parser.add_argument('--tucker_rank_nodes', type=int, default=32)
parser.add_argument('--tucker_rank_time', type=int, default=6)
parser.add_argument('--tucker_rank_features', type=int, default=4)
parser.add_argument('--ode_layers', type=int, default=2)
parser.add_argument('--ode_time', type=float, default=1.0)
parser.add_argument('--ode_solver', type=str, default='rk4', choices=['euler', 'midpoint', 'rk4', 'dopri5'])
parser.add_argument('--ode_euler_steps', type=int, default=4)
parser.add_argument('--time_slice', type=int, default=24)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--model_name', type=str, default='ablation_model')

args = parser.parse_args()


if __name__ == '__main__':
    suffix = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    torch.manual_seed(7)
    elogger = logger.Logger(
        f'test_run_log_tuckergode_batch_{args.batch_size}'
        f'_epochs_{args.epochs}'
        f'_dataset_{args.filename}'
        f'{suffix}'
    )
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    data_set = Data_load(args)
    # generate data_loader
    train_setting = [data_set['train_input'], data_set['train_target']]
    train_dataset = DatasetPEMS(train_setting)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=1, drop_last=True)

    eval_setting = [data_set['eval_input'], data_set['eval_target']]
    eval_dataset = DatasetPEMS(eval_setting)
    eval_loader = DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=True, num_workers=1, drop_last=True)

    test_setting = [data_set['test_input'], data_set['test_target']]
    test_dataset = DatasetPEMS(test_setting)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, num_workers=1, drop_last=True)

    num_of_nodes = data_set['num_nodes']
    input_features = data_set['input_features']
    output_features = 1;
    model = TensorGODEForecast(
        num_nodes=num_of_nodes,
        num_features=input_features,
        output_features = output_features,
        num_timesteps_input=args.timesteps_input,
        num_timesteps_output=args.timesteps_output,
        hidden_dim=args.nhid,
        rank_nodes=64,
        rank_time=24,
        rank_features=args.tucker_rank_features,
        num_ode_layers=args.ode_layers,
        ode_time=args.ode_time,
        ode_solver=args.ode_solver,
        euler_steps=args.ode_euler_steps,
    ).to(device)
    # init change lr fucntion
    batches_per_epoch = math.floor(data_set['train_input'].shape[0]/args.batch_size)
    lr_fn = learning_rate_with_decay(args, args.batch_size, batch_denom=args.batch_size,
                                     batches_per_epoch=batches_per_epoch, boundary_epochs=[50,200], decay_rates=[1,0.1, 0.1])

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    L2 = nn.MSELoss()

    for epoch in range(args.epochs):
        lr = lr_fn(epoch)
        # change learning rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr_fn(epoch)
        print(
            f"Epoch {epoch}, lr={optimizer.param_groups[0]['lr']}"
        )
        print("################################################s")
        print("Train Process")
        permutation = torch.randperm(data_set['train_input'].shape[0])
        epoch_training_losses = []
        loss_mean = 0.0
        is_best_for_now = False
        # train
        for i, [X_batch, y_batch] in enumerate(train_loader):
            model.train()
            optimizer.zero_grad()
            # indices = permutation[i:i+args.batch_size]
            # X_batch, y_batch = data_set['train_input'][indices], data_set['train_target'][indices]

            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            std = torch.tensor(data_set['data_std']).to(device)
            mean = torch.tensor(data_set['data_mean']).to(device)
            pred = model(X_batch)
            pred, y_batch = Un_Z_Score(pred, mean, std), Un_Z_Score(y_batch, mean, std)
            loss = L2(pred, y_batch)

            # 损失反向传播
            loss.backward()
            optimizer.step()
            epoch_training_losses.append(loss.detach().cpu().numpy())
        loss_mean = sum(epoch_training_losses)/len(epoch_training_losses)


        # test
        torch.cuda.empty_cache()
        with torch.no_grad():
            print("Evalution Process")
            model.eval()
            eval_input = data_set['eval_input']
            eval_target = data_set['eval_target']
            eval_input = eval_input.to(device)
            eval_target = eval_target.to(device)
            std = torch.tensor(data_set['data_std']).to(device)
            mean = torch.tensor(data_set['data_mean']).to(device)

            pred = model(eval_input)
            val_index = {}
            val_index['MAE'] = []
            val_index['RMSE'] = []
            val_index['sMAPE'] = []
            val_loss = []

            for item in range(1, args.time_slice+1):
                pred_index = pred[:, :, item - 1]
                val_target_index = eval_target[:, :, item - 1]
                pred_index, val_target_index = Un_Z_Score(pred_index, mean, std), Un_Z_Score(val_target_index, mean,
                                                                                             std)

                loss = L2(pred_index, val_target_index)
                val_loss.append(loss)



                filePath = f"./results/{args.filename}/run_log_tuckergode_{suffix}"
                if not os.path.exists(filePath):
                    os.makedirs(filePath)
                if ((epoch + 1) % 50 == 0) & (epoch != 0) & (epoch > 100):
                    np.savetxt(filePath + "/pred_" + str(epoch) + ".csv", pred_index.cpu(), delimiter=',')
                    np.savetxt(filePath + "/true_" + str(epoch) + ".csv", val_target_index.cpu(), delimiter=',')
                mae = MAE(val_target_index, pred_index)
                val_index['MAE'].append(mae)

                rmse = RMSE(val_target_index, pred_index)
                val_index['RMSE'].append(rmse)

                smape = SMAPE(val_target_index, pred_index)
                val_index['sMAPE'].append(smape)

        print("---------------------------------------------------------------------------------------------------")
        print("epoch: {}/{}".format(epoch, args.epochs))
        print("Training loss: {}".format(loss_mean))
        elogger.log("Epoch:{}".format(epoch))
        elogger.log(f"Training loss: {loss_mean}")
        for i in range(1, args.time_slice+1):
            print("time:{}, Evaluation loss:{}, MAE:{}, RMSE:{}, sMAPE:{}"
                  .format(i * 5, val_loss[-((args.time_slice) - i)],
                          val_index['MAE'][-((args.time_slice) - i)],
                          val_index['RMSE'][-((args.time_slice) - i)],
                          val_index['sMAPE'][-((args.time_slice) - i)]))
            elogger.log("time:{}, Evaluation loss:{}, MAE:{}, RMSE:{}, sMAPE:{}"
                        .format(i * 5, val_loss[-((args.time_slice) - i)],
                                val_index['MAE'][-((args.time_slice) - i)],
                                val_index['RMSE'][-((args.time_slice) - i)],
                                val_index['sMAPE'][-((args.time_slice) - i)]))
        elogger.log("-----------")
        print("---------------------------------------------------------------------------------------------------")
