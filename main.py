import math
import argparse
import datetime
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os

from data.loader import Data_load
__all__ = ["Data_load"]
from utils.utils import DatasetPEMS, learning_rate_with_decay
from model.registry import build_model
from methods.train import train_one_epoch
from methods.evaluate import evaluate, should_save_preds, log_eval_metrics
import logger

if torch.cuda.is_available():
    torch.cuda.current_device()
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

parser = argparse.ArgumentParser()
parser.add_argument('--filename', type=str, default='electricity',
                    help='TS: electricity/traffic/weather/ETTm1/exchange/solar_AL...; PEMS: pems08/pems04/...')
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--epochs', type=int, default=300)
parser.add_argument('--timesteps_input', type=int, default=48)
parser.add_argument('--timesteps_output', type=int, default=1)
parser.add_argument('--nhid', type=int, default=32,
                    help='number of hidden units per layer (default: 32)')
parser.add_argument('--tucker_rank_nodes', type=int, default=None)
parser.add_argument('--tucker_rank_time', type=int, default=None, help='Tucker time rank (default: half of timesteps_input)')
parser.add_argument('--tucker_rank_features', type=int, default=16)
parser.add_argument('--tucker_rank_nodes_ratio', type=float, default=0.6,
                    help='ratio to auto-derive rank_nodes from num_nodes (default: 0.3)')
parser.add_argument('--ode_layers', type=int, default=3)
parser.add_argument('--ode_time', type=float, default=1.0)
parser.add_argument('--ode_solver', type=str, default='rk4',
                    choices=['euler', 'midpoint', 'rk4', 'dopri5'])
parser.add_argument('--ode_euler_steps', type=int, default=4)
parser.add_argument('--time_slice', type=int, default=24)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--lr_milestones', type=int, nargs='+', default=[50, 200],
                    help='epoch boundaries for lr decay')
parser.add_argument('--lr_decay_rates', type=float, nargs='+', default=[0.1, 0.1],
                    help='multiplicative decay at each milestone (方案 A)')


# Step 3: 图结构参数
parser.add_argument('--graph_type', type=str, default='correlation',
                    choices=['none', 'identity', 'correlation', 'distance'],
                    help='adjacency construction method')
parser.add_argument('--graph_threshold', type=float, default=0.3,
                    help='correlation graph threshold')
parser.add_argument('--graph_sigma', type=float, default=0.1,
                    help='distance graph sigma')
parser.add_argument('--graph_thres', type=float, default=0.5,
                    help='distance graph threshold')
parser.add_argument('--max_rows', type=int, default=8640,
                    help='max time steps loaded from raw data 8640 2160 c 1440 720 2880, 4320')

parser.add_argument('--model_name', type=str, default='tgtn_tucker422')
parser.add_argument('--model', type=str, default='tgtn',
                    choices=['tuckergode', 'gcn', 'gat', 'dcrnn','mtgnn','stgcn','tuckergode_v2','tgtn', 'stode','sttgm', 'gru'],help='forecast model to train')
parser.add_argument('--mae_threshold', type=float, default=None,
                    help='只有当 avg_mae < threshold 时才保存预测结果')
args = parser.parse_args()


if __name__ == '__main__':
    suffix = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    torch.manual_seed(7)

    elogger = logger.Logger(
        f'test_run_log_{args.model_name}_batch_{args.batch_size}'
        f'_epochs_{args.epochs}'
        f'_dataset_{args.filename}'
        f'{suffix}'
    )

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    data_set = Data_load(args)

    train_loader = DataLoader(
        DatasetPEMS([data_set['train_input'], data_set['train_target']]),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=1,
        drop_last=True,
    )
    args._adj = data_set['adj']
    model = build_model(
        model_name=args.model,
        args=args,
        data_set=data_set,
        output_features=1,
    ).to(device)
    batches_per_epoch = math.floor(data_set['train_input'].shape[0] / args.batch_size)
    lr_fn = learning_rate_with_decay(
        args,
        args.batch_size,
        batch_denom=args.batch_size,
        batches_per_epoch=batches_per_epoch,
        boundary_epochs=args.lr_milestones,
        decay_rates=args.lr_decay_rates,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()
    results_dir = f"./results/{args.filename}/run_log_{args.model}_wotucker_{suffix}"
    # 第 1 处修改：最优模型跟踪变量初始化
    best_val_loss = float('inf')
    best_epoch = 0
    model_dir = f"./checkpoints/{args.filename}"
    os.makedirs(model_dir, exist_ok=True)

    for epoch in range(args.epochs):
        current_lr = lr_fn(epoch)
        for param_group in optimizer.param_groups:
            param_group['lr'] = current_lr
        print(f"Epoch {epoch}, lr={current_lr}")
        elogger.log(f"Epoch {epoch}, lr={current_lr}")

        print("################################################")
        print("Train Process")
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, data_set, device
        )

        torch.cuda.empty_cache()
        print("Evaluation Process")
        val_loss, val_index = evaluate(
            model=model,
            data_set=data_set,
            criterion=criterion,
            device=device,
            epoch=epoch,
            time_slice=args.time_slice,
            results_dir=results_dir,
            save_preds=should_save_preds(epoch),
            # mae_threshold=args.mae_threshold,
        )
        # save_preds=should_save_preds(epoch),
        log_eval_metrics(
            epoch=epoch,
            epochs=args.epochs,
            train_loss=train_loss,
            val_loss=val_loss,
            val_index=val_index,
            time_slice=args.time_slice,
            elogger=elogger,
            time_stride=5,
        )
       # 第 2 处修改：最优模型保存
        current_val_loss = val_loss[-1].item() if isinstance(val_loss[-1], torch.Tensor) else val_loss[-1]
        if current_val_loss < best_val_loss:
            best_val_loss = current_val_loss
            best_epoch = epoch
            best_model_path = os.path.join(model_dir, f"{args.model}_{suffix}_best.pth")
            torch.save(model.state_dict(), best_model_path)
            print(f"*** 新的最优模型已保存 (epoch={epoch}, val_loss={best_val_loss:.6f}) ***")
            elogger.log(f"Best model updated at epoch {epoch}, val_loss={best_val_loss:.6f}")
    # 第 3 处修改：训练结束后保存最终模型
    final_model_path = os.path.join(model_dir, f"{args.model}_{suffix}_final.pth")
    torch.save(model.state_dict(), final_model_path)
    print(f"\n训练完成！最优模型: epoch={best_epoch}, val_loss={best_val_loss:.6f}")
    print(f"最终模型已保存至: {final_model_path}")
    elogger.log(f"Training complete. Best: epoch={best_epoch}, val_loss={best_val_loss:.6f}")