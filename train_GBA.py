import numpy as np
import random
from torch.utils.data import DataLoader
from torchvision import transforms
import torch
import time
from time import gmtime, strftime
import os
from data.data_split import cyclic_split
from data.dataset import get_dataset_class, CustomTensorDataset_GBA, CustomTensorDataset_GBA_gap, CustomTensorDataset_GBA_seq, CustomTensorDataset_GBA_seq_gap
from data.transforms import ClassifyByThresholds
from trainer import NIMSTrainer_Germnay_Two
from utils import *
import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=RuntimeWarning)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def main(args):
    device = set_device(args)
    fix_seed(args.seed)
    g = torch.Generator()
    g.manual_seed(args.seed)
    

    # Set experiment name and use it as process name if possible
    experiment_name = get_experiment_name(args)
    current_time = strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))
    args.experiment_name = experiment_name+ "_" + current_time
    experiment_name = args.experiment_name
    
    print('Running Experiment'.center(30).center(80, "="))
    print(experiment_name)

    print("Using date intervals")
    print("#" * 80)
    for start, end in args.date_intervals:
        print("{} - {}".format(start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")))
    print('Start to load datasets......')
    

    save_path = '/home/jianer/PostRainBench/3_GBA_wandb_ConvLSTM/GBA_dataset/experiment/'
    # trn_x_1 = np.load(save_path + 'X_train_period1.npy')
    # trn_x_2= np.load(save_path + 'X_train_period2.npy')
    # trn_y_1 = np.load(save_path + 'y_train_period1.npy')
    # trn_y_2 = np.load(save_path + 'y_train_period2.npy')

    # tst_x = np.load(save_path + 'X_test_period.npy')
    # tst_y = np.load(save_path + 'y_test_period.npy')
    # vld_x = np.load(save_path + 'X_valid_period.npy')
    # vld_y = np.load(save_path + 'y_valid_period.npy')

    # 使用内存映射
    trn_x_1 = np.load(save_path + 'X_train_period1.npy', mmap_mode='r')
    trn_x_2 = np.load(save_path + 'X_train_period2.npy', mmap_mode='r')
    trn_y_1 = np.load(save_path + 'y_train_period1.npy', mmap_mode='r')
    trn_y_2 = np.load(save_path + 'y_train_period2.npy', mmap_mode='r')
    tst_x = np.load(save_path + 'X_test_period.npy', mmap_mode='r')
    tst_y = np.load(save_path + 'y_test_period.npy', mmap_mode='r')
    vld_x = np.load(save_path + 'X_valid_period.npy', mmap_mode='r')
    vld_y = np.load(save_path + 'y_valid_period.npy', mmap_mode='r')
    
    print('Load datasets in CPU memory successfully!')
    print("#" * 80)

    batch_size = args.batch_size
    train_dataset = CustomTensorDataset_GBA_gap(torch.from_numpy(trn_x_1),torch.from_numpy(trn_x_2), \
                                                    torch.from_numpy(trn_y_1), torch.from_numpy(trn_y_2), \
                                                    args.rain_thresholds, downscaling_t=4) #, sequence_length=args.seq_length)
    val_dataset = CustomTensorDataset_GBA(torch.from_numpy(vld_x),torch.from_numpy(vld_y), args.rain_thresholds, downscaling_t=4) #, sequence_length=args.seq_length)
    test_dataset = CustomTensorDataset_GBA(torch.from_numpy(tst_x),torch.from_numpy(tst_y), args.rain_thresholds, downscaling_t=4) #, sequence_length=args.seq_length)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    valid_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    for x, y, z in test_dataset:
        print(f'x shape: {x.shape}')
        print(f'y shape: {y.shape}')
        print(f'z shape: {z.shape}')
        break  # 打印一次后跳出循环
    for x, y, z in test_loader:
        print(f'x shape: {x.shape}')
        print(f'y shape: {y.shape}')
        print(f'z shape: {z.shape}')
        break  # 打印一次后跳出循环

    nwp_sample = torch.rand(1, args.seq_length*4, 84, 64, 64)
    model, criterion, dice_criterion = set_model(nwp_sample, device, args)
    normalization = None
    # Train model
    optimizer, scheduler = set_optimizer(model, args)
    wandb = None
    if args.use_two:
        nims_trainer = NIMSTrainer_Germnay_Two(wandb, model, criterion, dice_criterion, optimizer, scheduler, device,
                                train_loader, valid_loader, test_loader, experiment_name,
                                args, normalization=normalization)
    else:
        nims_trainer = NIMSTrainer_Germany(model, criterion, dice_criterion, optimizer, scheduler, device,
                        train_loader, valid_loader, test_loader, experiment_name,
                        args, normalization=normalization)
        
    nims_trainer.train()


if __name__ == '__main__':
    args = parse_args()
    main(args)
