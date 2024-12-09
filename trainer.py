import datetime
import json
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from evaluation.evaluate import evaluate_model,evaluate_model_Two,evaluate_model_Germany_Two,evaluate_model_Germany
from evaluation.metrics import compile_metrics,compile_metrics_Germany
from paths import *
from utils import save_evaluation_results_for_args
from torch.utils.tensorboard import SummaryWriter
from einops import rearrange
import torch.nn.functional as F


__all__ = ['NIMSTrainer','NIMSTrainer_Two','NIMSTrainer_Germany_Two']
    
# two-frame
class NIMSTrainer_Germnay_Two:
    """
    Provides functionality regarding training including save/loading models and experiment configurations.
    """
    def __init__(self, wandb, model, criterion, dice_criterion, optimizer, scheduler, device, train_loader, valid_loader, test_loader,
                 experiment_name, args, normalization=None):
        self.args = args
        self.wandb = wandb
        self.model = model
        self.model_name = args.model
        self.custom_name = args.custom_name
        self.criterion = criterion
        self.dice_criterion = dice_criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.num_epochs = args.num_epochs

        self.num_classes = args.num_classes
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader

        self.experiment_name = experiment_name
        self.log_dir = args.log_dir
        self.normalization = normalization
        self.rain_thresholds = args.rain_thresholds

        self.trained_weight = {'model': None,
                               'model_name': args.model,
                               'n_blocks': args.n_blocks,  # U-Net related
                               'start_channels': args.start_channels,  # U-Net related
                               'no_residual': args.no_residual,  # U-Net related
                               'no_skip': args.no_skip,  # U-Net related
                               'use_tte': args.use_tte,  # U-Net related
                               'num_layers': args.num_layers,  # ConvLSTM related
                               'hidden_dim': args.hidden_dim,  # ConvLSTM related
                               'kernel_size': args.kernel_size,  # ConvLSTM related
                               'start_dim': args.start_dim,  # MetNet related
                               'window_size': args.window_size,
                               'model_utc': args.model_utc,
                               'dry_sampling_rate': args.dry_sampling_rate,
                               'global_sampling_rate': args.global_sampling_rate,
                               'num_epochs': args.num_epochs,
                               'batch_size': args.batch_size,
                               'optimizer': args.optimizer,
                               'lr': args.lr,
                               'wd': args.wd,
                               'custom_name': args.custom_name,
                               'rain_thresholds': args.rain_thresholds,
                               'num_classes': args.num_classes,
                               'norm_max': normalization['max_values'] if normalization else None,
                               'norm_min': normalization['min_values'] if normalization else None}

        self.device_idx = int(args.device)
        self.model.to(self.device)

        # Hotfix - intermediate evaluation on test set
        self.intermediate_test = args.intermediate_test
        if args.intermediate_test:
            print("Intermediate evaluation on test set is enabled")
        else:
            print("Omitting intermediate evaluation on test set")

        self.log_dict = {
            'complete': False,
            'history': [],
        }
        self.log_json_path = get_train_log_json_path(self.experiment_name, makedirs=True)
        self.log_txt_path = get_train_log_txt_path(self.experiment_name, makedirs=True)
        self.history_path = get_train_log_history_path(self.experiment_name, makedirs=True)
        # self.writer = SummaryWriter(log_dir=self.log_dir,flush_secs=60)

    def save_trained_weight(self, epoch: int = None, step: int = None):
        """
        Save trained weights and experiment configurations to appropriate path.
        """
        if sum([epoch is not None, step is not None]) != 1:
            raise ValueError('Only one of `epoch` or `step` must be specified to save model')
        trained_weight_path = get_trained_model_path(self.experiment_name, epoch=epoch, step=step, makedirs=True)
        if os.path.isfile(trained_weight_path):
            os.remove(trained_weight_path)
        torch.save(self.trained_weight, trained_weight_path)

    def train(self):
        """
        Train the model by `self.num_epochs`.
        """
        self.model.train()
        val_best_csi = 0.0
        val_best_loss = float('inf')
        val_best_epoch = 0
        early_stop_count = 0
        test_best_summary = " "

        for epoch in range(1, self.num_epochs + 1):
            epoch_log_dict = dict()
            epoch_log_dict['epoch'] = epoch

            epoch_log = 'Epoch {:3d} / {:3d} (GPU {})'.format(epoch, self.num_epochs, self.device_idx)
            # print(epoch_log.center(40).center(80, '='))
            self._log(epoch_log.center(40).center(80, '='))

            # Run training epoch
            train_loss, train_loss_cls, train_loss_reg, confusion, metrics, metrics_t2p, metrics_ts = self._epoch(self.train_loader, mode='train')
            correct = confusion[np.diag_indices_from(confusion)].sum()
            accuracy = (correct / confusion.sum()).item()
            epoch_log_dict['loss'] = train_loss
            epoch_log_dict['loss_cls'] = train_loss_cls
            epoch_log_dict['loss_reg'] = train_loss_reg
            epoch_log_dict['accuracy'] = accuracy
            self._log('epoch train confusion:')
            self._log(np.array2string(confusion, separator=', '))

            # Save model
            self.trained_weight['model'] = self.model.state_dict()
            self.save_trained_weight(epoch=epoch, step=None)

            # Save/log train evaluation metrics
            train_csi, summary = save_evaluation_results_for_args(self.wandb, confusion, metrics, metrics_t2p, metrics_ts, epoch, self.args, "train", thresholds=self.rain_thresholds, loss=train_loss, loss_cls=train_loss_cls, loss_reg=train_loss_reg)
            self._log('Train Metrics '.center(40).center(80, '#'))
            self._log(summary)

            self.log_dict['history'].append(epoch_log_dict)

            # Validation evaluation
            _, val_loss, val_loss_cls, val_loss_reg, confusion, metrics, metrics_t2p, metrics_ts = evaluate_model_Germany_Two(self.model, self.valid_loader, self.args.rain_thresholds,
                                                         self.criterion, self.device, self.normalization)
            val_csi, summary = save_evaluation_results_for_args(self.wandb, confusion, metrics, metrics_t2p, metrics_ts, epoch, self.args, "val", thresholds=self.rain_thresholds, loss=val_loss, loss_cls=val_loss_cls, loss_reg=val_loss_reg)
            self._log('Validation Metrics '.center(40).center(80, '#'))
            self._log(summary)
            self._log('epoch valid confusion:')
            self._log(np.array2string(confusion, separator=', '))

            # Test evaluation
            _, test_loss, test_loss_cls, test_loss_reg, confusion, metrics, metrics_t2p, metrics_ts = evaluate_model_Germany_Two(self.model, self.test_loader, self.args.rain_thresholds,
                                                         self.criterion, self.device, self.normalization)
            test_csi, summary = save_evaluation_results_for_args(self.wandb, confusion, metrics, metrics_t2p, metrics_ts, epoch, self.args, "test", thresholds=self.rain_thresholds, loss=test_loss, loss_cls=test_loss_cls, loss_reg=test_loss_reg)
            self._log('Test Metrics '.center(40).center(80, '#'))
            self._log(summary)
            self._log('epoch test confusion:')
            self._log(np.array2string(confusion, separator=', '))

            # # CSI为优化指标
            # if val_csi >= val_best_csi:
            #     val_best_csi = val_csi
            #     val_best_epoch = epoch
            #     test_best_summary = summary
            # loss为优化指标
            if val_loss <= val_best_loss:
                val_best_epoch = epoch
                test_best_summary = summary
                if val_best_loss-val_loss< self.wandb.config['tolerate_loss']:
                    early_stop_count += 1
                else:
                    early_stop_count = 0
                val_best_loss = val_loss
            else:
                early_stop_count += 1

            with open(self.log_json_path, 'w') as f:
                json.dump(self.log_dict, f)

            history = pd.DataFrame(self.log_dict['history'])
            history.to_csv(self.history_path)

            if self.scheduler:
                self.scheduler.step(val_loss_reg)
            
            # self.writer.add_scalars("{} ".format(self.custom_name), {
            # "train": train_loss,
            # "valid": val_loss,
            # }, epoch)
                
            if early_stop_count >= self.wandb.config['early_stop']:
                break
            # wandb epoch logging
            ############################################################################################
            self.wandb.log({'epoch': epoch, 
                    'lr': (self.optimizer.param_groups[0])['lr'], 
                    'train_loss_reg': train_loss_reg, 
                    'train_loss_cls': train_loss_cls, 
                    'val_loss_reg': val_loss_reg, 
                    'val_loss_cls': val_loss_cls, 
                    'test_loss_reg': test_loss_reg, 
                    'test_loss_cls': test_loss_cls,
                    })
            ###########################################################################################
            
        self._log('Best Val Performance on Test'.center(40).center(80, '#'))
        self._log(str(val_best_epoch))
        self._log(test_best_summary)
            
        # Save history
        self.log_dict['complete'] = True
        with open(self.log_json_path, 'w') as f:
            json.dump(self.log_dict, f)

        self._log('Log files saved to the following paths:')
        self._log(self.log_txt_path)
        self._log(self.log_json_path)
        self._log(self.history_path)


    def _log(self, text):
        # print(text)
        with open(self.log_txt_path, 'a') as f:
            f.write(text + '\n')

    def _epoch(self, data_loader, mode):
        """
        Run a single epoch of inference or training (`mode="train"`) based on the supplied `mode`.
        :param data_loader:
        :param mode: "train" | "eval"
        """
        # pbar = tqdm(data_loader)
        total_loss = 0
        total_loss_cls = 0
        total_loss_reg = 0
        total_samples = 0

        confusion_matrix = np.zeros((self.num_classes, self.num_classes), dtype=int)
        metrics_by_threshold = defaultdict(list)  # metrics_by_threshold[threshold][step]: DataFrame
        metrics_by_threshold_t2p = defaultdict(list) 
        metrics_by_threshold_ts = defaultdict(list)
        for i, (images, target_cls, target_reg) in enumerate(data_loader):
            # Apply normalizations
            if self.normalization:
                with torch.no_grad():
                    for i, (max_val, min_val) in enumerate(zip(self.normalization['max_values'],
                                                               self.normalization['min_values'])):
                        if min_val < 0:
                            images[:, :, i, :, :] = images[:, :, i, :, :] / max(-min_val, max_val)
                        else:
                            images[:, :, i, :, :] = (images[:, :, i, :, :] - min_val) / (max_val - min_val)

            images = images.type(torch.FloatTensor).to(self.device)
            target_cls = target_cls.type(torch.LongTensor).to(self.device)
            target_reg = target_reg.type(torch.LongTensor).to(self.device)
            # print(target_cls.shape, target_reg.shape)
            # Modify
            if self.args.model=='STGAT':
                output,output2 = self.model(images, target_reg)
            else:
                output,output2 = self.model(images, target_reg)
            output2 = F.relu(output2)

            # print(output.shape, output2.shape)
            loss, loss_cls, loss_reg, pred_labels, target_labels = self.criterion(output, output2, target_cls, target_reg, mode=mode)
            if self.dice_criterion !=None:
                loss += self.dice_criterion(pred_labels, target_labels, self.device)
            _, predictions = output.detach().cpu().topk(1, dim=1, largest=True,
                                                        sorted=True)  # (batch_size, height, width)
            step_confusion, step_metrics_by_threshold, step_metrics_by_threshold_t2p, step_metrics_by_threshold_ts= compile_metrics_Germany(data_loader.dataset, predictions.numpy(),
                                                                        target_reg.detach().cpu().numpy(), self.args.rain_thresholds)
            confusion_matrix += step_confusion
            for threshold, metrics in step_metrics_by_threshold.items():
                metrics_by_threshold[threshold].append(metrics)

            for threshold, metrics in step_metrics_by_threshold_t2p.items():
                metrics_by_threshold_t2p[threshold].append(metrics)
            
            for threshold, metrics in step_metrics_by_threshold_ts.items():
                metrics_by_threshold_ts[threshold].append(metrics)

            if loss is None:
                continue

            total_loss += loss.item() * images.shape[0]
            total_loss_cls += loss_cls.item() * images.shape[0]
            total_loss_reg += loss_reg.item() * images.shape[0]
            total_samples += images.shape[0]

            # Apply backprop
            if mode == 'train':
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        # Collate evaluation results
        metrics_by_threshold = {t: pd.concat(metrics) for t, metrics in metrics_by_threshold.items()}
        metrics_by_threshold_t2p = {t: pd.concat(metrics) for t, metrics in metrics_by_threshold_t2p.items()}
        metrics_by_threshold_ts = {t: pd.concat(metrics) for t, metrics in metrics_by_threshold_ts.items()}
        # loss of one grid of one sample
        average_loss = total_loss / total_samples
        average_loss_cls = total_loss_cls / total_samples
        average_loss_reg = total_loss_reg / total_samples

        return average_loss, average_loss_cls, average_loss_reg, confusion_matrix, metrics_by_threshold, metrics_by_threshold_t2p, metrics_by_threshold_ts
