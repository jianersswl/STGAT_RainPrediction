from collections import defaultdict
from datetime import timedelta
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
import torch
from data.dataset import StandardDataset
from torch.utils.data import Subset
import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=RuntimeWarning)

def compile_metrics(dataset: StandardDataset, predictions: np.ndarray, timestamps: List,
                    thresholds: List[float]) -> Tuple[np.ndarray, Dict[float, pd.DataFrame]]:
    """
    :param dataset:
    :param predictions: shape=(batch_size, height, width), dtype=int
    :param timestamps:
    :param thresholds:
    :return: confusion_matrix, binary_metrics_by_threshold

    confusion: (prediction, target)

    pred\target |   false  |   true
    false       |   cn     |   miss
    true        |   fa     |   hit
    """
    assert (predictions.shape[0] == len(timestamps))
    if isinstance(dataset, Subset):
        dataset = dataset.dataset
    if isinstance(predictions, torch.Tensor):
        raise ValueError("`predictions` must be a numpy array")

    n_classes = len(thresholds) + 1
    padded_thresholds = [float('-inf')] + thresholds + [float('inf')]
    confusion_matrix = np.zeros((n_classes, n_classes), dtype=int)
    binary_metrics_by_threshold = defaultdict(lambda: defaultdict(list))
    for (origin, lead_time), prediction in zip(timestamps, predictions):
        local_confusion_matrix = np.zeros((n_classes, n_classes), dtype=int)
        target = dataset.aws_base_dataset.load_array(origin + timedelta(hours=lead_time), 0)
        valid = target != -9999

        # Populate local confusion matrix
        interval_list = zip(padded_thresholds[:-1], padded_thresholds[1:])
        preds_by_interval = []
        targets_by_interval = []
        for i, (start, end) in enumerate(interval_list):
            preds_by_interval.append(prediction == i)
            targets_by_interval.append((start <= target) & (target < end))
        for i, p in enumerate(preds_by_interval):
            for j, t in enumerate(targets_by_interval):
                local_confusion_matrix[i, j] = (valid & p & t).sum()

        # Populate binary metric dict[list]s
        for i, threshold in enumerate(thresholds, start=1):
            metrics = binary_metrics_by_threshold[threshold]
            metrics["origin"].append(origin.strftime("%Y%m%d_%H%M"))
            metrics["lead_time"].append(lead_time)
            metrics["hit"].append(local_confusion_matrix[i:, i:].sum())
            metrics["miss"].append(local_confusion_matrix[:i, i:].sum())
            metrics["fa"].append(local_confusion_matrix[i:, :i].sum())
            metrics["cn"].append(local_confusion_matrix[:i, :i].sum())

        confusion_matrix += local_confusion_matrix

    binary_metrics_by_threshold = {t: pd.DataFrame(metrics) for t, metrics in binary_metrics_by_threshold.items()}
    return confusion_matrix, binary_metrics_by_threshold


def compile_metrics_Germany(dataset: StandardDataset, predictions: np.ndarray, targets: np.ndarray,
                    thresholds: List[float]) -> Tuple[np.ndarray, Dict[float, pd.DataFrame]]:
    """
    :param dataset:
    :param predictions: shape=(batch_size, height, width), dtype=int
    :param timestamps:
    :param thresholds:
    :return: confusion_matrix, binary_metrics_by_threshold

    confusion: (prediction, target)

    pred\target |   false  |   true
    false       |   cn     |   miss
    true        |   fa     |   hit
    """
    # assert (predictions.shape[0] == len(timestamps))
    # if isinstance(dataset, Subset):
    #     dataset = dataset.dataset
    # if isinstance(predictions, torch.Tensor):
    #     raise ValueError("`predictions` must be a numpy array")

    n_classes = len(thresholds) + 1
    padded_thresholds = [float('-inf')] + thresholds + [float('inf')]
    # print(padded_thresholds)
    # 分类类别是行列含义
    confusion_matrix = np.zeros((n_classes, n_classes), dtype=int)
    # 一个batch一个hit miss cn fa，所以需要list
    binary_metrics_by_threshold = defaultdict(lambda: defaultdict(list))
    binary_metrics_by_threshold_t2p = defaultdict(lambda: defaultdict(list))
    binary_metrics_by_threshold_ts = defaultdict(lambda: defaultdict(list))

    # print('new metrics')
    for target, prediction in zip(targets, predictions):
        local_confusion_matrix = np.zeros((n_classes, n_classes), dtype=int)
        # target = dataset.aws_base_dataset.load_array(origin + timedelta(hours=lead_time), 0)
        valid = target != -1

        # Populate local confusion matrix
        interval_list = zip(padded_thresholds[:-1], padded_thresholds[1:])
        preds_by_interval = []
        targets_by_interval = []
        for i, (start, end) in enumerate(interval_list):
            # print(start, end)
            preds_by_interval.append(prediction == i)
            targets_by_interval.append((start <= target) & (target < end))
        # 好像有问题？？？
        # print(predictions)
        for i, p in enumerate(preds_by_interval):
            for j, t in enumerate(targets_by_interval):
                local_confusion_matrix[i, j] = (valid & p & t).sum()
                # if j==2:
                    # print(i, j)
                    # print(np.where(p==True))
                    # print(np.where(t==True))
                    # print((valid & p & t).sum())
                
        # Populate binary metric dict[list]s
        # 三分类这样计算没有问题吗？TS
        for i, threshold in enumerate(thresholds, start=1):
            metrics_ts = binary_metrics_by_threshold_ts[threshold]
            metrics_ts["hit"].append(local_confusion_matrix[i:, i:].sum())
            metrics_ts["miss"].append(local_confusion_matrix[:i, i:].sum())
            metrics_ts["fa"].append(local_confusion_matrix[i:, :i].sum())
            metrics_ts["cn"].append(local_confusion_matrix[:i, :i].sum())
        # 更改版
        # print([0] + thresholds)
        # for i, threshold in enumerate([0.0] + thresholds, start=0):
        #     metrics = binary_metrics_by_threshold[threshold]
        #     metrics_t2p = binary_metrics_by_threshold_t2p[threshold]
        #     if i==0:
        #         metrics["cn"].append(local_confusion_matrix[0, 0].sum())
        #         metrics["miss"].append(local_confusion_matrix[0, 1:].sum())
        #         metrics_t2p["cn"].append(local_confusion_matrix[0, 0].sum())
        #         metrics_t2p["fa"].append(local_confusion_matrix[1:, 0].sum())
        #         # 没意义
        #         metrics_t2p["hit"].append(0)
        #         metrics_t2p["miss"].append(0)
        #         metrics["fa"].append(0)
        #         metrics["hit"].append(0)
        #     if i==1:
        #         metrics["hit"].append(local_confusion_matrix[i, i].sum())
        #         metrics["miss"].append(local_confusion_matrix[1, 2].sum())
        #         metrics["fa"].append(local_confusion_matrix[1, 0].sum())
        #         metrics_t2p["hit"].append(local_confusion_matrix[i, i].sum())
        #         metrics_t2p["miss"].append(local_confusion_matrix[0, 1].sum())
        #         metrics_t2p["fa"].append(local_confusion_matrix[2, 1].sum())
        #         # 没意义
        #         metrics["cn"].append(0)
        #         metrics_t2p["cn"].append(0)
        #     if i==2:
        #         metrics["hit"].append(local_confusion_matrix[i, i].sum())
        #         metrics["fa"].append(local_confusion_matrix[2, :2].sum())
        #         metrics_t2p["hit"].append(local_confusion_matrix[i, i].sum())
        #         metrics_t2p["miss"].append(local_confusion_matrix[0:i, 2].sum())
        #         # 没意义
        #         metrics["miss"].append(0)
        #         metrics["cn"].append(0)
        #         metrics_t2p["cn"].append(0)
        #         metrics_t2p["fa"].append(0)

        for i, threshold in enumerate([0.0] + thresholds, start=0):
            metrics = binary_metrics_by_threshold[threshold]
            metrics_t2p = binary_metrics_by_threshold_t2p[threshold]
            metrics["cn"].append(0)
            metrics_t2p["cn"].append(0)
            metrics["hit"].append(local_confusion_matrix[i, i].sum())
            metrics_t2p["hit"].append(local_confusion_matrix[i, i].sum())

            if i==0:
                metrics["fa"].append(0)
                metrics_t2p["miss"].append(0)
            else:
                metrics["fa"].append(local_confusion_matrix[i, 0:i].sum())
                metrics_t2p["miss"].append(local_confusion_matrix[:i,i].sum())
            if i==n_classes-1:
                metrics["miss"].append(0)
                metrics_t2p["fa"].append(0)
            else:
                metrics["miss"].append(local_confusion_matrix[i, i+1:].sum())
                metrics_t2p["fa"].append(local_confusion_matrix[i+1: ,i].sum())
        confusion_matrix += local_confusion_matrix
        # print('batch target:')
        # print(np.where(target>1))
    binary_metrics_by_threshold = {t: pd.DataFrame(metrics) for t, metrics in binary_metrics_by_threshold.items()}
    binary_metrics_by_threshold_t2p = {t: pd.DataFrame(metrics) for t, metrics in binary_metrics_by_threshold_t2p.items()}
    binary_metrics_by_threshold_ts = {t: pd.DataFrame(metrics) for t, metrics in binary_metrics_by_threshold_ts.items()}
    return confusion_matrix, binary_metrics_by_threshold, binary_metrics_by_threshold_t2p, binary_metrics_by_threshold_ts


def compute_evaluation_metrics_eachclass(df: pd.DataFrame):
    """
    Compute evaluation metrics in-place for dataframe containing binary confusion matrix.
    """
    # print(len(df)) : (4,)
    # print('compute_evaluation_metrics')
    # print(df.shape)
    df = df.copy()
    # print(df)
    # # 原版
    # df['acc'] = (df['hit'] + df['cn']) / (df['hit'] + df['miss'] + df['fa'] + df['cn'])  # accuracy
    # df['pod'] = df['hit'] / (df['hit'] + df['miss'])  # probability of detection
    # df['far'] = df['fa'] / (df['hit'] + df['fa'])  # false alarm ratio
    # df['csi'] = df['hit'] / (df['hit'] + df['miss'] + df['fa'])  # critical success index
    # df['bias'] = (df['hit'] + df['fa']) / (df['hit'] + df['miss'])  # bias
    # df['hss'] = 2* (df['hit'] * df['cn']-df['miss']*df['fa']) / (df['miss']**2 + df['fa']**2 + 2*df['hit']*df['cn']+(df['miss']+df['fa'])*(df['hit']+df['cn']))  
    # 改进版
    if (df['hit'] + df['miss'] + df['fa'] + df['cn'])==0:
        df['acc'] = -1
    else:
        df['acc'] = (df['hit'] + df['cn']) / (df['hit'] + df['miss'] + df['fa'] + df['cn'])  # accuracy
    df['pod'] = 0#df['hit'] / (df['hit'] + df['miss'])  # probability of detection
    if (df['hit'] + df['fa'])==0:
        df['far'] = -1
    else:
        df['far'] = df['fa'] / (df['hit'] + df['fa'])  # false alarm ratio
    # df['msr'] = 
    if (df['hit'] + df['miss'] + df['fa'])==0:
        df['csi'] = -1
    else:
        df['csi'] = df['hit'] / (df['hit'] + df['miss'] + df['fa'])  # critical success index
    if (df['hit'] + df['miss'] + df['fa'])==0:
        df['msr'] = -1
    else:
        df['msr'] = df['miss'] / (df['hit'] + df['miss'] + df['fa'])  # miss rate
    df['bias'] = 0#(df['hit'] + df['fa']) / (df['hit'] + df['miss'])  # bias
    df['hss'] = 0#2* (df['hit'] * df['cn']-df['miss']*df['fa']) / (df['miss']**2 + df['fa']**2 + 2*df['hit']*df['cn']+(df['miss']+df['fa'])*(df['hit']+df['cn']))  
    return df

def compute_evaluation_metrics_total(df0: pd.DataFrame, df1: pd.DataFrame, df2: pd.DataFrame):
    """
    Compute evaluation metrics in-place for dataframe containing binary confusion matrix.
    """
    # print(len(df)) : (4,)
    # print('compute_evaluation_metrics')
    # print(df.shape)
    df0 = df0.copy()
    df1 = df1.copy()
    df2 = df2.copy()
    df_combined = pd.concat([df0, df1, df2])
    # 然后按列求和
    df = df_combined.groupby(df_combined.index).sum()

    df['acc'] = (df['hit'] + df['cn']) / (df['hit'] + df['miss'] + df['fa'] + df['cn'])  # accuracy
    df['pod'] = df['hit'] / (df['hit'] + df['miss'])  # probability of detection
    df['far'] = df['fa'] / (df['hit'] + df['fa'])  # false alarm ratio
    df['csi'] = df['hit'] / (df['hit'] + df['miss'] + df['fa'])  # critical success index
    df['bias'] = (df['hit'] + df['fa']) / (df['hit'] + df['miss'])  # bias
    df['hss'] = 2* (df['hit'] * df['cn']-df['miss']*df['fa']) / (df['miss']**2 + df['fa']**2 + 2*df['hit']*df['cn']+(df['miss']+df['fa'])*(df['hit']+df['cn']))  
    return df
