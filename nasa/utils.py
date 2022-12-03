import numpy as np
import random
import math
import os
import pandas as pd
import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import scipy.io
import functools

from math import sqrt

from scipy.signal import find_peaks

import matplotlib.pyplot as plt

Rated_Capacity = 1.1

def denormalize(data, rated_capacity=Rated_Capacity):
    return data * (2.035337591005598 - 1.15381833159625) + 1.15381833159625

def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def mean_absolute_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs(y_true - y_pred))

def root_mean_square_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return math.sqrt(np.mean((y_true - y_pred)**2))

def drop_outlier(array,count,bins):
    index = []
    range_ = np.arange(1,count,bins)
    for i in range_[:-1]:
        array_lim = array[i:i+bins]
        sigma = np.std(array_lim)
        mean = np.mean(array_lim)
        th_max,th_min = mean + sigma*2, mean - sigma*2
        idx = np.where((array_lim < th_max) & (array_lim > th_min))
        idx = idx[0] + i
        index.extend(list(idx))
    return np.array(index)


def build_sequences(text, window_size):
    #text:list of capacity
    x, y = [],[]
    for i in range(len(text) - window_size):
        sequence = text[i:i+window_size]
        target = text[i+1:i+1+window_size]

        x.append(sequence)
        y.append(target)

    return np.array(x), np.array(y)


##Hung##
#Change: add parameter "feature_names"
def get_train_test(data_dict, name, feature_names, window_size=8, ):
    data_sequence=data_dict[name][feature_names]
    train_data, test_data = data_sequence[:window_size+1], data_sequence[window_size+1:]
    train_x, train_y = build_sequences(text=train_data, window_size=window_size)
    for k, v in data_dict.items():
        if k != name:
            data_x, data_y = build_sequences(text=v[feature_names], window_size=window_size)
            train_x, train_y = np.r_[train_x, data_x], np.r_[train_y, data_y]
            
    return train_x, train_y, list(train_data), list(test_data)


def relative_error(y_test, y_predict, threshold):
    true_re, pred_re = len(y_test), 0
    for i in range(len(y_test)-1):
        if y_test[i] <= threshold >= y_test[i+1]:
            true_re = i - 1
            break
    for i in range(len(y_predict)-1):
        if y_predict[i] <= threshold:
            pred_re = i - 1
            break
    return abs(true_re - pred_re)/true_re if abs(true_re - pred_re)/true_re<=1 else 1


def evaluation(y_test, y_predict):
    mape = mean_absolute_percentage_error(y_test, y_predict)
    rmse = root_mean_square_error(y_test, y_predict)
    return mape, rmse
    
    
def setup_seed(seed):
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    os.environ['PYTHONHASHSEED'] = str(seed) # 为了禁止hash随机化，使得实验可复现。
    torch.manual_seed(seed) # 为CPU设置随机种子
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed) # 为当前GPU设置随机种子
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU，为所有GPU设置随机种子
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

def peak_visualization(Y_test, points_near=2, pin=None):
    Y_test = denormalize(Y_test)
    peak_idx = np.array(find_peaks(Y_test.reshape(-1, Y_test.shape[0])[0], height = 0.0001)[0])

    # fig, ax = plt.subplots(1, figsize=(12, 8))
    # ax.plot(Y_test, label='Actual Capacity')
    # ax.plot(peak_idx, Y_test[peak_idx], ".", color='brown')
    # print(f'Number of peak of Battery {pin} : {len(peak_idx)}')
    # ax.set(xlabel='Discharge cycles', ylabel='Capacity/Ah', title='Peak detection of Battery ' + pin)
    # plt.legend()

    peak_len = len(peak_idx)
    if points_near != 0:
        peak_idx = np.append(peak_idx, [peak_idx[-1] + i for i in range(1,points_near+1)])
        peak_idx = np.insert(peak_idx, peak_len - 1, [peak_idx[-1] - i for i in range(points_near,0,-1)])
        for idx in range(peak_len - 2, -1, -1):
            peak_idx = np.insert(peak_idx, idx + 1, [peak_idx[idx] + i for i in range(1,points_near+1)])
            peak_idx = np.insert(peak_idx, idx, [peak_idx[idx] - i for i in range(points_near,0,-1)])

    peak_idx = np.array(sorted(set(peak_idx)))
    peak_idx = np.intersect1d(peak_idx, range(len(Y_test)))

    # fig, ax = plt.subplots(1, figsize=(12, 8))
    # ax.plot(Y_test, label='Actual Capacity')
    # ax.plot(peak_idx, Y_test[peak_idx], ".", color='brown')
    # print(f'Number of peak of Battery {pin} : {len(peak_idx)}')
    # ax.set(xlabel='Discharge cycles', ylabel='Capacity/Ah', title='Group of peak detection of Battery ' + pin)
    # plt.legend()
    return peak_idx

def error_computation(model, X_test, Y_test, peak_idx):
    Y_pred = model.predict(X_test)
    Y_pred = denormalize(Y_pred)
    Y_test = denormalize(Y_test)

    peak_mape = mean_absolute_percentage_error(Y_test[peak_idx], Y_pred[peak_idx])
    peak_rmse = root_mean_square_error(Y_test[peak_idx], Y_pred[peak_idx])
    print(f'\tPeak Error: \nMAPE: {mape}\nRMSE: {rmse}')

    non_peak_idx = list(set(range(len(Y_test))) - set(peak_idx))

    non_peak_mape = mean_absolute_percentage_error(Y_test[non_peak_idx], Y_pred[non_peak_idx])
    non_peak_rmse = root_mean_square_error(Y_test[non_peak_idx], Y_pred[non_peak_idx])
    print(f'\tNon-Peak Error: \nMAPE: {mape}\nRMSE: {rmse}')

    mape = mean_absolute_percentage_error(Y_test, Y_pred)
    rmse = root_mean_square_error(Y_test, Y_pred)
    print(f'\tOverall Error: \nMAPE: {mape}\nRMSE: {rmse}')
    return rmse, mape, peak_rmse, peak_mape, non_peak_rmse, non_peak_mape

def print_num_params(model):
    print(f"Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

