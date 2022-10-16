import numpy as np
import math
import torch
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import mean_squared_error
import os
from math import sqrt
import random

def extract_batter_features(concat_df, global_df, battery_nb, window_size, prediction_interval, multichannel_features, extract_points_len, global_seq_features):

  df_copy = concat_df.copy()

  global_df_copy = global_df.copy()
    
  capacity = df_copy.groupby(['charge_nb']).mean()['interpolated_capacity'].to_numpy().reshape(-1, 1)

  global_feature = global_df_copy[global_seq_features].to_numpy()
    
  multi_channel_feature = df_copy[multichannel_features].to_numpy().reshape(-1, extract_points_len, len(multichannel_features))

  # sliding window
  def sliding_window(window_length, input_matrix):
    nb_items = input_matrix.shape[0]
    sub_window = np.array([np.arange(start=x, stop=x+window_length, step=1) for x in range(nb_items-window_length)])
    return input_matrix[sub_window]

  slided_mc_feature = sliding_window(window_size, multi_channel_feature)
  slided_glb_feature = sliding_window(window_size, global_feature)
#   print(slided_mc_feature.shape)
  slided_capacity = capacity[window_size:]
#   print(slided_capacity.shape)

  # shifting forecast interval
  input = (slided_mc_feature[:-prediction_interval], slided_glb_feature[:-prediction_interval])
  output = slided_capacity[prediction_interval:]

  return input, output

def denormalize(data, maximum, minimum):
    return data * (maximum - minimum) + minimum

def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def mean_absolute_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs(y_true - y_pred))

def root_mean_square_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return math.sqrt(np.mean((y_true - y_pred)**2))

def test(model, X_test, Y_test, starting_point=0, maximum=None, minimum=None):
    X_local, X_global = X_test
    X_local = torch.from_numpy(X_local).float()
    X_global = torch.from_numpy(X_global).float()
    model.eval()
    with torch.no_grad():
        Y_pred, _ = model(X_local, X_global)
        pred = denormalize(Y_pred[starting_point:], maximum=maximum, minimum=minimum)
    
    test = denormalize(Y_test[starting_point:], maximum=maximum, minimum=minimum)

    mape = mean_absolute_percentage_error(pred,test)
    mae = mean_absolute_error(pred,test)
    rsme = root_mean_square_error(pred,test)

    print('MAPE', mape, '\nMAE', mae, '\nRSME', rsme)
    print('-----------------------------------------')

    return mape, mae, rsme

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
    mse = mean_squared_error(y_test, y_predict)
    rmse = sqrt(mean_squared_error(y_test, y_predict))
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