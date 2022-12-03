import numpy as np
from FXP import FXP
from ARNS import ARNS
from CTARNS import CTARNS
from basic_models import GF, Battery_LSTM, Battery_SVR, MLP
from utils import print_num_params, setup_seed
import time
import os
from MC_LSTM import MC_LSTM
import torch
import pandas as pd
from tensorflow import keras

concat_df = pd.read_csv("./data/concat_df.csv")
rest_df = pd.read_csv("./data/rest_df.csv")

_concat_df = {}
_rest_df = {}
batteries = [5,6,7,18]
for bat in batteries:
    _concat_df[bat] = concat_df[concat_df['battery_nb'] == bat].copy().reset_index()
    _rest_df[bat] = rest_df[rest_df['battery_nb'] == bat].copy().reset_index()
    _concat_df[bat]['datetime'] = pd.to_datetime(_concat_df[bat]['datetime'])

del concat_df, rest_df

global_df = {}
for bat in batteries:
    global_df[bat] = _concat_df[bat].groupby(['charge_nb'],as_index=False).mean()[['charge_nb','interpolated_capacity']].sort_values(['charge_nb'])
    global_df[bat]['mean_previous_5'] = global_df[bat]['interpolated_capacity'].rolling(5).mean().shift(1).fillna(10)
    global_df[bat]['isPeak'] = np.where(global_df[bat]['interpolated_capacity'] > global_df[bat]['mean_previous_5'],1,0)
    global_df[bat]['x2'] = _rest_df[bat]['cc_capacity'].copy()
    global_df[bat]['x3'] = _rest_df[bat]['cv_capacity'].copy()
    global_df[bat]['rest_period'] = _rest_df[bat]['rest_period'].copy()
    temp_df = _concat_df[bat].groupby(['charge_nb']).last().reset_index()
    global_df[bat]['x4'] = temp_df['current_measured'].copy()
    global_df[bat]['x5'] = temp_df['voltage_measured'].copy()
    del temp_df
    global_df[bat].drop(['mean_previous_5'],axis=1,inplace=True)

# normalize data
column_to_norm = ['voltage_measured','current_measured','temperature_measured','interpolated_capacity']
max_values = {}
min_values = {}

full_df = pd.concat([_concat_df[bat] for bat in _concat_df],ignore_index=True)
for col in column_to_norm:
    max_values[col] =  full_df[col].max()
    min_values[col] =  full_df[col].min()
    for bat in _concat_df:
        _concat_df[bat][col]=(_concat_df[bat][col]-min_values[col])/(max_values[col]-min_values[col]) 

column_to_norm = ['x4','x5', 'interpolated_capacity']
full_df = pd.concat([global_df[bat] for bat in global_df],ignore_index=True)
for col in column_to_norm:
    if col not in max_values:
        max_values[col] =  full_df[col].max()
        min_values[col] =  full_df[col].min()
    for bat in _concat_df:
        global_df[bat][col]=(global_df[bat][col]-min_values[col])/(max_values[col]-min_values[col]) 

window_size=10
prediction_interval=1
multichannel_features = ['voltage_measured','current_measured','temperature_measured']
global_seq_features = ['x2','x3','x4','x5','isPeak','interpolated_capacity']
# global_seq_features = ['x5','isPeak','interpolated_capacity']
epochs = EPOCH = 10
BATTERIES = [5,6,7,18]
extract_points_len = 11
setup_seed(0)

# print(max_values['interpolated_capacity'],min_values['interpolated_capacity'])

###########################
fxp = FXP()
os.mkdir('FXP')

for bat in BATTERIES:
    batteries = BATTERIES.copy()
    batteries.remove(bat)
    batteries = batteries + [bat]

    X_train, Y_train, X_val, Y_val, X_test,Y_test = fxp.cross_validation(batteries, _concat_df, global_df, extract_points_len=extract_points_len, multichannel_features=multichannel_features, window_size=window_size, prediction_interval=prediction_interval, global_seq_features=global_seq_features, key='charge_nb', label='interpolated_capacity', random_state=7, shuffle=False, validation=0.1)

    model = fxp.build_model(model_mode='bert-trainposcode', feature_mode='LSTM_bi', use_AR=True, num_labels=1, out_embed_dim=32, hidden_dim=32, dropout=0.05, window_size=window_size, extract_points_len=extract_points_len, local_features=multichannel_features, global_features=global_seq_features)
    print_num_params(model)

    start = time.time()
    fxp.train(model, X_train, Y_train, X_val, Y_val, out_dir='FXP/' + str(batteries[-1]), EPOCH=3000, verbose=10, patience=25)
    end = time.time()

    with open('FXP/' + str(batteries[-1]) + '/result.txt', 'a') as f:
        f.write(f"Training time {end - start} seconds" +'\n')
    
    model = torch.load('FXP/' + str(batteries[-1]) + '/checkpoint.pt')

    fxp.test(model, X_test, Y_test, out_dir='FXP/' + str(batteries[-1]), pin=str(batteries[-1]))

########################3
gf = GF()
os.mkdir('GF')

for bat in BATTERIES:
    batteries = BATTERIES.copy()
    batteries.remove(bat)
    batteries = batteries + [bat]

    X_train, Y_train, X_val, Y_val, X_test,Y_test = gf.cross_validation(batteries, _concat_df, key='charge_nb', label='interpolated_capacity',validation=0.1)

    model = gf.build_model(k=5)
    print_num_params(model)

    start = time.time()
    gf.train(model, X_train, Y_train, out_dir='GF/' + str(batteries[-1]),lr=0.05,epochs=50000)
    stop = time.time()
    print(f"Traing time {stop - start} seconds")
    with open('GF/' + str(batteries[-1]) + '/result.txt', 'a') as f:
        f.write(f"Training time {stop - start} seconds" +'\n')
    model  = torch.load('GF/' + str(batteries[-1])+ '/checkpoint.pt')

    gf.test(model, X_test, Y_test, out_dir='GF/' + str(batteries[-1]), pin=str(batteries[-1]), points_near=2)

###########################3
svr = Battery_SVR()
# os.mkdir('SVR')

for bat in BATTERIES:
    batteries = BATTERIES.copy()
    batteries.remove(bat)
    batteries = batteries + [bat]

    X_train, Y_train, X_val, Y_val, X_test,Y_test = svr.cross_validation(batteries, _concat_df, key='charge_nb', label='interpolated_capacity', validation=0.1)

    model = svr.build_model()
    print(model.get_params())

    svr.train(model, X_train, Y_train, out_dir='SVR/' + str(batteries[-1]))
    svr.test(model, X_test, Y_test, out_dir='SVR/' + str(batteries[-1]), pin=str(batteries[-1]), points_near=2)


######################################################
mlp = MLP()
if not os.path.isdir('MLP'):
    os.mkdir('MLP')

for bat in BATTERIES:
    batteries = BATTERIES.copy()
    batteries.remove(bat)
    batteries = batteries + [bat]

    X_train, Y_train, X_val, Y_val, X_test,Y_test = mlp.cross_validation(batteries, _concat_df, global_df, extract_points_len=extract_points_len, multichannel_features=multichannel_features, window_size=window_size, prediction_interval=prediction_interval, global_seq_features = global_seq_features, key='charge_nb', label='interpolated_capacity', random_state=7, shuffle=False, validation=0.1)

    model = mlp.build_model(feature_size=10*11*3 + 10*6, hidden_size=[256,128])
    print_num_params(model)

    start = time.time()
    mlp.train(model, X_train, Y_train, X_val, Y_val, out_dir='MLP/' + str(batteries[-1]), LR=0.001, EPOCH=3000)
    end = time.time()

    with open('MLP/' + str(batteries[-1]) + '/result.txt', 'a') as f:
        f.write(f"Training time {end - start} seconds" +'\n')

    model  = torch.load('MLP/' + str(batteries[-1])+ '/checkpoint.pt')

    mlp.test(model, X_test, Y_test, out_dir='MLP/' + str(batteries[-1]), pin=str(batteries[-1]), points_near=2)

###############
lstm = Battery_LSTM()
if not os.path.isdir('LSTM'):
    os.mkdir('LSTM')

for bat in BATTERIES:
    batteries = BATTERIES.copy()
    batteries.remove(bat)
    batteries = batteries + [bat]

    X_train, Y_train, X_val, Y_val, X_test,Y_test = lstm.cross_validation(batteries, _concat_df, global_df, extract_points_len=extract_points_len, multichannel_features=multichannel_features, window_size=window_size, prediction_interval=prediction_interval, global_seq_features = global_seq_features, key='charge_nb', label='interpolated_capacity', random_state=7, shuffle=False, validation=0.1)

    model = lstm.build_model(input_size=11*3 + 6, hidden_dim=256, num_layers=1, n_class=1, mode='LSTM')
    print_num_params(model)


    start = time.time()
    lstm.train(model, X_train, Y_train, X_val, Y_val, out_dir='LSTM/' + str(batteries[-1]), LR=0.001, EPOCH=3000)
    end = time.time()

    with open('LSTM/' + str(batteries[-1]) + '/result.txt', 'a') as f:
        f.write(f"Training time {end - start} seconds" +'\n')

    model  = torch.load('LSTM/' + str(batteries[-1])+ '/checkpoint.pt')

    lstm.test(model, X_test, Y_test, out_dir='LSTM/' + str(batteries[-1]), pin=str(batteries[-1]), points_near=2)

##########################
window_size=10
prediction_interval=1
multichannel_features = ['voltage_measured','current_measured','temperature_measured']
global_seq_features = ['interpolated_capacity']
epochs = EPOCH = 1500
BATTERIES = [5,6,7,18]
extract_points_len = 11

mc_lstm = MC_LSTM()
os.mkdir('MC_LSTM')

for bat in BATTERIES:
    batteries = BATTERIES.copy()
    batteries.remove(bat)
    batteries = batteries + [bat]

    X_train, Y_train, X_val, Y_val, X_test,Y_test = mc_lstm.cross_validation(batteries, _concat_df, global_df, extract_points_len=extract_points_len, multichannel_features=multichannel_features, window_size=window_size, prediction_interval=prediction_interval, global_seq_features = global_seq_features, key='charge_nb', label='interpolated_capacity', random_state=7, shuffle=False, validation=0.1)

    model = mc_lstm.build_model((X_train[0].shape[1:], X_train[1].shape[1:], X_train[2].shape[1:], X_train[3].shape[1:]))

    start = time.time()
    mc_lstm.train(model, X_train, Y_train, X_val, Y_val, out_dir='MC_LSTM/' + str(batteries[-1]),epochs=1000,verbose=1)
    end = time.time()

    model.load_weights('MC_LSTM/' + str(batteries[-1]) + '/checkpoint_weight')
    mc_lstm.test(model, X_test, Y_test, out_dir='MC_LSTM/' + str(batteries[-1]), pin=str(batteries[-1]))
    with open('MC_LSTM/' + str(batteries[-1]) + '/result.txt', 'a') as f:
        f.write(f"Training time {end - start} seconds" +'\n')