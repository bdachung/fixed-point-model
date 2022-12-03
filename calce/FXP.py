from cs_interface import NestedModel
import numpy as np
from FXP_modules import Regressor
import functools
import torch.nn as nn
import torch
from utils import setup_seed, evaluation, relative_error, denormalize, peak_visualization, mean_absolute_percentage_error,root_mean_square_error
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import copy

class FXP(NestedModel):
    def build_model(self, model_mode, feature_mode, use_AR, num_labels, out_embed_dim, hidden_dim, dropout=0., window_size=8, extract_points_len=11, local_features=['voltage','current','resistance'], global_features = ['x_5', 'rest_period', 'capacity']):
        """model mode: bert, bert-newposcode, bert-trainposcode, lstm
           feature_mode: Conv1D_k3mk3, Conv1D_k5k5, LSTM_uni, LSTM_bi, FLatten 
        """
        return Regressor(model_mode, feature_mode, use_AR, num_labels, out_embed_dim, hidden_dim, dropout, window_size, extract_points_len, local_features, global_features)

    def extract_batter_features(self, concat_df, global_df, battery_nb, window_size=8, prediction_interval=1, extract_points_len=11, multichannel_features=['voltage','current','resistance'], global_seq_features = ['x5', 'rest_period', 'capacity'], key='cycle', label='capacity'):
        # choose only 1 battery at a time
        #   df_copy = concat_df[battery_nb].copy()
        df_copy = concat_df.copy()
        # rest time of each charge cycle
        #   global_df_copy = global_df[battery_nb].copy()
        global_df_copy = global_df.copy()
            
        capacity = df_copy.groupby([key]).mean()[label].to_numpy().reshape(-1, 1)

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

    def cross_validation(self, batteries, concat_df, global_df, extract_points_len=11, multichannel_features=['voltage','current','resistance'], window_size=8, prediction_interval=1, global_seq_features = ['x5', 'rest_period', 'capacity'],key='cycle', label='capacity', validation=None, random_state=7, shuffle=False):
        # last pin is the test set
        all_batteries = [self.extract_batter_features(concat_df[battery_nb].copy(), global_df[battery_nb].copy(), battery_nb, multichannel_features=multichannel_features, window_size=window_size, prediction_interval=prediction_interval, global_seq_features = global_seq_features, key=key, label=label) for battery_nb in batteries]

        # cross validation
        train_set = all_batteries[0:-1]
        val_set = all_batteries[-1]
        test_set = all_batteries[-1]

        mc_train = functools.reduce(lambda a,b: np.concatenate((a,b), axis=0), [x for (x, _),_ in train_set]).reshape(-1, window_size, extract_points_len, len(multichannel_features))
        glb_train = functools.reduce(lambda a,b: np.concatenate((a,b), axis=0), [x for (_, x),_ in train_set]).reshape(-1, window_size, len(global_seq_features))

        Y_train = functools.reduce(lambda a,b: np.concatenate((a,b), axis=0), [y for _,y in train_set])
        Y_train = Y_train.reshape(-1, 1)

        if validation is None:
            X_train = [mc_train, glb_train]

            (mc_val, glb_val), Y_val = val_set
            X_val = [mc_val, glb_val]

        else:
            mc_train, mc_val, glb_train, glb_val, Y_train, Y_val = train_test_split(mc_train, glb_train, Y_train, test_size=validation, random_state=random_state, shuffle=shuffle)
            X_train = [mc_train, glb_train]
            X_val = [mc_val, glb_val]

        (mc_test, glb_test), Y_test = test_set
        X_test = [mc_test, glb_test]

        return X_train, Y_train, X_val, Y_val, X_test,Y_test

    def train(self, model, X_train, Y_train, X_val, Y_val, out_dir, EPOCH=1000, LR=0.001, seed=0, weight_decay=0., device="cpu", clipping_value=None, verbose=2, patience=25):
        if not os.path.isdir(out_dir):
            os.mkdir(out_dir)
        setup_seed(seed)
        
        loss_list, y_ = [0], [] 
        rmse_list, mape_list = [], []
        best_rmse = None

        model = model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=weight_decay)
        criterion = nn.MSELoss()
    
        for epoch in range(EPOCH) :
            model.train()
            X_local, X_global = X_train
            X_local = np.array(X_local).astype(np.float32)
            X_global = np.array(X_global).astype(np.float32)
            Y = np.array(Y_train).astype(np.float32)
            X_local, X_global = torch.from_numpy(X_local).to(device), torch.from_numpy(X_global).to(device) 
            Y = torch.from_numpy(Y).to(device)
            output, attention_weights = model(X_local, X_global)
            downstream_loss = criterion(output, Y)
            loss = downstream_loss
            optimizer.zero_grad()
            loss.backward()
            if clipping_value:
                nn.utils.clip_grad_norm(model.parameters(), clipping_value)
            optimizer.step()

            if epoch%verbose==0 or epoch==(EPOCH-1):
                point_list = []
                x_local, x_global = X_val
                x_local = np.array(x_local).astype(np.float32)
                x_global = np.array(x_global).astype(np.float32)
                x_local, x_global,  = torch.from_numpy(x_local).to(device), torch.from_numpy(x_global).to(device)
                model.eval()
                with torch.no_grad():
                    pred, _ = model(x_local, x_global)
                point_list = list(pred[:,0].cpu().detach().numpy())
                y_.append(point_list)
                loss_list.append(loss)
                mape, rmse = evaluation(y_test=denormalize(Y_val).flatten(), y_predict=denormalize(np.array(y_[-1])))
                mape_list.append(mape)
                rmse_list.append(rmse)
                if best_rmse == None:
                    best_rmse = {}
                    best_rmse['value'] = rmse
                    best_rmse['epoch'] = epoch
                    best_rmse['model'] = copy.deepcopy(model)
                else:
                    if rmse < best_rmse['value']:
                        best_rmse['value'] = rmse
                        best_rmse['epoch'] = epoch
                        del best_rmse['model']
                        best_rmse['model'] = copy.deepcopy(model)
                re = relative_error(y_test=Y_val.flatten(), y_predict=y_[-1], threshold=0.7)
                print('epoch:{:<2d} | loss:{:<6.7f} | MAPE:{:<6.4f} | RMSE:{:<6.4f} | RE:{:<6.4f}'.format(epoch, loss, mape, rmse, re))
                print(f'current epoch: {epoch}, best epoch:' + str(best_rmse['epoch']))
                if (epoch - best_rmse['epoch']) / verbose >= patience:
                    print("Early stopping")
                    break
                
            if ((len(loss_list) > 1) and (abs(loss_list[-2] - loss_list[-1]) < 1e-7)):
                break
        torch.save(best_rmse['model'], os.path.join(out_dir,'checkpoint.pt'))
        mape, rmse = evaluation(y_test=denormalize(Y_val).flatten(), y_predict=denormalize(np.array(y_[-1])))
        re = relative_error(y_test=Y_val.flatten(), y_predict=y_[-1], threshold=0.7)

        return model, y_[-1], mape_list, rmse_list, re
    
    def test(self, model, X_test, Y_test, out_dir, points_near=0, device='cpu', starting_point=0, pin=None):
        X_local, X_global = X_test
        X_local = torch.from_numpy(X_local).float().to(device)
        X_global = torch.from_numpy(X_global).float().to(device)
        Y_test = torch.from_numpy(Y_test).float().to(device)
        peak_idx = peak_visualization(Y_test=Y_test,points_near=points_near,pin=pin)
        model.to(device)
        model.eval()
        with torch.no_grad():
            Y_pred, _ = model(X_local, X_global)
            pred = denormalize(Y_pred[starting_point:])
        
        test = denormalize(Y_test[starting_point:])

        plt.figure(figsize=(7,5))
        if starting_point == 0:
            plt.plot(pred)
            plt.plot(test)
            plt.legend(['Predict', 'Test'])
        else:
            plt.plot(np.concatenate((denormalize(Y_test[-60:starting_point]), pred)))
            plt.plot(denormalize(Y_test[-60:]))
            plt.legend(['Predict', 'Test'])
        # plt.plot(peak_idx, test[peak_idx], ".", color='brown')
        plt.savefig(out_dir + '/predict.png', format='png')

        peak_mape = mean_absolute_percentage_error(test[peak_idx], pred[peak_idx])
        peak_rmse = root_mean_square_error(test[peak_idx], pred[peak_idx])

        non_peak_idx = list(set(range(len(test))) - set(peak_idx))

        non_peak_mape = mean_absolute_percentage_error(test[non_peak_idx], pred[non_peak_idx])
        non_peak_rmse = root_mean_square_error(test[non_peak_idx], pred[non_peak_idx])

        mape = mean_absolute_percentage_error(test, pred)
        rmse = root_mean_square_error(test, pred)

        with open(out_dir + '/result.txt', 'a') as f:
            f.write("overall rmse:" + str(rmse) + '\n')
            f.write("overall mape:" + str(mape) + '\n')
            f.write("peak rmse:" + str(peak_rmse) + '\n')
            f.write("peak mape:" + str(peak_mape) + '\n')
            f.write("non peak rmse:" + str(non_peak_rmse) + '\n')
            f.write("non peak mape:" + str(non_peak_mape) + '\n')
        return pred, test 
    

