from cProfile import label
import torch
from torch import nn
import numpy as np
import functools
from torch.autograd import Variable
from cs_interface import NestedModel
from utils import setup_seed, evaluation, relative_error, denormalize, peak_visualization, mean_absolute_percentage_error,root_mean_square_error
import matplotlib.pyplot as plt
import os
from sklearn.svm import SVR 
import joblib
from sklearn.model_selection import train_test_split

class GFNet(nn.Module):
    def __init__(self, k=3, eps=1e-6):
        super(GFNet, self).__init__()
        self.eps = eps
        self.k = k
        self.w = torch.nn.Parameter(torch.randn((3, k)), requires_grad=True)
        
    def forward(self, x): 
        out = 0
        for i in range(self.k):
            out += self.w[0, i] * torch.exp(-torch.pow((x - self.w[1, i])/(self.w[2, i] + self.eps), 2)) 
        return out

class GF(NestedModel):
    def build_model(self, k=3, eps=1e-6):
        return GFNet(k=k,eps=eps)
    def extract_batter_features(self, concat_df, battery_nb, key, label):

        df_copy = concat_df.groupby(by=[key],as_index=False).mean()[[key,label]].copy()
        
        input = df_copy[key].values
        output = df_copy[label].values

        return input, output
    def cross_validation(self, batteries, concat_df, key='cycle', label='capacity', validation=None, random_state=7, shuffle=False):

        all_batteries = [self.extract_batter_features(concat_df[battery_nb].copy(), battery_nb, key=key, label=label) for battery_nb in batteries]

        # cross validation
        train_set = all_batteries[0:-1]
        val_set = all_batteries[-1]
        test_set = all_batteries[-1]

        X_train, Y_train = np.array([]), np.array([])

        for (X,Y) in train_set:
            X_train = np.append(X_train, X)
            Y_train = np.append(Y_train, Y)
        
        if validation is None:
            X_val, Y_val = val_set
        else:
            X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=validation, random_state=random_state, shuffle=shuffle)

        X_test, Y_test = test_set

        return X_train, Y_train, X_val, Y_val, X_test, Y_test

    def train(self, model, X_train, Y_train, out_dir, lr=1e-4, stop=1e-3, epochs=1000, device='cpu', patience=10):
        if not os.path.isdir(out_dir):
            os.mkdir(out_dir)

        model = model.to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.MSELoss()

        loss_list = []
        epoch = 1

        data = np.c_[X_train, Y_train]

        train_data = np.random.permutation(data)
        x, y_org = train_data[:,0], train_data[:,1]
        X, y = np.reshape(x, (-1, 1)).astype(np.float32), np.reshape(y_org, (-1, 1)).astype(np.float32)
        X, y = Variable(torch.from_numpy(X)).to(device), Variable(torch.from_numpy(y)).to(device)
        y_ = model(X)
        loss = criterion(y_, y)
        #print(loss.detach().numpy())
        optimizer.zero_grad()              # clear gradients for this training step
        loss.backward()                    # backpropagation, compute gradients
        optimizer.step()                   # apply gradients
        loss_list.append(loss.detach().numpy())
        
        best_loss = loss
        best_epoch = 0
        best_w = model.w

        while True:
            model.train()
            train_data = np.random.permutation(data)
            x, y_org = train_data[:,0], train_data[:,1]
            X, y = np.reshape(x, (-1, 1)).astype(np.float32), np.reshape(y_org, (-1, 1)).astype(np.float32)
            X, y = Variable(torch.from_numpy(X)).to(device), Variable(torch.from_numpy(y)).to(device)
            
            y_ = model(X)

            loss = criterion(y_, y)
            #print(loss.detach().numpy())
            optimizer.zero_grad()              # clear gradients for this training step
            loss.backward()                    # backpropagation, compute gradients
            optimizer.step()                   # apply gradients

            if ((loss - best_loss) < 1e-4):
                best_loss = loss
                best_epoch = epoch
                best_w = model.w
            
            if epoch % 100 == 0:
                loss_list.append(loss.detach().numpy())
                print(loss_list[-1])
            if (loss.detach().numpy() < stop) or (epoch > epochs) or ((epoch - best_epoch) == patience):
                break
                
            epoch +=1
        model.w = best_w
        torch.save(model, out_dir + '/checkpoint.pt')
        return model, loss_list, x, y_org #GF_model, loss_list, x_test, y_true

    def test(self, model, X_test, Y_test, out_dir, points_near=0, device='cpu', starting_point=0, pin=None):
        X = torch.from_numpy(X_test).float().to(device)
        peak_idx = peak_visualization(Y_test=Y_test,points_near=points_near,pin=pin)
        Y_test = torch.from_numpy(Y_test).float().to(device)
        model.to(device)
        model.eval()
        with torch.no_grad():
            Y_pred = model(X)
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

class Battery_SVR(NestedModel):
    def build_model(self, C=20, epsilon=0.0001, gamma=0.00001, cache_size=200,kernel='rbf', max_iter=-1, shrinking=True, tol=0.001, verbose=False):
        return SVR(C=C, epsilon=epsilon, gamma=gamma, cache_size=cache_size, kernel=kernel, max_iter=max_iter, shrinking=shrinking, tol=tol, verbose=verbose)
    
    def extract_batter_features(self, concat_df, battery_nb, key='cycle', label='capacity'):

        df_copy = concat_df.groupby(by=[key],as_index=False).mean()[[key, label]].copy()
        
        input = df_copy[key].values
        output = df_copy[label].values

        return input, output
    def cross_validation(self, batteries, concat_df, key='cycle', label='capacity', validation=None, random_state=7, shuffle=False):

        all_batteries = [self.extract_batter_features(concat_df[battery_nb].copy(), battery_nb, key=key, label=label) for battery_nb in batteries]

        # cross validation
        train_set = all_batteries[0:-1]
        val_set = all_batteries[-1]
        test_set = all_batteries[-1]

        X_train, Y_train = np.array([]), np.array([])

        for (X,Y) in train_set:
            X_train = np.append(X_train, X)
            Y_train = np.append(Y_train, Y)

        if validation is None:
            X_val, Y_val = val_set
        else:
            X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=validation, random_state=random_state, shuffle=shuffle)
        
        X_test, Y_test = test_set

        return X_train, Y_train, X_val, Y_val, X_test, Y_test
    
    def train(self, model, X_train, Y_train, out_dir):
        if not os.path.isdir(out_dir):
            os.mkdir(out_dir)
        model.fit(X_train.reshape(-1,1), Y_train.reshape(-1,1))

        joblib.dump(model, out_dir + "/checkpoint.pkl")

    def test(self, model, X_test, Y_test, out_dir, points_near=0, starting_point=0, pin=None):
        Y_pred = model.predict(X_test.reshape(-1, 1))
        peak_idx = peak_visualization(Y_test=Y_test,points_near=points_near,pin=pin)
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

    def load_model(self, model_path):
        return joblib.load(model_path)

class MLP_Net(nn.Module):
    def __init__(self, feature_size=8, hidden_size=[16, 8], out_dim=1):
        super(MLP_Net, self).__init__()
        self.feature_size, self.hidden_size = feature_size, hidden_size
        self.layer0 = nn.Linear(self.feature_size, self.hidden_size[0])
        self.layers = nn.ModuleList()
        ##Hung##
        [self.layers.append(nn.Sequential(nn.Linear(self.hidden_size[i], self.hidden_size[i+1]), nn.ReLU())) 
                       for i in range(len(self.hidden_size) - 1)]
        self.linear = nn.Linear(self.hidden_size[-1], out_dim)
 
    def forward(self, x):
        out = self.layer0(x)
        for layer in self.layers:
            out = layer(out)
        out = self.linear(out) 
        return out

class MLP(NestedModel):
    def build_model(self, feature_size=8, hidden_size=[64, 32], out_dim=1):
        return MLP_Net(feature_size, hidden_size, out_dim)

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

    def cross_validation(self, batteries, concat_df, global_df, extract_points_len=11, multichannel_features=['voltage','current','resistance'], window_size=8, prediction_interval=1, global_seq_features = ['x5', 'rest_period', 'capacity'], key='cycle', label='capacity', validation=None, random_state=7, shuffle=False):
        # last pin is the test set
        all_batteries = [self.extract_batter_features(concat_df[battery_nb].copy(), global_df[battery_nb].copy(), battery_nb, window_size=window_size, prediction_interval=prediction_interval, extract_points_len=extract_points_len, multichannel_features=multichannel_features, global_seq_features=global_seq_features, key=key, label=label) for battery_nb in batteries]

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
    
    def train(self, model, X_train, Y_train, X_val, Y_val, out_dir, EPOCH=1000, seed=0, dropout=0., LR=0.001, weight_decay=0, device='cpu', clipping_value=None):
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
            X_local = torch.reshape(X_local, (X_local.size(0),-1))
            X_global = torch.reshape(X_global, (X_global.size(0),-1))
            X = torch.cat([X_local,X_global], dim=-1)
            Y = torch.from_numpy(Y).to(device)
            output = model(X)
            downstream_loss = criterion(output, Y)
            loss = downstream_loss
            optimizer.zero_grad()
            loss.backward()
            if clipping_value:
                nn.utils.clip_grad_norm(model.parameters(), clipping_value)
            optimizer.step()

            if epoch%100==0 or epoch==(EPOCH-1):
                point_list = []
                x_local, x_global = X_val
                x_local = np.array(x_local).astype(np.float32)
                x_global = np.array(x_global).astype(np.float32)
                x_local, x_global,  = torch.from_numpy(x_local).to(device), torch.from_numpy(x_global).to(device)
                x_local = torch.reshape(x_local, (x_local.size(0),-1))
                x_global = torch.reshape(x_global, (x_global.size(0),-1))
                x = torch.cat([x_local,x_global], dim=-1)
                model.eval()
                with torch.no_grad():
                    pred = model(x)
    #                 print(pred)
                point_list = list(pred[:,0].cpu().detach().numpy())
                y_.append(point_list)
                loss_list.append(loss)
                mape, rmse = evaluation(y_test=denormalize(Y_val).flatten(), y_predict=denormalize(np.array(y_[-1])))
                mape_list.append(mape)
                rmse_list.append(rmse)
                if best_rmse == None:
                    best_rmse = {}
                    # best_rmse['name'] = 'checkpoint-' + str(epoch) + '.pt'
                    best_rmse['name'] = 'checkpoint.pt'
                    best_rmse['value'] = rmse
                    torch.save(model, os.path.join(out_dir,best_rmse['name']))
                else:
                    if rmse < best_rmse['value']:
                        os.remove(os.path.join(out_dir,best_rmse['name']))
                        best_rmse['name'] = 'checkpoint.pt'
                        best_rmse['value'] = rmse
                        torch.save(model, os.path.join(out_dir,best_rmse['name']))
                re = relative_error(y_test=Y_val.flatten(), y_predict=y_[-1], threshold=0.7)
                print('epoch:{:<2d} | loss:{:<6.7f} | MAPE:{:<6.4f} | RMSE:{:<6.4f} | RE:{:<6.4f}'.format(epoch, loss, mape, rmse, re))
                
    #         if (len(loss_list) > 1) and (abs(loss_list[-2] - loss_list[-1]) < 1e-9):
    #           break
        mape, rmse = evaluation(y_test=denormalize(Y_val).flatten(), y_predict=denormalize(np.array(y_[-1])))
        re = relative_error(y_test=Y_val.flatten(), y_predict=y_[-1], threshold=0.7)
        return model, y_[-1], mape_list, rmse_list, re

    def test(self, model, X_test, Y_test, out_dir, points_near=0, starting_point=0, device='cpu', pin=None):
        X_local, X_global = X_test
        X_local = np.array(X_local).astype(np.float32)
        X_global = np.array(X_global).astype(np.float32)
        Y = np.array(Y_test).astype(np.float32)
        X_local, X_global = torch.from_numpy(X_local).to(device), torch.from_numpy(X_global).to(device) 
        X_local = torch.reshape(X_local, (X_local.size(0),-1))
        X_global = torch.reshape(X_global, (X_global.size(0),-1))
        X = torch.cat([X_local,X_global], dim=-1)
        Y = torch.from_numpy(Y).to(device)
        peak_idx = peak_visualization(Y_test=Y_test,points_near=points_near,pin=pin)
        model.to(device)

        model.eval()
        with torch.no_grad():
            Y_pred = model(X)
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

class RNN_Net(nn.Module):
    def __init__(self, input_size, hidden_dim, num_layers, n_class=1, mode='LSTM'):
        super(RNN_Net, self).__init__()
        self.hidden_dim = hidden_dim
        self.cell = nn.LSTM(input_size=input_size, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)
        if mode == 'GRU':
            self.cell = nn.GRU(input_size=input_size, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)
        elif mode == 'RNN':
            self.cell = nn.RNN(input_size=input_size, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_dim, n_class)
 
    def forward(self, x):           # x shape: (batch_size, seq_len, input_size)
        out, _ = self.cell(x) 
#         print("out shape",out.shape)
        out = out[:,-1,:]
        out = out.reshape(-1, self.hidden_dim)
        out = self.linear(out)      # out shape: (batch_size, n_class=1)
        return out

class Battery_LSTM(NestedModel):
    def build_model(self, input_size, hidden_dim, num_layers, n_class=1, mode='LSTM'):
        return RNN_Net(input_size, hidden_dim, num_layers, n_class, mode)

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

    def cross_validation(self, batteries, concat_df, global_df, extract_points_len=11, multichannel_features=['voltage','current','resistance'], window_size=8, prediction_interval=1, global_seq_features=['x5', 'rest_period', 'capacity'], key='cycle', label='capacity', validation=None, random_state=7, shuffle=False):
        # last pin is the test set
        all_batteries = [self.extract_batter_features(concat_df[battery_nb].copy(), global_df[battery_nb].copy(), battery_nb, window_size=window_size, prediction_interval=prediction_interval, extract_points_len=extract_points_len, multichannel_features=multichannel_features, global_seq_features=global_seq_features, key=key, label=label) for battery_nb in batteries]

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
    
    def train(self, model, X_train, Y_train, X_val, Y_val, out_dir='.', EPOCH=1000, seed=0, dropout=0., LR=0.001, weight_decay=0, device='cpu', clipping_value=None):
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
            X_local = torch.reshape(X_local, (X_local.size(0),X_local.size(1),-1))
            X_global = torch.reshape(X_global, (X_global.size(0),X_global.size(1),-1))
            X = torch.cat([X_local,X_global], dim=-1)
            Y = torch.from_numpy(Y).to(device)
            output = model(X)
            downstream_loss = criterion(output, Y)
            loss = downstream_loss
            optimizer.zero_grad()
            loss.backward()
            if clipping_value:
                nn.utils.clip_grad_norm(model.parameters(), clipping_value)
            optimizer.step()

            if epoch%100==0 or epoch==(EPOCH-1):
                point_list = []
                x_local, x_global = X_val
                x_local = np.array(x_local).astype(np.float32)
                x_global = np.array(x_global).astype(np.float32)
                x_local, x_global,  = torch.from_numpy(x_local).to(device), torch.from_numpy(x_global).to(device)
                x_local = torch.reshape(x_local, (x_local.size(0),x_local.size(1),-1))
                x_global = torch.reshape(x_global, (x_global.size(0),x_global.size(1),-1))
                x = torch.cat([x_local,x_global], dim=-1)
                model.eval()
                with torch.no_grad():
                    pred = model(x)
    #                 print(pred)
                point_list = list(pred[:,0].cpu().detach().numpy())
                y_.append(point_list)
                loss_list.append(loss)
                mape, rmse = evaluation(y_test=denormalize(Y_val).flatten(), y_predict=denormalize(np.array(y_[-1])))
                mape_list.append(mape)
                rmse_list.append(rmse)
                if best_rmse == None:
                    best_rmse = {}
                    best_rmse['name'] = 'checkpoint-' + str(epoch) + '.pt'
                    best_rmse['value'] = rmse
                    torch.save(model, os.path.join(out_dir,best_rmse['name']))
                else:
                    if rmse < best_rmse['value']:
                        os.remove(os.path.join(out_dir,best_rmse['name']))
                        best_rmse['name'] = 'checkpoint-' + str(epoch) + '.pt'
                        best_rmse['value'] = rmse
                        torch.save(model, os.path.join(out_dir,best_rmse['name']))
                re = relative_error(y_test=Y_val.flatten(), y_predict=y_[-1], threshold=0.7)
                print('epoch:{:<2d} | loss:{:<6.7f} | MAPE:{:<6.4f} | RMSE:{:<6.4f} | RE:{:<6.4f}'.format(epoch, loss, mape, rmse, re))
                
    #         if (len(loss_list) > 1) and (abs(loss_list[-2] - loss_list[-1]) < 1e-9):
    #           break
        mape, rmse = evaluation(y_test=denormalize(Y_val).flatten(), y_predict=denormalize(np.array(y_[-1])))
        re = relative_error(y_test=Y_val.flatten(), y_predict=y_[-1], threshold=0.7)
        return model, y_[-1], mape_list, rmse_list, re

    def test(self, model, X_test, Y_test, out_dir, points_near=0, starting_point=0, device='cpu', pin=None):
        X_local, X_global = X_test
        X_local = np.array(X_local).astype(np.float32)
        X_global = np.array(X_global).astype(np.float32)
        Y = np.array(Y_test).astype(np.float32)
        X_local, X_global = torch.from_numpy(X_local).to(device), torch.from_numpy(X_global).to(device) 
        X_local = torch.reshape(X_local, (X_local.size(0),X_local.size(1),-1))
        X_global = torch.reshape(X_global, (X_global.size(0),X_global.size(1),-1))
        X = torch.cat([X_local,X_global], dim=-1)
        Y = torch.from_numpy(Y).to(device)
        peak_idx = peak_visualization(Y_test=Y_test,points_near=points_near,pin=pin)
        model.to(device)
        model.eval()
        with torch.no_grad():
            Y_pred = model(X)
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
    


