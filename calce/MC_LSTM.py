import numpy as np
from cs_interface import NestedModel
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, ConvLSTM2D, TimeDistributed, InputLayer, Input, Lambda, LSTM, Bidirectional
import os
import matplotlib.pyplot as plt
import functools
from utils import peak_visualization, denormalize, mean_absolute_percentage_error,root_mean_square_error
from sklearn.model_selection import train_test_split

class MC_LSTM(NestedModel):
    def build_model(self, input_shape, hidden_node=21, extract_points_len=11, multichannel_features=['voltage','current','resistance'], window_size=8):
        input_shape_V, input_shape_A, input_shape_T, input_shape_C = input_shape
        output_node = 1
        time_step = window_size
        channel_sample = extract_points_len
        capacity_sample = 1
        feature = 1

        # Creating the layers
        input_layer_V = Input(shape=input_shape_V, name='Voltage_Input') # multichannel input (V,I,T)
        input_layer_A = Input(shape=input_shape_A, name='Ampere_Input')
        input_layer_T = Input(shape=input_shape_T, name='Temperature_Input')
        input_layer_C = Input(shape=input_shape_C, name='Capacity_Input')
        
        LSTM_V = LSTM(hidden_node, input_shape=(time_step, channel_sample, feature), activation='relu', name='Voltage_LSTM')(input_layer_V)
        LSTM_A = LSTM(hidden_node, input_shape=(time_step, channel_sample, feature), activation='relu', name='Ampere_LSTM')(input_layer_A)
        LSTM_T = LSTM(hidden_node, input_shape=(time_step, channel_sample, feature), activation='relu', name='Temperature_LSTM')(input_layer_T)
        LSTM_C = LSTM(hidden_node, input_shape=(time_step, capacity_sample, feature), activation='relu', name='Capacity_LSTM')(input_layer_C)

        output_layer = tf.concat(axis=1, values=[LSTM_V, LSTM_A, LSTM_T, LSTM_C])
        output_layer = Dense(output_node, name='Output')(output_layer)

        # Defining the model by specifying the input and output layers
        model = Model(inputs=[input_layer_V, input_layer_A, input_layer_T, input_layer_C], outputs=output_layer)
        model.compile(optimizer='adam',loss='mse',metrics=[tf.keras.metrics.RootMeanSquaredError()])
        print(model.summary())
        return model

    def extract_batter_features(self, df, rest_df, battery_name, extract_points_len=11, multichannel_features=['voltage','current','resistance'], window_size=8, prediction_interval=1, global_seq_features=['capacity'], key='cycle', label='capacity'):
        """df: local features, rest_df: global features"""
        # choose only 1 battery at a time
        df_copy = df[battery_name].copy()

        capacity = df_copy.groupby([key]).mean()[label].to_numpy().reshape(-1, 1)

        global_feature = np.concatenate(capacity)
        multi_channel_feature = df_copy[multichannel_features].to_numpy().reshape(-1, extract_points_len, len(multichannel_features))

        # sliding window
        def sliding_window(window_length, input_matrix):
            nb_items = input_matrix.shape[0]
            sub_window = np.array([np.arange(start=x, stop=x+window_length, step=1) for x in range(nb_items-window_length)])
            return input_matrix[sub_window]

        slided_V_feature = sliding_window(window_size, multi_channel_feature[:,:,0])
        slided_A_feature = sliding_window(window_size, multi_channel_feature[:,:,1])
        slided_T_feature = sliding_window(window_size, multi_channel_feature[:,:,2])
        slided_C_feature = sliding_window(window_size, global_feature)
        #slided_forecast_rest_time = rest_time[window_size:]
        slided_capcity = capacity[window_size:]

        # shifting forecast interval
        input = (slided_V_feature[:-prediction_interval], 
                slided_A_feature[:-prediction_interval], 
                slided_T_feature[:-prediction_interval], 
                slided_C_feature[:-prediction_interval])
        output = slided_capcity[prediction_interval:]

        return input, output

    # cross validation
    def cross_validation(self, batteries, df, rest_df, extract_points_len=11, multichannel_features=['voltage','current','resistance'], window_size=8, prediction_interval=1, global_seq_features=['capacity'], key='cycle', label='capacity', validation=None, random_state=7, shuffle=False):
        # last pin is the test sets
        all_batteries = [self.extract_batter_features(df, rest_df, battery_nb, window_size=window_size, prediction_interval=prediction_interval, extract_points_len=extract_points_len, multichannel_features=multichannel_features, global_seq_features=global_seq_features, key=key, label=label) for battery_nb in batteries]
        # cross validation
        train_set = all_batteries[0:-1]
        val_set = all_batteries[-1]
        test_set = all_batteries[-1]

        V_train = functools.reduce(lambda a,b: np.concatenate((a,b), axis=0), [x for (x, _, _, _),_ in train_set]).reshape(-1, window_size, extract_points_len)
        A_train = functools.reduce(lambda a,b: np.concatenate((a,b), axis=0), [x for (_, x, _, _),_ in train_set]).reshape(-1, window_size, extract_points_len)
        T_train = functools.reduce(lambda a,b: np.concatenate((a,b), axis=0), [x for (_, _, x, _),_ in train_set]).reshape(-1, window_size, extract_points_len)
        C_train = functools.reduce(lambda a,b: np.concatenate((a,b), axis=0), [x for (_, _, _, x),_ in train_set]).reshape(-1, window_size, len(global_seq_features))
        
        Y_train = functools.reduce(lambda a,b: np.concatenate((a,b), axis=0), [y for _,y in train_set])
        Y_train = Y_train.reshape(-1, 1)
        
        if validation is None:
            X_train = [V_train, A_train, T_train, C_train]

            (V_val, A_val, T_val, C_val), Y_val = val_set
            X_val = [V_val, A_val, T_val, C_val]
        else:
            V_train, V_val, A_train, A_val, T_train, T_val, C_train, C_val, Y_train, Y_val = train_test_split(V_train, A_train, T_train, C_train, Y_train, test_size=validation, random_state=random_state, shuffle=shuffle)

            X_train = [V_train, A_train, T_train, C_train]
            X_val = [V_val, A_val, T_val, C_val]

        (V_test, A_test, T_test, C_test), Y_test = test_set
        X_test = [V_test, A_test, T_test, C_test]

        return X_train, Y_train, X_val, Y_val, X_test,Y_test

    def train(self, model, X_train, Y_train, X_val, Y_val, out_dir, epochs=1000, batch_size=32, validation_split=0.1, verbose=0, shuffle=False):
        tf.keras.backend.clear_session()
        history = None
        if not os.path.isdir(out_dir):
            os.mkdir(out_dir)
        # fit network
        checkpoint_filepath = out_dir + '/checkpoint_weight'
        model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_filepath,
            save_weights_only=True,
            monitor='val_root_mean_squared_error',
            mode='min',
            save_best_only=True)
        es = tf.keras.callbacks.EarlyStopping(monitor='val_root_mean_squared_error', mode='min', verbose=1, patience=25)
        history = model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_val, Y_val), verbose=verbose, callbacks=[model_checkpoint_callback, es])
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title("Model Loss")
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend(['Train', 'Valid'])

        plt.savefig(out_dir + '/history_loss.png', format='png')
        plt.close()

        # model.save(out_dir + '/checkpoint.h5', save_format="h5")

        print("Train successful")

        return model, history
    
    def test(self, model, X_test, Y_test, out_dir, starting_point=0, pin=None):
        Y_pred = model.predict(X_test)
        peak_idx = peak_visualization(Y_test=Y_test,points_near=0,pin=pin)
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
        plt.close()

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

