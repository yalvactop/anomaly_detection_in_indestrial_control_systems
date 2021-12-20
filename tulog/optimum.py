# general imports 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler

from utils import plot, plot_ts, plot_rws, plot_error, unroll_ts
from orion.data import load_signal, load_anomalies

from model import hyperparameters
from tadgan import TadGAN
import tensorflow as tf

from orion.primitives.timeseries_anomalies import find_anomalies

from tadgan import score_anomalies

import json

import concurrent.futures
import os

def time_segments_aggregate(X, interval, time_column, method=['mean']):
    """Aggregate values over given time span.
    Args:
        X (ndarray or pandas.DataFrame):
            N-dimensional sequence of values.
        interval (int):
            Integer denoting time span to compute aggregation of.
        time_column (int):
            Column of X that contains time values.
        method (str or list):
            Optional. String describing aggregation method or list of strings describing multiple
            aggregation methods. If not given, `mean` is used.
    Returns:
        ndarray, ndarray:
            * Sequence of aggregated values, one column for each aggregation method.
            * Sequence of index values (first index of each aggregated segment).
    """
    if isinstance(X, np.ndarray):
        X = pd.DataFrame(X)

    X = X.sort_values(time_column).set_index(time_column)

    if isinstance(method, str):
        method = [method]

    start_ts = X.index.values[0]
    max_ts = X.index.values[-1]

    values = list()
    index = list()
    while start_ts <= max_ts:
        end_ts = start_ts + interval
        subset = X.loc[start_ts:end_ts]
        aggregated = [
            getattr(subset, agg)(skipna=True).values
            for agg in method
        ]
        values.append(np.concatenate(aggregated))
        index.append(start_ts)
        start_ts = end_ts

    return np.asarray(values), np.asarray(index)


def rolling_window_sequences(X, index, window_size, target_size, step_size, target_column,
                             drop=None, drop_windows=False):
    """Create rolling window sequences out of time series data.
    The function creates an array of input sequences and an array of target sequences by rolling
    over the input sequence with a specified window.
    Optionally, certain values can be dropped from the sequences.
    Args:
        X (ndarray):
            N-dimensional sequence to iterate over.
        index (ndarray):
            Array containing the index values of X.
        window_size (int):
            Length of the input sequences.
        target_size (int):
            Length of the target sequences.
        step_size (int):
            Indicating the number of steps to move the window forward each round.
        target_column (int):
            Indicating which column of X is the target.
        drop (ndarray or None or str or float or bool):
            Optional. Array of boolean values indicating which values of X are invalid, or value
            indicating which value should be dropped. If not given, `None` is used.
        drop_windows (bool):
            Optional. Indicates whether the dropping functionality should be enabled. If not
            given, `False` is used.
    Returns:
        ndarray, ndarray, ndarray, ndarray:
            * input sequences.
            * target sequences.
            * first index value of each input sequence.
            * first index value of each target sequence.
    """
    out_X = list()
    out_y = list()
    X_index = list()
    y_index = list()
    target = X[:, target_column]
#     print("X.shape: ", X.shape)
#     print("target.shape: ", target.shape)

    if drop_windows:
        if hasattr(drop, '__len__') and (not isinstance(drop, str)):
            if len(drop) != len(X):
                raise Exception('Arrays `drop` and `X` must be of the same length.')
        else:
            if isinstance(drop, float) and np.isnan(drop):
                drop = np.isnan(X)
            else:
                drop = X == drop

    start = 0
    max_start = len(X) - window_size - target_size + 1
    while start < max_start:
        end = start + window_size

        if drop_windows:
            drop_window = drop[start:end + target_size]
            to_drop = np.where(drop_window)[0]
            if to_drop.size:
                start += to_drop[-1] + 1
                continue

        out_X.append(X[start:end])
        out_y.append(target[end:end + target_size])
        X_index.append(index[start])
        y_index.append(index[end])
        start = start + step_size

    return np.asarray(out_X), np.asarray(out_y), np.asarray(X_index), np.asarray(y_index)


def run_tadgan(df_train, df_test_init, techniques):
    
#     signal = 'SWaT_Dataset_Attack_v1.csv'
#     df_test = pd.read_csv(signal)
    df_test = df_test_init.iloc[:7*len(df_test_init.index)//8]

    window_size = 100 #these hyperparameters will be defined after grid search
    epoch = 50
    learning_rate = 0.0005
    latent_dim = 20
    batch_size = 16
    comb = "mult"
    step_size = 1
    drop_windows = False

    prev_state = False
    anomalies = []
    for ind in df_test.index:
        if prev_state == False and df_test['Normal/Attack'][ind] == True:
            start = df_test['timestamp'][ind]
        if prev_state == True and df_test['Normal/Attack'][ind] == False:
            stop = df_test['timestamp'][ind-1]
            anomalies.append([start, stop])

        prev_state = df_test['Normal/Attack'][ind]


    known_anomalies = pd.DataFrame(anomalies, columns=['start', 'end'])
    print("known_anomalies:")
    print(known_anomalies)

    del df_train["Normal/Attack"]
    del df_test["Normal/Attack"]

    X_tsa_train, index_train = time_segments_aggregate(df_train, interval=1000000000, time_column='timestamp')
    X_tsa_test, index_test = time_segments_aggregate(df_test, interval=1000000000, time_column='timestamp')

    imp = SimpleImputer()
    X_imp_test = imp.fit_transform(X_tsa_test)
    X_imp_train = imp.fit_transform(X_tsa_train)

    scaler = MinMaxScaler(feature_range=(-1, 1)) ## for the gradients to converge faster == try 0-1 StandardScaler()
    X_scl_train = scaler.fit_transform(X_imp_train)
    X_scl_test = scaler.fit_transform(X_imp_test)


    X_rws_train, y_train, X_index_train, y_index_train = rolling_window_sequences(X_scl_train, index_train, 
                                                      window_size=window_size, 
                                                      target_size=X_scl_train.shape[1], 
                                                      step_size=step_size,
                                                      target_column=X_scl_train.shape[1]-1)

    X_rws_test, y_test, X_index_test, y_index_test = rolling_window_sequences(X_scl_test, index_test, 
                                                      window_size=window_size, 
                                                      target_size=X_scl_test.shape[1], 
                                                      step_size=step_size,
                                                      target_column=X_scl_test.shape[1]-1)

    ##set model architecture

    hyperparameters["epochs"] = epoch

    hyperparameters["shape"] = (window_size, X_rws_train.shape[2]) # based on the window size
    hyperparameters["critic_x_input_shape"] = (window_size, X_rws_train.shape[2])
    hyperparameters["encoder_input_shape"] = (window_size, X_rws_train.shape[2])
    hyperparameters["layers_generator"][1]["parameters"]["units"] = window_size//2
    hyperparameters["generator_reshape_shape"] = (window_size//2, 1)
    hyperparameters["layers_encoder"][0]["parameters"]["layer"]["parameters"]["units"] = window_size
    hyperparameters["layers_critic_z"][1]["parameters"]["units"] = window_size
    hyperparameters["layers_critic_z"][4]["parameters"]["units"] = window_size

    hyperparameters["optimizer"] = "keras.optimizers.Adam"
    hyperparameters["learning_rate"] = learning_rate

    hyperparameters["latent_dim"] = latent_dim
    hyperparameters["generator_input_shape"] = (latent_dim, 1)
    hyperparameters["critic_z_input_shape"] = (latent_dim, 1)
    hyperparameters["encoder_reshape_shape"] = (latent_dim, 1)
    hyperparameters["layers_encoder"][2]["parameters"]["units"] = latent_dim

    hyperparameters["batch_size"] = batch_size
    hyperparameters["layers_generator"][3]["parameters"]["layer"]["parameters"]["units"] = batch_size
    hyperparameters["layers_generator"][5]["parameters"]["layer"]["parameters"]["units"] = batch_size
    hyperparameters["layers_critic_x"][0]["parameters"]["filters"] = batch_size
    hyperparameters["layers_critic_x"][3]["parameters"]["filters"] = batch_size
    hyperparameters["layers_critic_x"][6]["parameters"]["filters"] = batch_size
    hyperparameters["layers_critic_x"][9]["parameters"]["filters"] = batch_size

    hyperparameters["layers_generator"][6]["parameters"]["layer"]["parameters"]["units"] = X_rws_train.shape[2]
    hyperparameters["layers_critic_x"][-1]["parameters"]["units"] = X_rws_train.shape[2]

    tgan = TadGAN(**hyperparameters)
    tgan.fit(X_rws_train)
    tgan.save_model(str(hyperparameters["epochs"]) + techniques + "_GPU_train")

    X_hat, critic = tgan.predict(X_rws_test) # predict using model

    error, true_index, true, pred = score_anomalies(X_rws_test, X_hat, critic, X_index_test, rec_error_type="dtw", comb="mult")
    #find the anomaly score of the prediction
    pred = np.array(pred).mean(axis=2)
    

    intervals_window = find_anomalies(error, index_test, 
                           window_size_portion=0.33, 
                           window_step_size_portion=0.1, 
                           fixed_threshold=True) # leave this part for now
    
    if len(intervals_window) == 0:
        return "NO ANOMALIES FOUND"
    
    anomalies_window = pd.DataFrame(intervals_window, columns=['start', 'end', 'score']) #find the anomalies to compare with the new anomalies extracted at the beginning
    del anomalies_window["score"]

    score = 0
    overall_count = 0
    for ind in range(len(known_anomalies)):# compare known and predicted anomalies to find the score. we can use this score to compare the efficacy of the feature selection algorithms
        for i in range(known_anomalies["start"][ind], known_anomalies["end"][ind], 1000000000):
            overall_count += 1
            for j in range(len(anomalies_window)):
                if anomalies_window["start"][j] <= i <= anomalies_window["end"][j]:
                    score += 1

    with open('tuning/results', 'a') as fout:
        fout.write(str(score) + " / " + str(overall_count) + "  " + "window_size-" + str(window_size) + "_epoch-" + str(epoch) + "_learning_rate-0.0005_latent_dim-" + str(latent_dim) + "_batch_size-512_comb-mult_" + techniques)
        fout.write("\n")

    fig1, ax1 = plt.subplots()
    ax1.plot(range(len(tgan.total_cx_loss)), np.array(tgan.total_cx_loss)[:,0], "-b")
    ax1.plot(range(len(tgan.total_cx_loss)), np.array(tgan.total_cx_loss)[:,1], "-g")
    ax1.plot(range(len(tgan.total_cx_loss)), np.array(tgan.total_cx_loss)[:,-2], "-y")
    ax1.plot(range(len(tgan.total_cx_loss)), np.array(tgan.total_cx_loss)[:,-1], "-r")
    ax1.set_title("CX")
    fig1.savefig('tuning/cx_window_size-' + str(window_size) + "_" + techniques + '_epoch-' + str(epoch) + '_learning_rate-0.0005_latent_dim-' + str(latent_dim) + '_batch_size-512_comb-mult.png')
    fig2, ax2 = plt.subplots()
    ax2.plot(range(len(tgan.total_cz_loss)), np.array(tgan.total_cz_loss)[:,0], "-b")
    ax2.plot(range(len(tgan.total_cz_loss)), np.array(tgan.total_cz_loss)[:,1], "-g")
    ax2.plot(range(len(tgan.total_cz_loss)), np.array(tgan.total_cz_loss)[:,-2], "-y")
    ax2.plot(range(len(tgan.total_cz_loss)), np.array(tgan.total_cz_loss)[:,-1], "-r")
    ax2.set_title("CZ")
    fig2.savefig('tuning/cz_window_size-' + str(window_size) + "_" + techniques + '_epoch-' + str(epoch) + '_learning_rate-0.0005_latent_dim-' + str(latent_dim) + '_batch_size-512_comb-mult.png')

    fig3, ax3 = plt.subplots()
    ax3.plot(range(len(tgan.total_g_loss)), np.array(tgan.total_g_loss)[:,0], "-b")
    ax3.plot(range(len(tgan.total_g_loss)), np.array(tgan.total_g_loss)[:,1], "-g")
    ax3.plot(range(len(tgan.total_g_loss)), np.array(tgan.total_g_loss)[:,-2], "-y")
    ax3.plot(range(len(tgan.total_g_loss)), np.array(tgan.total_g_loss)[:,-1], "-r")
    ax3.set_title("G")
    fig3.savefig('tuning/g_window_size-' + str(window_size) + "_" + techniques + '_epoch-' + str(epoch) + '_learning_rate-0.0005_latent_dim-' + str(latent_dim) + '_batch_size-512_comb-mult.png')

    plt.rcParams['figure.figsize'] = [30, 20]
    df_test.plot(x="timestamp")

    for ind in range(len(known_anomalies)):
        plt.axvspan(known_anomalies["start"][ind], known_anomalies["end"][ind], color='red', alpha=0.5)
    for ind in range(len(intervals_window)):
        plt.axvspan(anomalies_window["start"][ind], anomalies_window["end"][ind], color='blue', alpha=0.5)

    plt.savefig('tuning/output_window_size-' + str(window_size) + "_" + techniques + '_epoch-' + str(epoch) + '_learning_rate-0.0005_latent_dim-' + str(latent_dim) + '_batch_size-512_comb-mult.png')


    return str(score) + " / " + str(overall_count)

       
if __name__ == '__main__':
    signal = 'swat.csv'

    df = pd.read_csv(signal)

    #df = df.iloc[:5*len(df.index)//8] #use smaller dataset for test and bigger for real test :)
    #df = df.iloc[:len(df.index)//2]
    df = df.iloc[:7500]

    #window_sizes = [100]#[50, 100,  200,  300,  400,  500,  600,  700,  800,  900, 1000]#
    #latent_dim = [20]#[10, 20, 30, 40, 50]#
    run_tadgan(df, "bare")
    
    #with concurrent.futures.ProcessPoolExecutor(max_workers=25) as executor:
    #    for i in reversed(window_sizes):
    #        for j in reversed(latent_dim):
    #            #params = (i, j) #pack
    #            #executor.submit(main, params)
    #            run_tadgan(df, i, j)
