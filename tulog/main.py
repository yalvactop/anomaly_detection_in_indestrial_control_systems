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


def main():
    df_init_train = pd.read_csv('SWaT_Dataset_Normal_v1.csv')
    df_init_test = pd.read_csv('SWaT_Dataset_Attack_v1.csv')
    rows = len(df_init_train.index)
#     rows = 40000
    df_train = df_init_train.iloc[21600:rows]
    df_test = df_init_test.iloc[:7*len(df_init_test.index)//8]
    rows = rows - 21600

    print()
    print("ROW COUNT: ", rows)
    print()

    prev_state = "Normal"
    anomalies = []
    for ind in df_test.index:
        #print(df['timestamp'][ind], df['Normal/Attack'][ind])
        if prev_state == "Normal" and df_test['Normal/Attack'][ind] == "Attack":
            start = df_test['timestamp'][ind]
        if prev_state == "Attack" and df_test['Normal/Attack'][ind] == "Normal":
            stop = df_test['timestamp'][ind-1]
            anomalies.append([start, stop])

        prev_state = df_test['Normal/Attack'][ind]

    known_anomalies = pd.DataFrame(anomalies, columns=['start', 'end'])

    del df_train["Normal/Attack"]  
    del df_test["Normal/Attack"]                                  #CHANGE THIS!!!

    print("Before time segment train")
    X_tsa_train, index_train = time_segments_aggregate(df_train, interval=1000000000, time_column='timestamp')
    print("Before time segment test")
    X_tsa_test, index_test = time_segments_aggregate(df_test, interval=1000000000, time_column='timestamp')

    print("Before imputer")
    imp = SimpleImputer()
    X_imp_train = imp.fit_transform(X_tsa_train)
    X_imp_test = imp.fit_transform(X_tsa_test)

    print("Before minmax scaler")
    scaler = MinMaxScaler(feature_range=(-1, 1)) ## for the gradients to converge faster
    X_scl_train = scaler.fit_transform(X_imp_train)
    X_scl_test = scaler.fit_transform(X_imp_test)

    fig1, ax1 = plt.subplots()
    ax1.plot(X_scl_train)
    ax1.set_title("X_scl_train")
    fig1.savefig('grid_search/X_scl_train.png')

    ##################### tuning starts here #####################

    window_size = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    epoch = [50, 100]
    learning_rate = [0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1]
    latent_dim = [5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    batch_size = [16, 32, 64, 128, 256, 512]
    comb = ["mult", "sum", "rec"]
    
    print("before tuning for loops")

    for e in epoch:
        for window in window_size:
            for rate in learning_rate:
                for dim in latent_dim:
                    for batch in batch_size:
                        for c in comb:
                            try:
                                tf.keras.backend.clear_session()
                                params = (X_scl_train, X_scl_test, index_train, index_test, known_anomalies, window, e, rate, dim, batch, c) #pack
                                tune(params)
                            except Exception as ex:
                                print(ex)
                                print("PATLAK PARAMS: " + str((window, e, rate, dim, batch, c)))

    
def tune(params):
    
    (X_scl_train, X_scl_test, index_train, index_test, known_anomalies, window_size, epoch, learning_rate, latent_dim, batch_size, comb) = params #unpack
    print("before rolling window")
    
    X_rws_train, y_train, X_index_train, y_index_train = rolling_window_sequences(X_scl_train, index_train, 
                                                      window_size=window_size, 
                                                      target_size=X_scl_train.shape[1], 
                                                      step_size=1,
                                                      target_column=X_scl_train.shape[1]-1)
    X_rws_test, y_test, X_index_test, y_index_test = rolling_window_sequences(X_scl_test, index_test, 
                                                      window_size=window_size, 
                                                      target_size=X_scl_test.shape[1], 
                                                      step_size=1,
                                                      target_column=X_scl_test.shape[1]-1)
    print("after rolling window")
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
    
                    
    res = dict()
    res["window_size"] = window_size
    res["epoch"] = epoch
    res["learning_rate"] = learning_rate
    res["latent_dim"] = latent_dim
    res["batch_size"] = batch_size
    res["comb"] = comb


    print("before tgan.fit")
    tgan = TadGAN(**hyperparameters)
    tgan.fit(X_rws_train)
    print("after tgan.fit")
    
    X_hat, critic = tgan.predict(X_rws_test)

    print("after tgan.predict")
    error, true_index, true, pred = score_anomalies(X_rws_test, X_hat, critic, X_index_test, rec_error_type="dtw", comb=comb)
    pred = np.array(pred).mean(axis=2)
    print("after score_anomalies")
    
    # find anomalies
    intervals_window = find_anomalies(error, index_test, 
                               window_size_portion=0.33, 
                               window_step_size_portion=0.1, 
                               fixed_threshold=True) # leave this part for now
    if len(intervals_window) == 0:
        return "NO ANOMALIES FOUND"
    print("after find_anomalies")
    anomalies_window = pd.DataFrame(intervals_window, columns=['start', 'end', 'score'])
    del anomalies_window["score"]
    
    score = 0
    overall_count = 0
    for ind in range(len(known_anomalies)):
        for i in range(known_anomalies["start"][ind], known_anomalies["end"][ind], 1000000000):
            
            overall_count += 1
            for j in range(len(anomalies_window)):
                if anomalies_window["start"][j] <= i <= anomalies_window["end"][j]:
                    score += 1
                    
    print("after score calculation")
    res["score"] = str(score) + " / " + str(overall_count)

    with open('grid_search/result', 'a') as fout:
        fout.write(json.dumps(str(os.getpid()) + "  " + str(list(res.values()))))
        fout.write("\n")
    

main()