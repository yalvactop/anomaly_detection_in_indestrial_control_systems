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

from multiprocessing import Process
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
    print("X.shape: ", X.shape)
    print("target.shape: ", target.shape)

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
    with tf.device("gpu:0"):
        signal = 'swat.csv'

        df = pd.read_csv(signal)

        prev_state = "Normal"
        anomalies = []
        for ind in df.index:
            #print(df['timestamp'][ind], df['Normal/Attack'][ind])
            if prev_state == "Normal" and df['Normal/Attack'][ind] == "Attack":
                start = df['timestamp'][ind]
            if prev_state == "Attack" and df['Normal/Attack'][ind] == "Normal":
                stop = df['timestamp'][ind-1]
                anomalies.append([start, stop])

            prev_state = df['Normal/Attack'][ind]

        known_anomalies = pd.DataFrame(anomalies, columns=['start', 'end'])

        del df["Normal/Attack"]

        #df = df.iloc[:5*len(df.index)//8]
        #df = df.iloc[:len(df.index)//2]
        df = df.iloc[:7500]                                   #CHANGE THIS!!!

        X_tsa, index = time_segments_aggregate(df, interval=1000000000, time_column='timestamp')

        imp = SimpleImputer()
        X_imp = imp.fit_transform(X_tsa)

        scaler = MinMaxScaler(feature_range=(-1, 1)) ## for the gradients to converge faster
        X_scl = scaler.fit_transform(X_imp)

        fig1, ax1 = plt.subplots()
        ax1.plot(X_scl)
        ax1.set_title("X_scl")
        fig1.savefig('tuning/X_scl.png')

        ##################### tuning starts here #####################

        processes = []
        max_processes = 31

        window_size = np.linspace(start=50, stop=1050, num=11, dtype=int)#[100]#
        epoch = [1]#np.linspace(start=50, stop=500, num=10, dtype=int)#[1]#
        learning_rate = [0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1]#[0.0005]#
        latent_dim = [20]#np.linspace(start=10, stop=110, num=11, dtype=int)#[20]#
        batch_size = [64]#[32, 64, 128, 256, 512]#[64]#
        comb = ["mult"]#["mult", "sum", "rec"]#["mult"]#
        
        p_count = 0

        for window in window_size:
            for e in epoch:
                for rate in learning_rate:
                    for dim in latent_dim:
                        for batch in batch_size:
                            for c in comb:
                                if p_count == max_processes:
                                    print("PROCESS COUNT REACHED MAX. WAITING FOR PROCESSES TO BE COMPLETED TO CONTINUE")
                                    for proc in processes:
                                        proc.join()
                                    print("PROCESSES ARE COMPLETED! CONTINUE..")
                                    p_count = 0
                                else:
                                    p = Process(target=tune, args=(X_scl, index, known_anomalies, window, e, rate, dim, batch, c))
                                    processes.append(p)
                                    p.start()
                                    p_count += 1



        #print(tune(X_scl, index, known_anomalies))

    
def tune(X_scl, index, known_anomalies, window_size, epoch, learning_rate, latent_dim, batch_size, comb):
    
    
    X_rws, y, X_index, y_index = rolling_window_sequences(X_scl, index, 
                                                      window_size=window_size, 
                                                      target_size=51, 
                                                      step_size=1,
                                                      target_column=50)
    
    hyperparameters["epochs"] = epoch
    
    hyperparameters["shape"] = (window_size, 51) # based on the window size
    hyperparameters["critic_x_input_shape"] = (window_size, 51)
    hyperparameters["encoder_input_shape"] = (window_size, 51)
    hyperparameters["layers_generator"][1]["parameters"]["units"] = window_size//2
    hyperparameters["generator_reshape_shape"] = (window_size//2, 1)
    hyperparameters["layers_encoder"][0]["parameters"]["layer"]["parameters"]["units"] = window_size
    hyperparameters["layers_critic_z"][1]["parameters"]["units"] = window_size
    hyperparameters["layers_critic_z"][4]["parameters"]["units"] = window_size
    
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
    
                    
    res = dict()
    res["window_size"] = window_size
    res["batch_size"] = batch_size
    res["epoch"] = epoch
    res["learning_rate"] = learning_rate
    res["latent_dim"] = latent_dim
    res["comb"] = comb
    res["batch_size"] = batch_size

    with open('tuning/resultfile', 'a') as fout:
        fout.write(json.dumps(str(os.getpid()) + "  " + str(list(res.values()))))
        fout.write("\n")


    tgan = TadGAN(**hyperparameters)
    tgan.fit(X_rws)
    
    X_hat, critic = tgan.predict(X_rws)

    error, true_index, true, pred = score_anomalies(X_rws, X_hat, critic, X_index, rec_error_type="dtw", comb=comb)
    pred = np.array(pred).mean(axis=2)
    
    # find anomalies
    intervals_window = find_anomalies(error, index, 
                               window_size_portion=0.33, 
                               window_step_size_portion=0.1, 
                               fixed_threshold=True) # leave this part for now
    anomalies_window = pd.DataFrame(intervals_window, columns=['start', 'end', 'score'])
    del anomalies_window["score"]
    
    score = 0
    
    for ind in range(len(known_anomalies)):
        for i in range(known_anomalies["start"][ind], known_anomalies["end"][ind], 1000000000):
            for j in range(len(anomalies_window)):
                if anomalies_window["start"][j] <= i <= anomalies_window["end"][j]:
                    score += 1
                    
    res["score"] = score

    with open('tuning/resultfile', 'a') as fout:
        fout.write(json.dumps(str(os.getpid()) + "  " + str(score)))
        fout.write("\n")
    
main()