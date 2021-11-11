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


def main(window, dim):
    signal = 'swat.csv'

    df = pd.read_csv(signal)

    prev_state = "Normal"
    anomalies = []
    for ind in df.index:
        if prev_state == "Normal" and df['Normal/Attack'][ind] == "Attack":
            start = df['timestamp'][ind]
        if prev_state == "Attack" and df['Normal/Attack'][ind] == "Normal":
            stop = df['timestamp'][ind-1]
            anomalies.append([start, stop])

        prev_state = df['Normal/Attack'][ind]

    known_anomalies = pd.DataFrame(anomalies, columns=['start', 'end']) #identify known anomalies

    del df["Normal/Attack"] # delete because not needed

    df = df.iloc[:5*len(df.index)//8] #use smaller dataset for test and bigger for real test :)
    #df = df.iloc[:7500]                                   #CHANGE THIS!!

    X = df
    interval=1000000000
    time_column='timestamp'
    method=['mean']

    if isinstance(X, np.ndarray):
        X = pd.DataFrame(X)

    X = X.sort_values(time_column).set_index(time_column)

    if isinstance(method, str):
        method = [method]

    start_ts = X.index.values[0]
    max_ts = X.index.values[-1]

    values = list()
    index = list()
    while start_ts <= max_ts:# find time segments and create a new array
        end_ts = start_ts + interval
        subset = X.loc[start_ts:end_ts]
        aggregated = [
            getattr(subset, agg)(skipna=True).values
            for agg in method
        ]
        values.append(np.concatenate(aggregated))
        index.append(start_ts)
        start_ts = end_ts

    X_tsa = np.asarray(values)
    index = np.asarray(index)

    imp = SimpleImputer()
    X_imp = imp.fit_transform(X_tsa) #imputer

    scaler = MinMaxScaler(feature_range=(-1, 1)) #normalize between -1 and 1
    X_scl = scaler.fit_transform(X_imp)

    X = X_scl
    window_size = window #these hyperparameters will be defined after grid search
    epoch = 200
    learning_rate = 0.0005
    latent_dim = dim
    batch_size = 512
    comb = "mult"

    target_size=51
    step_size=1
    target_column=50
    drop_windows = False

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
    while start < max_start:#create new array based on the window size
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

    X_rws = np.asarray(out_X)
    y = np.asarray(out_y)
    X_index = np.asarray(X_index)
    y_index = np.asarray(y_index)

    ##set model architecture

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

    tgan = TadGAN(**hyperparameters) #train model
    tgan.fit(X_rws)

    X_hat, critic = tgan.predict(X_rws) # predict using model

    error, true_index, true, pred = score_anomalies(X_rws, X_hat, critic, X_index, rec_error_type="dtw", comb=comb) 
    #find the anomaly score of the prediction
    pred = np.array(pred).mean(axis=2)

    intervals_window = find_anomalies(error, index, 
                               window_size_portion=0.33, 
                               window_step_size_portion=0.1, 
                               fixed_threshold=True) # leave this part for now
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
        fout.write(str(score) + " / " + str(overall_count) + "  " + "g_5_8dataset_window_size-" + str(window) + "_epoch-200_learning_rate-0.0005_latent_dim-" + str(dim) + "_batch_size-512_comb-mult")
        fout.write("\n")

    fig1, ax1 = plt.subplots()
    ax1.plot(range(len(tgan.total_cx_loss)), np.array(tgan.total_cx_loss)[:,0], "-b")
    ax1.plot(range(len(tgan.total_cx_loss)), np.array(tgan.total_cx_loss)[:,1], "-g")
    ax1.plot(range(len(tgan.total_cx_loss)), np.array(tgan.total_cx_loss)[:,-2], "-y")
    ax1.plot(range(len(tgan.total_cx_loss)), np.array(tgan.total_cx_loss)[:,-1], "-r")
    ax1.set_title("CX")
    fig1.savefig('tuning/cx_5_8dataset_window_size-' + str(window) + '_epoch-200_learning_rate-0.0005_latent_dim-' + str(dim) + '_batch_size-512_comb-mult.png')
    fig2, ax2 = plt.subplots()
    ax2.plot(range(len(tgan.total_cz_loss)), np.array(tgan.total_cz_loss)[:,0], "-b")
    ax2.plot(range(len(tgan.total_cz_loss)), np.array(tgan.total_cz_loss)[:,1], "-g")
    ax2.plot(range(len(tgan.total_cz_loss)), np.array(tgan.total_cz_loss)[:,-2], "-y")
    ax2.plot(range(len(tgan.total_cz_loss)), np.array(tgan.total_cz_loss)[:,-1], "-r")
    ax2.set_title("CZ")
    fig2.savefig('tuning/cz_5_8dataset_window_size-' + str(window) + '_epoch-200_learning_rate-0.0005_latent_dim-' + str(dim) + '_batch_size-512_comb-mult.png')

    fig3, ax3 = plt.subplots()
    ax3.plot(range(len(tgan.total_g_loss)), np.array(tgan.total_g_loss)[:,0], "-b")
    ax3.plot(range(len(tgan.total_g_loss)), np.array(tgan.total_g_loss)[:,1], "-g")
    ax3.plot(range(len(tgan.total_g_loss)), np.array(tgan.total_g_loss)[:,-2], "-y")
    ax3.plot(range(len(tgan.total_g_loss)), np.array(tgan.total_g_loss)[:,-1], "-r")
    ax3.set_title("G")
    fig3.savefig('tuning/g_5_8dataset_window_size-' + str(window) + '_epoch-200_learning_rate-0.0005_latent_dim-' + str(dim) + '_batch_size-512_comb-mult.png')

    plt.rcParams['figure.figsize'] = [30, 20]
    df.plot(x="timestamp")

    for ind in range(35):
        plt.axvspan(known_anomalies["start"][ind], known_anomalies["end"][ind], color='red', alpha=0.5)
    for ind in range(len(intervals_window)):
        plt.axvspan(anomalies_window["start"][ind], anomalies_window["end"][ind], color='blue', alpha=0.5)

    plt.savefig('tuning/output_5_8dataset_window_size-' + str(window) + '_epoch-200_learning_rate-0.0005_latent_dim-' + str(dim) + '_batch_size-512_comb-mult.png')

        
        

if __name__ == '__main__':

    window_sizes = [50, 100,  200,  300,  400,  500,  600,  700,  800,  900, 1000]#[50]#
    latent_dim = [10, 20, 30, 40, 50]#[10]#
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=40) as executor:
        for i in reversed(window_sizes):
            for j in reversed(latent_dim):
                try:
                    params = (i, j) #pack
                    executor.submit(main, params)
                    #main(i, j)
                except:
                    continue
