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

from orion.primitives.timeseries_anomalies import find_anomalies
    
from tadgan import score_anomalies

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
    
    window_size = np.linspace(start=50, stop=1050, num=2)
    step_size = np.linspace(start=1, stop=100, num=2)
    epoch = np.linspace(start=1, stop=10, num=2)
    learning_rate = np.linspace(start=1, stop=10, num=2)
    latent_dim = np.linspace(start=1, stop=10, num=2)
    rec_error_type = ["dtw", "point", "area"]
    comb = ["mult", "sum", "rec"]
    
#     uuuu.append(window_size)
#     uuuu.append(step_size)
#     uuuu.append(epoch)
#     uuuu.append(learning_rate)
#     uuuu.append(latent_dim)
#     uuuu.append(rec_error_type)
#     uuuu.append(comb)
    
#     for item in uuuu:

    #df = pd.read_csv('result.csv')
    res = dict()
    res["window_size"] = 100
    res["step_size"] = 1
    res["epoch"] = 10
    res["learning_rate"] = 0.0005
    res["latent_dim"] = 20
    res["rec_error_type"] = "dtw"
    res["comb"] = "mult"
    res["score"] = tune(X_scl, index, known_anomalies)
    res_df = pd.DataFrame.from_dict(res)
    res_df.to_csv('tuning/result.csv', index=False, sep=',', encoding='utf-8')
    
    print(tune(X_scl, index, known_anomalies))
    
    
def tune(X_scl, index, known_anomalies, window_size=100, step_size=1, epoch=10, learning_rate=0.0005, latent_dim=20, rec_error_type="dtw", comb="mult"):
    
    X_rws, y, X_index, y_index = rolling_window_sequences(X_scl, index, 
                                                      window_size=100, 
                                                      target_size=51, 
                                                      step_size=1,
                                                      target_column=50)
    
    hyperparameters["epochs"] = 10
    
    hyperparameters["shape"] = (100, 51) # based on the window size
    hyperparameters["critic_x_input_shape"] = (100, 51)
    hyperparameters["encoder_input_shape"] = (100, 51)
    
    hyperparameters["learning_rate"] = 0.0005
    
    hyperparameters["latent_dim"] = 20
    hyperparameters["generator_input_shape"] = (20, 1)
    hyperparameters["critic_z_input_shape"] = (20, 1)
    hyperparameters["encoder_reshape_shape"] = (20, 1)
    hyperparameters["layers_encoder"][2]["parameters"]["units"] = 20
    
    hyperparameters["batch_size"] = 64


    tgan = TadGAN(**hyperparameters)
    tgan.fit(X_rws)
    
    X_hat, critic = tgan.predict(X_rws)

    error, true_index, true, pred = score_anomalies(X_rws, X_hat, critic, X_index, rec_error_type="dtw", comb="mult")
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
                    
    return score
    
main()