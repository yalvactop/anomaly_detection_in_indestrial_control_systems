import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from keras import metrics
import keras
import tensorflow as tf
import os


def generate_datasets_for_training(data, window_size,scale=True, scaler_type=StandardScaler):
    _l = len(data) 
    data = scaler_type().fit_transform(data)
    Xs = []
    Ys = []
    for i in range(0, (_l - window_size)):
        # because this is an autoencoder - our Ys are the same as our Xs. No need to pull the next sequence of values
        Xs.append(data[i:i+window_size])
        Ys.append(data[i:i+window_size])
    tr_x, ts_x, tr_y, ts_y = [np.array(x) for x in train_test_split(Xs, Ys)]
    assert tr_x.shape[2] == ts_x.shape[2] == (data.shape[1] if (type(data) == np.ndarray) else len(data))
    return  (tr_x.shape[2], tr_x, tr_y, ts_x, ts_y)


df = pd.read_csv("swat.csv")
df = df.iloc[:5*len(df.index)//8]
#df = df.iloc[:7500]

y = df["Normal/Attack"]
del df["Normal/Attack"]

epochs = 200
batch_size = 512
window_length = 100

early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', min_delta=1e-2, patience=5, verbose=0, mode='auto',
    baseline=None, restore_best_weights=True)

feats, X, Y, XX, YY = generate_datasets_for_training(df, window_length)

model = keras.Sequential()
model.add(keras.layers.LSTM(64, kernel_initializer='he_uniform', batch_input_shape=(None, window_length, feats), return_sequences=True, name='encoder_1'))
model.add(keras.layers.LSTM(32, kernel_initializer='he_uniform', return_sequences=True, name='encoder_2'))
model.add(keras.layers.LSTM(16, kernel_initializer='he_uniform', return_sequences=False, name='encoder_3'))
model.add(keras.layers.RepeatVector(window_length, name='encoder_decoder_bridge'))
model.add(keras.layers.LSTM(16, kernel_initializer='he_uniform', return_sequences=True, name='decoder_1'))
model.add(keras.layers.LSTM(32, kernel_initializer='he_uniform', return_sequences=True, name='decoder_2'))
model.add(keras.layers.LSTM(64, kernel_initializer='he_uniform', return_sequences=True, name='decoder_3'))
model.add(keras.layers.TimeDistributed(keras.layers.Dense(feats)))
model.compile(loss="mse",optimizer='adam')
model.build()
print(model.summary())

model.fit(x=X, y=Y, validation_data=(XX, YY), epochs=epochs, batch_size=batch_size, shuffle=True, callbacks=[early_stop])

model.save("ae")