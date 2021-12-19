from sklearn.datasets import make_classification
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import BatchNormalization
from matplotlib import pyplot
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score

import pandas as pd
import numpy as np

def ae_train():
    print("TRAINING")
    df = pd.read_csv("SWaT_Dataset_Normal_v1.csv")
    df = df.iloc[21600:]

    y = df["Normal/Attack"]
    timestamp = df["timestamp"]
    del df["Normal/Attack"]
    del df["timestamp"]

    X = np.array(df)
    n_inputs = X.shape[1]
    # split into train test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)
    # scale data
    t = MinMaxScaler()
    t.fit(X_train)
    X_train = t.transform(X_train)
    X_test = t.transform(X_test)

    visible = Input(shape=(n_inputs,))

    e = Dense(n_inputs*2)(visible)
    e = BatchNormalization()(e)
    e = LeakyReLU()(e)

    e = Dense(n_inputs)(e)
    e = BatchNormalization()(e)
    e = LeakyReLU()(e)

    n_bottleneck = round(float(n_inputs) / 2.0)
    bottleneck = Dense(n_bottleneck)(e)

    d = Dense(n_inputs)(bottleneck)
    d = BatchNormalization()(d)
    d = LeakyReLU()(d)

    d = Dense(n_inputs*2)(d)
    d = BatchNormalization()(d)
    d = LeakyReLU()(d)

    output = Dense(n_inputs, activation='linear')(d)

    model = Model(inputs=visible, outputs=output)

    model.compile(optimizer='adam', loss='mse')

    history = model.fit(X_train, X_train, epochs=200, batch_size=512, verbose=2, validation_data=(X_test,X_test))

    encoder = Model(inputs=visible, outputs=bottleneck)

    pyplot.plot(history.history['loss'], label='train')
    pyplot.plot(history.history['val_loss'], label='test')
    pyplot.legend()
    pyplot.savefig("ae_history_chart.png")
    
    
def ae_test(X):
    encoder = load_model('encoder.h5', compile=False)
    return encoder.predict(X)

if __name__ == '__main__':
    ae_train()