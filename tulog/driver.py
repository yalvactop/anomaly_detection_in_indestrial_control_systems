from ae_encoder import ae_test
from feature_selection import select_features
from pca import run_pca
from optimum import run_tadgan
import pandas as pd
import numpy as np

def main():
    
    signal = 'swat.csv'
    df = pd.read_csv(signal)
    rows = 37500
    #df = df.iloc[:5*len(df.index)//8]
    df = df.iloc[:rows]
    y = df["Normal/Attack"]
    timestamp = df["timestamp"]
    del df["Normal/Attack"]
    del df["timestamp"]
    
    y = np.array(y, dtype=str)
    y_binary = []
    for i in y:
        if i == "Normal":
            y_binary.append(False)  #false == normal
        else:
            y_binary.append(True)   #true == attack

    y_binary = np.array(y_binary)
    
    #feature selection
    #use the feature selection table to eliminate some features
    feature_selection_df = select_features(df, y_binary)
    selected_features = []
    for i in range(1, len(feature_selection_df["Total"])):
        if feature_selection_df["Total"][i] > 3:
            selected_features.append(feature_selection_df["Feature"][i])
    df_fs = df[selected_features]
    
    #feature extraction
    #pca
    df_pca = run_pca(df)
    print("df_pca.shape: " ,df_pca.shape)
    
    #ae
    X_ae = ae_test(df.to_numpy()) #X is numpy. returns the extracted features dataset
    df_ae = pd.DataFrame(data=X_ae)
    print("X_ae.shape: ", X_ae.shape)
    
    #try these different techniques and their combination with tadgan
    df_pca["Normal/Attack"] = y
    df_pca["timestamp"] = timestamp
    pca_score = run_tadgan(df_pca, "pca_" + str(rows) + "_")
    
    df_ae["Normal/Attack"] = y
    df_ae["timestamp"] = timestamp
    ae_score = run_tadgan(df_ae, "autoencoder_" + str(rows) + "_")
    
    df_fs["Normal/Attack"] = y
    df_fs["timestamp"] = timestamp
    fs_score = run_tadgan(df_fs, "feature_selection_" + str(rows) + "_")
    
    df["Normal/Attack"] = y
    df["timestamp"] = timestamp
    bare_score = run_tadgan(df, "bare_" + str(rows) + "_")
    
    print("bare_score: ", bare_score)
    print("fs_score: ", fs_score)
    print("pca_score: ", pca_score)
    print("ae_score: ", ae_score)

    
if __name__ == '__main__':
    main()