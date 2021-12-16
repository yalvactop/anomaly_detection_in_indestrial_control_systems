from ae_encoder import ae_test
from feature_selection import select_features
from pca import run_pca
from optimum import run_tadgan
import pandas as pd
import numpy as np
import tensorflow as tf

def main():
    
    df_init_train = pd.read_csv('SWaT_Dataset_Normal_v1.csv')
    df_init_test = pd.read_csv('SWaT_Dataset_Attack_v1.csv')
    rows = len(df_init_train.index)
    rows = 40000
    df_train = df_init_train.iloc[21600:rows]
    df_test = df_init_test#.iloc[:rows]
    rows = rows - 21600

    print()
    print("ROW COUNT: ", rows)
    print()
    
    
    y_train = df_train["Normal/Attack"]
    timestamp_train = list(df_train["timestamp"])
    del df_train["Normal/Attack"]
    del df_train["timestamp"]

    y_train = np.array(y_train, dtype=str)
    y_binary_train = []
    for i in y_train:
        if i == "Normal":
            y_binary_train.append(False)  #false == normal
        else:
            y_binary_train.append(True)   #true == attack

    y_binary_train = np.array(y_binary_train)
    
    
    y_test = df_test["Normal/Attack"]
    timestamp_test = list(df_test["timestamp"])
    del df_test["Normal/Attack"]
    del df_test["timestamp"]

    y_test = np.array(y_test, dtype=str)
    y_binary_test = []
    for i in y_test:
        if i == "Normal":
            y_binary_test.append(False)  #false == normal
        else:
            y_binary_test.append(True)   #true == attack

    y_binary_test = np.array(y_binary_test)

#     #feature selection
#     #use the feature selection table to eliminate some features
#     feature_selection_df = select_features(df, y_binary)
#     selected_features = []
#     print(feature_selection_df["Total"])
#     for i in range(1, len(feature_selection_df["Total"])):
#         if feature_selection_df["Total"][i] > 3:
#             selected_features.append(feature_selection_df["Feature"][i])
#     df_fs = df[selected_features]
#     print(df_fs)
#     print("df_fs.shape: ", df_fs.shape)
#     print()
#     return

    #feature extraction
    #pca
    df_pca_dim_train, _ = run_pca(df_train, 5) # keep only 5 dimensions
    print()
    print("df_pca_dim_train.shape: " ,df_pca_dim_train.shape)
    print()
    df_pca_var_train, pca_dim = run_pca(df_train, .99)# keep as many dimension as need as long as the variance is 0.99
    print()
    print("df_pca_var_train.shape: " ,df_pca_var_train.shape)
    print()

    #ae
    X_ae_train = ae_test(df_train.to_numpy()) #X is numpy. returns the extracted features dataset
    df_ae_train = pd.DataFrame(data=X_ae_train)
    print("df_ae_train.shape: ", df_ae_train.shape)

#     #feature selection
#     #use the feature selection table to eliminate some features
#     feature_selection_df = select_features(df, y_binary)
#     selected_features = []
#     print(feature_selection_df["Total"])
#     for i in range(1, len(feature_selection_df["Total"])):
#         if feature_selection_df["Total"][i] > 3:
#             selected_features.append(feature_selection_df["Feature"][i])
#     df_fs = df[selected_features]
#     print(df_fs)
#     print("df_fs.shape: ", df_fs.shape)
#     print()
#     return

    #feature extraction
    #pca
    df_pca_dim_test, _ = run_pca(df_test, 5) # keep only 5 dimensions
    print()
    print("df_pca_dim_test.shape: " ,df_pca_dim_test.shape)
    print()
    df_pca_var_test, _ = run_pca(df_test, pca_dim)# keep as many dimension as need as long as the variance is 0.99
    print()
    print("df_pca_var_test.shape: " ,df_pca_var_test.shape)
    print()

    #ae
    X_ae_test = ae_test(df_test.to_numpy()) #X is numpy. returns the extracted features dataset
    df_ae_test = pd.DataFrame(data=X_ae_test)
    print("df_ae_test.shape: ", df_ae_test.shape)

    #try these different techniques and their combination with tadgan
    df_pca_dim_train["Normal/Attack"] = y_binary_train
    df_pca_dim_train["timestamp"] = timestamp_train
    df_pca_dim_test["Normal/Attack"] = y_binary_test
    df_pca_dim_test["timestamp"] = timestamp_test
    print()
    print("Training for PCA for 5 dimensions")
    print()
    pca_dim_score = run_tadgan(df_pca_dim_train, df_pca_dim_test, str(rows) + "_rows_GPU_FINAL_pca_dim-5")

    df_pca_var_train["Normal/Attack"] = y_binary_train
    df_pca_var_train["timestamp"] = timestamp_train
    df_pca_var_test["Normal/Attack"] = y_binary_test
    df_pca_var_test["timestamp"] = timestamp_test
    print()
    print("Training for PCA for .99 variance")
    print()
    pca_var_score = run_tadgan(df_pca_var_train, df_pca_var_test, str(rows) + "_rows_GPU_FINAL_rows_pca_var-099")

    df_ae_train["Normal/Attack"] = y_binary_train
    df_ae_train["timestamp"] = timestamp_train
    df_ae_test["Normal/Attack"] = y_binary_test
    df_ae_test["timestamp"] = timestamp_test
    print()
    print("Training for autoencoder")
    print()
    ae_score = run_tadgan(df_ae_train, df_ae_test, str(rows) + "_rows_GPU_FINAL_rows_autoencoder_")

#     df_fs["Normal/Attack"] = y_binary
#     df_fs["timestamp"] = timestamp
#     print()
#     print("Training for feature selection")
#     print()
#     fs_score = run_tadgan(df_fs, str(rows) + "_rows_GPU_FINAL_rows_feature_selection_")

    df_train["Normal/Attack"] = y_binary_train
    df_train["timestamp"] = timestamp_train
    df_test["Normal/Attack"] = y_binary_test
    df_test["timestamp"] = timestamp_test
    print()
    print("Bare training")
    print()
    bare_score = run_tadgan(df_train, df_test, str(rows) + "_rows_GPU_FINAL_rows_bare_")

    print()
    print("bare_score: ", bare_score)
#     print("fs_score: ", fs_score)
    print("pca_5_score: ", pca_dim_score)
    print("pca_099_score: ", pca_var_score)
    print("ae_score: ", ae_score)

    
if __name__ == '__main__':
    main()