from ae_encoder import ae_test
from feature_selection import select_features
from pca import run_pca
from optimum import run_tadgan
import pandas as pd
import numpy as np
import tensorflow as tf

def main():
    
    signal = 'SWaT_Dataset_Normal_v1.csv'
    df = pd.read_csv(signal)

    rows = len(df.index)
    print(df.head())
    print()
    print("ROW COUNT: ", rows)
    print()
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
#     feature_selection_df = select_features(df, y_binary)
#     selected_features = []
#     for i in range(1, len(feature_selection_df["Total"])):
#         if feature_selection_df["Total"][i] > 3:
#             selected_features.append(feature_selection_df["Feature"][i])
#     df_fs = df[selected_features]
#     print()
#     print("df_fs.shape: ", df_fs.shape)
#     print()

#     #feature extraction
#     #pca
#     df_pca_dim = run_pca(df, 5) # keep only 5 dimensions
#     print()
#     print("df_pca_dim.shape: " ,df_pca_dim.shape)
#     print()
#     df_pca_var = run_pca(df, .99)# keep as many dimension as need as long as the variance is 0.99
#     print()
#     print("df_pca_var.shape: " ,df_pca_var.shape)
#     print()

#     #ae
#     X_ae = ae_test(df.to_numpy()) #X is numpy. returns the extracted features dataset
#     df_ae = pd.DataFrame(data=X_ae)
#     print("X_ae.shape: ", X_ae.shape)

#     #try these different techniques and their combination with tadgan
#     df_pca_dim["Normal/Attack"] = y
#     df_pca_dim["timestamp"] = timestamp
#     print()
#     print("Training for PCA for 5 dimensions")
#     print()
#     pca_score = run_tadgan(df_pca_dim, str(rows) + "_rows_GPU_FINAL_pca_dim-5")

#     df_pca_var["Normal/Attack"] = y
#     df_pca_var["timestamp"] = timestamp
#     print()
#     print("Training for PCA for .99 variance")
#     print()
#     pca_score = run_tadgan(df_pca_var, str(rows) + "_rows_GPU_FINAL_rows_pca_var-099")

#     df_ae["Normal/Attack"] = y
#     df_ae["timestamp"] = timestamp
#     print()
#     print("Training for autoencoder")
#     print()
#     ae_score = run_tadgan(df_ae, str(rows) + "_rows_GPU_FINAL_rows_autoencoder_")

#     df_fs["Normal/Attack"] = y
#     df_fs["timestamp"] = timestamp
#     print()
#     print("Training for feature selection")
#     print()
#     fs_score = run_tadgan(df_fs, str(rows) + "_rows_GPU_FINAL_rows_feature_selection_")

    df["Normal/Attack"] = y
    df["timestamp"] = timestamp
    print()
    print("Bare training")
    print()
    bare_score = run_tadgan(df, str(rows) + "_rows_GPU_FINAL_rows_bare_")

    print()
    print("bare_score: ", bare_score)
#     print("fs_score: ", fs_score)
#     print("pca_09_score: ", pca_score)
#     print("ae_score: ", ae_score)

    
if __name__ == '__main__':
    main()