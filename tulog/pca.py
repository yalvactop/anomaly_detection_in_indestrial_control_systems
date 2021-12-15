import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def run_pca(df, pca_param):

    feature_names = np.array(df.columns, dtype=str)

    # Separating out the features
    x = df.loc[:, feature_names[1:]].values
    x = StandardScaler().fit_transform(x)

    pca = PCA(pca_param)
    principalComponents = pca.fit_transform(x)
    principal_df = pd.DataFrame(data = principalComponents)

    print("pca.n_components_: ", pca.n_components_)
    return principal_df