import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_selection import SelectKBest, chi2, RFE, SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

def cor_selector(X, y,num_feats):
    cor_list = []
    feature_name = X.columns.tolist()
    # calculate the correlation with y for each feature
    for i in X.columns.tolist():
        cor = np.corrcoef(X[i], y)[0, 1]
        cor_list.append(cor)
    # replace NaN with 0
    cor_list = [0 if np.isnan(i) else i for i in cor_list]
    # feature name
    cor_feature = X.iloc[:,np.argsort(np.abs(cor_list))[-num_feats:]].columns.tolist()
    # feature selection? 0 for not select, 1 for select
    cor_support = [True if i in cor_feature else False for i in feature_name]
    return cor_support, cor_feature

def select_features(df, y_binary):

    feature_names = np.array(df.columns, dtype=str)
    feature_names = feature_names[1:]

    # Separating out the features
    x = df.loc[:, feature_names].values
    x_scl = StandardScaler().fit_transform(x)

    df_scl = pd.DataFrame(data = x_scl)

    #high corr
    cor_support, cor_feature = cor_selector(df_scl, y_binary, len(feature_names))

    #chi
    X_norm = MinMaxScaler().fit_transform(df_scl)
    chi_selector = SelectKBest(chi2, k=len(feature_names))
    chi_selector.fit(X_norm, y_binary)
    chi_support = chi_selector.get_support()
    chi_feature = df_scl.loc[:,chi_support].columns.tolist()

    #rfe
#     rfe_selector = RFE(estimator=LogisticRegression(solver='lbfgs', multi_class='auto', max_iter=1000), n_features_to_select=len(feature_names), step=10, verbose=5)
#     rfe_selector.fit(X_norm, y_binary)
#     rfe_support = rfe_selector.get_support()
#     rfe_feature = df_scl.loc[:,rfe_support].columns.tolist()

    #logistic regression
#     embeded_lr_selector = SelectFromModel(LogisticRegression(solver='lbfgs', multi_class='auto', penalty="l2", max_iter=1000), max_features=len(feature_names))
#     embeded_lr_selector.fit(X_norm, y_binary)

#     embeded_lr_support = embeded_lr_selector.get_support()
#     embeded_lr_feature = df_scl.loc[:,embeded_lr_support].columns.tolist()

    #random forest classifier
    embeded_rf_selector = SelectFromModel(RandomForestClassifier(n_estimators=100), max_features=len(feature_names))
    embeded_rf_selector.fit(df_scl, y_binary)

    embeded_rf_support = embeded_rf_selector.get_support()
    embeded_rf_feature = df_scl.loc[:,embeded_rf_support].columns.tolist()

    #result
    # put all selection together
    feature_selection_df = pd.DataFrame({'Feature':feature_names, 'Pearson':cor_support, 'Chi-2':chi_support,
                                        'Random Forest':embeded_rf_support})
    # count the selected times for each feature
    feature_selection_df['Total'] = np.sum(feature_selection_df, axis=1)
    # display the top 100
    feature_selection_df = feature_selection_df.sort_values(['Total','Feature'] , ascending=False)
    feature_selection_df.index = range(1, len(feature_selection_df)+1)
    return feature_selection_df