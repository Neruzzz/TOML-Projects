import pandas as pd
from mlxtend.feature_selection import SequentialFeatureSelector as sfs
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt


dir_path = "/Users/Imanol/OneDrive/Escritorio/Master/Q2/TOML/TOML-Projects/Project-3/Data/"
Dataframe = pd.read_csv(dir_path + "norm_all_data.csv")
Dataframe["date"] = pd.to_datetime(Dataframe["date"])


X = Dataframe.drop(["RefSt", "date"], axis = 1)
Y = Dataframe["RefSt"]

lr = LinearRegression()

sfs_rmse = sfs(lr, k_features=3, forward=True, scoring="neg_root_mean_squared_error")
sfs_rmse = sfs_rmse.fit(X,Y)
features_rmse = list(sfs_rmse.k_feature_names_)

sfs_mae = sfs(lr, k_features=3, forward=True, scoring="neg_median_absolute_error")
sfs_mae = sfs_mae.fit(X,Y)
features_mae = list(sfs_mae.k_feature_names_)

sfs_r2 = sfs(lr, k_features=3, forward=True, scoring="r2")
sfs_r2 = sfs_r2.fit(X,Y)
features_r2 = list(sfs_r2.k_feature_names_)

print()
print(features_rmse)
print(features_mae)
print(features_r2)
print()
print(sfs_rmse.subsets_[1]['cv_scores'])
print(sfs_mae.subsets_[1]['cv_scores'])
print(sfs_r2.subsets_[1]['cv_scores'])


