import pandas as pd
from mlxtend.feature_selection import SequentialFeatureSelector as sfs
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

Dataframe = pd.read_csv("all_data.csv")
Dataframe["date"] = pd.to_datetime(Dataframe["date"])
Dataframe["Sensor_O3"] = Dataframe["Sensor_O3"].str.replace(".","", regex = True)
Dataframe["Sensor_O3"] = pd.to_numeric(Dataframe["Sensor_O3"])

X = Dataframe.drop(["Sensor_O3", "RefSt", "date"], axis = 1)
Y = Dataframe["RefSt"]

lr = LinearRegression()
sfs_rmse = sfs(lr, k_features=5, forward=True, verbose=2, scoring="neg_root_mean_squared_error")
sfs_rmse = sfs_rmse.fit(X,Y)
features_rmse = list(sfs_rmse.k_feature_names_)
scores_rmse = sfs_rmse.k_score_
print(features_rmse)
print(scores_rmse)

