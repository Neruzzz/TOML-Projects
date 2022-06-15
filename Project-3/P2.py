import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

Dataframe = pd.read_csv("norm_all_data.csv")
Dataframe["date"] = pd.to_datetime(Dataframe["date"])

xTrain, xTest, yTrain, yTest = train_test_split()