import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn import metrics
import numpy as np

Dataframe = pd.read_csv("norm_all_data.csv")
Dataframe["date"] = pd.to_datetime(Dataframe["date"])

xTrain, xTest, yTrain, yTest = train_test_split(Dataframe.drop(["RefSt", "date"], axis = 1), Dataframe["RefSt"], test_size = 0.3)

rr = linear_model.Ridge()

alphas = np.linspace(1, 100, num = 100, dtype = int)
coefficients = []
R2 = []
RMSE = []
MAE = []

print()
for a in alphas:
    rr.set_params(alpha = a)
    rr.fit(xTrain, yTrain)
    coefficients.append(rr.coef_)
    prediction = rr.predict(xTest)

    print("METRICS FOR ALPHA = " + str(a))
    print("R²: " + str(metrics.r2_score(yTest, prediction)))
    print("RMSE: " + str(metrics.mean_squared_error(yTest, prediction, squared = False)))
    print("MAE: " + str(metrics.mean_absolute_error(yTest, prediction)))
    print()
