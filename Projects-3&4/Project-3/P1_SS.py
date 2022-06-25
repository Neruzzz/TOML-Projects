import pandas as pd
from mlxtend.feature_selection import SequentialFeatureSelector as sfs
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.model_selection import train_test_split
import numpy as np
from tabulate import tabulate
import matplotlib.pyplot as plt

def table_creation(headers, data, file):
    table = {}
    for i, h in enumerate(headers):
        table.update({h: data[i]})
    with open("./MetricTables/"+file, 'w', encoding="utf-8") as file:
        file.write(tabulate(table, headers='keys', tablefmt='fancy_grid'))
        file.close()
    return True

dir_path = "/Users/Imanol/OneDrive/Escritorio/Master/Q2/TOML/TOML-Projects/Project-3/Data/"
Dataframe = pd.read_csv(dir_path + "norm_all_data.csv")
Dataframe["date"] = pd.to_datetime(Dataframe["date"])


X = Dataframe.drop(["RefSt", "date"], axis = 1)
Y = Dataframe["RefSt"]

xTrain, xTest, yTrain, yTest = train_test_split(X, Y, test_size = 0.3)

lr = LinearRegression()

Predictions = pd.DataFrame()
Predictions['RefSt'] = yTest
Predictions['Sensor_O3'] = xTest['Sensor_O3']
Predictions['date'] = Dataframe['date']

features = np.linspace(1, 6, num = 6, dtype = int)

R2_lr = []
RMSE_lr = []
MAE_lr = []

for n in features:

    sfs_r2 = sfs(lr, k_features=int(n), forward=True, scoring="r2")
    sfs_r2 = sfs_r2.fit(xTrain,yTrain)
    features_r2 = list(sfs_r2.k_feature_names_)

    print(features_r2)

    xTrain_n = xTrain[features_r2]
    xTest_n = xTest[features_r2]

    lr.fit(xTrain_n, yTrain)
    prediction_lr = lr.predict(xTest_n)
    Predictions["LR_Prediction"] = prediction_lr

    print()
    print("LINEAR REGRESSION WITH" + str(n) + " FEATURES")
    print("R²: " + str(metrics.r2_score(yTest, prediction_lr)))
    R2_lr.append(metrics.r2_score(yTest, prediction_lr))
    print("RMSE: " + str(metrics.mean_squared_error(yTest, prediction_lr, squared=False)))
    RMSE_lr.append(metrics.mean_squared_error(yTest, prediction_lr, squared=False))
    print("MAE: " + str(metrics.mean_absolute_error(yTest, prediction_lr)))
    MAE_lr.append(metrics.mean_absolute_error(yTest, prediction_lr))
    print()

    ax1 = Predictions.plot(x='date', y='RefSt')
    Predictions.plot(x='date', y='LR_Prediction', ax=ax1, title='Linear regression for ' + str(n) + ' features')
    plt.show()

table_creation(['Number of features', 'R²', 'RMSE', 'MAE'], [features, R2_lr, RMSE_lr, MAE_lr], 'P1_lr_table.txt')


plt.title("Subset Selection Linear Regression. Metrics vs  number of features")
plt.xlabel('Number of features')
plt.ylabel('Metric value')
plt.plot(features, R2_lr, color='red', label = "R²")
plt.plot(features, RMSE_lr, color='blue', label = "RMSE")
plt.plot(features, MAE_lr, color='green', label = "MAE")
plt.legend(loc = "center left")
plt.show()


