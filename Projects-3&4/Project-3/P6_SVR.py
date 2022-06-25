import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn import metrics
from tabulate import tabulate
import seaborn as sns

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

X = Dataframe.drop(["RefSt", "date", "Sensor_NO2", "Sensor_NO", "Sensor_SO2"], axis = 1)
Y = Dataframe["RefSt"]

xTrain, xTest, yTrain, yTest = train_test_split(X, Y, test_size = 0.3)

Predictions = pd.DataFrame()
Predictions['RefSt'] = yTest
Predictions['Sensor_O3'] = xTest['Sensor_O3']
Predictions['date'] = Dataframe['date']

svr = SVR()

C = np.linspace(500 , 1, num = 10, dtype = int)
coefficients_svr = []

R2_svr = []
RMSE_svr = []
MAE_svr = []

print()
for c in C:
    svr.set_params(C = c, gamma = 'auto')
    svr.fit(xTrain, yTrain)
    prediction_svr = svr.predict(xTest)

    Predictions['SVR_Prediction'] = prediction_svr

    print("SVR WITH C = " + str(c))
    print("R²: " + str(metrics.r2_score(yTest, prediction_svr)))
    R2_svr.append(metrics.r2_score(yTest, prediction_svr))
    print("RMSE: " + str(metrics.mean_squared_error(yTest, prediction_svr, squared = False)))
    RMSE_svr.append(metrics.mean_squared_error(yTest, prediction_svr, squared = False))
    print("MAE: " + str(metrics.mean_absolute_error(yTest, prediction_svr)))
    MAE_svr.append(metrics.mean_absolute_error(yTest, prediction_svr))
    print()

    ax1 = Predictions.plot(x='date', y='RefSt')
    Predictions.plot(x='date', y='SVR_Prediction', ax=ax1, title='SVR for C = ' + str(c))
    plt.show()

    '''sns_svr = sns.lmplot(x='RefSt', y='SVR_Prediction', data=Predictions, fit_reg=True, line_kws={'color': 'orange'}).set(title='SVR for C = ' + str(c))
    sns_svr.set(ylim=(-2, 3))
    sns_svr.set(xlim=(-2, 3))
    plt.show()'''


table_creation(['C', 'R²', 'RMSE', 'MAE'], [C, R2_svr, RMSE_svr, MAE_svr], 'P6_svr_table.txt')

plt.title("SVR. Metrics vs  C (regularization)")
plt.xlabel('C')
plt.ylabel('Metric value')
plt.plot(C, R2_svr, color='red', label = "R²")
plt.plot(C, RMSE_svr, color='blue', label = "RMSE")
plt.plot(C, MAE_svr, color='green', label = "MAE")
plt.legend(loc = "center left")
plt.show()
