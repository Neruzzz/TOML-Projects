import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.kernel_ridge import KernelRidge
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


####################################### KRR POLY #####################################

'''krr = KernelRidge(kernel='poly', degree = 3) # poly with degree 3

alpha_krr = np.linspace(0, 1000, num = 10, dtype = float)
coefficients_krr = []

R2_krr = []
RMSE_krr = []
MAE_krr = []

print()
for a in alpha_krr:
    krr.set_params(alpha = a)
    krr.fit(xTrain, yTrain)
    prediction_krr = krr.predict(xTest)

    Predictions['KRR_Prediction'] = prediction_krr

    print("KERNEL RIDGE REGRESSION (POLY) WITH ALPHA = " + str(a))
    print("R²: " + str(metrics.r2_score(yTest, prediction_krr)))
    R2_krr.append(metrics.r2_score(yTest, prediction_krr))
    print("RMSE: " + str(metrics.mean_squared_error(yTest, prediction_krr, squared = False)))
    RMSE_krr.append(metrics.mean_squared_error(yTest, prediction_krr, squared = False))
    print("MAE: " + str(metrics.mean_absolute_error(yTest, prediction_krr)))
    MAE_krr.append(metrics.mean_absolute_error(yTest, prediction_krr))
    print()

    ax1 = Predictions.plot(x='date', y='RefSt')
    Predictions.plot(x='date', y='KRR_Prediction', ax=ax1, title='Kernel Ridge Regression POLY with alpha = ' + str(a))
    plt.show()

    sns_krr = sns.lmplot(x='RefSt', y='KRR_Prediction', data=Predictions, fit_reg=True, line_kws={'color': 'orange'}).set(title='Kernel Ridge Regression POLY with alpha = ' + str(a))
    sns_krr.set(ylim=(-2, 3))
    sns_krr.set(xlim=(-2, 3))
    plt.show()


table_creation(['Alpha value', 'R²', 'RMSE', 'MAE'], [alpha_krr, R2_krr, RMSE_krr, MAE_krr], 'P4_krr_poly_table.txt')

plt.title("Kernel Ridge Regression POLY. Metrics vs  alpha value")
plt.xlabel('Alpha value')
plt.ylabel('Metric value')
plt.plot(alpha_krr, R2_krr, color='red', label = "R²")
plt.plot(alpha_krr, RMSE_krr, color='blue', label = "RMSE")
plt.plot(alpha_krr, MAE_krr, color='green', label = "MAE")
plt.legend(loc = "center left")
plt.show()'''



####################################### KRR RBF #####################################

krr = KernelRidge(kernel='rbf')

alpha_krr = np.linspace(1, 50, num = 10, dtype = float)
coefficients_krr = []

R2_krr = []
RMSE_krr = []
MAE_krr = []

print()
for a in alpha_krr:
    krr.set_params(alpha = a)
    krr.fit(xTrain, yTrain)
    prediction_krr = krr.predict(xTest)

    Predictions['KRR_Prediction'] = prediction_krr

    print("KERNEL RIDGE REGRESSION (RBF) WITH ALPHA = " + str(a))
    print("R²: " + str(metrics.r2_score(yTest, prediction_krr)))
    R2_krr.append(metrics.r2_score(yTest, prediction_krr))
    print("RMSE: " + str(metrics.mean_squared_error(yTest, prediction_krr, squared = False)))
    RMSE_krr.append(metrics.mean_squared_error(yTest, prediction_krr, squared = False))
    print("MAE: " + str(metrics.mean_absolute_error(yTest, prediction_krr)))
    MAE_krr.append(metrics.mean_absolute_error(yTest, prediction_krr))
    print()

    ax1 = Predictions.plot(x='date', y='RefSt')
    Predictions.plot(x='date', y='KRR_Prediction', ax=ax1, title='Kernel Ridge Regression RBF with alpha = ' + str(a))
    plt.show()

    sns_krr = sns.lmplot(x='RefSt', y='KRR_Prediction', data=Predictions, fit_reg=True, line_kws={'color': 'orange'}).set(title='Kernel Ridge Regression RBF with alpha = ' + str(a))
    sns_krr.set(ylim=(-2, 3))
    sns_krr.set(xlim=(-2, 3))
    plt.show()


table_creation(['Alpha value', 'R²', 'RMSE', 'MAE'], [alpha_krr, R2_krr, RMSE_krr, MAE_krr], 'P4_krr_rbf_table.txt')

plt.title("Kernel Ridge Regression RBF. Metrics vs  alpha value")
plt.xlabel('Alpha value')
plt.ylabel('Metric value')
plt.plot(alpha_krr, R2_krr, color='red', label = "R²")
plt.plot(alpha_krr, RMSE_krr, color='blue', label = "RMSE")
plt.plot(alpha_krr, MAE_krr, color='green', label = "MAE")
plt.legend(loc = "center left")
plt.show()