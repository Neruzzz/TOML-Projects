import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
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

X = Dataframe.drop(["RefSt", "date"], axis = 1)
Y = Dataframe["RefSt"]


xTrain, xTest, yTrain, yTest = train_test_split(X, Y, test_size = 0.3)

Predictions = pd.DataFrame()
Predictions['RefSt'] = yTest
Predictions['Sensor_O3'] = xTest['Sensor_O3']
Predictions['date'] = Dataframe['date']

knn = KNeighborsRegressor()

neighbors_knn = np.linspace(1, 20, num = 10, dtype = int)
coefficients_knn = []

R2_knn = []
RMSE_knn = []
MAE_knn = []

print()
for n in neighbors_knn:
    knn.set_params(n_neighbors = n)
    knn.fit(xTrain, yTrain)
    prediction_knn = knn.predict(xTest)

    Predictions['KNN_Prediction'] = prediction_knn

    print("KERNEL RIDGE REGRESSION (RBF) WITH ALPHA = " + str(n))
    print("R²: " + str(metrics.r2_score(yTest, prediction_knn)))
    R2_knn.append(metrics.r2_score(yTest, prediction_knn))
    print("RMSE: " + str(metrics.mean_squared_error(yTest, prediction_knn, squared = False)))
    RMSE_knn.append(metrics.mean_squared_error(yTest, prediction_knn, squared = False))
    print("MAE: " + str(metrics.mean_absolute_error(yTest, prediction_knn)))
    MAE_knn.append(metrics.mean_absolute_error(yTest, prediction_knn))
    print()

    ax1 = Predictions.plot(x='date', y='RefSt')
    Predictions.plot(x='date', y='KNN_Prediction', ax=ax1, title='K-Nearest Neighbors with neighbors = ' + str(n))
    plt.show()

    '''sns_knn = sns.lmplot(x='RefSt', y='KNN_Prediction', data=Predictions, fit_reg=True, line_kws={'color': 'orange'}).set(title='K-Nearest Neighbors with neighbors = ' + str(n))
    sns_knn.set(ylim=(-2, 3))
    sns_knn.set(xlim=(-2, 3))
    plt.show()'''


table_creation(['Number of neighbors', 'R²', 'RMSE', 'MAE'], [neighbors_knn, R2_knn, RMSE_knn, MAE_knn], 'P3_knn_table.txt')

plt.title("K-Nearest Neighbors. Metrics vs  Number of neighbors")
plt.xlabel('Number of neighbors')
plt.ylabel('Metric value')
plt.plot(neighbors_knn, R2_knn, color='red', label = "R²")
plt.plot(neighbors_knn, RMSE_knn, color='blue', label = "RMSE")
plt.plot(neighbors_knn, MAE_knn, color='green', label = "MAE")
plt.legend(loc = "center left")
plt.show()