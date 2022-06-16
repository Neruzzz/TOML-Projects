import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn import metrics
import numpy as np
from tabulate import tabulate

def table_creation(headers, data, file):
    table = {}
    for i, h in enumerate(headers):
        table.update({h: data[i]})
    with open("./"+file, 'w', encoding="utf-8") as file:
        file.write(tabulate(table, headers='keys', tablefmt='fancy_grid'))
        file.close()
    return True

Dataframe = pd.read_csv("norm_all_data.csv")
Dataframe["date"] = pd.to_datetime(Dataframe["date"])

xTrain, xTest, yTrain, yTest = train_test_split(Dataframe.drop(["RefSt", "date"], axis = 1), Dataframe["RefSt"], test_size = 0.3)

rr = linear_model.Ridge()

alphas = np.linspace(1, 1000, num = 500, dtype = int)
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
    R2.append(metrics.r2_score(yTest, prediction))
    print("RMSE: " + str(metrics.mean_squared_error(yTest, prediction, squared = False)))
    RMSE.append(metrics.mean_squared_error(yTest, prediction, squared = False))
    print("MAE: " + str(metrics.mean_absolute_error(yTest, prediction)))
    MAE.append(metrics.mean_absolute_error(yTest, prediction))
    print()

table_creation(['Alpha', 'R²', 'RMSE', 'MAE'], [alphas, R2, RMSE, MAE], 'P2_rr_table.txt')

ax = plt.gca()
ax.plot(alphas, coefficients)
ax.set_xscale('log')
plt.axis('tight')
plt.title("Coefficient values vs alpha values")
plt.xlabel('Alpha value')
plt.ylabel('Coefficient value')
plt.show()


plt.title("Metrics vs alpha value")
plt.xlabel('Alpha value')
plt.ylabel('Metric value')
plt.plot(alphas, R2, color='red', label = "R²")
plt.plot(alphas, RMSE, color='blue', label = "RMSE")
plt.plot(alphas, MAE, color='green', label = "MAE")
plt.legend(loc = "center left")
plt.show()

