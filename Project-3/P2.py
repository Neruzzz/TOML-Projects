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

xTrain, xTest, yTrain, yTest = train_test_split(Dataframe.drop(["RefSt", "date", "Sensor_NO2", "Sensor_NO", "Sensor_SO2"], axis = 1), Dataframe["RefSt"], test_size = 0.3)

######################## RIDGE REGRESSION #######################################
rr = linear_model.Ridge()

alphas_rr = np.linspace(1, 1000, num = 10, dtype = int)
coefficients_rr = []
R2_rr = []
RMSE_rr = []
MAE_rr = []

print()
for a in alphas_rr:
    rr.set_params(alpha = a)
    rr.fit(xTrain, yTrain)
    coefficients_rr.append(rr.coef_)
    prediction_rr = rr.predict(xTest)

    print("METRICS FOR ALPHA = " + str(a))
    print("R²: " + str(metrics.r2_score(yTest, prediction_rr)))
    R2_rr.append(metrics.r2_score(yTest, prediction_rr))
    print("RMSE: " + str(metrics.mean_squared_error(yTest, prediction_rr, squared = False)))
    RMSE_rr.append(metrics.mean_squared_error(yTest, prediction_rr, squared = False))
    print("MAE: " + str(metrics.mean_absolute_error(yTest, prediction_rr)))
    MAE_rr.append(metrics.mean_absolute_error(yTest, prediction_rr))
    print()

table_creation(['Alpha', 'R²', 'RMSE', 'MAE'], [alphas_rr, R2_rr, RMSE_rr, MAE_rr], 'P2_rr_table.txt')

ax = plt.gca()
ax.plot(alphas_rr, coefficients_rr)
plt.axis('tight')
plt.legend(("Sensor_O3 coefficient", "Temp coefficient", "RelHum coefficient"))
plt.title("Coefficient values vs alpha values")
plt.xlabel('Alpha value')
plt.ylabel('Coefficient value')
plt.show()


plt.title("Metrics vs alpha value")
plt.xlabel('Alpha value')
plt.ylabel('Metric value')
plt.plot(alphas_rr, R2_rr, color='red', label = "R²")
plt.plot(alphas_rr, RMSE_rr, color='blue', label = "RMSE")
plt.plot(alphas_rr, MAE_rr, color='green', label = "MAE")
plt.legend(loc = "center left")
plt.show()

######################## LASSO REGRESSION #######################################
lasso = linear_model.Lasso()

alphas_lasso = np.linspace(0, 1000, num = 10, dtype = float)
coefficients_lasso = []
R2_lasso = []
RMSE_lasso = []
MAE_lasso = []

print()
for a in alphas_lasso:
    lasso.set_params(alpha = a)
    lasso.fit(xTrain, yTrain)
    coefficients_lasso.append(rr.coef_)
    prediction_lasso = lasso.predict(xTest)

    print("METRICS FOR ALPHA = " + str(a))
    print("R²: " + str(metrics.r2_score(yTest, prediction_lasso)))
    R2_lasso.append(metrics.r2_score(yTest, prediction_lasso))
    print("RMSE: " + str(metrics.mean_squared_error(yTest, prediction_lasso, squared = False)))
    RMSE_lasso.append(metrics.mean_squared_error(yTest, prediction_lasso, squared = False))
    print("MAE: " + str(metrics.mean_absolute_error(yTest, prediction_lasso)))
    MAE_lasso.append(metrics.mean_absolute_error(yTest, prediction_lasso))
    print()

table_creation(['Alpha', 'R²', 'RMSE', 'MAE'], [alphas_lasso, R2_lasso, RMSE_lasso, MAE_lasso], 'P2_lasso_table.txt')

ax = plt.gca()
ax.plot(alphas_lasso, coefficients_lasso)
plt.axis('tight')
plt.legend(("Sensor_O3 coefficient", "Temp coefficient", "RelHum coefficient"))
plt.title("Coefficient values vs alpha values")
plt.xlabel('Alpha value')
plt.ylabel('Coefficient value')
plt.show()


plt.title("Metrics vs alpha value")
plt.xlabel('Alpha value')
plt.ylabel('Metric value')
plt.plot(alphas_lasso, R2_lasso, color='red', label = "R²")
plt.plot(alphas_lasso, RMSE_lasso, color='blue', label = "RMSE")
plt.plot(alphas_lasso, MAE_lasso, color='green', label = "MAE")
plt.legend(loc = "center left")
plt.show()
