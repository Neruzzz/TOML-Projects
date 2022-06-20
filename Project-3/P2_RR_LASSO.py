import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn import metrics
import numpy as np
from tabulate import tabulate
import seaborn as sns


def table_creation(headers, data, file):
    table = {}
    for i, h in enumerate(headers):
        table.update({h: data[i]})
    with open("./MetricTables/" + file, 'w', encoding="utf-8") as file:
        file.write(tabulate(table, headers='keys', tablefmt='fancy_grid'))
        file.close()
    return True
dir_path = "/Users/Imanol/OneDrive/Escritorio/Master/Q2/TOML/TOML-Projects/Project-3/Data/"
Dataframe = pd.read_csv(dir_path + "norm_all_data.csv")
Dataframe["date"] = pd.to_datetime(Dataframe["date"])

xTrain, xTest, yTrain, yTest = train_test_split(Dataframe.drop(["RefSt", "date"], axis = 1), Dataframe["RefSt"], test_size = 0.3) # , "Sensor_NO2", "Sensor_NO", "Sensor_SO2"

Predictions = pd.DataFrame()
Predictions['RefSt'] = yTest
Predictions['Sensor_O3'] = xTest['Sensor_O3']
Predictions['date'] = Dataframe['date']

######################## RIDGE REGRESSION #######################################
rr = linear_model.Ridge()

alphas_rr = np.linspace(0, 1000, num = 10, dtype = int)
coefficients_rr = []
R2_rr = []
RMSE_rr = []
MAE_rr = []

print()
for a in alphas_rr:
    rr.set_params(alpha = a)
    rr.fit(xTrain, yTrain)
    prediction_rr = rr.predict(xTest)
    coefficients_rr.append(rr.coef_)
    Predictions['RR_Prediction'] = rr.intercept_ + rr.coef_[0] * xTest['Sensor_O3'] + rr.coef_[1] * xTest['Temp'] + rr.coef_[2] * xTest['RelHum'] + rr.coef_[3] * xTest['Sensor_NO2'] + rr.coef_[4] * xTest['Sensor_NO'] + rr.coef_[5] * xTest['Sensor_SO2']

    print("RIDGE REGRESSION METRICS FOR ALPHA = " + str(a))
    print("R²: " + str(metrics.r2_score(yTest, prediction_rr)))
    R2_rr.append(metrics.r2_score(yTest, prediction_rr))
    print("RMSE: " + str(metrics.mean_squared_error(yTest, prediction_rr, squared = False)))
    RMSE_rr.append(metrics.mean_squared_error(yTest, prediction_rr, squared = False))
    print("MAE: " + str(metrics.mean_absolute_error(yTest, prediction_rr)))
    MAE_rr.append(metrics.mean_absolute_error(yTest, prediction_rr))
    print()

    ax1 = Predictions.plot(x='date', y='RefSt')
    Predictions.plot(x='date', y='RR_Prediction', ax=ax1, title='Ridge Regression for alpha = ' + str(a))
    plt.show()
    '''sns_rr = sns.lmplot(x='RefSt', y='RR_Prediction', data=Predictions, fit_reg=True, line_kws={'color': 'orange'}).set(title='Ridge Regression for alpha = ' + str(a))
    sns_rr.set(ylim=(-2, 3))
    sns_rr.set(xlim=(-2, 3))
    plt.show()'''

table_creation(['Alpha', 'R²', 'RMSE', 'MAE'], [alphas_rr, R2_rr, RMSE_rr, MAE_rr], 'P2_rr_table.txt')

ax2 = plt.gca()
ax2.plot(alphas_rr, coefficients_rr)
plt.axis('tight')
plt.legend(("Sensor_O3 coefficient", "Temp coefficient", "RelHum coefficient", "Sensor NO2", "Sensor NO", "Sensor SO2"))
plt.title("Ridge Regression. Coefficient values vs alpha values")
plt.xlabel('Alpha value')
plt.ylabel('Coefficient value')
plt.show()


plt.title("Ridge Regression. Metrics vs alpha value")
plt.xlabel('Alpha value')
plt.ylabel('Metric value')
plt.plot(alphas_rr, R2_rr, color='red', label = "R²")
plt.plot(alphas_rr, RMSE_rr, color='blue', label = "RMSE")
plt.plot(alphas_rr, MAE_rr, color='green', label = "MAE")
plt.legend(loc = "center left")
plt.show()

######################## LASSO REGRESSION #######################################
lasso = linear_model.Lasso()

alphas_lasso = np.linspace(0, 1, num = 10, dtype = float)
coefficients_lasso = []
R2_lasso = []
RMSE_lasso = []
MAE_lasso = []

print()
for a in alphas_lasso:
    lasso.set_params(alpha = a)
    lasso.fit(xTrain, yTrain)
    coefficients_lasso.append(lasso.coef_)
    prediction_lasso = lasso.predict(xTest)
    Predictions['LASSO_Prediction'] = lasso.intercept_ + lasso.coef_[0] * xTest['Sensor_O3'] + lasso.coef_[1]


    print("LASSO REGRESSION METRICS FOR ALPHA = " + str(a))
    print("R²: " + str(metrics.r2_score(yTest, prediction_lasso)))
    R2_lasso.append(metrics.r2_score(yTest, prediction_lasso))
    print("RMSE: " + str(metrics.mean_squared_error(yTest, prediction_lasso, squared = False)))
    RMSE_lasso.append(metrics.mean_squared_error(yTest, prediction_lasso, squared = False))
    print("MAE: " + str(metrics.mean_absolute_error(yTest, prediction_lasso)))
    MAE_lasso.append(metrics.mean_absolute_error(yTest, prediction_lasso))
    print()

    '''ax3 = Predictions.plot(x='date', y='RefSt')
    Predictions.plot(x='date', y='LASSO_Prediction', ax=ax3, title='Lasso Regression for alpha = ' + str(a))
    plt.show()
    sns_lasso = sns.lmplot(x='RefSt', y='LASSO_Prediction', data=Predictions, fit_reg=True, line_kws={'color': 'orange'}).set(title='Lasso Regression for alpha = ' + str(a))
    sns_lasso.set(ylim=(-2, 3))
    sns_lasso.set(xlim=(-2, 3))
    plt.show()'''

table_creation(['Alpha', 'R²', 'RMSE', 'MAE'], [alphas_lasso, R2_lasso, RMSE_lasso, MAE_lasso], 'P2_lasso_table.txt')



ax4 = plt.gca()
ax4.plot(alphas_lasso, coefficients_lasso)
plt.axis('tight')
plt.legend(("Sensor_O3 coefficient", "Temp coefficient", "RelHum coefficient", "Sensor NO2", "Sensor NO", "Sensor SO2"))
plt.title("Lasso Regression. Coefficient values vs alpha values Lasso Regression")
plt.xlabel('Alpha value')
plt.ylabel('Coefficient value')
plt.show()


plt.title("Lasso Regression. Metrics vs alpha value.")
plt.xlabel('Alpha value')
plt.ylabel('Metric value')
plt.plot(alphas_lasso, R2_lasso, color='red', label = "R²")
plt.plot(alphas_lasso, RMSE_lasso, color='blue', label = "RMSE")
plt.plot(alphas_lasso, MAE_lasso, color='green', label = "MAE")
plt.legend(loc = "center left")
plt.show()

