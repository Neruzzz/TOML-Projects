import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
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

xTrain, xTest, yTrain, yTest = train_test_split(Dataframe.drop(["RefSt", "date"], axis = 1), Dataframe["RefSt"], test_size = 0.3) # , "Sensor_NO2", "Sensor_NO", "Sensor_SO2"

Predictions = pd.DataFrame()
Predictions['RefSt'] = yTest
Predictions['Sensor_O3'] = xTest['Sensor_O3']
Predictions['date'] = Dataframe['date']

SS = StandardScaler()
rf = RandomForestClassifier()

xTrain = SS.fit_transform(xTrain)
xTest = SS.fit_transform(xTest)
yTrain = SS.fit_transform(yTrain)
yTest = SS.fit_transform(yTest)

estimators_rf = np.linspace(0, 1000, num = 10, dtype = int)
coefficients_rf = []

R2_rf = []
RMSE_rf = []
MAE_rf = []

print()
for n in estimators_rf:
    rf.set_params(n_estimators = n)
    rf.fit(xTrain, yTrain)
    prediction_rf = rf.predict(xTest)
    coefficients_rf.append(rf.coef_)
    Predictions['RF_Prediction'] = rf.intercept_ + rf.coef_[0] * xTest['Sensor_O3'] + rf.coef_[1] * xTest['Temp'] + rf.coef_[2] * xTest['RelHum'] + rf.coef_[3] * xTest['Sensor_NO2'] + rf.coef_[4] * xTest['Sensor_NO'] + rf.coef_[5] * xTest['Sensor_SO2']

    print("RANDOM FOREST WITH " + str(n) + " TREES")
    print("R²: " + str(metrics.r2_score(yTest, prediction_rf)))
    R2_rf.append(metrics.r2_score(yTest, prediction_rf))
    print("RMSE: " + str(metrics.mean_squared_error(yTest, prediction_rf, squared = False)))
    RMSE_rf.append(metrics.mean_squared_error(yTest, prediction_rf, squared = False))
    print("MAE: " + str(metrics.mean_absolute_error(yTest, prediction_rf)))
    MAE_rf.append(metrics.mean_absolute_error(yTest, prediction_rf))
    print()

    ax1 = Predictions.plot(x='date', y='RefSt')
    Predictions.plot(x='date', y='RF_Prediction', ax=ax1, title='Random Forest for = ' + str(n) + " trees")
    plt.show()
    sns_rf = sns.lmplot(x='RefSt', y='RF_Prediction', data=Predictions, fit_reg=True, line_kws={'color': 'orange'}).set(title='Random Forest for alpha = ' + str(a))
    sns_rf.set(ylim=(-2, 3))
    sns_rf.set(xlim=(-2, 3))
    plt.show()

table_creation(['Number of trees', 'R²', 'RMSE', 'MAE'], [estimators_rf, R2_rf, RMSE_rf, MAE_rf], 'P5_rf_table.txt')

ax2 = plt.gca()
ax2.plot(estimators_rf, coefficients_rf)
plt.axis('tight')
plt.legend(("Sensor_O3 coefficient", "Temp coefficient", "RelHum coefficient", "Sensor NO2", "Sensor NO", "Sensor SO2"))
plt.title("Random Forest. Coefficient values vs number of estimators")
plt.xlabel('Number of trees')
plt.ylabel('Coefficient value')
plt.show()


plt.title("Random Forest. Metrics vs  number of estimators (trees)")
plt.xlabel('Number of trees')
plt.ylabel('Metric value')
plt.plot(estimators_rf, R2_rf, color='red', label = "R²")
plt.plot(estimators_rf, RMSE_rf, color='blue', label = "RMSE")
plt.plot(estimators_rf, MAE_rf, color='green', label = "MAE")
plt.legend(loc = "center left")
plt.show()





'''classifing the features according to their importance
feature_imp = pd.Series(classifier.feature_importances_,index=[i for i in range(21)]).sort_values(ascending=False)

# printing
print(feature_imp)'''