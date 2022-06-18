import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestRegressor
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

rf = RandomForestRegressor()

estimators_rf = np.linspace(1, 20, num = 5, dtype = int)
coefficients_rf = []

R2_rf = []
RMSE_rf = []
MAE_rf = []

print()
for n in estimators_rf:
    rf.set_params(n_estimators = n)
    rf.fit(xTrain, yTrain)
    prediction_rf = rf.predict(xTest)

    Predictions['RF_Prediction'] = prediction_rf

    print(n, prediction_rf)
    print("RANDOM FOREST WITH " + str(n) + " TREES")
    print("R²: " + str(metrics.r2_score(yTest, prediction_rf)))
    R2_rf.append(metrics.r2_score(yTest, prediction_rf))
    print("RMSE: " + str(metrics.mean_squared_error(yTest, prediction_rf, squared = False)))
    RMSE_rf.append(metrics.mean_squared_error(yTest, prediction_rf, squared = False))
    print("MAE: " + str(metrics.mean_absolute_error(yTest, prediction_rf)))
    MAE_rf.append(metrics.mean_absolute_error(yTest, prediction_rf))
    print()

    ax1 = Predictions.plot(x='date', y='RefSt')
    Predictions.plot(x='date', y='RF_Prediction', ax=ax1, title='Random Forest for ' + str(n) + ' trees.')
    plt.show()

    sns_rf = sns.lmplot(x='RefSt', y='RF_Prediction', data=Predictions, fit_reg=True, line_kws={'color': 'orange'}).set(title='Random Forest for ' + str(n) + ' trees.')
    sns_rf.set(ylim=(-2, 3))
    sns_rf.set(xlim=(-2, 3))
    plt.show()


table_creation(['Number of trees', 'R²', 'RMSE', 'MAE'], [estimators_rf, R2_rf, RMSE_rf, MAE_rf], 'P5_rf_table.txt')


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