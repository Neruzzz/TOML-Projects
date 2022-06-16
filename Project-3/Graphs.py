import pandas as pd
import matplotlib.pyplot as plt

Dataframe = pd.read_csv("all_data.csv")
Dataframe["date"] = pd.to_datetime(Dataframe["date"])
Dataframe["Sensor_O3"] = Dataframe["Sensor_O3"].str.replace(".","", regex = "True")
Dataframe["Sensor_O3"] = pd.to_numeric(Dataframe["Sensor_O3"])
Dataframe.info()

# Ozone as a function of time
Dataframe.plot(x = "date", y = "Sensor_O3", kind = "line", color = "r")
plt.title("O3 sensor as a function of time")
plt.ylabel("Ozone in KOhms")
plt.show()

# Ozone reference data as a function of time
Dataframe.plot(x = "date", y = "RefSt", kind = "line", color="r")
plt.title("Scatter plot RefSt vs Sensor_O3")
plt.ylabel("Ozone in µg/m³")
plt.show()

# Scatterplots Sensor_O3 vs ...
Dataframe.plot(x = "Sensor_O3", y = "RefSt", kind = "scatter", color="r")
plt.title("Scatter plot RefSt vs Sensor_O3")
plt.ylabel("Ozone in µg/m³")
plt.xlabel("Ozone in KOhms")
plt.show()

Dataframe.plot(x = "Temp", y = "Sensor_O3", kind = "scatter", color="r")
plt.title("Scatter plot Sensor_O3 vs Temp")
plt.ylabel("Ozone in KOhms")
plt.xlabel("Temperature in ºC")
plt.show()

Dataframe.plot(x = "RelHum", y = "Sensor_O3", kind = "scatter", color="r")
plt.title("Scatter plot Sensor_O3 vs RelHum")
plt.ylabel("Ozone in KOhms")
plt.xlabel("RelHum in %")
plt.show()

Dataframe.plot(x = "Sensor_NO2", y = "Sensor_O3", kind = "scatter", color="r")
plt.title("Scatter plot Sensor_O3 vs Sensor_NO2")
plt.ylabel("Ozone in KOhms")
plt.xlabel("Sensor_NO2")
plt.show()

Dataframe.plot(x = "Sensor_NO", y = "Sensor_O3", kind = "scatter", color="r")
plt.title("Scatter plot Sensor_O3 vs Sensor_NO")
plt.ylabel("Ozone in KOhms")
plt.xlabel("Sensor_NO")
plt.show()

Dataframe.plot(x = "Sensor_SO2", y = "Sensor_O3", kind = "scatter", color="r")
plt.title("Scatter plot Sensor_O3 vs Sensor_SO2")
plt.ylabel("Ozone in KOhms")
plt.xlabel("Sensor_SO2")
plt.show()

# Scatterplots RefSt vs ...
Dataframe.plot(x = "Temp", y = "RefSt", kind = "scatter", color="r")
plt.title("Scatter plot RefSt vs Temp")
plt.ylabel("Ozone in µg/m³")
plt.xlabel("Temperature in ºC")
plt.show()

Dataframe.plot(x = "RelHum", y = "RefSt", kind = "scatter", color="r")
plt.title("Scatter plot RefSt vs RelHum")
plt.ylabel("Ozone in µg/m³")
plt.xlabel("RelHum in %")
plt.show()

Dataframe.plot(x = "Sensor_NO2", y = "RefSt", kind = "scatter", color="r")
plt.title("Scatter plot RefSt vs Sensor_NO2")
plt.ylabel("Ozone in µg/m³")
plt.xlabel("Sensor_NO2")
plt.show()

Dataframe.plot(x = "Sensor_NO", y = "RefSt", kind = "scatter", color="r")
plt.title("Scatter plot RefSt vs Sensor_NO")
plt.ylabel("Ozone in µg/m³")
plt.xlabel("Sensor_NO")
plt.show()

Dataframe.plot(x = "Sensor_SO2", y = "RefSt", kind = "scatter", color="r")
plt.title("Scatter plot RefSt vs Sensor_SO2")
plt.ylabel("Ozone in µg/m³")
plt.xlabel("Sensor_SO2")
plt.show()





