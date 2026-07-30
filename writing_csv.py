import numpy as np
import pandas as pd
import csv

data=pd.DataFrame(columns=["x","y1","y2"])

x_vals=np.arange(1,100,0.5)

y1_vals=np.sin(x_vals)
y2_vals=np.cos(x_vals)

data["x"]=x_vals
data["y1"]=y1_vals
data["y2"]=y2_vals

data.to_csv("data.csv",index=False)

#using the csv to plot the data

data_reading=pd.read_csv("data.csv")
import matplotlib.pyplot as plt

plt.plot(data_reading["x"],data_reading["y1"],label="y1")
plt.plot(data_reading["x"],data_reading["y2"],label="y2")

plt.xlabel="x"
plt.legend()

plt.show()
