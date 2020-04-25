# -*- coding: utf-8 -*-
"""
Created on Sat Apr 25 15:47:13 2020

@author: Vaibhav
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:,1:2].values
y = dataset.iloc[:,2:3].values

from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state =0)

regressor.fit(X, y)

y_pred = regressor.predict([[6.5]])

x_grid = np.arange(min(X), max(X), 0.01)
x_grid = x_grid.reshape(len(x_grid), 1)
plt.scatter(X, y, color ='red')
plt.plot(x_grid, regressor.predict(x_grid), color ='blue')
plt.xlabel('Position')
plt.ylabel('Salry')
plt.title('Salary vs Postion (Decision tree regresson)')