#!/home/finney/.conda/envs/fin/bin/python

from sklearn.linear_model import LinearRegression
import numpy as np
from numpy import genfromtxt

data = genfromtxt("combined", delimiter=",")
x = data[:, :-1]
y = data[:, -1]

reg = LinearRegression().fit(x, y)
pred = reg.predict(x)

diff = []
for i in range(208):
	diff.append(abs(y[i] - pred[i]))
print(np.mean(diff))
print(np.std(diff))
