#!/home/finney/.conda/envs/fin/bin/python

from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import numpy as np
from numpy import genfromtxt

data = genfromtxt("combined", delimiter=",")
x = data[:, :-1]
y = data[:, -1]

#reg = svm.SVR()
reg = make_pipeline(StandardScaler(), SVR(kernel="linear", C=1.0, epsilon=0.2))

reg.fit(x, y)
pred = reg.predict(x)

diff = []
for i in range(208):
	diff.append(abs(y[i] - pred[i]))
print(np.mean(diff))
print(np.std(diff))
