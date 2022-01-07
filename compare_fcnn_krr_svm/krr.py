#!/home/finney/.conda/envs/fin/bin/python

from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import learning_curve
from sklearn.kernel_ridge import KernelRidge
import numpy as np
from numpy import genfromtxt

data = genfromtxt("combined", delimiter=",")
x = data[:, :-1]
y = data[:, -1]

krr = KernelRidge(alpha=1.0)
krr.fit(x, y)
pred = krr.predict(x)
print(krr.score(x, y))

diff = []
for i in range(208):
	diff.append(abs(y[i] - pred[i]))
print(np.mean(diff))
print(np.std(diff))
