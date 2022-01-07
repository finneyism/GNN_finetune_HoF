#!/home/finney/.conda/envs/fin/bin/python

import numpy as np

data_train = np.loadtxt("train_des.csv", delimiter=",")
train_0 = data_train[0, :]
train_1 = data_train[1, :]
train_2 = data_train[2, :]
train_3 = data_train[3, :]
train_4 = data_train[4, :]
train_5 = data_train[5, :]

data_test = np.loadtxt("test_des.csv", delimiter=",")
test_0 = data_test[0, :]
test_1 = data_test[1, :]
test_2 = data_test[2, :]
test_3 = data_test[3, :]
test_4 = data_test[4, :]
test_5 = data_test[5, :]

d_array = np.ones((62, 146))
for i in range(62):
	for j in range(146):
		d = np.sqrt((train_0[j]-test_0[i]) ** 2 
		+ (train_1[j]-test_1[i]) ** 2 
		+ (train_2[j]-test_2[i]) ** 2 
		+ (train_3[j]-test_3[i]) ** 2 
		+ (train_4[j]-test_4[i]) ** 2
		+ (train_5[j]-test_5[i]) ** 2)
		d_array[i, j] = d

rho_list = []
for i in range(62):
	rho = 0
	for j in range(146):
		rho += 1 / d_array[i, j]
	rho_list.append(rho)

error = np.loadtxt("gnn_expt_fcnn_error_testdata.csv", delimiter=",")[:, 3]

import matplotlib.pyplot as plt
plt.scatter(rho_list, error, marker=".")
plt.title("Error v.s. data density", fontsize=18)
plt.xlabel("Data density", fontsize=15)
plt.ylabel("Error (kcal/mol)", fontsize=15)
plt.grid(ls="--")
plt.savefig("density_vs_error.eps")

