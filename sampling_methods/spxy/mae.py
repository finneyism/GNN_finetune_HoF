import numpy as np

train_tar = np.loadtxt("train_tar.csv", delimiter=",")
train_pred = np.loadtxt("train_results.csv", delimiter = ",")
train_len = len(train_tar)
train_error = 0
for i in range(train_len):
	train_error += abs(train_tar[i] - train_pred[i])
print(f"training MAE = {train_error/train_len}")

test_tar = np.loadtxt("test_tar.csv", delimiter=",")
test_pred = np.loadtxt("test_results.csv", delimiter = ",")
test_len = len(test_tar)
test_error = 0
for i in range(test_len):
	test_error += abs(test_tar[i] - test_pred[i])
print(f"testing MAE = {test_error/test_len}")

