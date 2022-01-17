#!/home/finney/.conda/envs/fin/bin/python

import numpy as np
from split import kennardstone, spxy

data_x = np.loadtxt("combined", delimiter=",", usecols=(0, 1, 2, 3, 4, 5))
data_y = np.loadtxt("combined", delimiter=",", usecols=(6))

train, test = spxy(data_x, data_y)

with open("combined", 'r') as f:
    lines = f.readlines()
    for i in train:
        with open("spxy_train.csv", "a+") as F:
            F.write(lines[i])
    for j in test:
        with open("spxy_test.csv", "a+") as F1:
            F1.write(lines[j])

with open("/home/finney/DATA/hof_me/fcnn/schnet_predicted/plot/dft_hof.csv", 'r') as f:
    lines = f.readlines()
    for i in test:
        with open("spxy_test_dft.csv", "a+") as F:
            F.write(lines[i])
