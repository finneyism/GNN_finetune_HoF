#!/home/finney/.conda/envs/fin/bin/python

import matplotlib.pyplot as plt
import numpy as np

#f = np.loadtxt("train_exp_lr.log")
f = np.loadtxt("train.log")
epoch = f[:,0]
train_loss = f[:,1] 
val_loss = f[:,2]

plt.plot(epoch, train_loss, label="test loss")
plt.plot(epoch, val_loss, label="validation loss")
plt.xlabel("Number of epochs")
plt.ylabel("Loss")
plt.legend()
#plt.savefig("loss_exp_lr.png")
plt.savefig("loss.png")

