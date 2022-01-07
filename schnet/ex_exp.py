#!/home/finney/.conda/envs/fin/bin/python

import torch
import torch.nn as nn
from torch_geometric.nn import SchNet
from torch_geometric.loader import DataLoader
from torch_geometric.datasets import QM9
from statistics import mean, stdev
import numpy as np
from qm9_test.qm9_me import QM9
import os

device = torch.device("cuda", 1)

# hyperparameters
epoch = 100
#start_epoch = 1
#learning_rate = 0.001
#start_epoch = 69 
#learning_rate = 0.0005
#start_epoch = 159
#learning_rate = 0.0001
start_epoch = 253 
learning_rate = 0.00005
#start_epoch = 227 
#learning_rate = 0.00001
#start_epoch = 295 
#start_epoch = 364
#learning_rate = 0.000005
#epoch = 300
#start_epoch = 663
#learning_rate = 0.0000001
#start_epoch = 962
#learning_rate = 0.00000005
#start_epoch = 1261
#learning_rate = 0.00000001



#model = SchNet()
model = torch.load(f"ex_exp_models/model{start_epoch}.pth")
train_data = torch.load("my_dataset/train_ex_exp")
val_data = torch.load("my_dataset/val_ex_exp")
test_data = torch.load("my_dataset/test_ex_exp")

#dataset = QM9(".")
#perm = np.arange(len(dataset))
#np.random.shuffle(perm)
#perm = perm.tolist()
#val_num = int(0.1 * len(dataset))
#test_num = int(0.2 * len(dataset))
#train_num = len(dataset) - val_num - test_num
#train_data = dataset[perm[:train_num]]
#val_data = dataset[perm[train_num:(train_num+val_num)]]
#test_data = dataset[perm[(train_num+val_num):]]
#torch.save(train_data, "my_dataset/train_ex_exp")
#torch.save(val_data, "my_dataset/val_ex_exp")
#torch.save(test_data, "my_dataset/test_ex_exp")

train_loader = DataLoader(train_data, batch_size=256)
test_loader = DataLoader(test_data, batch_size=256)
val_loader = DataLoader(val_data, batch_size=256)

model = model.to(device)

loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

def train(dataloader, model, loss_fn, optimizer, device):
	model.train()
	for data in dataloader:
		data = data.to(device)
		pred = model(data.z, data.pos, data.batch)

		loss = loss_fn(pred.view(-1), data.y[:, 0])

        # backpropagation
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

#    if (i+1) % 10 == 0:
#        print(f"epoch: {i+1}, loss: {loss:>7f}")
	return loss.item()

def validation(dataloader, model, loss_fn, device):
	model.eval()
	test_loss = []
	with torch.no_grad():
		for data in dataloader:
			data = data.to(device)
			pred = model(data.z, data.pos, data.batch)
			test_loss.append(loss_fn(pred.view(-1), data.y[:, 0]).item())
#            print(f"test loss: {test_loss:>7f}")
	return test_loss

# start training
for i in range(start_epoch, (start_epoch+epoch)):
	loss = train(train_loader, model, loss_fn, optimizer, device)
	test_loss = validation(val_loader, model, loss_fn, device)
#	if i % 5 == 0:
	test_avg = mean(test_loss)
	test_std = stdev(test_loss)
	with open("ex_exp.log", "a+") as f:
		f.write(f"{i}\t{loss}\t{test_avg}\t{test_std}\n")
	torch.save(model, f"ex_exp_models/model{i}.pth")
#	print(f"epoch: {i}, train loss: {loss}")
#	print(f"validation loss: {test_avg}+/-{test_std}")
