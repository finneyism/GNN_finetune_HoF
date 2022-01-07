#!/home/finney/.conda/envs/fin/bin/python

import torch
from torch_geometric.nn import SchNet
from torch_geometric.loader import DataLoader
from statistics import mean, stdev

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#model = SchNet()
model = torch.load("ex_exp_models/model253.pth")

train_data = torch.load("my_dataset/train_ex_exp")
test_data = torch.load("my_dataset/test_ex_exp")
val_data = torch.load("my_dataset/val_ex_exp")
#print(train_data, test_data, val_data)

train_loader = DataLoader(train_data, batch_size=256)
test_loader = DataLoader(test_data, batch_size=256)
val_loader = DataLoader(val_data, batch_size=256)

model = model.to(device)
model.eval()

# train set
maes_train = []
for data in train_loader:
	data = data.to(device)
	with torch.no_grad():
		pred = model(data.z, data.pos, data.batch)
	mae = (pred.view(-1) - data.y[:, 0]).abs()
	maes_train.append(mae)
mae_train = torch.cat(maes_train, dim=0)
# validation set
maes_val = []
for i, data in enumerate(val_loader):
	data = data.to(device)
	with torch.no_grad():
		pred = model(data.z, data.pos, data.batch)
	mae = (pred.view(-1) - data.y[:, 0]).abs()
	maes_val.append(mae)
mae_val = torch.cat(maes_val, dim=0)
# test set
maes_test = []
for data in test_loader:
	data = data.to(device)
	with torch.no_grad():
		pred = model(data.z, data.pos, data.batch)
	mae = (pred.view(-1) - data.y[:, 0]).abs()
	maes_test.append(mae)
mae_test = torch.cat(maes_test, dim=0)

with open("evaluation_of_model", "a+") as f:
	f.write(f"train: {mae_train.mean()}\t{mae_train.std()}\n")
	f.write(f"validation: {mae_val.mean()}\t{mae_val.std()}\n")
	f.write(f"test: {mae_test.mean()}\t{mae_test.std()}\n")

