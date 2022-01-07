#!/home/finney/.conda/envs/fin/bin/python

import torch
from torch_geometric.nn import SchNet
from torch_geometric.loader import DataLoader
from statistics import mean, stdev
from qm9_exp import QM9

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#model = SchNet()
model = torch.load("/home/finney/DATA/hof_me/training/schnet/ex_exp_models/model253.pth")

exp_data = QM9(".")

exp_loader = DataLoader(exp_data, batch_size=256)

model = model.to(device)
model.eval()

# exp set
maes_exp = []
for data in exp_loader:
	data = data.to(device)
	with torch.no_grad():
		pred = model(data.z, data.pos, data.batch)
		names = data.name
	mae = (pred.view(-1) - data.y[:, 0]).abs()
	maes_exp.append(mae)
mae_exp = torch.cat(maes_exp, dim=0)
predicted_values = pred.view(-1)

with open("values_predicted", "a+") as F:
	for i in range(208):
		F.write(f"{names[i]},{predicted_values[i]}\n")

