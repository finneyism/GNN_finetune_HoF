#!/home/finney/.conda/envs/fin/bin/python

pred_list = []
exp_list = []

with open("values_predicted", 'r') as f:
	lines = f.readlines()
	for line in lines:
		pred_list.append(line.split(",")[1].strip())

with open("gdb_exp", 'r') as f:
	lines = f.readlines()
	for line in lines:
		exp_list.append(line.split(":")[3].strip())

with open("exp.atomnumber", 'r') as f:
	lines = f.readlines()
	i = 0
	for line in lines:
		with open("combined", "a+") as F:
			F.write(f"{line.strip()},{pred_list[i]},{exp_list[i]}\n")
			i += 1
