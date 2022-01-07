#!/home/finney/.conda/envs/fin/bin/python

import numpy as np

ls = []
with open("../values_predicted", "r") as F:
	lins = F.readlines()
	for l in lins:
		ls.append(l.split(",")[0])
with open("/home/finney/DATA/hof_me/training/schnet/raw/gdb9_hof.csv") as f:
	lines = f.readlines()
	for line in lines:
		if line.split(",")[0] in ls:
			with open("dft_hof.csv", "a+") as a:
				a.write(f"{line.split(',')[0]},{line.split(',')[-1]}")
