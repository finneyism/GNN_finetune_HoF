#!/home/finney/.conda/envs/fin/bin/python

ls = [4, 5, 6, 8, 12, 15, 20, 21, 22, 26, 27, 28, 30, 36, 38, 40, 41, 42, 46, 48, 52, 60, 69, 72, 74, 75, 79, 82, 84, 94, 95, 102, 116, 117, 122, 123, 128, 133, 135, 14, 142, 144, 147, 155, 163, 164, 167, 168, 169, 170, 172, 176, 180, 182, 184, 188, 189, 199, 201,  203, 204, 208]

with open("dft_hof.csv", 'r') as f:
	lines = f.readlines()
	for l in ls:
		with open("dft_test.csv", "a+") as F:
			F.write(lines[int(l)-1])
