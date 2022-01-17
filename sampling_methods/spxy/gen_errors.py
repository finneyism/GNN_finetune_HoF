import numpy as np

for i in range(100):
	with open(f"bootstrap_{i}.csv", 'r') as f:
		line = f.readlines()[0]
		with open("bootstrap_errors.csv", "a+") as F:
			F.write(line)
