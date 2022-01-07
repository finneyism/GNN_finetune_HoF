#!/home/finney/.conda/envs/fin/bin/python

import matplotlib.pyplot as plt
import numpy as np

dft = np.loadtxt("dft_hof.csv", delimiter=",", usecols=(1))
pred = np.loadtxt("values_predicted", delimiter=",", usecols=(1))

plt.scatter(dft, pred, marker="o", facecolors="none", edgecolors="darkgreen")
corr = np.corrcoef(dft, pred)[0, 1]
plt.axline((0,0), slope=1, linestyle=":", color="black")
plt.grid(ls="--", color="grey")
plt.xlim([-200, 170])
plt.ylim([-200, 170])
plt.title("SchNet-predicted HoFs v.s. DFT HoFs", fontsize=18)
plt.xlabel("DFT heats of formation (kcal/mol)", fontsize=15)
plt.ylabel("SchNet-predicted HoFs (kcal/mol)", fontsize=15)
plt.text(-150, 100, "${R^2}$"+f"={corr:.4f}", fontsize=15)
plt.savefig("schnet_dft.png")
