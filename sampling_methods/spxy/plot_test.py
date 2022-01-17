#!/home/finney/.conda/envs/fin/bin/python

import numpy as np
import matplotlib.pyplot as plt

test_expt = np.loadtxt("test_tar.csv", delimiter=",")
test_schnet = np.loadtxt("test_des.csv", delimiter=",", usecols=((-1)))
test_fcnn = np.loadtxt("test_results.csv", delimiter=",")
error = np.loadtxt("errors.csv", delimiter=",")

corr = np.corrcoef(test_expt, test_fcnn)[0,1]
# plot testing molecules with error bars
plt.errorbar(test_expt, test_fcnn, yerr=error, fmt=".", ecolor="orange", elinewidth=3, capsize=4)
plt.title("SPXY sampling method with error bars", fontsize=18)
plt.xlabel("Experimental heats of formation (kcal/mol)", fontsize=15)
plt.ylabel("Fine-tuned HoFs (kcal/mol)", fontsize=15)
plt.axline((0, 0), slope=1, linestyle=":", color="black")
plt.grid(ls="--")
plt.text(-80, 0, "${R^2}$"+f"={corr:.4f}", fontsize=15)
plt.savefig("test_errors_spxy.eps")

## plot tesing v.s. dft
#dft = np.loadtxt("../spxy_test_dft.csv", delimiter=",", usecols=(1))
#
#corr_dft = np.corrcoef(dft, test_expt)[0,1]
#plt.scatter(test_expt, test_fcnn, marker="o", facecolors="none", edgecolors="darkgreen", label="Fine-tuned HoFs, ${R^2}$="+f"{corr:.4f}")
#plt.scatter(test_expt, dft, marker=".", color="darkred", label="DFT HoFs, ${R^2}$="+f"{corr_dft:.4f}")
#plt.axline((0, 0), slope=1, linestyle=":", color="black")
#plt.grid(ls="--", color="grey")
#plt.xlim([-100, 40])
#plt.ylim([-100, 40])
#plt.ylabel("Calculated HoFs (kcal/mol)", fontsize=15)
#plt.xlabel("Experimental heats of formation (kcal/mol)", fontsize=15)
#plt.title("Calculated HoFs v.s. experimental HoFs", fontsize=18)
#plt.legend()
#plt.savefig("fcnn_vs_expt.eps")

## plot schnet v.s. expt
#corr_schnet = np.corrcoef(test_schnet, test_expt)[0,1]
#plt.scatter(test_expt, test_schnet, marker="o", facecolors="none", edgecolors="darkgreen")
#plt.axline((0, 0), slope=1, linestyle=":", color="black")
#plt.grid(ls="--", color="grey")
#plt.xlim([-100, 40])
#plt.ylim([-100, 40])
#plt.ylabel("Pre-trained HoFs (kcal/mol)", fontsize=15)
#plt.xlabel("Experimental heats of formation (kcal/mol)", fontsize=15)
#plt.title("Pre-trained HoFs v.s. experimental HoFs", fontsize=18)
#plt.text(-80, 0, "${R^2}$"+f"={corr_schnet:.4f}", fontsize=15)
#plt.savefig("schnet_vs_expt.eps")

