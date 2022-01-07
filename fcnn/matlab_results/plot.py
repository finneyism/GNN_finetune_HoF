#!/home/finney/.conda/envs/fin/bin/python
#/Users/finney/opt/anaconda3/bin/python3.8

import matplotlib.pyplot as plt
import numpy as np

data = np.loadtxt("gnn_expt_fcnn_error_testdata.csv", delimiter=",")
expt = data[:,1]
fcnn = data[:,2]
error = data[:,3]
gnn = data[:,0]
#
## plot testing molecules with error bars
#plt.errorbar(expt, fcnn, yerr=error, fmt=".", ecolor="orange", elinewidth=3, capsize=4)
#plt.title("Testing molecules for training FCNN", fontsize=18)
#plt.xlabel("Experimental heats of formation (kcal/mol)", fontsize=15)
#plt.ylabel("FCNN-predicted HoFs (kcal/mol)", fontsize=15)
#corr = np.corrcoef(expt, fcnn)[0,1]
#plt.axline((0, 0), slope=1, linestyle=":", color="black")
#plt.grid(ls="--")
#plt.text(-100, 50, "${R^2}$"+f"={corr:.4f}", fontsize=15)
#plt.savefig("test_errors.eps")


dft = np.loadtxt("/home/finney/DATA/hof_me/fcnn/schnet_predicted/plot/dft_test.csv", delimiter=",", usecols=(1))

# plot gnn v.s. expt
corr_gnn = np.corrcoef(expt, gnn)[0,1]
plt.scatter(gnn, expt, marker="o", facecolors="none", edgecolors="darkgreen")
plt.axline((0, 0), slope=1, linestyle=":", color="grey")
plt.xlim([-150, 100])
plt.ylim([-150, 100])
plt.ylabel("SchNet-predicted HoFs (kcal/mol)", fontsize=15)
plt.xlabel("Experimental heats of formation (kcal/mol)", fontsize=15)
plt.title("SchNet-predicted HoFs v.s. experimental HoFs", fontsize=18)
plt.grid(ls="--")
plt.text(-120, 50, "${R^2}$"+f"={corr_gnn:.4f}", fontsize=15)
plt.savefig("schnet_vs_expt.eps")

## plot fcnn v.s. expt
#corr_fcnn = np.corrcoef(fcnn, expt)[0,1]
#corr_dft = np.corrcoef(dft, expt)[0,1]
#plt.scatter(expt, fcnn, marker="o", facecolors="none", edgecolors="darkgreen", label="FCNN-predicted, ${R^2}$="+f"{corr_fcnn:.4f}")
#plt.scatter(expt, dft, marker="1", color="darkred", label="DFT, ${R^2}$="+f"{corr_dft:.4f}")
#plt.axline((0, 0), slope=1, linestyle=":", color="black")
#plt.grid(ls="--", color="grey")
#plt.xlim([-150, 100])
#plt.ylim([-150, 100])
#plt.ylabel("Calculated HoFs (kcal/mol)", fontsize=15)
#plt.xlabel("Experimental heats of formation (kcal/mol)", fontsize=15)
#plt.title("Calculated HoFs v.s. experimental HoFs", fontsize=18)
#plt.legend()
#plt.savefig("fcnn_vs_expt.eps")
