import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns


mu_params = [-1, 0, 1]
sd_params = [0.5, 1, 1.5]
x = np.linspace(-7, 7, 100)
f, ax = plt.subplots(len(mu_params), len(sd_params), sharex=True, sharey=True)
for i in range(3):
    for j in range(3):
        mu = mu_params[i]
        sd = sd_params[j]
        y = stats.norm(mu, sd).pdf(x)
        ax[i, j].plot(x, y)
        ax[i, j].plot(
            0, 0, label="$\\mu$ ={:3.2f}\n$\\sigma$ = {:3.2f}".format(mu, sd), alpha=0
        )
        ax[i, j].legend(fontsize=8)

plt.savefig("img101.png")
