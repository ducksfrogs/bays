import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


def metropolis(func, steps=10000):
    """ A very simple metropolis
    """

    samples = np.zeros(steps)
    old_x = func.mean()
    old_prob = func.pdf(old_x)

    for i in range(steps):
        new_x = old_x + np.random.normal(0, 0.5)
        new_prob = func.pdf(new_x)
        acceptance = new_prob / old_prob
        if acceptance >= np.random.random():
            samples[i] = new_x
            old_x = new_x
            old_prob = new_prob

        else:
            samples[i] = old_x
    return samples


func = stats.beta(0.4, 2)
samples = metropolis(func=func)

x = np.linspace(0.01, 0.99, 100)
y = func.pdf(x)

plt.plot(x, y, "r-", lw=3, label="True distribution")
plt.hist(samples, bins=30, normed=True, label="Estimated distribution")
plt.xlabel("$x$", fontsize=14)
plt.ylabel("$pdf(x)", fontsize=14)

plt.legend(fontsize=14)
plt.savefig("img023.png")
