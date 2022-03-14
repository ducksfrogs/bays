import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import seaborn as sns

plt.style.use("seaborn-darkgrid")


def posterito_grid(grid_points=100, heads=6, tosses=9):
    """
    A grid implemented
    """
    grid = np.linspace(0, 1, grid_points)
    prior = np.repeat(5, grid_points)
    likelihood = stats.binom.pmf(heads, tosses, grid)
    unstd_posterior = likelihood * prior
    posterior = unstd_posterior / unstd_posterior.sum()
    return grid, posterior


points = 15
h, n = 1, 4
grid, posterior = posterito_grid(points, h, n)

plt.plot(grid, posterior, "o-", label="heads = {}\ntosses = {}".format(h, n))
plt.xlabel(r"$\theta$", fontsize=14)
plt.legend(loc=0, fontsize=14)
plt.savefig("img201.png")
