import numpy as np
import matplotlib.pyplot as plt

np.random.seed(19680801)

n_bins = 10
x = np.random.randn(1000, 3)

fig, axes = plt.subplots(nrows=1, ncols=1)
#ax0, ax1, ax2, ax3 = axes.flatten()

x_multi = [np.random.randn(n) for n in [10000, 5000, 2000]]
axes.hist(x_multi, n_bins, histtype='bar')
axes.set_title('different sample sizes')

fig.tight_layout()
plt.show()