#%%
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np

gaussian_samples = np.random.multivariate_normal(mean=[1,2], cov=[[1, 0.5], [0.5, 1]], size=4000)
x, y = gaussian_samples[:, 0], gaussian_samples[:, 1]
# %%

plt.scatter(x, y)
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.title("Scatter plot of Gaussian samples")
plt.grid()
plt.show()

#%% plot a kde of the samples using stats.gaussian_kde

xmin, xmax, ymin, ymax = x.min(), x.max(), y.min(), y.max()
X, Y = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
positions = np.vstack([X.ravel(), Y.ravel()])

kernel = stats.gaussian_kde(gaussian_samples.T)
Z = np.reshape(kernel(positions).T, X.shape)

plt.imshow(np.rot90(Z), cmap=plt.cm.gist_earth_r, extent=[xmin, xmax, ymin, ymax])
plt.colorbar(label="Density")
plt.xlim(xmin, xmax)
plt.ylim(ymin, ymax)
plt.show()