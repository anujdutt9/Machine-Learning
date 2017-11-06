import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')


np.random.seed(0)

# Generate a dataset and plot it
X, y = datasets.make_moons(200, noise=0.20)
plt.scatter(X[:,0], X[:,1], s=40, c=y, cmap=plt.cm.Spectral)
plt.title('Input Dataset')
plt.show()