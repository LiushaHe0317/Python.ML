
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import style
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.mlab import bivariate_normal
from matplotlib import cm

x = y = np.arange(-3,3,0.05)
X, Y = np.meshgrid(x, y)

Z1 = bivariate_normal(X, Y, 1, 1, 0, 0)
Z2 = bivariate_normal(X, Y, 1.5, 0.5, 1, 1)
Z = Z2 - Z1

X = X*10
Y = Y*10
Z = Z*500

Z = Z.reshape(X.shape)

style.use('ggplot'), style.use('dark_background')
f1 = plt.figure()

ax = f1.add_subplot(221, projection = '3d')
ax.plot_surface(X, Y, Z, cmap = cm.coolwarm)

f1.show()
