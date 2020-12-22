import numpy as np
import pandas as pd 
import random
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def hypercube(x):
    """
    function for coding per hypercube
    :param x: pd.Series, column of dataframe
    :return: pd.Series
    """
    min_v = np.min(x)
    max_v = np.max(x)
    return x.apply(lambda y: 2 * ((y - min_v) / (max_v - min_v)) - 1)

dataname = 'iris'
data = pd.read_csv(dataname+'.csv')

y = data['Species']

data = data.drop(['Species'], axis=1)
data = data.apply(lambda x: hypercube(x))


pca = PCA(n_components=2)
X_r = pca.fit(data).transform(data)
target_names = y.unique()

plt.figure()
colors = ['navy', 'turquoise', 'darkorange']
lw = 2

for color, i, target_name in zip(colors, target_names, target_names):
    plt.scatter(X_r[y == i, 0], X_r[y == i, 1], color=color, alpha=.8, lw=lw,
                label=target_name)

plt.legend(loc='best', shadow=False, scatterpoints=1)
plt.title('PCA of Iris dataset')

plt.savefig(dataname + '_real.png')