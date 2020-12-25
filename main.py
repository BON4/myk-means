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

class Kmean:
    def __init__(self, data, n_clusters):
        self.n_clusters = n_clusters
        self.data = data.to_numpy()
        self.clusters_cord = []
        for k in range(n_clusters):
            cluster = []
            for i in data:
                #cluster.append(random.uniform(-1, 1))
                cluster.append(random.gauss(np.mean(data[i]), np.std(data[i])))
            #self.clusters_cord.append(self.data[k])
            self.clusters_cord.append(cluster)
        
        self.clusters_cord = np.reshape(self.clusters_cord, (self.n_clusters, len(self.data[0])))

    #Рассчитать расстояние между вектором x и центром c кластера
    def calculate_mean(self):
        self.distanse_arr = np.array([])
        for c in self.clusters_cord:
            distance = np.array([])
            for x in self.data:
                distance = np.append(distance, np.sqrt(np.dot(np.transpose(np.subtract(x, c)), np.subtract(x, c))))
            self.distanse_arr = np.append(self.distanse_arr, distance)
        self.distanse_arr = np.reshape(self.distanse_arr, (self.n_clusters, len(self.data)))

    #Найти минимальное расстояние
    def find_min_for_datasamples(self):
        self.best_min_cluster = []
        for i in range(len(self.data)):
            self.best_min_cluster = np.append(self.best_min_cluster, np.where(self.distanse_arr[:,i] == min(self.distanse_arr[:,i])))
            
    #Пересчитать центроиды
    def find_new_center(self):
        new_cluster_cord = np.array([])
        for i in range(len(self.clusters_cord)):
            summ = np.array([])
            for k,m in zip(self.data, self.best_min_cluster):
                if m == i:
                    summ = np.append(summ, k)
            summ = np.reshape(summ, (-1, len(self.data[0])))
            new_cord = np.dot(1/len(self.data), sum(summ))

            #Sometimes new_cord is equal to 0 not array. So we need to create array out of 0
            if np.array([new_cord]).shape == (1,):
                new_cord = np.full(shape=len(self.data[0]),fill_value=new_cord,dtype=np.float64)

            new_cluster_cord = np.append(new_cluster_cord, new_cord)

        new_cluster_cord = np.reshape(new_cluster_cord, (self.n_clusters, len(self.data[0])))

        self.clusters_cord = new_cluster_cord

    def transform(self):
        eps = 0.000001
        count = 0

        a = self.clusters_cord
        self.calculate_mean()
        self.find_min_for_datasamples()
        self.find_new_center()
        b = self.clusters_cord
        while np.max(np.abs(np.subtract(a,b))) > eps:
            a = b
            self.calculate_mean()
            self.find_min_for_datasamples()
            self.find_new_center()
            b = self.clusters_cord
            count = count +1 

        self.find_min_for_datasamples()
        print("Кол-во итераций: ", count)

dataname = 'iris'
data = pd.read_csv(dataname+'.csv')

data.info()
print(data.describe())

#Need to drop ID !
df = data.drop(['Species', 'Id'], axis=1)

df = df.apply(lambda x: hypercube(x))

df = df.sample(frac=1)

km = Kmean(df, 3)

km.transform()

#PRINTING

print("---матрица координат центроидов кластеров---")
print(km.clusters_cord, '\n')

print("--матрица  расстояний  объектов  кластера  до  его центроида--")
arr_dist_cluster = pd.DataFrame(data=None, columns=['Distanse', 'Classter'])
for i in range(len(km.data)):
    arr_dist_cluster = arr_dist_cluster.append({'Classter': f"{np.where(km.distanse_arr[:,i] == min(km.distanse_arr[:,i]))[0][0]}",
                                                 'Distanse': min(km.distanse_arr[:,i])},
                                               ignore_index=True)

print(arr_dist_cluster.head(), '\n')

arr_sum_dist_cluster = {}
for i in range(len(km.data)):
    cluster = f"{np.where(km.distanse_arr[:,i] == min(km.distanse_arr[:,i]))[0][0]}"
    if cluster in arr_sum_dist_cluster:
        arr_sum_dist_cluster[cluster] += min(km.distanse_arr[:,i])
    else:
        arr_sum_dist_cluster[cluster] = min(km.distanse_arr[:,i])



for i in arr_sum_dist_cluster:
    print(f"Класстер - {i} сумма расстояний: {arr_sum_dist_cluster[i]}")


#PCA

pca = PCA(n_components=2)
X_r = pca.fit(df).transform(df)
y = km.best_min_cluster
target_names = [0.0, 1.0, 2.0]

plt.figure()
colors = ['navy', 'turquoise', 'darkorange']
lw = 2

for color, i, target_name in zip(colors, [0, 1, 2], target_names):
    plt.scatter(X_r[y == i, 0], X_r[y == i, 1], color=color, alpha=.8, lw=lw,
                label=target_name)

plt.legend(loc='best', shadow=False, scatterpoints=1)
plt.title('PCA of Wine dataset')

plt.savefig(dataname + '.png')