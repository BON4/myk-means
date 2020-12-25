import numpy as np
import pandas as pd 
import random
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris
from sklearn.datasets import load_wine
import matplotlib.pyplot as plt

def plot(x, v, u, c):

    fig = plt.figure()
    u = np.argmax(u, axis=0)
    x = x.T

    for j in range(c):
        plt.scatter(x[0][u == j], x[1][u == j], alpha=0.8)
        plt.scatter(v[j][0], v[j][1], s=500 , marker="x")
        
    plt.savefig('wine_gk.png')



class GustafsonKessel:
    def __init__(self, data, clusters, fuzzifier, error_tolerance, iterations):
        self.z = data
        self.c = clusters
        self.m = fuzzifier
        self.e = error_tolerance
        self.l = iterations
        
    def compute_distance(self, x, V, F, n, m):
        F = F+0.00001*np.random.rand(n, n)
        zk_V = np.expand_dims(x.reshape(x.shape[0],1,-1) - V.reshape(-1,V.shape[0],1), axis=3)
        det = np.power(np.linalg.det(F)+0.00001, 1 / n)
        inv = np.linalg.inv(F)
        detinv = np.reshape(det, (-1, 1, 1)) * inv
        temp = np.matmul(zk_V.transpose((0, 1, 3, 2)), detinv)
        zk_V = np.squeeze(zk_V)
        temp = np.squeeze(temp)
        zk_V = np.einsum('ijk->jik', zk_V)
        temp = np.einsum('ijk->jki', temp)
        output = np.matmul(temp, zk_V).squeeze().T
        output = output[0, :, :]
        return np.fmax(output, 0.00001).T
    
    def update_partition_matrix(self, dist, m):
        exp_dist = dist ** (2.0/m-1)
        return exp_dist / np.sum(exp_dist)
    
    def compute_clusters(self, x, U, m):
        exp_U = np.power(U,m) 
        p1 = exp_U.dot(x.T)
        p2 = np.sum(exp_U,  axis=1)
        p2 = np.reshape(p2, (-1, 3))
        return np.divide(p1,p2.T)
    
    def compute_cluter_covariance_matrix(self, U, x, V, m, n):
        exp_U = np.power(U,m) 
        mx1 = np.ones((3,n))
        xV = x - (V.T.dot(mx1))
        xV = xV[0].T
        xV2 = np.multiply(xV, xV[:, np.newaxis])
        P1 = (np.sum(exp_U) * xV2)
        P2 = np.sum(exp_U)
        return np.divide(P1,P2)
    
    def cmeans(self):
        Datapoints,Features = self.z.shape

        x = self.z.T

        V = np.empty((self.l, self.c, Features))
        V[0] = np.array(x[:,np.random.choice(x.shape[1], size = self.c)].T)
        
        U = np.zeros((self.l, self.c, Datapoints))
        U[0] = np.array(x[np.random.choice(x.shape[0], size = self.c),:])
        
        F = np.zeros((self.l, self.c, Datapoints, Datapoints))

        i = 0
        distance = 0
        while True:
            
            F[i] = self.compute_cluter_covariance_matrix(U[i], x, V[i], self.m, Datapoints)
            
            for j in range(self.c):
                distance = self.compute_distance(x,V[i],F[i][j],Datapoints,Features)
                
            U[i] = self.update_partition_matrix(distance, self.m)            

            V[i+1] = self.compute_clusters(x, U[i], self.m)
            
            if np.linalg.norm(U[i - 1] - U[i]) < self.e:
                break
                
            i += 1

        return V[i],U[i]


d = load_wine()

df= pd.DataFrame(d.data)


gk = GustafsonKessel(d.data, 3, 1.2,0.1,200)

v, u = gk.cmeans()

print(v)

plot(d.data, v, u, 3)