import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
import numpy as np

#################### st hyper param for KMeans ####################
K = 3
#################### en hyper param for KMeans ####################

#################### st prepare data ####################
## let's generate a two-dimensional dataset containing four distinct blobs.
# from sklearn.datasets.samples_generator import make_blobs
# X, y_true = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)
# print(type(X))
# print(X.shape)

excel = pd.read_excel('../input/clustering_KYG.xlsx')
# x, y 값 선택
df = pd.DataFrame(excel, columns=['LFC', 'significance'])
# input 값을 (n, 2) shape의 numpy array 로 만들기
X = df.to_numpy()
#################### en prepare data ####################

kmeans = KMeans(n_clusters=K)
kmeans.fit(X)
y_kmeans = kmeans.predict(X)

plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, cmap='viridis')
# plt.show()

centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='black', alpha=0.5)
# plt.show()


from sklearn.metrics import pairwise_distances_argmin


def find_clusters(X, n_clusters, rseed=2):
    # 1. Randomly choose clusters
    rng = np.random.RandomState(rseed)
    i = rng.permutation(X.shape[0])[:n_clusters]
    centers = X[i]

    while True:
        # 2a. Assign labels based on closest center
        labels = pairwise_distances_argmin(X, centers)

        # 2b. Find new centers from means of points
        new_centers = np.array([X[labels == i].mean(0) for i in range(n_clusters)])

        # 2c. Check for convergence
        if np.all(centers == new_centers):
            break
        centers = new_centers
        plt.scatter(centers[:, 0], centers[:, 1], alpha=0.5)
    # plt.show()

    return centers, labels


centers, labels = find_clusters(X, K)
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis')
# plt.show()
