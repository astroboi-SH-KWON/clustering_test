import time
import os
WORK_DIR = os.getcwd() + "/"
PROJECT_NAME = WORK_DIR.split("/")[-2]

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
import numpy as np

#################### st hyper param for KMeans ####################
K = 4
#################### en hyper param for KMeans ####################

#################### st prepare data ####################
excel = pd.read_excel('../input/clustering_KYG.xlsx')
# x, y 값 선택
# df = pd.DataFrame(excel, columns=['significance', 'LFC'])
df = pd.DataFrame(excel, columns=['LFC', 'significance'])
# input 값을 (n, 2) shape의 numpy.ndarray 로 만들기
X = df.to_numpy()

# X[:, 1] = - np.log(X[:, 1])

# Min-Max Normalization
print('X.shape', X.shape)
# for i in range(X.shape[1]):
#     X[:, i] = (X[:, i] - min(X[:, i])) / (max(X[:, i]) - min(X[:, i]))
i = 1
X[:, i] = (X[:, i] - min(X[:, i])) / (max(X[:, i]) - min(X[:, i]))
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
plt.show()

def get_elbow_point():
    distortions = []
    K = range(1, 10)
    for k in K:
        kmeanModel = KMeans(n_clusters=k)
        kmeanModel.fit(X)
        distortions.append(kmeanModel.inertia_)

    plt.figure(figsize=(16, 8))
    plt.plot(K, distortions, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Distortion')
    plt.title('The Elbow Method showing the optimal k')
    plt.show()

if __name__ == '__main__':
    start_time = time.perf_counter()
    print("start [ " + PROJECT_NAME + " ]>>>>>>>>>>>>>>>>>>")
    # get_elbow_point()
    # kmeans, y_kmeans = do_kmeans()
    # get_plt_of_scikit_kmeans(kmeans, y_kmeans).show()
    # do_expect_max_for_kmanes().show()
    print("::::::::::: %.2f seconds ::::::::::::::" % (time.perf_counter() - start_time))