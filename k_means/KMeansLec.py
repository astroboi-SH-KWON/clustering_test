import time
import os
WORK_DIR = os.getcwd() + "/"
PROJECT_NAME = WORK_DIR.split("/")[-2]

import matplotlib.pyplot as plt
import seaborn as sns; sns.set()  # for plot styling
import numpy as np

# TODO 000
"""
let's generate a two-dimensional dataset containing four distinct blobs. 
"""
def make_blobs_data():
    # from sklearn.datasets.samples_generator import make_blobs
    from sklearn.datasets import make_blobs
    x, y_true = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)
    print('make_blobs', type(x))  # numpy.ndarray
    print(x.shape)
    # plt.scatter(x[:, 0], x[:, 1], s=50)
    # plt.show()
    return x, y_true


def make_moons_data():
    from sklearn.datasets import make_moons
    x, y = make_moons(200, noise=.05, random_state=0)
    print('make_moons', type(x))  # numpy.ndarray
    print(x.shape)
    # plt.scatter(x[:, 0], x[:, 1], s=50)
    # plt.show()
    return x, y


def get_plt_of_scikit_kmeans(x_np_arr, kmeans, y_kmeans):
    plt.scatter(x_np_arr[:, 0], x_np_arr[:, 1], c=y_kmeans, s=50, cmap='viridis')

    centers = kmeans.cluster_centers_
    plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)
    return plt


# TODO 001
"""
Let's visualize the results by plotting the data colored by these labels. 
We will also plot the cluster centers as determined by the k-means estimator:
"""
def do_kmeans(x_np_arr, n_clstrs=4):
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=n_clstrs)
    kmeans.fit(x_np_arr)
    y_kmeans = kmeans.predict(x_np_arr)
    return kmeans, y_kmeans


# TODO 002 Expectation–Maximization
"""
The k-Means algorithm is simple enough that we can write it in a few lines of code. 
The following is a very basic implementation:

do_expect_max_for_kmanes()
"""
def find_clusters(x_np_arr, n_clstrs, rseed=2):
    from sklearn.metrics import pairwise_distances_argmin

    # 1. Randomly choose clusters
    rng = np.random.RandomState(rseed)
    i = rng.permutation(x_np_arr.shape[0])[:n_clstrs]
    centers = x_np_arr[i]

    while True:
        # TODO E-step
        # 2a. Assign labels based on closest center
        labels = pairwise_distances_argmin(x_np_arr, centers)

        # TODO M-step
        # 2b. Find new centers from means of points
        new_centers = np.array([x_np_arr[labels == i].mean(0) for i in range(n_clstrs)])

        # 2c. Check for convergence
        if np.all(centers == new_centers):
            break
        centers = new_centers

        ## opt : draw centroid
        # plt.scatter(centers[:, 0], centers[:, 1], alpha=0.5)
        # plt.show()
    return centers, labels


def do_expect_max_for_kmanes(x_np_arr, n_clstrs=4, rseed=2):
    centers, labels = find_clusters(x_np_arr, n_clstrs, rseed)
    plt.scatter(x_np_arr[:, 0], x_np_arr[:, 1], c=labels, s=50, cmap='viridis')
    return plt


def do_kmeans_with_param(x_np_arr, n_clstrs=4, random_state=0):
    from sklearn.cluster import KMeans
    labels = KMeans(n_clstrs, random_state=random_state).fit_predict(x_np_arr)
    plt.scatter(x_np_arr[:, 0], x_np_arr[:, 1], c=labels, s=50, cmap='viridis')
    return plt


if __name__ == '__main__':
    start_time = time.perf_counter()
    print("start [ " + PROJECT_NAME + " ]>>>>>>>>>>>>>>>>>>")
    # # generate blobs data
    # blobs_x, blobs_y = make_blobs_data()

    # TODO 001
    # kmeans, y_kmeans = do_kmeans(blobs_x)
    # get_plt_of_scikit_kmeans(blobs_x, kmeans, y_kmeans).show()

    # TODO 002 Expectation–Maximization
    # do_expect_max_for_kmanes(blobs_x).show()

    # TODO 003
    """
    For example, if we use a different random seed in our simple procedure
    , the particular starting guesses lead to poor results
    random seed = 0
    """
    # do_expect_max_for_kmanes(blobs_x, rseed=0).show()
    # # of_cluster = 6
    # do_expect_max_for_kmanes(blobs_x, n_clstrs=6).show()

    # TODO 004 The number of clusters must be selected beforehand (n_clusters is hyper param)
    # do_kmeans_with_param(blobs_x, n_clstrs=6).show()
    # TODO 005 k-means is limited to linear cluster boundaries (moons)
    # # generate moon data
    moon_x, moon_y = make_moons_data()
    do_kmeans_with_param(moon_x, 2).show()

    print("::::::::::: %.2f seconds ::::::::::::::" % (time.perf_counter() - start_time))


