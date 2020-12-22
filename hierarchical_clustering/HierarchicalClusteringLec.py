import time
import os
WORK_DIR = os.getcwd() + "/"
PROJECT_NAME = WORK_DIR.split("/")[-2]

import pandas as pd
import numpy as np
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as shc


"""
Rat toxicogenomic study reveals analytical consistency across microarray platforms
"""
#################### st prepare data ####################
excel = pd.read_excel('../input/clustering_KYG.xlsx')
# x, y 값 선택
# df = pd.DataFrame(excel, columns=['significance', 'LFC'])
df = pd.DataFrame(excel, columns=['LFC', 'significance'])
labels = pd.DataFrame(excel, columns=['Gene'])
# input 값을 (n, 2) shape의 numpy.ndarray 로 만들기
X = df.to_numpy()
LABL_ARR = labels.to_numpy()

# X[:, 1] = - np.log(X[:, 1])

# Min-Max Normalization
# for i in range(X.shape[1]):
#     X[:, i] = (X[:, i] - min(X[:, i])) / (max(X[:, i]) - min(X[:, i]))
i = 1
# X[:, i] = (X[:, i] - min(X[:, i])) / (max(X[:, i]) - min(X[:, i]))

X[:, 1] = - np.log(X[:, 1])

METHOD_ARR = ['single', 'complete', 'average', 'centroid', 'weighted', 'median', 'ward']
print('X.shape', X.shape)
#################### en prepare data ####################


def make_scatter_plt_by_AgglomerativeClustering(x_np_arr, mthod, n_clstr=2, affi='euclidean'):
    cluster = AgglomerativeClustering(n_clusters=n_clstr, affinity=affi, linkage=mthod)
    cluster.fit_predict(x_np_arr)

    # set size of window
    plt.figure(figsize=(10, 7))
    plt.scatter(x_np_arr[:, 0], x_np_arr[:, 1], c=cluster.labels_, cmap='rainbow')
    return plt


def make_dendrogram_then_scatter(x_np_arr, mthod_arr, n_clstr=2):
    for mthod in mthod_arr:
        # draw dendrogram without labels
        make_dendrogram_by_method(x_np_arr, None, mthod).show()

        if mthod in ['ward', 'complete', 'average', 'single']:
            make_scatter_plt_by_AgglomerativeClustering(x_np_arr, mthod, n_clstr).show()


def make_dendrogram_by_method(x_np_arr, lbls, mthod):
    # set size of window
    plt.figure(figsize=(10, 7))
    plt.title(mthod + " Dendograms")
    if lbls is None:
        # draw dendrogram without labels
        dend = shc.dendrogram(shc.linkage(x_np_arr, method=mthod))
    else:
        # draw dendrogram with labels
        dend = shc.dendrogram(shc.linkage(x_np_arr, method=mthod), labels=lbls)
    return plt


def make_dendrogram(x_np_arr, lbls, mthod_arr):
    for mthod in mthod_arr:
        # draw dendrogram with labels
        make_dendrogram_by_method(x_np_arr, lbls, mthod).show()


if __name__ == '__main__':
    start_time = time.perf_counter()
    print("start [ " + PROJECT_NAME + " ]>>>>>>>>>>>>>>>>>>")
    make_dendrogram_then_scatter(X, METHOD_ARR)
    # make_dendrogram(X, LABL_ARR, ['ward'])
    print("::::::::::: %.2f seconds ::::::::::::::" % (time.perf_counter() - start_time))