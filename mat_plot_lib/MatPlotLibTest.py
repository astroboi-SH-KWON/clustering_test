import time
import os

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import math

WORK_DIR = os.getcwd() + "/"
PROJECT_NAME = WORK_DIR.split("/")[-2]

#################### st prepare data ####################
# from sklearn.datasets.samples_generator import make_blobs
# X, y_true = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)
# print(type(X))
# print(X.shape)

excel = pd.read_excel('../input/clustering_KYG.xlsx')
# x, y 값 선택
DF = pd.DataFrame(excel, columns=['LFC', 'significance'])
# DF = pd.DataFrame(excel, columns=['LFC', 'guide_score'])
# DF['significance'] = -math.log(DF['significance'], 10)
print(DF['significance'])
# for i in range(len(DF['significance'])):
#     DF['significance'].loc[i] = -math.log(DF['significance'].loc[i], 10)
# min-Max
min_x = min(DF['significance'])
max_x = max(DF['significance'])
for i in range(len(DF['significance'])):
    DF['significance'].loc[i] = (DF['significance'].loc[i] - min_x) / (max_x - min_x)
print(DF['significance'])
#################### en prepare data ####################

"""
kind : str
    ‘line’ : line plot (default)
    ‘bar’ : vertical bar plot
    ‘barh’ : horizontal bar plot
    ‘hist’ : histogram
    ‘box’ : boxplot
    ‘kde’ : Kernel Density Estimation plot
    ‘density’ : same as ‘kde’
    ‘area’ : area plot
    ‘pie’ : pie plot
    ‘scatter’ : scatter plot
    ‘hexbin’ : hexbin plot
"""
def matplotlib_pandas():
    print(DF)

    kind_arr = [
        'line'
        , 'bar'
        , 'barh'
        , 'hist'
        , 'box'
        , 'kde'
        , 'density'
        , 'area'
        , 'pie'
        , 'scatter'
        , 'hexbin'
    ]

    for kind in kind_arr:
        print(kind)
        if kind == 'area':
            # stacked=False bc some columns have negative value
            DF.plot(kind=kind, stacked=False)
        elif kind == 'pie':
            # pie requires either y column or 'subplots=True'
            DF.plot(kind=kind, y="significance")
        elif kind == 'scatter' or kind == 'hexbin':
            # scatter/hexbin requires an x and y column
            DF.plot(kind=kind, x='LFC', y="significance")
        else:
            DF.plot(kind=kind)
        plt.show()


"""
kind{ “scatter” | “kde” | “hist” | “hex” | “reg” | “resid” }
Kind of plot to draw
"""
def jointplot():
    kind_arr = [
        'scatter'
        , 'kde'
        , 'hist'
        , 'hex'
        , 'reg'
        , 'resid'
    ]

    for kind in kind_arr:
        print(kind)
        sns.jointplot(x='LFC', y='significance', data=DF, kind=kind)
        plt.show()


if __name__ == '__main__':
    start_time = time.perf_counter()
    print("start [ " + PROJECT_NAME + " ]>>>>>>>>>>>>>>>>>>")
    # matplotlib_pandas()
    jointplot()
    print("::::::::::: %.2f seconds ::::::::::::::" % (time.perf_counter() - start_time))
