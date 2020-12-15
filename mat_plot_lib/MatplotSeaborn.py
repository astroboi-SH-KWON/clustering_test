import time
import os

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

WORK_DIR = os.getcwd() + "/"
PROJECT_NAME = WORK_DIR.split("/")[-2]


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
    iris = sns.load_dataset('iris')
    df = pd.DataFrame(iris)
    df.plot(kind='pie')
    print(df)
    plt.show()


"""
kind{ “scatter” | “kde” | “hist” | “hex” | “reg” | “resid” }
Kind of plot to draw
"""
def jointplot_iris():
    iris = sns.load_dataset('iris')
    print(iris)

    # x='[column name]', y='[column name]', data=[target data]
    # sns.jointplot(x='sepal_length', y='sepal_width', data=iris)
    # kind='[kind of plots e.g. 'scatter', 'hist', 'hex', 'kde', 'reg', 'resid']'
    sns.jointplot(x='sepal_length', y='sepal_width', data=iris, kind='hist')
    plt.show()


def pairplot_multiple_dataset():
    iris = sns.load_dataset('iris')
    # hue : color
    sns.pairplot(iris, hue='species')
    plt.show()


def heatmap_titanic():
    titanic = sns.load_dataset('titanic')
    titanic_size = titanic.pivot_table(index='who', columns='alive', aggfunc='size')
    sns.heatmap(titanic_size, cmap=sns.light_palette('crimson', as_cmap=True), annot=True, fmt='d')
    print(titanic_size)
    plt.show()


def heatmap_flight():
    flights = sns.load_dataset('flights')
    print(flights)
    print(flights.T)

    flights_passensgers = flights.pivot('month', 'year', 'passengers')
    sns.heatmap(flights_passensgers, annot=True, fmt='d')
    plt.show()


def boxplot_tips():
    tips = sns.load_dataset('tips')
    # sns.boxplot(x='sex', y='total_bill', hue='day', data=tips)
    sns.boxplot(x='day', y='total_bill', hue='sex', data=tips)
    plt.show()


def boxenplot_tips():
    tips = sns.load_dataset('tips')
    # sns.boxenplot(x='sex', y='total_bill', hue='day', data=tips)
    sns.boxenplot(x='day', y='total_bill', hue='sex', data=tips)
    plt.show()


if __name__ == '__main__':
    start_time = time.perf_counter()
    print("start [ " + PROJECT_NAME + " ]>>>>>>>>>>>>>>>>>>")
    # matplotlib_pandas()
    # jointplot_iris()
    # pairplot_multiple_dataset()
    # heatmap_titanic()
    # heatmap_flight()
    boxplot_tips()
    print("::::::::::: %.2f seconds ::::::::::::::" % (time.perf_counter() - start_time))


