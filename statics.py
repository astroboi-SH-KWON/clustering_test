from statistics import mean
from statistics import stdev
import numpy as np
import math
from bioinfokit import visuz


def fc_diff(x, y):
    return (stdev(x) - stdev(y))


def fc_ratio(x, y):
    return mean(x) / mean(y)


def t_stat(x, y):
    return sum(math.log(x) - math.log(y)) / stdev(math.log(x + y))

gene_1_c = [150, 200, 250]
gene_1_t = [1, 50, 100]

x = gene_1_c
y = gene_1_t
print(fc_diff(x, y), 'fc_dif')
print(fc_ratio(x, y), 'fc_ratio')
# print(t_stat(x, y), 't_stat')

print()
gene_2_c = [101.1, 101.2, 101.3]
gene_2_t = [100.1, 100.2, 100.3]

x = gene_2_c
y = gene_2_t
print(fc_diff(x, y), 'fc_dif')
print(fc_ratio(x, y), 'fc_ratio')
# print(t_stat(x, y), 't_stat')


print((sum(np.log(x)/np.log(2)) - sum(np.log(y)/np.log(2))) / stdev(np.log(x + y)/np.log(2)))

print()

print(mean(gene_1_c))
print(gene_1_c[0] - gene_2_t[0] - mean(gene_1_c))
print((-(gene_1_c[0] - gene_2_t[0] - mean(gene_1_c)))**2)
val = 0.0
for i in range(len(gene_1_c)):
    val += math.log((-(gene_1_c[i] - gene_1_t[i] - mean(gene_1_c)))**2, 2)
print(math.log((-(gene_1_c[0] - gene_1_t[0] - mean(gene_1_c)))**2, 2))



print('d_i')
x = gene_1_c
y = gene_1_t
bunja = sum(x) - sum(y)
a = (1/len(x) + 1/len(y))/(len(x) + len(y) - 2)
# s_i = ( a * ( np.sum(x - np.mean(x)) + np.sum(y - np.mean(y)) ) )**(1/2)
# s_i = 0.0
exp_x = mean(x)
exp_y = mean(y)
sum_x = 0.0
sum_y = 0.0
for i in range(len(x)):
    sum_x += (x[i] - exp_x)**2
    sum_y += (y[i] - exp_y)**2

s_i = ( a * ( sum_x + sum_y ) )**(1/2)

print(a)
print(bunja/(s_i))

print((sum(x) - sum(y)) / sum(y))

import pandas as pd
excel = pd.read_excel('./input/clustering_KYG.xlsx')
# x, y 값 선택
df = pd.DataFrame(excel, columns=['LFC', 'significance'])
print(visuz.gene_exp.volcano(lfc=df['LFC']))