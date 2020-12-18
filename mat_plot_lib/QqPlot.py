import statsmodels.api as sm
from scipy.stats import shapiro
import pylab
import pandas as pd
import numpy as np

"""
normality test
https://machinelearningmastery.com/a-gentle-introduction-to-normality-tests-in-python/
"""
#################### st prepare data ####################


excel = pd.read_excel('../input/clustering_KYG.xlsx')
excel_4KCBE2nd = pd.read_excel('../input/clustering_KYG_4KCBE2nd.xlsx')
# x, y 값 선택
df = pd.DataFrame(excel, columns=['significance', 'LFC'])
# df = pd.DataFrame(excel_4KCBE2nd, columns=['Distance'])
# input 값을 (n, 2) shape의 numpy array 로 만들기
X = df.to_numpy()
# Z-Score Normalization
# X = (X - np.mean(X)) / np.std(X)
# Min-Max Normalization
# X = (X - min(X)) / (max(X) - min(X))


# X = (X[:, 0] - min(X[:, 0])) / (max(X[:, 0]) - min(X[:, 0]))
# print(X)
#################### en prepare data ####################

# Quantile-Quantile Plot
# Quantile-Quantile Plot
sm.qqplot(X[:, 1], line='45')
pylab.show()


# Shapiro-Wilk Test
# Shapiro-Wilk Test
stat, p = shapiro(X[:, 1])
print('Statistics=%.3f, p=%.3f' % (stat, p))
# interpret
alpha = 0.05
if p > alpha:
	print('Sample looks Gaussian (fail to reject H0)')
else:
	print('Sample does not look Gaussian (reject H0)')
