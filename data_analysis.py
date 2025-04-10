import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

data = pd.read_csv(r'Breast Cancer Coimbra_MLR\dataR2.csv')

X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

n = np.count_nonzero(y==1)

X1 = pd.DataFrame({'age': X[:n,0],
                   'bmi': X[:n,1],
                   'glucose': X[:n,2],
                   'insulin': X[:n,3],
                   'homa': X[:n,4],
                   'leptin': X[:n,5],
                   'adiponectin': X[:n,6],
                   'resistin': X[:n,7],
                   'mcp.1': X[:n,8]})

X2 = pd.DataFrame({'age': X[n:,0],
                   'bmi': X[n:,1],
                   'glucose': X[n:,2],
                   'insulin': X[n:,3],
                   'homa': X[n:,4],
                   'leptin': X[n:,5],
                   'adiponectin': X[n:,6],
                   'resistin': X[n:,7],
                   'mcp.1': X[n:,8]})

mean1 = X1.mean()
mean2 = X2.mean()
std1 = X1.std()
std2 = X2.std()

print('Class 1:\nmean values:\n', mean1, '\nstandard deviation:\n', std1,
      '\nClass 2:\nmean values:\n', mean2, '\nstandard deviation:\n', std2)

X1.hist(bins=40, figsize=(10, 6), edgecolor='black')
plt.tight_layout()
plt.show()

X2.hist(bins=40, figsize=(10, 6), edgecolor='black')
plt.tight_layout()
plt.show()