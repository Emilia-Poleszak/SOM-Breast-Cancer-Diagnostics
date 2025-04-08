import statistics as st
import numpy as np
import pandas as pd

data = pd.read_csv(r'Breast Cancer Coimbra_MLR\dataR2.csv')

X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

n = np.count_nonzero(y==1)

means1 = []
std1 = []
means2 = []
std2 = []

for i in range(len(X[0])-1):
    means1.append(st.mean(X[:-n,i]))
    means2.append(st.mean(X[-n:,i]))
    std1.append(st.stdev(X[:-n,i]))
    std2.append(st.stdev(X[-n:,i]))

print("Klasa 1")
for i in range(len(means1)):
    print(means1[i])
    print(std1[i])

print("Klasa 2")
for i in range(len(means2)):
    print(means2[i])
    print(std2[i])