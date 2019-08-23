import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale

df_pj = pd.read_csv("data/transacoes_pj.csv")
df_pj.dateTime = pd.to_datetime(df_pj['dateTime'], errors='coerce')

df_pj.dropna(subset=['productTotal'], inplace=True)
df_pj.dropna(subset=['cep'], inplace=True)

X = np.column_stack([df_pj.cep, df_pj.productTotal])

kmean = KMeans(init='k-means++', n_clusters=4)
kmean.fit(X)
c = kmean.predict(X)

plt.scatter(X[:,1], X[:,0], c=c)
