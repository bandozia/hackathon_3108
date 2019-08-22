import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale

lojas_df = pd.read_csv("data/lojas_norm.csv")
lojas_df.dropna(subset=['encrypted_5_zipcode','faturamento_total','produtos_vendidos'], inplace=True)

lojas_df.dropna(subset=['debito','credito'], inplace=True)

Xdf = lojas_df[lojas_df.transacoes_total >= 0.005]

X = np.column_stack([Xdf.encrypted_5_zipcode, Xdf.faturamento_total,Xdf.produtos_vendidos])

kmean = KMeans(init='k-means++', n_clusters=4)
kmean.fit(X)
c = kmean.predict(X)

plt.scatter(X[:,0], X[:,2], c=c)
