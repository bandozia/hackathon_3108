import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale
from sklearn.decomposition import PCA
from sklearn import metrics
from sklearn.metrics import pairwise_distances

lojas_df = pd.read_csv("data/lojas_enc.csv")
lojas_df.encrypted_5_zipcode = pd.to_numeric(lojas_df.encrypted_5_zipcode, errors='coerce')

lojas_df = lojas_df[lojas_df.faturamento_total < 50000]
lojas_df.dropna(subset=['produtos_vendidos','transacoes_total','faturamento_total'], inplace=True)

subset = lojas_df[['produtos_vendidos','transacoes_total','faturamento_total']]
subset = (subset - subset.min()) / (subset.max() - subset.min())

X = np.column_stack([subset.produtos_vendidos,subset.transacoes_total,subset.faturamento_total])

sqr_dist = []
for i in range(1,15):
    kmeans = KMeans(init='k-means++', n_clusters=i)
    kmeans.fit(X)
    sqr_dist.append(kmeans.inertia_)

plt.plot(sqr_dist,'bx-')

kmeans = KMeans(init='k-means++', n_clusters=4)
kmeans.fit(X)
predicted = kmeans.predict(X)


plt.scatter(X[:,1], X[:,2], c=predicted)
metrics.silhouette_score(X, predicted,metric='euclidean')

lojas_df['cluster'] = predicted

sazon_set = lojas_df[['encrypted_5_zipcode','periodo_0','periodo_1','periodo_2','periodo_3','periodo_4']].dropna()
sazon_set = (sazon_set - sazon_set.min()) / (sazon_set.max() - sazon_set.min())

X = np.column_stack([sazon_set.encrypted_5_zipcode, sazon_set.periodo_0,sazon_set.periodo_1,sazon_set.periodo_2,sazon_set.periodo_3,sazon_set.periodo_4])
kmeans = KMeans(init='k-means++', n_clusters=5)
kmeans.fit(X)
predicted = kmeans.predict(X)
metrics.silhouette_score(X, predicted,metric='euclidean')

lojas_df['ticket_medio'] = lojas_df.faturamento_total / lojas_df.transacoes_total


lojas_df[lojas_df.cluster == 0].transacoes_total.mean()
lojas_df[lojas_df.cluster == 0].produtos_vendidos.mean()
lojas_df[lojas_df.cluster == 0].ticket_medio.mean()

lojas_df[lojas_df.cluster == 1].transacoes_total.mean()
lojas_df[lojas_df.cluster == 1].produtos_vendidos.mean()
lojas_df[lojas_df.cluster == 1].ticket_medio.mean()

lojas_df[lojas_df.cluster == 2].transacoes_total.mean()
lojas_df[lojas_df.cluster == 2].produtos_vendidos.mean()
lojas_df[lojas_df.cluster == 2].ticket_medio.mean()

lojas_df[lojas_df.cluster == 3].transacoes_total.mean()
lojas_df[lojas_df.cluster == 3].produtos_vendidos.mean()
lojas_df[lojas_df.cluster == 3].ticket_medio.mean()
