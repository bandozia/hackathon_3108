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
#subset = (subset - subset.min()) / (subset.max() - subset.min())
subset = (subset - subset.mean()) / subset.std()

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

pca = PCA(n_components=2)
reduced = pca.fit_transform(X)

plt.style.use('default')
plt.style.use('ggplot')
plt.scatter(X[:,0], X[:,2], c=predicted, alpha=0.5)
plt.scatter(reduced[:,0], reduced[:,1], c=predicted)
plt.savefig("scatter.jpg")
metrics.silhouette_score(X, predicted,metric='euclidean')

lojas_df['cluster'] = predicted

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


sazon_subset = lojas_df[['transacoes_total','faturamento_total','periodo_0','periodo_1','periodo_2','periodo_3','periodo_4']]
#sazon_subset = (sazon_subset - sazon_subset.min()) / (sazon_subset.max() - sazon_subset.min())
sazon_subset = (sazon_subset - sazon_subset.mean()) / sazon_subset.std()
X = np.column_stack([sazon_subset.transacoes_total,sazon_subset.faturamento_total,sazon_subset.periodo_1,sazon_subset.periodo_1,sazon_subset.periodo_2,sazon_subset.periodo_3,sazon_subset.periodo_4])

kmeans = KMeans(init='k-means++', n_clusters=4)
kmeans.fit(X)
predicted = kmeans.predict(X)

pca = PCA(n_components=1)
reduced_x = pca.fit_transform(X[:,2:6])
reduced_y = pca.fit_transform(X[:,0:2])



plt.scatter(X[:,1], reduced_x, c=predicted)

lojas_df['sazon_cluster'] = predicted

lojas_df['ticket_medio'] = lojas_df.faturamento_total / lojas_df.transacoes_total

plt.style.available

((lojas_df.dinheiro + lojas_df.debito + lojas_df.deposito + lojas_df.transferencia + lojas_df.credito + lojas_df.cheque + lojas_df.crediario) == 1).value_counts()

pag_set = lojas_df[((lojas_df.dinheiro + lojas_df.debito + lojas_df.deposito + lojas_df.transferencia + lojas_df.credito + lojas_df.cheque + lojas_df.crediario) == 1)]

pag_set["digital"] = pag_set.credito + pag_set.debito
pag_set["analogico"] = pag_set.dinheiro + pag_set.deposito + pag_set.transferencia + pag_set.cheque + pag_set.crediario

import random

clusters_features = pd.DataFrame(columns=['faturamento_medio','ticket_medio','trans_medio'])
for i in range(0, 4):
    clusters_features = clusters_features.append({"faturamento_medio": random.randint(10,50),"ticket_medio": 5,"trans_medio": 40}, ignore_index=True)

clusters_features
