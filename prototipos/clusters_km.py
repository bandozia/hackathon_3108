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

subset = lojas_df[['encrypted_5_zipcode','produtos_vendidos','transacoes_total','faturamento_total']].dropna()
subset = (subset - subset.min()) / (subset.max() - subset.min())

X = np.column_stack([subset.encrypted_5_zipcode, subset.produtos_vendidos,subset.transacoes_total,subset.faturamento_total])
kmeans = KMeans(init='k-means++', n_clusters=3)
kmeans.fit(X)
predicted = kmeans.predict(X)

plt.scatter(X[:,0], X[:,1], c=predicted)
metrics.silhouette_score(X, predicted,metric='euclidean')

sazon_set = lojas_df[['encrypted_5_zipcode','periodo_0','periodo_1','periodo_2','periodo_3','periodo_4']].dropna()
sazon_set = (sazon_set - sazon_set.min()) / (sazon_set.max() - sazon_set.min())

X = np.column_stack([sazon_set.encrypted_5_zipcode, sazon_set.periodo_0,sazon_set.periodo_1,sazon_set.periodo_2,sazon_set.periodo_3,sazon_set.periodo_4])
kmeans = KMeans(init='k-means++', n_clusters=5)
kmeans.fit(X)
predicted = kmeans.predict(X)
metrics.silhouette_score(X, predicted,metric='euclidean')
