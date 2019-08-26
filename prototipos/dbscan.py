import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.decomposition import PCA

lojas_df = pd.read_csv("data/lojas_enc.csv")
lojas_df.encrypted_5_zipcode = pd.to_numeric(lojas_df.encrypted_5_zipcode, errors='coerce')
subset = lojas_df[['encrypted_5_zipcode','produtos_vendidos','transacoes_total','faturamento_total']].dropna()

#minmax
subset = (subset - subset.min()) / (subset.max() - subset.min())
#zscore
#subset = (subset - subset.mean()) / subset.std()

X = np.column_stack([subset.encrypted_5_zipcode, subset.produtos_vendidos,subset.transacoes_total,subset.faturamento_total])

clustering = DBSCAN(metric='euclidean', eps=0.3)
clustering.fit(X)
metrics.silhouette_score(X, clustering.labels_,metric='euclidean')

pca = PCA(n_components=2)
reduced = pca.fit_transform(X)
plt.scatter(reduced[:,0], reduced[:,1], c=clustering.labels_)

sazon_set = lojas_df[['encrypted_5_zipcode','periodo_0','periodo_1','periodo_2','periodo_3','periodo_4']].dropna()
sazon_set = (sazon_set - sazon_set.min()) / (sazon_set.max() - sazon_set.min())

X = np.column_stack([sazon_set.encrypted_5_zipcode, sazon_set.periodo_0,sazon_set.periodo_1,sazon_set.periodo_2,sazon_set.periodo_3,sazon_set.periodo_4])
clustering = DBSCAN(metric='euclidean')
clustering.fit(X)

plt.scatter(X[:,0], X[:,1], c=clustering.labels_)
metrics.silhouette_score(X, clustering.labels_,metric='euclidean')

clientes = pd.read_csv("data/clientes_pf.csv")
X = np.column_stack([clientes.total_produtos, clientes.valor_total])
clustering = DBSCAN(metric='euclidean')
clustering.fit(X)
plt.scatter(X[:,0], X[:,1], c=clustering.labels_)
metrics.silhouette_score(X, clustering.labels_,metric='euclidean')
