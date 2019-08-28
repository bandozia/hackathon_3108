import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.externals import joblib
from sklearn.decomposition import PCA
from sklearn import metrics
import matplotlib.pyplot as plt


def train_profile_clustering(csv_path, k=4):
    df = pd.read_csv(csv_path)
    df = df[df["faturamento_total"] < 50000]
    df.dropna(subset=["produtos_vendidos", "transacoes_total", "faturamento_total"], inplace=True)
    subset = df[["produtos_vendidos", "transacoes_total", "faturamento_total"]]
    X = np.column_stack([subset["produtos_vendidos"], subset["transacoes_total"], subset["faturamento_total"]])
    km = KMeans(n_clusters=k, random_state=7)
    km.fit(X)        
    joblib.dump(km, "models/km_profile.pkl")
    pred = km.predict(X)
    reduced = PCA(n_components=2).fit_transform(X)
    plt.style.use("ggplot")
    plt.scatter(reduced[:, 0], reduced[:, 1], c=pred, alpha=0.5)
    score = metrics.silhouette_score(X, pred)
    plt.title("k=%s, silhueta: %s" % (k, round(score)))
    plt.savefig("plots/last_trained_profile_cluster.jpg")
    

def load_profile_clustering():
    km = joblib.load("models/km_profile.pkl")
    return km


def get_clusters_fatures(model, csv_path):
    df = pd.read_csv(csv_path)
    df = df[df["faturamento_total"] < 50000]
    df.dropna(subset=["produtos_vendidos", "transacoes_total", "faturamento_total"], inplace=True)
    subset = df[["produtos_vendidos", "transacoes_total", "faturamento_total"]]
    X = np.column_stack([subset["produtos_vendidos"], subset["transacoes_total"], subset["faturamento_total"]])
    pred = model.predict(X)
    df["cluster"] = pred
    df["ticket_medio"] = df["faturamento_total"] / df["transacoes_total"]
    clusters_features = {}
    for i in range(0, len(df["cluster"].value_counts())):
        clusters_features[i] = {
            "faturamento_medio": df[df["cluster"] == i].faturamento_total.mean(),
            "ticket_medio" : df[df["cluster"] == i].ticket_medio.mean(),
            "trans_medio": df[df["cluster"] == i].transacoes_total.mean()
        }

    return clusters_features



#train_profile_clustering( "../../data/lojas_enc.csv", 4)
#model = load_profile_clustering()
#print(get_clusters_fatures(model, "../../data/lojas_enc.csv"))
