import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import joblib
from sklearn.decomposition import PCA
from sklearn import metrics
import matplotlib.pyplot as plt
import sys
import argparse


def train_profile_clustering(csv_path, k=4, crop=None):
    df = pd.read_csv(csv_path)

    if crop is not None:
        df = df[(df["faturamento_total"] > crop[0]) & (df["faturamento_total"] < crop[1])]

    df.dropna(subset=["produtos_vendidos", "transacoes_total", "faturamento_total", "periodo_0", "periodo_1", "periodo_2", "periodo_3", "periodo_4"], inplace=True)
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
    plt.title("k=%s, silhueta: %s" % (k, score))
    plt.savefig("plots/last_trained_profile_cluster.jpg")


def load_profile_clustering():
    km = joblib.load("models/km_profile.pkl")
    return km


def evaluate_model(model, csv_path, crop=None):
    df = pd.read_csv(csv_path)

    if crop is not None:
        df = df[(df["faturamento_total"] > crop[0]) & (df["faturamento_total"] < crop[1])]

    df.dropna(subset=["produtos_vendidos", "transacoes_total", "faturamento_total"], inplace=True)
    subset = df[["produtos_vendidos", "transacoes_total", "faturamento_total"]]
    X = np.column_stack([subset["produtos_vendidos"], subset["transacoes_total"], subset["faturamento_total"]])
    pred = model.predict(X)
    df["cluster"] = pred
    df["ticket_medio"] = df["faturamento_total"] / df["transacoes_total"]

    clusters_features = pd.DataFrame(columns=["faturamento_medio", "ticket_medio", "trans_medio"])
    for i in range(0, len(df["cluster"].value_counts())):
        clusters_features = clusters_features.append({
            "faturamento_medio": df[df["cluster"] == i].faturamento_total.mean(),
            "ticket_medio": df[df["cluster"] == i].ticket_medio.mean(),
            "trans_medio": df[df["cluster"] == i].transacoes_total.mean()
        }, ignore_index=True)

    m_score = metrics.silhouette_score(X, pred)
    cluster_norm = clusters_features / clusters_features.sum()
    print(m_score)
    print(cluster_norm.std())
    return np.array(cluster_norm.std() * m_score)


def consume(df):
    model = load_profile_clustering()
    X = np.column_stack([df["produtos_vendidos"], df["transacoes_total"], df["faturamento_total"]])
    pred = model.predict(X)
    return pred


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="treine, consuma ou avalie o modelo")

    parser.add_argument("-train", action="store", dest="train_path", required=False)
    parser.add_argument("--k", action="store", dest="k", type=int, default=4)
    parser.add_argument("--crop", action="store", dest="crop", nargs=2, type=int, default=None)

    parser.add_argument("-evaluate", action="store", dest="eval_path", required=False)
    parser.add_argument("-consume", action="store", dest="data_csv", required=False)

    args = parser.parse_args()

    if args.train_path is not None:
        train_profile_clustering(args.train_path, args.k, args.crop)
        print("modelo treinado.")
    elif args.eval_path is not None:
        model_score = evaluate_model(load_profile_clustering(), args.eval_path, args.crop)
        print(model_score)
    elif args.data_csv is not None:
        df = pd.read_csv(args.data_csv)
        pred = consume(df)
        print(pred)

