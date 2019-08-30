import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import joblib
from sklearn.decomposition import PCA
from sklearn import metrics
import matplotlib.pyplot as plt
import sys
import argparse


def train_profile_clustering(csv_path, k=4):
    df = pd.read_csv(csv_path)

    subset = df[["produtos_vendidos", "transacoes_total", "faturamento_total", "periodo_0", "periodo_1",
                 "periodo_2", "periodo_3", "periodo_4"]].dropna()

    X = np.column_stack([subset["produtos_vendidos"], subset["transacoes_total"],
                         subset["faturamento_total"], subset["periodo_0"], subset["periodo_1"],
                         subset["periodo_2"], subset["periodo_3"], subset["periodo_4"]])

    km = KMeans(n_clusters=k, random_state=7)
    km.fit(X)
    joblib.dump(km, "models/km_profile.pkl")


def load_profile_clustering():
    km = joblib.load("models/km_profile.pkl")
    return km


def evaluate_model(model, csv_path, crop=None):
    df = pd.read_csv(csv_path)

    if crop is not None:
        df = df[(df["faturamento_total"] > crop[0]) & (df["faturamento_total"] < crop[1])]

    subset = df[["produtos_vendidos", "transacoes_total", "faturamento_total", "periodo_0", "periodo_1",
                 "periodo_2", "periodo_3", "periodo_4"]].dropna()

    X = np.column_stack([subset["produtos_vendidos"], subset["transacoes_total"],
                         subset["faturamento_total"], subset["periodo_0"], subset["periodo_1"],
                         subset["periodo_2"], subset["periodo_3"], subset["periodo_4"]])

    pred = model.predict(X)
    df["cluster"] = pred
    df["ticket_medio"] = df["faturamento_total"] / df["transacoes_total"]

    periodos = PCA(n_components=1).fit_transform(np.column_stack([subset["periodo_1"], subset["periodo_2"], subset["periodo_3"], subset["periodo_4"]]))
    df["periodos"] = periodos

    clusters_features = pd.DataFrame(columns=["faturamento_medio", "ticket_medio", "trans_medio", "sazonalidade_diaria"])
    for i in range(0, len(df["cluster"].value_counts())):
        clusters_features = clusters_features.append({
            "faturamento_medio": df[df["cluster"] == i].faturamento_total.mean(),
            "ticket_medio": df[df["cluster"] == i].ticket_medio.mean(),
            "trans_medio": df[df["cluster"] == i].transacoes_total.mean(),
            "sazonalidade_diaria": df[df["cluster"] == i].periodos.mean()
        }, ignore_index=True)

    m_score = metrics.silhouette_score(X, pred)
    cluster_norm = clusters_features / clusters_features.sum()
    print("silhouette_score: %s" % m_score)
    print(cluster_norm.std())
    return np.array(cluster_norm.std() * m_score)


def consume(df):
    model = load_profile_clustering()
    X = np.column_stack([df["produtos_vendidos"], df["transacoes_total"], df["faturamento_total"],
                         df["periodo_0"],df["periodo_1"],df["periodo_2"],df["periodo_3"],df["periodo_4"]])
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
        train_profile_clustering(args.train_path, args.k)
        print("modelo treinado.")
    elif args.eval_path is not None:
        model_score = evaluate_model(load_profile_clustering(), args.eval_path, args.crop)
        print(model_score)
    elif args.data_csv is not None:
        df = pd.read_csv(args.data_csv)
        pred = consume(df)
        print(pred)

