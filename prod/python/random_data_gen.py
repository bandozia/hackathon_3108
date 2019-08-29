import numpy as np
import pandas as pd
import random
import argparse


def generate_data(n):
    df = pd.DataFrame(columns=["id", "produtos_vendidos", "faturamento_total", "transacoes_total"])
    tbase = pd.read_csv("../../data/lojas_enc.csv")
    pmin = tbase.produtos_vendidos.min()
    pmax = tbase.produtos_vendidos.max()
    d = round((tbase.produtos_vendidos / tbase.transacoes_total).max())
    for i in range(0, n):
        p = random.randint(pmin, pmax)
        pv = random.randint(1, 10000)
        t = pv / random.randint(1, d)
        df = df.append({"id": i, "produtos_vendidos": pv, "faturamento_total": p * pv, "transacoes_total": t}, ignore_index=True)

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="gerador de dados fajutos para teste do modelo")
    parser.add_argument("-dest", action="store", dest="csv_path", default="teste_data/test.csv")
    parser.add_argument("--n", action="store", dest="n", type=int, default=20)
    args = parser.parse_args()

    df = generate_data(args.n)
    df.to_csv(args.csv_path, index=False)
    print("base fajuta salva em %s" % args.csv_path)
    print(df.head())
    print("...")
