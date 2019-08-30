import numpy as np
import pandas as pd
import random
import argparse


def generate_data(n):
    df = pd.DataFrame(columns=["id", "produtos_vendidos", "faturamento_total", "transacoes_total", "periodo_0", "periodo_1", "periodo_2", "periodo_3", "periodo_4"])
    for i in range(0, n):
        p = random.randint(10, 1000)
        pv = random.randint(1, 100000)
        t = pv / random.randint(1, 50)
        pers = np.random.dirichlet(np.ones(5), size=1)
        df = df.append({
            "id": i, "produtos_vendidos": pv, "faturamento_total": p * pv, "transacoes_total": t,
            "periodo_0": pers[:, 0][0] * t, "periodo_1": pers[:, 1][0] * t, "periodo_2": pers[:, 2][0] * t,
            "periodo_3": pers[:, 3][0] * t, "periodo_4": pers[:, 4][0] * t
        }, ignore_index=True)

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
