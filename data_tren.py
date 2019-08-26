#%% imports
import numpy as np
import pandas as pd
import glob
import re

#%% carregar os dataframes
print("carregando df lojas...")
lojas_df = pd.read_csv("data/stores_details.csv")
print("carregando df transacoes...")
trans_df = pd.read_csv("data/raw_part-00000.csv", error_bad_lines=False)
for f in glob.glob("data/separados/*.{}".format("csv")):
    trans_df = trans_df.append(pd.read_csv(f, error_bad_lines=False),ignore_index=True)

#%% tratar colunas
print("tratando variaveis minimas[dateTime, productTotal]...")
trans_df['dateTime'] = pd.to_datetime(trans_df['dateTime'], errors='coerce')
trans_df['productTotal'] = pd.to_numeric(trans_df['productTotal'], errors='coerce')
trans_df.dropna(subset=['productTotal'], inplace=True)
trans_df.productTotal.isna().value_counts()
trans_df.dateTime.isna().value_counts()

#%% enrriquecer
produtos_vendidos = trans_df.encrypted_cnpj.value_counts()
def count_prod(cnpj):
    if cnpj in produtos_vendidos.index:
        return produtos_vendidos.loc[cnpj]
    else:
        return None


faturamento_total = trans_df.groupby("encrypted_cnpj").productTotal.sum()
def get_faturamento(cnpj):
    if cnpj in faturamento_total.index:
        return faturamento_total.loc[cnpj]
    else:
        return None


transacoes_total = trans_df.groupby(["encrypted_saleid","encrypted_cnpj"], as_index=False).agg('count')
transacoes_total.set_index("encrypted_cnpj", inplace=True)
def count_unique_trans(cnpj):
    if cnpj in transacoes_total.index:
        res = transacoes_total.loc[cnpj].count()
        if isinstance(res, pd.Series):
            return res.iloc[0]
        else:
            return 1
    else:
        return None


print("enrriquecendo df com a quantidade total de produtos...")
lojas_df['produtos_vendidos'] = lojas_df.encrypted_cnpj.apply(lambda x : count_prod(x))
print("enrriquecendo df o faturamento total...")
lojas_df['faturamento_total'] = lojas_df.encrypted_cnpj.apply(lambda x: get_faturamento(x))
print("enrriquecendo com o total de transacoes unicas por cnpj [esse vai demorar]...")
lojas_df['transacoes_total'] = lojas_df.encrypted_cnpj.apply(lambda x : count_unique_trans(x))

print("removendo dados que nao estao nos dois datasets [lojas e transacoes]")
c = lojas_df.produtos_vendidos.isna().value_counts().loc[True]
lojas_df.dropna(subset=['produtos_vendidos'], inplace=True)

print("%d observacoes removidas" % c)

print("salvando dataframe de lojas enrriquecido...")
lojas_df.to_csv("data/lojas_enc.csv", index=False)
print("salvando dataframe de transacoes concatenado [esse e grande]")
trans_df.to_csv("data/transacoes.csv", index=False)
print("tudo feito")
