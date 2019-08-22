import numpy as np
import pandas as pd
from scipy import stats

lojas_df = pd.read_csv("data/lojas_enc.csv")
lojas_df.drop(['encrypted_domain','cnae'], axis=1, inplace=True)
lojas_df.encrypted_5_zipcode = pd.to_numeric(lojas_df.encrypted_5_zipcode, errors='coerce')

lojas_df.encrypted_5_zipcode = (lojas_df.encrypted_5_zipcode - lojas_df.encrypted_5_zipcode.min()) / (lojas_df.encrypted_5_zipcode.max() - lojas_df.encrypted_5_zipcode.min())
lojas_df.produtos_vendidos = (lojas_df.produtos_vendidos - lojas_df.produtos_vendidos.min()) / (lojas_df.produtos_vendidos.max() - lojas_df.produtos_vendidos.min())
lojas_df.faturamento_total = (lojas_df.faturamento_total - lojas_df.faturamento_total.min()) / (lojas_df.faturamento_total.max() - lojas_df.faturamento_total.min())
lojas_df.transacoes_total = (lojas_df.transacoes_total - lojas_df.transacoes_total.min()) / (lojas_df.transacoes_total.max() - lojas_df.transacoes_total.min())

lojas_df.to_csv("data/lojas_norm.csv", index=False)
print("concluido. removidas as colunas domain e cnae. Normalizadas zipcode, produtos_vendidos,transacoes_total por min/max de cada coluna")
