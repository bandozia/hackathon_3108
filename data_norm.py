import numpy as np
import pandas as pd

lojas_df = pd.read_csv("data/lojas_enc.csv")
#lojas_df.drop(['encrypted_domain','cnae'], axis=1, inplace=True)
lojas_df.encrypted_5_zipcode = pd.to_numeric(lojas_df.encrypted_5_zipcode, errors='coerce')

lojas_df.encrypted_5_zipcode = (lojas_df.encrypted_5_zipcode - lojas_df.encrypted_5_zipcode.min()) / (lojas_df.encrypted_5_zipcode.max() - lojas_df.encrypted_5_zipcode.min())
lojas_df.produtos_vendidos = (lojas_df.produtos_vendidos - lojas_df.produtos_vendidos.min()) / (lojas_df.produtos_vendidos.max() - lojas_df.produtos_vendidos.min())
lojas_df.faturamento_total = (lojas_df.faturamento_total - lojas_df.faturamento_total.min()) / (lojas_df.faturamento_total.max() - lojas_df.faturamento_total.min())
lojas_df.transacoes_total = (lojas_df.transacoes_total - lojas_df.transacoes_total.min()) / (lojas_df.transacoes_total.max() - lojas_df.transacoes_total.min())

lojas_df.to_csv("data/lojas_norm.csv", index=False)

print("gerando base de transacoes de pessoa fisica e juridica")
trans_df = pd.read_csv("data/transacoes.csv")
trans_df['dateTime'] = pd.to_datetime(trans_df['dateTime'], errors='coerce')
trans_df['productTotal'] = pd.to_numeric(trans_df['productTotal'], errors='coerce')
#trans_df['productTotal'] = (trans_df.productTotal - trans_df.productTotal.min()) / (trans_df.productTotal.max() - trans_df.productTotal.min())

trans_pf = trans_df.dropna(subset=['encrypted_buyer_cpf'])
trans_pj = trans_df.dropna(subset=['encrypted_buyer_cnpj'])

trans_pf[trans_pf.encrypted_buyer_cpf == "3264383032613763326131653863326564326130623736343733396136306164"].dateTime.value_counts()
