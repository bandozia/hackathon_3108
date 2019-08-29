import numpy as np
import pandas as pd

print("gerando base de transacoes de pessoa fisica e juridica")
trans_df = pd.read_csv("data/transacoes.csv")
trans_df['dateTime'] = pd.to_datetime(trans_df['dateTime'], errors='coerce')
trans_df['periodo_venda_norm_4'] = trans_df.dateTime.apply(lambda x: round(((x.hour+1) / 24) * 4))
trans_df['productTotal'] = pd.to_numeric(trans_df['productTotal'], errors='coerce')

trans_pf = trans_df.dropna(subset=['encrypted_buyer_cpf'])

clientes_df = pd.DataFrame()
total_produtos = trans_pf.encrypted_buyer_cpf.value_counts()
clientes_df['cpf'] = total_produtos.index
clientes_df['total_produtos'] = total_produtos.values

valor_total = trans_pf.groupby('encrypted_buyer_cpf').productTotal.sum()
clientes_df['valor_total'] = clientes_df.cpf.apply(lambda x : valor_total.loc[x])

per_0 = trans_pf[trans_pf.periodo_venda_norm_4 == 0].encrypted_buyer_cpf.value_counts()
per_1 = trans_pf[trans_pf.periodo_venda_norm_4 == 1].encrypted_buyer_cpf.value_counts()
per_2 = trans_pf[trans_pf.periodo_venda_norm_4 == 2].encrypted_buyer_cpf.value_counts()
per_3 = trans_pf[trans_pf.periodo_venda_norm_4 == 3].encrypted_buyer_cpf.value_counts()
per_4 = trans_pf[trans_pf.periodo_venda_norm_4 == 4].encrypted_buyer_cpf.value_counts()

def count_sazon(cpf, periodo):
    if cpf in periodo.index:
        return periodo.loc[cpf]
    else:
        return 0

clientes_df['periodo_0'] = clientes_df.cpf.apply(lambda x : count_sazon(x, per_0))
clientes_df['periodo_1'] = clientes_df.cpf.apply(lambda x : count_sazon(x, per_1))
clientes_df['periodo_2'] = clientes_df.cpf.apply(lambda x : count_sazon(x, per_2))
clientes_df['periodo_3'] = clientes_df.cpf.apply(lambda x : count_sazon(x, per_3))
clientes_df['periodo_4'] = clientes_df.cpf.apply(lambda x : count_sazon(x, per_4))

clientes_df.to_csv("data/clientes_pf.csv", index=False)

lojas_df = pd.read_csv("data/lojas_enc.csv")
#lojas_df.encrypted_5_zipcode = pd.to_numeric(lojas_df.encrypted_5_zipcode, errors='coerce')
#lojas_df.encrypted_5_zipcode = (lojas_df.encrypted_5_zipcode - lojas_df.encrypted_5_zipcode.min()) / (lojas_df.encrypted_5_zipcode.max() - lojas_df.encrypted_5_zipcode.min())
#lojas_df.produtos_vendidos = (lojas_df.produtos_vendidos - lojas_df.produtos_vendidos.min()) / (lojas_df.produtos_vendidos.max() - lojas_df.produtos_vendidos.min())
#lojas_df.faturamento_total = (lojas_df.faturamento_total - lojas_df.faturamento_total.min()) / (lojas_df.faturamento_total.max() - lojas_df.faturamento_total.min())
#lojas_df.transacoes_total = (lojas_df.transacoes_total - lojas_df.transacoes_total.min()) / (lojas_df.transacoes_total.max() - lojas_df.transacoes_total.min())

((lojas_df.dinheiro + lojas_df.debito + lojas_df.deposito + lojas_df.transferencia + lojas_df.credito + lojas_df.cheque + lojas_df.crediario) == 1).value_counts()

per_loja_0 = trans_df[trans_df.periodo_venda_norm_4 == 0].encrypted_cnpj.value_counts()
per_loja_1 = trans_df[trans_df.periodo_venda_norm_4 == 1].encrypted_cnpj.value_counts()
per_loja_2 = trans_df[trans_df.periodo_venda_norm_4 == 2].encrypted_cnpj.value_counts()
per_loja_3 = trans_df[trans_df.periodo_venda_norm_4 == 3].encrypted_cnpj.value_counts()
per_loja_4 = trans_df[trans_df.periodo_venda_norm_4 == 4].encrypted_cnpj.value_counts()

def count_sazon_loja(cnpj, periodo):
    if cnpj in periodo.index:
        return periodo.loc[cnpj]
    else:
        return 0

lojas_df['periodo_0'] = lojas_df.encrypted_cnpj.apply(lambda x : count_sazon(x, per_loja_0))
lojas_df['periodo_1'] = lojas_df.encrypted_cnpj.apply(lambda x : count_sazon(x, per_loja_1))
lojas_df['periodo_2'] = lojas_df.encrypted_cnpj.apply(lambda x : count_sazon(x, per_loja_2))
lojas_df['periodo_3'] = lojas_df.encrypted_cnpj.apply(lambda x : count_sazon(x, per_loja_3))
lojas_df['periodo_4'] = lojas_df.encrypted_cnpj.apply(lambda x : count_sazon(x, per_loja_4))

lojas_df.to_csv("data/lojas_enc.csv", index=False)

(lojas_df.produtos_vendidos / lojas_df.transacoes_total).mean()
