import pandas as pd



# Caminho do arquivo parquet

arquivo = "df_experiments.parquet" 


# Ler o parquet

df = pd.read_parquet(arquivo)



# Mostrar as 10 primeiras linhas

print(df.head(10))
