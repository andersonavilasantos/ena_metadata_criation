#!/usr/bin/env python3

# -*- coding: utf-8 -*-



import polars as pl

from pathlib import Path



# ---------------------------------------------------

# Leitura dos arquivos

# ---------------------------------------------------



base_dir = Path(".")



df_ct_taxonomy = pl.read_parquet(base_dir / "df_ct_taxonomy.parquet")

df_tax_id      = pl.read_parquet(base_dir / "df_tax_id.parquet")

df_experiments = pl.read_parquet(base_dir / "df_experiments.parquet")



print("Carregados:")

print("  df_ct_taxonomy :", df_ct_taxonomy.height, "linhas")

print("  df_tax_id      :", df_tax_id.height, "linhas")

print("  df_experiments :", df_experiments.height, "linhas")



# ---------------------------------------------------

# Regex de data (igual antes)

# ---------------------------------------------------



date_regex = (

    r'^[12][0-9]{3}(-(0[1-9]|1[0-2])(-(0[1-9]|[12][0-9]|3[01])'

    r'(T[0-9]{2}:[0-9]{2}(:[0-9]{2})?Z?([+-][0-9]{1,2})?)?)?)?'

    r'(/[0-9]{4}(-[0-9]{2}(-[0-9]{2}'

    r'(T[0-9]{2}:[0-9]{2}(:[0-9]{2})?Z?([+-][0-9]{1,2})?)?)?)?)?$'

)



# ---------------------------------------------------

# 1) join taxonomy + tax_id (NCBI), com fallback pro domain

# ---------------------------------------------------



df_samples = df_ct_taxonomy.join(df_tax_id, on="sample_name", how="left")



df_samples = df_samples.with_columns(

    pl.when(pl.col("tax_id").is_null())

      .then(pl.col("domain"))

      .otherwise(pl.col("tax_id"))

      .alias("tax_id")

)



# ---------------------------------------------------

# 2) join com df_experiments: usar RUN (experiment da esquerda = run da direita)

# ---------------------------------------------------



df_samples = df_samples.join(

    df_experiments,

    left_on="experiment",   # em df_ct_taxonomy é o RUN

    right_on="run",         # em df_experiments é o RUN

    how="left"

)



# ---------------------------------------------------

# 3) Normalizações (país, completeness, data)

# ---------------------------------------------------



df_samples = df_samples.with_columns(

    # só a parte antes de ":" em geographic location (country and/or sea)

    pl.col("geographic location (country and/or sea)")

      .str.split(":")

      .list.get(0)

      .alias("geographic location (country and/or sea)"),



    # completeness score: se quiser garantir 100.0

    pl.when(pl.col("completeness score") == 100.0)

      .then(pl.lit(100.0))

      .otherwise(pl.col("completeness score"))

      .alias("completeness score"),



    # normalização/validação de collection date

    pl.when(

        pl.col("collection date")

        .cast(pl.Utf8)

        .str.replace(r'^(\d{4})$', r'\1-01-01')

        .str.replace(r'^(\d{4})-(\d{2})$', r'\1-\2-01')

        .str.replace(r'^(\d{4})(\d{2})(\d{2})$', r'\1-\2-\3')

        .str.contains(date_regex)

        .fill_null(False)

    )

    .then(

        pl.col("collection date")

        .cast(pl.Utf8)

        .str.replace(r'^(\d{4})$', r'\1-01-01')

        .str.replace(r'^(\d{4})-(\d{2})$', r'\1-\2-01')

        .str.replace(r'^(\d{4})(\d{2})(\d{2})$', r'\1-\2-\3')

    )

    .otherwise(pl.lit("missing"))

    .alias("collection date"),

)



# ---------------------------------------------------

# 4) Selecionar e ordenar as colunas como na saída esperada

# ---------------------------------------------------



cols_final = [

    "sample_name",

    "completeness score",

    "contamination score",

    "organism",

    "tax_id",

    "metagenomic source",

    "sample derived from",

    "project name",

    "completeness software",

    "binning software",

    "assembly quality",

    "binning parameters",

    "taxonomic identity marker",

    "isolation_source",

    "collection date",

    "geographic location (latitude)",

    "geographic location (longitude)",

    "broad-scale environmental context",

    "local environmental context",

    "environmental medium",

    "geographic location (country and/or sea)",

    "genome coverage",

    "assembly software",

    "platform",

    "ENA-CHECKLIST",

]



# algumas colunas podem não existir (dependendo do df_experiments),

# então filtramos pelas que realmente existem:

cols_final = [c for c in cols_final if c in df_samples.columns]



df_samples = df_samples.select(cols_final)



print("df_samples final:", df_samples.height, "linhas")

print(df_samples.head())



# opcional: converter tudo para string, se for exigência do ENA

df_samples_str = df_samples.with_columns(

    [pl.col(c).cast(pl.Utf8) for c in df_samples.columns]

)



df_samples_str.write_parquet(base_dir / "df_samples.parquet")

df_samples_str.write_csv(base_dir / "df_samples.tsv", separator="\t")


