#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script para:

1) Buscar metadados no ENA para cada RUN em df_marmags_final.parquet
   - Gera/atualiza df_experiments.parquet (com retomada e backoff).
   - Gera/atualiza failed_runs.parquet.

2) Construir df_ct_taxonomy a partir de df_marmags_final.parquet
   - Gera df_ct_taxonomy.parquet.

3) Consultar NCBI para tax_id específico de cada organism
   - Gera/atualiza df_tax_id.parquet (com retomada e backoff e schema fixo!).

4) Montar df_samples (tabela final por MAG) juntando tudo
   - Calcula genome coverage = base_count / genome_length (fallback = "9")
   - Gera df_samples.parquet e df_samples.tsv.
"""

import time
import re
from pathlib import Path
from io import StringIO

import requests
import pandas as pd
import polars as pl
from tqdm import tqdm

# ---------------------------------------------------
# CONFIG: nome da coluna com tamanho do genoma no df_marmags
# ---------------------------------------------------
# Troque aqui se no seu df_marmags o nome da coluna de tamanho
# do genoma for outro.
GENOME_LENGTH_COL = "BBTools_scaf_bp"

# ---------------------------------------------------
# Helpers gerais
# ---------------------------------------------------

def chunked(iterable, size):
    """Quebra um iterável em blocos de tamanho `size`."""
    for i in range(0, len(iterable), size):
        yield iterable[i:i + size]

def fix_coordinate_string(coord_str):
    """Conserta string de coordenadas, deixando W/S negativas."""
    if coord_str is None:
        return None, None
    coord_str = str(coord_str).strip()
    pattern = r'(\d*\.?\d+)\s*([NS])\s+(\d*\.?\d+)\s*([EW])'
    match = re.search(pattern, coord_str, re.IGNORECASE)
    if match:
        lat_dir = match.group(2).upper()
        lat_value = -1 * float(match.group(1)) if lat_dir == "S" else float(match.group(1))
        lon_dir = match.group(4).upper()
        lon_value = -1 * float(match.group(3)) if lon_dir == "W" else float(match.group(3))
        return lat_value, lon_value
    return "missing", "missing"

# ---------------------------------------------------
# Parte 1: ENA – df_experiments
# ---------------------------------------------------

def process_single_run(run_acc: str) -> pl.DataFrame:
    """
    Processa um único RUN:
    - ENA browser XML: pega EXPERIMENT, STUDY, SAMPLE, atributos.
    - ENA portal API filereport: tenta pegar base_count (read base count).
    Retorna um DataFrame polars com 1 linha.
    """
    # 1) ENA browser XML (RUN)
    run_url = f"https://www.ebi.ac.uk/ena/browser/api/xml/{run_acc}"
    run_df = pd.read_xml(
        run_url,
        xpath=f"//RUN[@accession='{run_acc}']/EXPERIMENT_REF"
    )
    if run_df is None or run_df.empty:
        raise RuntimeError(f"RUN sem EXPERIMENT_REF: {run_acc}")

    experiment_acc = run_df["accession"][0]
    experiment_url = f"https://www.ebi.ac.uk/ena/browser/api/xml/{experiment_acc}"

    project_acc = pd.read_xml(
        experiment_url,
        xpath=f"//EXPERIMENT[@accession='{experiment_acc}']/STUDY_REF"
    )["accession"][0]

    project_name = pd.read_xml(
        f"https://www.ebi.ac.uk/ena/browser/api/xml/{project_acc}",
        xpath=f"//STUDY[@accession='{project_acc}']/DESCRIPTOR"
    )["STUDY_TITLE"][0]

    sample_acc = pd.read_xml(
        experiment_url,
        xpath=f"//EXPERIMENT[@accession='{experiment_acc}']/DESIGN/SAMPLE_DESCRIPTOR"
    )["accession"][0]

    sample_attributes = pd.read_xml(
        f"https://www.ebi.ac.uk/ena/browser/api/xml/{sample_acc}",
        xpath=f"//SAMPLE[@accession='{sample_acc}']/SAMPLE_ATTRIBUTES/SAMPLE_ATTRIBUTE"
    )
    df_sample = pl.from_pandas(sample_attributes)

    # 2) ENA portal API (filereport) – base_count
    base_count = None
    try:
        portal_url = (
            "https://www.ebi.ac.uk/ena/portal/api/filereport"
            f"?accession={run_acc}"
            "&result=read_run"
            "&fields=run_accession,base_count,read_count"
            "&format=tsv&download=true"
        )
        resp = requests.get(portal_url, timeout=30)
        if resp.status_code == 200 and resp.text.strip():
            df_portal = pd.read_csv(StringIO(resp.text), sep="\t")
            if not df_portal.empty and "base_count" in df_portal.columns:
                val = df_portal.loc[0, "base_count"]
                if pd.notna(val):
                    base_count = int(val)
    except Exception:
        base_count = None

    # função auxiliar para pegar TAGs de SAMPLE
    def try_get(tag_list, default="missing"):
        for tag in tag_list:
            try:
                return df_sample.filter(pl.col("TAG") == tag)["VALUE"].item()
            except Exception:
                pass
        return default

    organism = try_get(["organism"])
    collection_date = try_get(["collection_date", "collection date"])
    env_broad_scale = try_get(["env_broad_scale", "env_biome"])
    env_local_scale = try_get(["env_local_scale", "env_feature"])
    env_medium = try_get(["env_medium", "env_material"])
    geo_loc_name = try_get(["geo_loc_name", "geo loc name"])
    isolation_source = try_get(["isolation_source"])

    try:
        latlon_str = try_get(["lat_lon", "lat lon"])
        lat_value, lon_value = fix_coordinate_string(latlon_str)
    except Exception:
        lat_value, lon_value = "missing", "missing"

    base_count_str = str(base_count) if base_count is not None else None

    df_experiment = pl.DataFrame(
        {
            "run": [run_acc],
            "experiment": [experiment_acc],
            "metagenomic source": [organism],
            "sample derived from": [sample_acc],
            "project name": [project_name],
            "completeness software": ["CheckM"],
            "binning software": ["metaWRAP"],
            "assembly quality": ["Many fragments with little to no review of assembly other than reporting of standard assembly statistics"],
            "binning parameters": ["Default metaWRAP parameters"],
            "taxonomic identity marker": ["Single-copy marker gene set"],
            "isolation_source": [isolation_source],
            "collection date": [collection_date],
            "geographic location (latitude)": [lat_value],
            "geographic location (longitude)": [lon_value],
            "broad-scale environmental context": [env_broad_scale],
            "local environmental context": [env_local_scale],
            "environmental medium": [env_medium],
            "geographic location (country and/or sea)": [geo_loc_name],
            # nova coluna: pares de base lidos do ENA Portal API
            "read base count": [base_count_str],
            # valor de coverage será recalculado depois por MAG
            "genome coverage": ["9"],
            "assembly software": ["SPAdes"],
            "platform": ["ILLUMINA"],
            "ENA-CHECKLIST": ["ERC000047"],
        }
    ).with_columns(pl.all().cast(pl.Utf8))

    return df_experiment

def build_df_experiments(df_marmags: pl.DataFrame) -> pl.DataFrame:
    """
    Constrói/atualiza df_experiments.parquet com retomada e backoff.
    Usa a coluna 'experiments' de df_marmags como lista de RUNs.
    """
    experiments_path = Path("df_experiments.parquet")
    if experiments_path.exists():
        print("Carregando df_experiments existente...")
        df_experiments = pl.read_parquet(experiments_path)
        completed_runs = set(df_experiments["run"].to_list())
    else:
        df_experiments = pl.DataFrame()
        completed_runs = set()

    runs_all = (
        df_marmags["experiments"]
        .drop_nulls()
        .unique()
        .to_list()
    )
    runs = [r for r in runs_all if r not in completed_runs]

    print(f"Total de RUNs no dataset: {len(runs_all)}")
    print(f"RUNs já concluídos:       {len(completed_runs)}")
    print(f"RUNs restantes:           {len(runs)}")

    backoff = [0, 30, 60, 120, 240]
    failed_runs = []

    for batch_idx, batch in enumerate(chunked(runs, 150), start=1):
        print(f"\n=== Processando bloco {batch_idx} com {len(batch)} RUNs ===")
        for run_acc in tqdm(batch, desc=f"Bloco {batch_idx}"):
            if run_acc in completed_runs:
                continue

            success = False
            for attempt, wait in enumerate(backoff):
                if wait > 0:
                    print(f"[{run_acc}] Tentativa {attempt+1}, esperando {wait}s...")
                    time.sleep(wait)
                try:
                    out = process_single_run(run_acc)
                    df_experiments = pl.concat([df_experiments, out])
                    completed_runs.add(run_acc)
                    success = True
                    break
                except Exception as e:
                    print(f"[ERRO] RUN {run_acc} tentativa {attempt+1}: {e}")
                    if attempt == len(backoff) - 1:
                        failed_runs.append(run_acc)

        # checkpoint parcial
        df_experiments.write_parquet("df_experiments.parquet")
        print(f"Checkpoint df_experiments: {df_experiments.height} linhas salvas.")

    if failed_runs:
        pl.DataFrame({"run": failed_runs}).write_parquet("failed_runs.parquet")
        print(f"failed_runs.parquet salvo com {len(failed_runs)} RUNs.")

    return df_experiments

# ---------------------------------------------------
# Parte 2: df_ct_taxonomy
# ---------------------------------------------------

def build_df_ct_taxonomy(df_marmags: pl.DataFrame) -> pl.DataFrame:
    """
    Monta df_ct_taxonomy a partir de df_marmags_final.
    """
    df = df_marmags.select(
        [
            "sample_name",
            pl.col("experiments").alias("experiment"),
            pl.col("CheckM_Completeness").alias("completeness score"),
            pl.col("CheckM_Contamination").alias("contamination score"),
            pl.col("GTDB_Tk_Domain").alias("gtdb_domain"),
            "GTDB_Tk_Phylum",
            "GTDB_Tk_Class",
            "GTDB_Tk_Order",
            "GTDB_Tk_Family",
            "GTDB_Tk_Genus",
            "GTDB_Tk_Species",
        ]
    )

    df = df.with_columns(
        # organism
        pl.when(
            pl.col("GTDB_Tk_Species").is_not_null()
            & (pl.col("GTDB_Tk_Species") != "NA")
            & (pl.col("GTDB_Tk_Species") != "")
        )
        .then(pl.col("GTDB_Tk_Species"))
        .when(pl.col("GTDB_Tk_Genus").is_not_null())
        .then(
            pl.col("GTDB_Tk_Genus")
            + pl.when(pl.col("gtdb_domain") == "Bacteria")
              .then(pl.lit(" bacterium"))
              .otherwise(pl.lit(" archaeon"))
        )
        .otherwise(
            pl.when(pl.col("gtdb_domain") == "Bacteria")
            .then(pl.lit("uncultured bacterium"))
            .otherwise(pl.lit("uncultured archaeon"))
        )
        .alias("organism"),
        # domain como tax_id genérico
        pl.when(pl.col("gtdb_domain") == "Bacteria")
        .then(pl.lit("77133"))
        .otherwise(pl.lit("115547"))
        .alias("domain"),
    )

    df = df.select(
        [
            "sample_name",
            "experiment",
            "completeness score",
            "contamination score",
            "domain",
            "organism",
        ]
    )

    df.write_parquet("df_ct_taxonomy.parquet")
    print("df_ct_taxonomy.parquet salvo.")
    return df

# ---------------------------------------------------
# Parte 3: df_tax_id com schema fixo
# ---------------------------------------------------

BACKOFF_TIMES_NCBI = [0, 30, 60, 120, 240]

def get_taxonomy_id_with_retry(query: str):
    """
    Consulta NCBI com backoff.
    Retorna tax_id (string) ou None.
    """
    url = (
        "https://api.ncbi.nlm.nih.gov/datasets/v2/taxonomy/"
        f"taxon_suggest/{query}?tax_rank_filter=higher_taxon&exact_match=true"
    )

    for wait in BACKOFF_TIMES_NCBI:
        if wait > 0:
            print(f"[NCBI] '{query}' – aguardando {wait}s antes da próxima tentativa...")
            time.sleep(wait)

        try:
            r = requests.get(url, timeout=30)
            if r.status_code == 200:
                data = r.json()
                if data.get("sci_name_and_ids"):
                    return data["sci_name_and_ids"][0]["tax_id"]
                return None
        except Exception as e:
            print(f"[NCBI ERRO] '{query}': {e}")

    print(f"[NCBI FALHA] '{query}' após todas as tentativas.")
    return None

def build_df_tax_id(df_ct_taxonomy: pl.DataFrame) -> pl.DataFrame:
    """
    Constrói/atualiza df_tax_id.parquet a partir de df_ct_taxonomy
    com retomada, backoff e schema fixo Utf8.
    """
    path = Path("df_tax_id.parquet")

    if path.exists():
        df_tax_id = pl.read_parquet(path)
        processed = set(df_tax_id["sample_name"].to_list())
    else:
        df_tax_id = pl.DataFrame(
            {
                "sample_name": pl.Series([], dtype=pl.Utf8),
                "tax_id": pl.Series([], dtype=pl.Utf8),
            }
        )
        processed = set()

    print(f"Amostras já processadas em df_tax_id: {len(processed)}")

    new_rows = []
    failed = []

    for row in tqdm(df_ct_taxonomy.iter_rows(named=True), desc="NCBI taxonomy"):
        sample = row["sample_name"]
        organism = row["organism"]

        if sample in processed:
            continue

        tax_id = get_taxonomy_id_with_retry(organism)

        if tax_id is None:
            failed.append(sample)

        new_rows.append({"sample_name": sample, "tax_id": tax_id})
        processed.add(sample)

        # checkpoint a cada 50
        if len(new_rows) >= 50:
            df_new = pl.DataFrame(new_rows).with_columns(
                pl.col("sample_name").cast(pl.Utf8),
                pl.col("tax_id").cast(pl.Utf8),
            )
            df_tax_id = pl.concat([df_tax_id, df_new])
            df_tax_id.write_parquet(path)
            new_rows = []
            print("Checkpoint df_tax_id salvo.")

    if new_rows:
        df_new = pl.DataFrame(new_rows).with_columns(
            pl.col("sample_name").cast(pl.Utf8),
            pl.col("tax_id").cast(pl.Utf8),
        )
        df_tax_id = pl.concat([df_tax_id, df_new])
        df_tax_id.write_parquet(path)
        print("df_tax_id final salvo.")

    if failed:
        pl.DataFrame({"sample_name": failed}).write_parquet("df_tax_id_failed.parquet")
        print(f"df_tax_id_failed.parquet salvo ({len(failed)} amostras).")

    return df_tax_id

# ---------------------------------------------------
# Parte 4: df_samples final (join + coverage)
# ---------------------------------------------------

def build_df_samples(
    df_marmags: pl.DataFrame,
    df_ct_taxonomy: pl.DataFrame,
    df_tax_id: pl.DataFrame,
    df_experiments: pl.DataFrame,
) -> pl.DataFrame:
    """
    Monta df_samples (tabela final por MAG), juntando:
    - df_ct_taxonomy  (sample_name, completeness, contamination, domain, organism)
    - df_tax_id       (tax_id)  [fallback para domain se tax_id==null]
    - df_experiments  (metagenomic source, sample derived from, project name, etc., read base count)
    - df_marmags      (genome_length via GENOME_LENGTH_COL)
    Calcula genome coverage = read_base_count / genome_length (fallback "9").
    """
    # 1) taxonomy + tax_id (fallback para domain)
    df_samples = df_ct_taxonomy.join(df_tax_id, on="sample_name", how="left")

    df_samples = df_samples.with_columns(
        pl.when(pl.col("tax_id").is_null())
        .then(pl.col("domain"))
        .otherwise(pl.col("tax_id"))
        .alias("tax_id")
    )

    # 2) join com df_experiments
    # df_ct_taxonomy["experiment"] contém o RUN (ex.: SRR...), então precisamos
    # ligar esse campo com df_experiments["run"].
    df_samples = df_samples.join(
        df_experiments,
        left_on="experiment",
        right_on="run",
        how="left",
    )

    # 3) adicionar tamanho do genoma a partir do df_marmags
    # (ajuste GENOME_LENGTH_COL se necessário)
    df_genome_len = df_marmags.select(
        [
            "sample_name",
            pl.col(GENOME_LENGTH_COL).alias("genome_length"),
        ]
    )

    df_samples = df_samples.join(df_genome_len, on="sample_name", how="left")

    # 4) calcular genome coverage = base_count / genome_length (fallback "9")
    df_samples = df_samples.with_columns(
        pl.when(
            pl.col("read base count").is_not_null()
            & pl.col("genome_length").is_not_null()
            & (pl.col("genome_length").cast(pl.Float64) > 0)
        )
        .then(
            (
                pl.col("read base count").cast(pl.Float64)
                / pl.col("genome_length").cast(pl.Float64)
            )
            .round(2)
            .cast(pl.Utf8)
        )
        .otherwise(pl.lit("9"))
        .alias("genome coverage")
    )

    # 5) selecionar colunas na ordem "bonita" parecida com seu exemplo
    cols_order = [
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
        "read base count",
        "genome_length",
    ]

    # mantém apenas as colunas que realmente existem
    cols_order = [c for c in cols_order if c in df_samples.columns]

    df_samples = df_samples.select(cols_order)

    # salvar
    df_samples.write_parquet("df_samples.parquet")
    df_samples.write_csv("df_samples.tsv", separator="\t")

    print("df_samples.parquet e df_samples.tsv salvos.")
    return df_samples

# ---------------------------------------------------
# Main
# ---------------------------------------------------

def main():
    print("Carregando df_marmags_final.parquet...")
    df_marmags = pl.read_parquet("df_marmags_final.parquet")

    df_experiments = build_df_experiments(df_marmags)
    df_ct_taxonomy = build_df_ct_taxonomy(df_marmags)
    df_tax_id = build_df_tax_id(df_ct_taxonomy)
    df_samples = build_df_samples(df_marmags, df_ct_taxonomy, df_tax_id, df_experiments)

    print("\n=== Tudo concluído ===")
    print("df_experiments:", df_experiments.height)
    print("df_ct_taxonomy:", df_ct_taxonomy.height)
    print("df_tax_id:", df_tax_id.height)
    print("df_samples:", df_samples.height)

if __name__ == "__main__":
    main()
