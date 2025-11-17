#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script para:
1) Buscar metadados no ENA para cada RUN em df_marmags_final.parquet
   - Gera/atualiza df_experiments.parquet (com retomada, backoff e paralelização).
   - Gera/atualiza failed_runs.parquet (RUNs que falharam definitivamente).

2) Montar df_ct_taxonomy a partir de df_marmags_final.parquet
   - Gera df_ct_taxonomy.parquet.

3) Consultar NCBI para obter tax_id específico de cada 'organism' em df_ct_taxonomy
   - Gera/atualiza df_tax_id.parquet (com retomada, backoff e paralelização).
"""

import time
import re
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

import requests
import pandas as pd
import polars as pl
from tqdm import tqdm


# ---------------------------------------------------
# Configuração de paralelização
# ---------------------------------------------------

# Ajusta se quiser usar mais/menos threads
MAX_ENA_WORKERS = 10     # threads para ENA (cuidado com rate limit)
MAX_NCBI_WORKERS = 20    # threads para NCBI

BACKOFF_TIMES_ENA = [0, 30, 60, 120, 240]    # segundos
BACKOFF_TIMES_NCBI = [0, 30, 60, 120, 240]   # segundos


# ---------------------------------------------------
# Helpers gerais
# ---------------------------------------------------

def chunked(iterable, size):
    """Quebra uma lista em blocos de tamanho 'size'."""
    for i in range(0, len(iterable), size):
        yield iterable[i:i+size]


def fix_coordinate_string(coord_str):
    """Conserta string de coordenadas, deixando W/S negativas."""
    if coord_str is None:
        return "missing", "missing"

    coord_str = str(coord_str).strip()

    # Pattern to match latitude and longitude with directions
    pattern = r'(\d*\.?\d+)\s*([NS])\s+(\d*\.?\d+)\s*([EW])'
    match = re.search(pattern, coord_str, re.IGNORECASE)

    if match:
        lat_dir = match.group(2).upper()
        lat_value = -1 * float(match.group(1)) if lat_dir == "S" else float(match.group(1))

        lon_dir = match.group(4).upper()
        lon_value = -1 * float(match.group(3)) if lon_dir == "W" else float(match.group(3))

        return lat_value, lon_value

    return "missing", "missing"  # caso não bata o padrão


# ---------------------------------------------------
# Parte 1: ENA – construção de df_experiments
# ---------------------------------------------------

def process_single_run(run_acc: str) -> pl.DataFrame:
    """
    Faz todo o fluxo RUN -> EXPERIMENT -> STUDY -> SAMPLE -> atributos
    para um único run e retorna um DataFrame polars com 1 linha.
    Dispara exceção se algo der errado, para o loop de retry tratar.
    """

    # 1) RUN
    run_url = f"https://www.ebi.ac.uk/ena/browser/api/xml/{run_acc}"
    run_df = pd.read_xml(
        run_url,
        xpath=f"//RUN[@accession='{run_acc}']/EXPERIMENT_REF"
    )
    if run_df is None or run_df.empty:
        raise RuntimeError(f"RUN sem EXPERIMENT_REF: {run_acc}")

    experiment_acc = run_df["accession"][0]

    # 2) EXPERIMENT
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

    # 3) SAMPLE
    sample_attributes = pd.read_xml(
        f"https://www.ebi.ac.uk/ena/browser/api/xml/{sample_acc}",
        xpath=f"//SAMPLE[@accession='{sample_acc}']/SAMPLE_ATTRIBUTES/SAMPLE_ATTRIBUTE"
    )
    df_sample = pl.from_pandas(sample_attributes)

    # ---- Extração dos campos de interesse ----
    try:
        organism = df_sample.filter(pl.col("TAG") == "organism")["VALUE"].item()
    except Exception:
        organism = "missing"

    # collection_date: aceita "collection_date" e "collection date"
    try:
        collection_date = df_sample.filter(
            pl.col("TAG").is_in(["collection_date", "collection date"])
        )["VALUE"].item()
    except Exception:
        collection_date = "missing"

    try:
        env_broad_scale = df_sample.filter(pl.col("TAG") == "env_broad_scale")["VALUE"].item()
    except Exception:
        env_broad_scale = "missing"

    if env_broad_scale == "missing":
        try:
            env_broad_scale = df_sample.filter(pl.col("TAG") == "env_biome")["VALUE"].item()
        except Exception:
            env_broad_scale = "missing"

    try:
        env_local_scale = df_sample.filter(pl.col("TAG") == "env_local_scale")["VALUE"].item()
    except Exception:
        env_local_scale = "missing"

    if env_local_scale == "missing":
        try:
            env_local_scale = df_sample.filter(pl.col("TAG") == "env_feature")["VALUE"].item()
        except Exception:
            env_local_scale = "missing"

    try:
        env_medium = df_sample.filter(pl.col("TAG") == "env_medium")["VALUE"].item()
    except Exception:
        env_medium = "missing"

    if env_medium == "missing":
        try:
            env_medium = df_sample.filter(pl.col("TAG") == "env_material")["VALUE"].item()
        except Exception:
            env_medium = "missing"

    # geo_loc_name: aceita "geo_loc_name" e "geo loc name"
    try:
        geo_loc_name = df_sample.filter(
            pl.col("TAG").is_in(["geo_loc_name", "geo loc name"])
        )["VALUE"].item()
    except Exception:
        geo_loc_name = "missing"

    try:
        isolation_source = df_sample.filter(pl.col("TAG") == "isolation_source")["VALUE"].item()
    except Exception:
        isolation_source = "missing"

    # lat/lon: aceita "lat_lon" e "lat lon"
    try:
        latlon_str = df_sample.filter(
            pl.col("TAG").is_in(["lat_lon", "lat lon"])
        )["VALUE"].item()
        lat_value, lon_value = fix_coordinate_string(latlon_str)
    except Exception:
        lat_value, lon_value = "missing", "missing"

    if env_broad_scale == "missing" and env_local_scale == "missing" and env_medium == "missing":
        try:
            gold_ecosystem = (
                df_sample.filter(pl.col("TAG") == "GOLD Ecosystem Classification")["VALUE"]
                .item()
                .split(" | ")
            )
            env_broad_scale, env_local_scale, env_medium = gold_ecosystem[1], gold_ecosystem[2], gold_ecosystem[3]
        except Exception:
            gold_ecosystem = "missing"

    df_experiment = pl.DataFrame({
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
        "geographic location (latitude)": [str(lat_value)],
        "geographic location (longitude)": [str(lon_value)],
        "broad-scale environmental context": [env_broad_scale],
        "local environmental context": [env_local_scale],
        "environmental medium": [env_medium],
        "geographic location (country and/or sea)": [geo_loc_name],
        "genome coverage": ["9"],
        "assembly software": ["SPAdes"],
        "platform": ["ILLUMINA"],
        "ENA-CHECKLIST": ["ERC000047"],
    }).with_columns(pl.all().cast(str))

    return df_experiment


def build_df_experiments(df_marmags: pl.DataFrame) -> pl.DataFrame:
    """
    Constrói/atualiza df_experiments.parquet com retomada, backoff e paralelização.
    Usa a coluna 'experiments' de df_marmags como lista de RUNs.
    """
    experiments_path = Path("df_experiments.parquet")
    failed_path = Path("failed_runs.parquet")

    if experiments_path.exists():
        print("Carregando df_experiments existente para retomar...")
        df_experiments = pl.read_parquet(experiments_path)
        completed_runs = set(df_experiments.get_column("run").to_list())
    else:
        print("Nenhum df_experiments encontrado, iniciando do zero...")
        df_experiments = pl.DataFrame()
        completed_runs = set()

    runs_all = (
        df_marmags
        .get_column("experiments")
        .drop_nulls()
        .unique()
        .to_list()
    )

    pending_runs = [r for r in runs_all if r not in completed_runs]

    print(f"Total de RUNs no dataset: {len(runs_all)}")
    print(f"RUNs já concluídos:       {len(completed_runs)}")
    print(f"RUNs restantes:           {len(pending_runs)}")

    # Se não tiver nada pra fazer, só retorna
    if not pending_runs:
        print("Nenhum RUN pendente para ENA.")
        return df_experiments

    # Estado compartilhado
    lock = Lock()
    failed_runs = []

    if failed_path.exists():
        # carrega falhas anteriores (opcional, para histórico)
        df_failed_prev = pl.read_parquet(failed_path)
        failed_runs = df_failed_prev.get_column("run").to_list()

    def worker(run_acc: str):
        nonlocal df_experiments, failed_runs

        # retry + backoff por RUN
        for attempt_idx, wait_seconds in enumerate(BACKOFF_TIMES_ENA):
            if wait_seconds > 0:
                print(f"[ENA:{run_acc}] Tentativa {attempt_idx+1}, esperando {wait_seconds}s...")
                time.sleep(wait_seconds)

            try:
                df_exp_run = process_single_run(run_acc)
                # Atualiza df_experiments e salva imediatamente (thread-safe)
                with lock:
                    df_experiments = pl.concat([df_experiments, df_exp_run])
                    df_experiments.write_parquet(experiments_path)
                return True
            except Exception as e:
                print(f"[ERRO ENA] RUN {run_acc} falhou na tentativa {attempt_idx+1}: {e}")

        # Falha definitiva
        with lock:
            failed_runs.append(run_acc)
            pl.DataFrame({"run": failed_runs}).write_parquet(failed_path)
        print(f"[FALHA DEFINITIVA ENA] RUN {run_acc}")
        return False

    # Paralelização com ThreadPoolExecutor
    futures = []
    with ThreadPoolExecutor(max_workers=MAX_ENA_WORKERS) as executor:
        for run_acc in pending_runs:
            futures.append(executor.submit(worker, run_acc))

        for _ in tqdm(as_completed(futures), total=len(futures), desc="ENA (runs)"):
            pass  # só para a barra de progresso

    print("\nProcessamento ENA finalizado.")
    print(f"Total de registros em df_experiments: {df_experiments.height}")
    if failed_runs:
        print(f"RUNs que falharam definitivamente ({len(failed_runs)}).")
    else:
        print("Nenhum RUN ficou em falha definitiva.")

    return df_experiments


# ---------------------------------------------------
# Parte 2: Construção de df_ct_taxonomy a partir de df_marmags
# ---------------------------------------------------

def build_df_ct_taxonomy(df_marmags: pl.DataFrame) -> pl.DataFrame:
    """
    Monta df_ct_taxonomy a partir de df_marmags_final, no formato desejado.
    """
    df_ct_taxonomy = df_marmags.select([
        pl.col("sample_name"),
        pl.col("experiments").alias("experiment"),
        pl.col("CheckM_Completeness").alias("completeness score"),
        pl.col("CheckM_Contamination").alias("contamination score"),
        pl.col("GTDB_Tk_Domain").alias("gtdb_domain"),
        pl.col("GTDB_Tk_Phylum"),
        pl.col("GTDB_Tk_Class"),
        pl.col("GTDB_Tk_Order"),
        pl.col("GTDB_Tk_Family"),
        pl.col("GTDB_Tk_Genus"),
        pl.col("GTDB_Tk_Species"),
    ])

    # Campo "organism"
    df_ct_taxonomy = df_ct_taxonomy.with_columns(
        pl.when(
            pl.col("GTDB_Tk_Species").is_not_null()
            & (pl.col("GTDB_Tk_Species") != "")
            & (pl.col("GTDB_Tk_Species") != "NA")
        )
        .then(pl.col("GTDB_Tk_Species"))
        .when(
            pl.col("GTDB_Tk_Genus").is_not_null()
            & (pl.col("GTDB_Tk_Genus") != "")
            & (pl.col("GTDB_Tk_Genus") != "NA")
        )
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
        .alias("organism")
    )

    # "domain" como tax_id genérico
    df_ct_taxonomy = df_ct_taxonomy.with_columns(
        pl.when(pl.col("gtdb_domain") == "Bacteria")
          .then(pl.lit("77133"))    # bacterium
          .otherwise(pl.lit("115547"))  # archaeon
          .alias("domain")
    )

    df_ct_taxonomy = df_ct_taxonomy.select([
        "sample_name",
        "experiment",
        "completeness score",
        "contamination score",
        "domain",
        "organism",
    ])

    df_ct_taxonomy.write_parquet("df_ct_taxonomy.parquet")
    print("df_ct_taxonomy.parquet salvo.")
    return df_ct_taxonomy


# ---------------------------------------------------
# Parte 3: NCBI – df_tax_id com backoff + retomada + paralelização
# ---------------------------------------------------

def get_taxonomy_id_with_retry(query: str) -> str | None:
    """
    Consulta o endpoint do NCBI com backoff.
    Retorna tax_id (string) ou None.
    """
    url = f"https://api.ncbi.nlm.nih.gov/datasets/v2/taxonomy/taxon_suggest/{query}?tax_rank_filter=higher_taxon&exact_match=true"

    for attempt_idx, wait_seconds in enumerate(BACKOFF_TIMES_NCBI):
        if wait_seconds > 0:
            print(f"[NCBI:{query}] Esperando {wait_seconds}s antes da tentativa {attempt_idx+1}...")
            time.sleep(wait_seconds)

        try:
            response = requests.get(url, timeout=30)

            if response.status_code == 200:
                data = response.json()
                if data and "sci_name_and_ids" in data and data["sci_name_and_ids"]:
                    return data["sci_name_and_ids"][0]["tax_id"]
                # 200 mas sem resultado
                return None

            print(f"[NCBI:{query}] HTTP {response.status_code} na tentativa {attempt_idx+1}.")
        except Exception as e:
            print(f"[ERRO NCBI] Falha ao consultar '{query}' tentativa {attempt_idx+1}: {e}")

    print(f"[FALHA DEFINITIVA NCBI] '{query}' após todas as tentativas.")
    return None


def build_df_tax_id(df_ct_taxonomy: pl.DataFrame) -> pl.DataFrame:
    """
    Constrói/atualiza df_tax_id.parquet a partir de df_ct_taxonomy,
    com retomada (checkpoint), backoff e paralelização.
    """
    path_out = Path("df_tax_id.parquet")
    failed_path = Path("df_tax_id_failed.parquet")

    if path_out.exists():
        print("Carregando df_tax_id existente (retomada)...")
        df_tax_id = pl.read_parquet(path_out)
        processed_samples = set(df_tax_id.get_column("sample_name").to_list())
    else:
        print("Nenhum df_tax_id encontrado, iniciando do zero...")
        df_tax_id = pl.DataFrame({"sample_name": [], "tax_id": []})
        processed_samples = set()

    print(f"Amostras já processadas: {len(processed_samples)}")

    rows = list(df_ct_taxonomy.iter_rows(named=True))
    pending = [(r["sample_name"], r["organism"]) for r in rows if r["sample_name"] not in processed_samples]

    print(f"Amostras pendentes para NCBI: {len(pending)}")

    if not pending:
        print("Nenhuma amostra pendente para NCBI.")
        return df_tax_id

    lock = Lock()
    failed_samples = []

    if failed_path.exists():
        df_failed_prev = pl.read_parquet(failed_path)
        failed_samples = df_failed_prev.get_column("sample_name").to_list()

    def worker(sample_name: str, organism: str):
        nonlocal df_tax_id, failed_samples

        tax_id = get_taxonomy_id_with_retry(organism)

        with lock:
            df_new = pl.DataFrame({"sample_name": [sample_name], "tax_id": [tax_id]})
            df_tax_id = pl.concat([df_tax_id, df_new])
            df_tax_id.write_parquet(path_out)

            if tax_id is None:
                failed_samples.append(sample_name)
                pl.DataFrame({"sample_name": failed_samples}).write_parquet(failed_path)

        return True

    futures = []
    with ThreadPoolExecutor(max_workers=MAX_NCBI_WORKERS) as executor:
        for sample_name, organism in pending:
            futures.append(executor.submit(worker, sample_name, organism))

        for _ in tqdm(as_completed(futures), total=len(futures), desc="NCBI (samples)"):
            pass

    print("\nProcesso NCBI finalizado!")
    print(f"Total de linhas em df_tax_id: {df_tax_id.height}")
    print(f"Amostras com falha definitiva: {len(failed_samples)}")

    return df_tax_id


# ---------------------------------------------------
# Main
# ---------------------------------------------------

def main():
    # 1) Carrega df_marmags_final (ajuste o caminho se estiver em outro lugar)
    print("Carregando df_marmags_final.parquet...")
    df_marmags = pl.read_parquet("df_marmags_final.parquet")

    # 2) ENA – df_experiments (paralelo)
    df_experiments = build_df_experiments(df_marmags)

    # 3) df_ct_taxonomy (sequencial, rápido)
    df_ct_taxonomy = build_df_ct_taxonomy(df_marmags)

    # 4) NCBI – df_tax_id (paralelo)
    df_tax_id = build_df_tax_id(df_ct_taxonomy)

    print("\n=== Tudo concluído com sucesso ===")
    print(f"df_experiments: {df_experiments.height} linhas")
    print(f"df_ct_taxonomy: {df_ct_taxonomy.height} linhas")
    print(f"df_tax_id:      {df_tax_id.height} linhas")


if __name__ == "__main__":
    main()
