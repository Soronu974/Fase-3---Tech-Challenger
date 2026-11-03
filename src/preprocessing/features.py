"""Construção dos datasets de modelagem a partir da Gold (engenharia de atributos).

Cuidado com data leakage (ver docs/decisoes_analiticas.md):
* o alvo é derivado de `taxa_alfabetizacao`; logo NENHUMA coluna calculada a partir da
  taxa/média do MESMO ciclo pode ser preditor (ranking, diff vs média, variação, média da
  região, proporções por nível, média de português...);
* o conjunto "temporal" usa só informação do ciclo ANTERIOR (lag), que existe antes de o
  alvo ser conhecido;
* estatísticas aprendidas (medianas, médias, categorias) são calculadas só dentro dos folds,
  porque ficam dentro do sklearn.Pipeline (preprocessing/pipeline.py).
"""
from __future__ import annotations

import pandas as pd

from src.config import GROUP, LIMIAR_ALFABETIZACAO, TARGET

LAG_MAP = {
    "taxa_alfabetizacao": "lag_taxa",
    "media_portugues": "lag_media",
    "diff_taxa_vs_media_ufs": "lag_diff_vs_media_ufs",
    "media_regiao_taxa": "lag_media_regiao",
}


def add_target(df: pd.DataFrame, limiar: float = LIMIAR_ALFABETIZACAO) -> pd.DataFrame:
    out = df.copy()
    out[TARGET] = (out["taxa_alfabetizacao"] >= limiar).astype(int)
    out[GROUP] = out["sigla_uf"]
    return out


def _lag_frame(prev: pd.DataFrame) -> pd.DataFrame:
    out = prev[["sigla_uf", "rede", *LAG_MAP]].rename(columns=LAG_MAP)
    out["lag_ranking_pct"] = (prev["ranking_taxa_uf"] / prev["qtd_ufs_ano_rede"]).values
    return out


def build_structural_dataset(gold: pd.DataFrame, limiar: float = LIMIAR_ALFABETIZACAO) -> pd.DataFrame:
    """Todas as linhas (ano x UF x rede); preditores apenas territoriais/rede/ano."""
    return add_target(gold, limiar).reset_index(drop=True)


def build_leakage_demo_dataset(gold: pd.DataFrame, limiar: float = LIMIAR_ALFABETIZACAO) -> pd.DataFrame:
    return add_target(gold, limiar).reset_index(drop=True)


def build_temporal_dataset(gold: pd.DataFrame, ano_alvo: int = 2024,
                           limiar: float = LIMIAR_ALFABETIZACAO) -> pd.DataFrame:
    """Linhas do ciclo `ano_alvo` com o desempenho do ciclo anterior (lag) como preditor.

    Linhas sem par no ciclo anterior (AC, RJ-estadual, TO-estadual...) ficam com lag = NaN
    e são tratadas pelo imputador do pipeline (ajustado só no treino de cada fold).
    """
    cur = gold[gold.ano == ano_alvo]
    prev = gold[gold.ano == ano_alvo - 1]
    d = cur.merge(_lag_frame(prev), on=["sigla_uf", "rede"], how="left")
    return add_target(d, limiar).reset_index(drop=True)


def build_forecast_frame(gold: pd.DataFrame, ano_base: int = 2024) -> pd.DataFrame:
    """Features do último ciclo observado formatadas como `lag_*` para prever o próximo ciclo."""
    base = gold[gold.ano == ano_base]
    ids = base[["sigla_uf", "nome_uf", "regiao", "rede", "rede_descricao", "taxa_alfabetizacao"]]
    return ids.merge(_lag_frame(base), on=["sigla_uf", "rede"], how="left")
