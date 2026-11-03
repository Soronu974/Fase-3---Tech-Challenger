"""Leitura e validação da camada Gold (contrato de dados)."""
from __future__ import annotations

import pandas as pd

from src.config import DATA_GOLD

REQUIRED_COLUMNS = [
    "ano", "sigla_uf", "nome_uf", "regiao", "serie", "rede", "rede_descricao",
    "taxa_alfabetizacao", "media_portugues", "diferenca_ponto_corte_743",
    "media_acima_ponto_corte", "ranking_taxa_uf", "qtd_ufs_ano_rede", "media_ufs_taxa",
    "diff_taxa_vs_media_ufs", "media_regiao_taxa",
]
KEY = ["ano", "sigla_uf", "serie", "rede"]


def validate_gold(df: pd.DataFrame) -> dict:
    """Regras de qualidade; levanta ValueError em violações graves e devolve um relatório."""
    missing = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Colunas ausentes na Gold: {sorted(missing)}")
    if df.duplicated(KEY).any():
        raise ValueError("Chave (ano, sigla_uf, serie, rede) duplicada na Gold.")
    if not df.taxa_alfabetizacao.between(0, 100).all():
        raise ValueError("taxa_alfabetizacao fora do intervalo [0, 100].")
    if df[["sigla_uf", "regiao", "rede"]].isna().any().any():
        raise ValueError("Chaves de relacionamento com valores ausentes.")
    nivel_cols = [c for c in df.columns if c.startswith("proporcao_aluno_nivel_")]
    soma = df[nivel_cols].sum(axis=1, min_count=1)
    soma = soma[soma.notna()]
    return {
        "linhas": int(len(df)),
        "colunas": int(df.shape[1]),
        "anos": sorted(int(a) for a in df.ano.unique()),
        "ufs": int(df.sigla_uf.nunique()),
        "redes": sorted(int(r) for r in df.rede.unique()),
        "duplicatas_chave": 0,
        "nulos_por_coluna": {c: int(v) for c, v in df.isna().sum().items() if v > 0},
        "linhas_com_distribuicao_niveis": int(len(soma)),
        "soma_niveis_min_max": [float(soma.min()), float(soma.max())] if len(soma) else None,
    }


def load_gold(path=DATA_GOLD) -> tuple[pd.DataFrame, dict]:
    df = pd.read_csv(path)
    return df, validate_gold(df)
