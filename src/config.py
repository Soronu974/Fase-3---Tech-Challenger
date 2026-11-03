"""Configuração central do projeto (caminhos, semente, definição do alvo e conjuntos de variáveis).

Toda "decisão analítica" mora aqui, para que a pipeline seja reproduzível e auditável.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_GOLD = ROOT / "data" / "gold" / "br_inep_avaliacao_alfabetizacao_uf_gold.csv"
DATA_EXTERNAL = ROOT / "data" / "external"
DATA_PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"
IMAGES = ROOT / "images"
MODELS = ROOT / "models"

RANDOM_STATE = 42

# --- Definição do alvo --------------------------------------------------------
# A base gold é agregada (ano x UF x rede). O alvo binário é "a maioria dos alunos da
# UF/rede atingiu o patamar de alfabetização" (taxa >= LIMIAR_ALFABETIZACAO).
# O limiar é um parâmetro: com a tabela oficial de metas, basta trocá-lo por taxa >= meta
# em preprocessing/features.py.
LIMIAR_ALFABETIZACAO = 50.0   # % de alunos alfabetizados
PONTO_CORTE_SAEB = 743        # pontos na escala Saeb (Alfabetiza Brasil 2023)
TARGET = "alfabetizacao_ok"   # 1 = taxa >= limiar ; 0 = abaixo (risco educacional)
GROUP = "grupo"               # UF: todas as redes de uma UF caem no mesmo fold

# --- Validação ----------------------------------------------------------------
N_SPLITS = 5
N_REPEATS = 5
INNER_SPLITS = 3
TEST_SPLITS_HOLDOUT = 4       # 1 fold de 4 (~25% das UFs) vira teste final

# --- Conjuntos de variáveis (feature sets) ------------------------------------
# estrutural     : contexto territorial/rede/ano (sem nenhuma medida de desempenho)
# temporal       : desempenho do CICLO ANTERIOR (2023) para prever o ciclo seguinte (2024)
# temporal_parcimonioso : apenas a taxa do ciclo anterior (modelo mínimo de comparação)
# vazamento_demo : inclui a média de proficiência do próprio ciclo -> DEMONSTRAÇÃO de
#                  data leakage; nunca usar como modelo final.
FEATURE_SETS = {
    "estrutural": {
        "numeric": ["ano"],
        "categorical": ["regiao", "rede_descricao"],
        "descricao": "Contexto territorial, rede e ano (sem medidas de desempenho).",
    },
    "temporal": {
        "numeric": ["lag_taxa", "lag_media", "lag_diff_vs_media_ufs",
                    "lag_media_regiao", "lag_ranking_pct"],
        "categorical": ["regiao", "rede_descricao"],
        "descricao": "Desempenho do ciclo anterior (2023) + território/rede para prever 2024.",
    },
    "temporal_parcimonioso": {
        "numeric": ["lag_taxa"],
        "categorical": [],
        "modelos": ["baseline_dummy", "regressao_logistica"],
        "descricao": "Somente a taxa do ciclo anterior (modelo mínimo; testa se o resto agrega sinal).",
    },
    "vazamento_demo": {
        "numeric": ["media_portugues"],
        "categorical": ["regiao", "rede_descricao"],
        "descricao": "Inclui a média de proficiência do mesmo ciclo (vazamento de alvo).",
    },
}
