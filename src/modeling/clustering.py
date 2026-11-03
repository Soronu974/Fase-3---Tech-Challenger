"""Agrupamento de UFs com trajetórias parecidas (aprendizado não supervisionado, uso descritivo)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.config import RANDOM_STATE


def uf_matrix(gold: pd.DataFrame) -> pd.DataFrame:
    """Uma linha por UF: nível (rede pública) em 2023/2024, variação e gap estadual-municipal."""
    pub = gold[gold.rede == 5].pivot_table(index="sigla_uf", columns="ano", values="taxa_alfabetizacao")
    pub.columns = [f"taxa_publica_{c}" for c in pub.columns]
    pub["variacao_pp"] = pub["taxa_publica_2024"] - pub["taxa_publica_2023"]
    g24 = gold[(gold.ano == 2024) & gold.rede.isin([2, 3])].pivot_table(
        index="sigla_uf", columns="rede", values="taxa_alfabetizacao")
    pub["gap_estadual_municipal_2024"] = g24[2] - g24[3]
    reg = gold.drop_duplicates("sigla_uf").set_index("sigla_uf")["regiao"]
    return pub.join(reg)


def cluster_ufs(M: pd.DataFrame, k_range=range(2, 6), seed: int = RANDOM_STATE):
    X = M.drop(columns=["regiao"])
    prep = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())
    Z = prep.fit_transform(X)
    sil = {k: float(silhouette_score(Z, KMeans(k, n_init=20, random_state=seed).fit_predict(Z))) for k in k_range}
    k = max(sil, key=sil.get)
    labels = KMeans(k, n_init=20, random_state=seed).fit_predict(Z)
    # rótulos legíveis ordenados pelo nível médio em 2024
    ordem = (X.assign(c=labels).groupby("c")["taxa_publica_2024"].mean().sort_values(ascending=False).index.tolist())
    nomes = {1: ["Único"], 2: ["Alto desempenho", "Baixo desempenho"],
             3: ["Alto desempenho", "Intermediário", "Baixo desempenho"],
             4: ["Alto desempenho", "Médio-alto", "Médio-baixo", "Baixo desempenho"],
             5: ["Alto desempenho", "Médio-alto", "Intermediário", "Médio-baixo", "Baixo desempenho"]}[k]
    nome_por_cluster = {c: nomes[i] for i, c in enumerate(ordem)}
    out = M.copy()
    out["cluster"] = [nome_por_cluster[c] for c in labels]
    pcs = PCA(n_components=2, random_state=seed).fit_transform(Z)
    out["pc1"], out["pc2"] = pcs[:, 0], pcs[:, 1]
    perfil = out.drop(columns=["pc1", "pc2", "regiao"]).groupby("cluster").agg(["mean"]).round(2)
    perfil.columns = [c[0] for c in perfil.columns]
    perfil["n_ufs"] = out.groupby("cluster").size()
    perfil["ufs"] = out.groupby("cluster").apply(lambda d: ", ".join(sorted(d.index)), include_groups=False)
    return out, sil, k, perfil.sort_values("taxa_publica_2024", ascending=False)
