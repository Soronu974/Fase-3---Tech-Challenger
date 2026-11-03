"""Gráficos do projeto (todos salvos em images/ quando `save` é informado)."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import roc_curve

from src.config import IMAGES, LIMIAR_ALFABETIZACAO, PONTO_CORTE_SAEB

sns.set_theme(style="whitegrid", context="notebook")
PAL = {"Norte": "#2a9d8f", "Nordeste": "#e76f51", "Centro-Oeste": "#e9c46a", "Sudeste": "#264653", "Sul": "#8ab17d"}


def _save(fig, save):
    fig.tight_layout()
    if save:
        IMAGES.mkdir(parents=True, exist_ok=True)
        fig.savefig(IMAGES / save, dpi=150, bbox_inches="tight")
    return fig


def fig_regioes(gold, save="01_taxa_por_regiao.png"):
    d = gold[gold.rede == 5]
    fig, ax = plt.subplots(figsize=(9, 4.8))
    sns.boxplot(data=d, x="regiao", y="taxa_alfabetizacao", hue="ano", palette="Blues", ax=ax,
                order=["Norte", "Nordeste", "Centro-Oeste", "Sudeste", "Sul"])
    sns.stripplot(data=d, x="regiao", y="taxa_alfabetizacao", hue="ano", dodge=True, palette={2023: "k", 2024: "k"}, size=3,
                  alpha=.6, ax=ax, legend=False, order=["Norte", "Nordeste", "Centro-Oeste", "Sudeste", "Sul"])
    ax.axhline(LIMIAR_ALFABETIZACAO, ls="--", c="crimson", lw=1)
    ax.text(4.45, LIMIAR_ALFABETIZACAO + .8, f"limiar do alvo ({LIMIAR_ALFABETIZACAO:.0f}%)", color="crimson",
            ha="right", fontsize=9)
    ax.set(title="Taxa de alfabetização por região (rede pública, 1 ponto = 1 UF)", xlabel="", ylabel="% alfabetizados")
    return _save(fig, save)


def fig_heatmap_uf(gold, save="02_heatmap_uf.png"):
    w = gold[gold.rede == 5].pivot_table(index="sigla_uf", columns="ano", values="taxa_alfabetizacao")
    w = w.sort_values(2024, ascending=False)
    fig, ax = plt.subplots(figsize=(4.2, 8))
    sns.heatmap(w, annot=True, fmt=".1f", cmap="RdYlGn", center=LIMIAR_ALFABETIZACAO, cbar_kws={"label": "%"}, ax=ax)
    ax.set(title="Taxa pública por UF e ciclo", xlabel="", ylabel="")
    return _save(fig, save)


def fig_variacao(gold, save="03_variacao_2023_2024.png"):
    d = gold[(gold.ano == 2024) & (gold.rede == 5)].dropna(subset=["var_taxa_pp_2023_2024"])
    d = d.sort_values("var_taxa_pp_2023_2024")
    fig, ax = plt.subplots(figsize=(7, 7))
    cores = ["#c0392b" if v < 0 else "#2e8b57" for v in d.var_taxa_pp_2023_2024]
    ax.barh(d.sigla_uf, d.var_taxa_pp_2023_2024, color=cores)
    ax.axvline(0, c="k", lw=.8)
    ax.set(title="Variação da taxa pública 2023→2024 (p.p.)", xlabel="pontos percentuais", ylabel="")
    return _save(fig, save)


def fig_correlacao(gold, save="04_correlacao.png"):
    cols = ["taxa_alfabetizacao", "media_portugues", "diff_taxa_vs_media_ufs", "ranking_taxa_uf"] + \
           [c for c in gold.columns if c.startswith("proporcao_aluno_nivel_")]
    c = gold[gold.ano == 2024][cols].corr(method="spearman")
    fig, ax = plt.subplots(figsize=(9, 7.5))
    sns.heatmap(c, annot=True, fmt=".2f", cmap="coolwarm", center=0, vmin=-1, vmax=1, ax=ax, annot_kws={"size": 7})
    ax.set(title="Correlação de Spearman (2024): variáveis do mesmo ciclo são quase determinísticas")
    return _save(fig, save)


def fig_vazamento(gold, save="05_vazamento_media_vs_taxa.png"):
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.scatterplot(data=gold, x="media_portugues", y="taxa_alfabetizacao", hue="regiao", palette=PAL, ax=ax)
    ax.axvline(PONTO_CORTE_SAEB, c="gray", ls=":")
    ax.axhline(LIMIAR_ALFABETIZACAO, c="crimson", ls="--", lw=1)
    ax.set(title="Média de proficiência x taxa: relação quase perfeita (fonte de vazamento)",
           xlabel="média de língua portuguesa (Saeb)", ylabel="% alfabetizados")
    return _save(fig, save)


def fig_comparacao_modelos(resumo, save="06_comparacao_modelos.png"):
    """resumo: colunas conjunto, modelo, roc_auc_media, roc_auc_desvio."""
    conjuntos = list(resumo.conjunto.unique())
    modelos = list(resumo.modelo.unique())
    w = 0.8 / len(modelos)
    fig, ax = plt.subplots(figsize=(10, 5))
    for i, m in enumerate(modelos):
        d = resumo[resumo.modelo == m].set_index("conjunto").reindex(conjuntos)
        ax.bar(np.arange(len(conjuntos)) + i * w, d.roc_auc_media, w, yerr=d.roc_auc_desvio, label=m, capsize=2)
    ax.set_xticks(np.arange(len(conjuntos)) + w * (len(modelos) - 1) / 2)
    ax.set_xticklabels(conjuntos)
    ax.axhline(0.5, c="gray", ls=":")
    ax.set(ylim=(0, 1.08), title="ROC AUC por conjunto de variáveis (nested CV agrupado por UF; barra = desvio entre repetições)",
           xlabel="", ylabel="ROC AUC")
    ax.title.set_fontsize(10)
    ax.legend(fontsize=8, loc="lower right")
    return _save(fig, save)


def fig_roc(curvas, save="07_roc_temporal.png"):
    fig, ax = plt.subplots(figsize=(5.8, 5.2))
    for nome, (y, p) in curvas.items():
        fpr, tpr, _ = roc_curve(y, p)
        ax.plot(fpr, tpr, label=nome)
    ax.plot([0, 1], [0, 1], "k:", lw=1)
    ax.set(title="Curvas ROC (previsões fora da amostra)", xlabel="FPR", ylabel="TPR")
    ax.legend(fontsize=8)
    return _save(fig, save)


def fig_confusao(cm, titulo, save="08_matriz_confusao.png"):
    fig, ax = plt.subplots(figsize=(4.2, 3.8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                xticklabels=["Risco (<limiar)", "OK (>=limiar)"], yticklabels=["Risco (<limiar)", "OK (>=limiar)"])
    ax.set(title=titulo, xlabel="previsto", ylabel="real")
    return _save(fig, save)


def fig_importancia(imp, titulo, save="09_importancia_permutacao.png"):
    fig, ax = plt.subplots(figsize=(7, 4))
    d = imp.iloc[::-1]
    ax.barh(d.index, d.queda_auc_media, xerr=d.desvio, color="#3a6ea5", alpha=.9)
    ax.axvline(0, c="k", lw=.8)
    ax.set(title=titulo, xlabel="queda na ROC AUC ao permutar a variável")
    return _save(fig, save)


def fig_clusters(df, save="10_clusters_ufs.png"):
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    sns.scatterplot(data=df, x="pc1", y="pc2", hue="cluster", style="regiao", s=110, ax=ax)
    for uf, r in df.iterrows():
        ax.annotate(uf, (r.pc1, r.pc2), xytext=(4, 4), textcoords="offset points", fontsize=8)
    ax.set(title="Clusters de UFs (K-Means, projeção PCA)")
    ax.legend(fontsize=7, bbox_to_anchor=(1.01, 1), loc="upper left")
    return _save(fig, save)


def fig_risco(risco, save="11_risco_proxima_edicao.png"):
    d = risco[risco.rede == 5].sort_values("prob_risco", ascending=True)
    cores = d.prob_risco.apply(lambda p: "#c0392b" if p >= .5 else ("#e9a23b" if p >= .25 else "#2e8b57"))
    fig, ax = plt.subplots(figsize=(7, 7.5))
    ax.barh(d.sigla_uf, d.prob_risco, color=cores)
    ax.axvline(.5, c="k", ls="--", lw=.8)
    ax.set(xlim=(0, 1), title="Probabilidade estimada de a rede pública ficar abaixo do limiar na próxima edição",
           xlabel="P(taxa < limiar)")
    ax.title.set_fontsize(10)
    return _save(fig, save)
