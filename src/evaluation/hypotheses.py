"""Testes estatísticos que sustentam as hipóteses analíticas da EDA (rede 5 = pública, 1 linha por UF/ano)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def _row(hip, teste, n, estat, p, leitura_h0):
    return {"hipotese": hip, "teste": teste, "n": int(n), "estatistica": float(estat),
            "p_valor": float(p),
            "conclusao": "rejeita H0 (p<0,05)" if p < 0.05 else "não rejeita H0 (p>=0,05)",
            "H0": leitura_h0}


def hypothesis_tests(gold: pd.DataFrame) -> pd.DataFrame:
    rows = []
    pub = gold[gold.rede == 5]

    # H1: a taxa difere entre regiões (Kruskal-Wallis por ciclo)
    for ano in sorted(pub.ano.unique()):
        d = pub[pub.ano == ano]
        grupos = [g.taxa_alfabetizacao.values for _, g in d.groupby("regiao")]
        s, p = stats.kruskal(*grupos)
        rows.append(_row(f"H1 regiões diferem ({ano})", "Kruskal-Wallis", len(d), s, p,
                         "a taxa tem a mesma distribuição em todas as regiões"))

    # H2: forte persistência entre ciclos (Spearman 2023 x 2024)
    w = pub.pivot_table(index="sigla_uf", columns="ano", values="taxa_alfabetizacao").dropna()
    rho, p = stats.spearmanr(w[2023], w[2024])
    rows.append(_row("H2 persistência 2023→2024", f"Spearman (rho={rho:.2f})", len(w), rho, p,
                     "não há associação entre a taxa de 2023 e a de 2024"))

    # H3: houve mudança sistemática 2023→2024 (Wilcoxon pareado por UF)
    s, p = stats.wilcoxon(w[2024], w[2023])
    rows.append(_row("H3 mudança sistemática 2023→2024", "Wilcoxon pareado", len(w), s, p,
                     "a mediana das diferenças 2024−2023 é zero"))

    # H4: estadual x municipal (2024, pareado por UF)
    p24 = gold[(gold.ano == 2024) & gold.rede.isin([2, 3])].pivot_table(
        index="sigla_uf", columns="rede", values="taxa_alfabetizacao").dropna()
    s, p = stats.wilcoxon(p24[2], p24[3])
    rows.append(_row("H4 estadual ≠ municipal (2024)", "Wilcoxon pareado", len(p24), s, p,
                     "estadual e municipal têm a mesma mediana na mesma UF"))

    # H5: a rede estadual é mais volátil que a municipal (|variação 2023→2024|)
    v = gold[(gold.ano == 2024) & gold.rede.isin([2, 3])].dropna(subset=["var_taxa_pp_2023_2024"])
    v = v.assign(abs_var=v.var_taxa_pp_2023_2024.abs()).pivot_table(
        index="sigla_uf", columns="rede", values="abs_var").dropna()
    s, p = stats.wilcoxon(v[2], v[3])
    rows.append(_row("H5 estadual mais volátil que municipal", "Wilcoxon pareado em |Δ|", len(v), s, p,
                     "a magnitude da variação é a mesma nas duas redes"))
    return pd.DataFrame(rows)
