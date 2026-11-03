"""Interpretabilidade: importância por permutação (validada em folds), importâncias nativas e SHAP."""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.inspection import permutation_importance
from sklearn.model_selection import StratifiedGroupKFold


def cv_permutation_importance(pipe, X, y, groups, n_splits=5, n_repeats_cv=3, n_repeats_perm=20, seed=42):
    """Importância por permutação medida em dados NÃO vistos no treino (folds agrupados por UF).

    Permuta a coluna ORIGINAL (antes do one-hot), então `regiao` aparece como um bloco só.
    Métrica = queda na ROC AUC.
    """
    rows = []
    for r in range(n_repeats_cv):
        cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed + r)
        for tr, va in cv.split(X, y, groups):
            if len(np.unique(y[va])) < 2:
                continue
            est = clone(pipe).fit(X.iloc[tr], y[tr])
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = permutation_importance(est, X.iloc[va], y[va], scoring="roc_auc",
                                             n_repeats=n_repeats_perm, random_state=seed + r)
            rows.append(pd.Series(res.importances_mean, index=X.columns))
    imp = pd.concat(rows, axis=1)
    return (pd.DataFrame({"queda_auc_media": imp.mean(axis=1), "desvio": imp.std(axis=1)})
            .sort_values("queda_auc_media", ascending=False))


def cv_group_permutation_importance(pipe, X, y, groups, feature_groups: dict, n_splits=5, n_repeats_cv=3,
                                    n_repeats_perm=20, seed=42):
    """Permutação em BLOCO: embaralha juntas todas as colunas de um grupo (ex.: todas as `lag_*`).

    Necessária porque as medidas de desempenho anterior são quase colineares (lag_taxa, lag_media,
    lag_ranking_pct...): permutar uma de cada vez subestima a importância do conjunto.
    """
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(seed)
    rows = []
    for r in range(n_repeats_cv):
        cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed + r)
        for tr, va in cv.split(X, y, groups):
            if len(np.unique(y[va])) < 2:
                continue
            est = clone(pipe).fit(X.iloc[tr], y[tr])
            Xva, yva = X.iloc[va].reset_index(drop=True), y[va]
            base = roc_auc_score(yva, est.predict_proba(Xva)[:, 1])
            row = {}
            for nome, cols in feature_groups.items():
                quedas = []
                for _ in range(n_repeats_perm):
                    Xp = Xva.copy()
                    idx = rng.permutation(len(Xp))
                    Xp[cols] = Xva[cols].values[idx]
                    quedas.append(base - roc_auc_score(yva, est.predict_proba(Xp)[:, 1]))
                row[nome] = np.mean(quedas)
            rows.append(pd.Series(row))
    imp = pd.concat(rows, axis=1)
    return (pd.DataFrame({"queda_auc_media": imp.mean(axis=1), "desvio": imp.std(axis=1)})
            .sort_values("queda_auc_media", ascending=False))


def native_importance(fitted_pipe) -> pd.DataFrame | None:
    """Coeficientes (modelos lineares) ou feature_importances_ (árvores) nas colunas já transformadas."""
    names = fitted_pipe.named_steps["prep"].get_feature_names_out()
    model = fitted_pipe.named_steps["model"]
    if hasattr(model, "coef_"):
        vals = model.coef_.ravel()
        col = "coeficiente (log-odds por desvio-padrão)"
    elif hasattr(model, "feature_importances_"):
        vals = model.feature_importances_
        col = "importancia_impureza"
    else:
        return None
    return (pd.DataFrame({"variavel": names, col: vals})
            .sort_values(col, key=np.abs, ascending=False).reset_index(drop=True))


def shap_summary(fitted_pipe, X, out_png) -> str:
    """Gera o gráfico SHAP quando a biblioteca `shap` está instalada; caso contrário informa e segue.

    Modelos lineares usam LinearExplainer; árvores usam TreeExplainer.
    """
    try:
        import shap
        import matplotlib.pyplot as plt
    except ImportError:
        return "shap não instalado: usando importância por permutação (pip install shap para habilitar)"
    prep = fitted_pipe.named_steps["prep"]
    model = fitted_pipe.named_steps["model"]
    Xt = pd.DataFrame(prep.transform(X), columns=prep.get_feature_names_out())
    if hasattr(model, "coef_"):
        explainer = shap.LinearExplainer(model, Xt)
    else:
        explainer = shap.TreeExplainer(model)
    values = explainer.shap_values(Xt)
    if isinstance(values, list):          # versões antigas devolvem lista por classe
        values = values[1]
    values = np.asarray(values)
    if values.ndim == 3:                  # (n, features, classes)
        values = values[:, :, 1]
    plt.figure()
    shap.summary_plot(values, Xt, show=False)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close()
    return f"SHAP salvo em {out_png}"
