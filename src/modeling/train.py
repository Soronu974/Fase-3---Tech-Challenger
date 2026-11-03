"""Treino, otimização e validação (nested CV agrupado por UF) dos modelos candidatos."""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold

from src.config import (GROUP, INNER_SPLITS, N_REPEATS, N_SPLITS, RANDOM_STATE, TARGET,
                        TEST_SPLITS_HOLDOUT)
from src.evaluation.metrics import compute_metrics, confusion
from src.preprocessing.pipeline import build_model_pipeline


def candidates(seed: int = RANDOM_STATE) -> dict:
    """Modelos candidatos e grades de hiperparâmetros (pequenas: poucos dados, evitar overfitting)."""
    return {
        "baseline_dummy": (DummyClassifier(strategy="prior"), {}),
        "regressao_logistica": (
            LogisticRegression(max_iter=5000),
            {"model__C": [0.01, 0.1, 1.0, 10.0, 100.0]}),
        "random_forest": (
            RandomForestClassifier(n_estimators=150, random_state=seed, n_jobs=1),
            {"model__max_depth": [2, 4], "model__min_samples_leaf": [2, 5]}),
        "gradient_boosting": (
            GradientBoostingClassifier(random_state=seed),
            {"model__n_estimators": [50, 150], "model__max_depth": [1, 2], "model__learning_rate": [0.05, 0.1]}),
    }


def _search(est, grid, numeric, categorical, seed):
    pipe = build_model_pipeline(est, numeric, categorical)
    cv = StratifiedGroupKFold(n_splits=INNER_SPLITS, shuffle=True, random_state=seed)
    return GridSearchCV(pipe, grid or {}, cv=cv, scoring="roc_auc", n_jobs=1,
                        refit=True, error_score=np.nan)


def nested_cv(name, est, grid, X, y, groups, numeric, categorical,
              n_repeats=N_REPEATS, n_splits=N_SPLITS, seed=RANDOM_STATE):
    """Validação aninhada: ajuste de hiperparâmetros no laço interno, desempenho no laço externo.

    Os folds são agrupados por UF (as três redes de uma UF são quase réplicas umas das outras
    e NÃO podem ficar em lados opostos da divisão) e estratificados pelo alvo.
    """
    per_repeat, oof_all, best_params = [], [], []
    for r in range(n_repeats):
        outer = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed + r)
        oof = np.full(len(y), np.nan)
        for tr, te in outer.split(X, y, groups):
            gs = _search(est, grid, numeric, categorical, seed)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                gs.fit(X.iloc[tr], y[tr], groups=groups[tr])
            oof[te] = gs.predict_proba(X.iloc[te])[:, 1]
            best_params.append(str(gs.best_params_))
        per_repeat.append(compute_metrics(y, oof))
        oof_all.append(oof)
    metrics = pd.DataFrame(per_repeat)
    return {"modelo": name, "metricas_por_repeticao": metrics,
            "oof_media": np.nanmean(oof_all, axis=0), "melhores_params": best_params}


def holdout_eval(name, est, grid, X, y, groups, numeric, categorical, seed=RANDOM_STATE):
    """Teste final com UFs inteiramente separadas do treino (um fold de StratifiedGroupKFold)."""
    outer = StratifiedGroupKFold(n_splits=TEST_SPLITS_HOLDOUT, shuffle=True, random_state=seed)
    tr, te = next(outer.split(X, y, groups))
    gs = _search(est, grid, numeric, categorical, seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gs.fit(X.iloc[tr], y[tr], groups=groups[tr])
    proba = gs.predict_proba(X.iloc[te])[:, 1]
    return {"modelo": name, "n_treino": len(tr), "n_teste": len(te),
            "ufs_teste": sorted(set(groups[te])), "cv_auc_treino": float(gs.best_score_),
            "metricas_teste": compute_metrics(y[te], proba), "matriz_confusao": confusion(y[te], proba),
            "y_teste": y[te], "proba_teste": proba, "melhores_params": gs.best_params_}


def persistence_baseline(df: pd.DataFrame, limiar: float) -> dict:
    """Baseline sem aprendizado: 'quem estava acima do limiar em 2023 continua acima em 2024'."""
    lag = df["lag_taxa"].fillna(df["lag_taxa"].median())
    return compute_metrics(df[TARGET].values, (lag.values / 100.0), threshold=limiar / 100.0)


def fit_final(name, est, grid, X, y, groups, numeric, categorical, seed=RANDOM_STATE):
    gs = _search(est, grid, numeric, categorical, seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gs.fit(X, y, groups=groups)
    return gs.best_estimator_, gs.best_params_, float(gs.best_score_)
