"""Pré-processamento integrado ao modelo (sklearn Pipeline + ColumnTransformer).

Tudo é *aprendido apenas no treino* de cada fold, pois está dentro do Pipeline:
imputação (mediana / moda), indicador de ausência, padronização e one-hot.
Esse é o principal mecanismo contra data leakage no pré-processamento.
"""
from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def build_preprocessor(numeric: list[str], categorical: list[str]) -> ColumnTransformer:
    num = Pipeline([
        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
        ("scaler", StandardScaler()),
    ])
    cat = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer(
        [("num", num, numeric), ("cat", cat, categorical)],
        remainder="drop", verbose_feature_names_out=False,
    )


def build_model_pipeline(estimator, numeric: list[str], categorical: list[str]) -> Pipeline:
    return Pipeline([("prep", build_preprocessor(numeric, categorical)), ("model", estimator)])
