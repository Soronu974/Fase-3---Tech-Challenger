"""Executa o projeto de ponta a ponta:  python -m src.run_pipeline [--quick]

Gold -> validação -> EDA/hipóteses -> modelagem (3 conjuntos de variáveis) -> interpretação ->
clusters -> previsão de risco. Saídas em reports/, images/, models/ e data/processed/.
"""
from __future__ import annotations

import argparse
import json
import time

import joblib
import numpy as np
import pandas as pd

from src.config import (DATA_PROCESSED, FEATURE_SETS, LIMIAR_ALFABETIZACAO, MODELS, N_REPEATS, RANDOM_STATE,
                        REPORTS, TARGET)
from src.evaluation.hypotheses import hypothesis_tests
from src.evaluation.interpret import (cv_group_permutation_importance, cv_permutation_importance,
                                      native_importance, shap_summary)
from src.modeling.clustering import cluster_ufs, uf_matrix
from src.modeling.train import candidates, fit_final, holdout_eval, nested_cv, persistence_baseline
from src.preprocessing.features import (build_forecast_frame, build_leakage_demo_dataset,
                                        build_structural_dataset, build_temporal_dataset)
from src.preprocessing.load import load_gold
from src.visualization import plots

BUILDERS = {"estrutural": build_structural_dataset, "temporal": build_temporal_dataset,
            "temporal_parcimonioso": build_temporal_dataset, "vazamento_demo": build_leakage_demo_dataset}
BASELINES = ["baseline_dummy", "baseline_persistencia"]


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main(quick: bool = False):
    for d in (REPORTS, DATA_PROCESSED, MODELS):
        d.mkdir(parents=True, exist_ok=True)
    n_rep = 1 if quick else N_REPEATS
    res = {"limiar": LIMIAR_ALFABETIZACAO, "n_repeats": n_rep}

    # 1) Gold + qualidade ------------------------------------------------------
    gold, qual = load_gold()
    (REPORTS / "qualidade_dados.json").write_text(json.dumps(qual, indent=2, ensure_ascii=False))
    log(f"Gold carregada: {qual['linhas']} linhas, {qual['ufs']} UFs")

    # 2) EDA -------------------------------------------------------------------
    hyp = hypothesis_tests(gold)
    hyp.to_csv(REPORTS / "testes_hipoteses.csv", index=False)
    for f in (plots.fig_regioes, plots.fig_heatmap_uf, plots.fig_variacao, plots.fig_correlacao, plots.fig_vazamento):
        f(gold)
    log("EDA e testes de hipótese concluídos")

    # 3) Modelagem -------------------------------------------------------------
    linhas, holds, oof_store, params_store = [], [], {}, {}
    for conjunto, cfg in FEATURE_SETS.items():
        df = BUILDERS[conjunto](gold)
        num, cat = cfg["numeric"], cfg["categorical"]
        X, y, grp = df[num + cat], df[TARGET].values, df["grupo"].values
        reps = min(n_rep, 2) if conjunto == "vazamento_demo" else n_rep
        log(f"[{conjunto}] n={len(df)} | prevalência classe 1 = {y.mean():.2f} | repetições = {reps}")
        for nome, (est, grid) in candidates().items():
            if nome not in cfg.get("modelos", list(candidates())):
                continue
            t0 = time.time()
            r = nested_cv(nome, est, grid, X, y, grp, num, cat, n_repeats=reps)
            m = r["metricas_por_repeticao"]
            linha = {"conjunto": conjunto, "modelo": nome, "n": len(df)}
            for c in m.columns:
                linha[f"{c}_media"], linha[f"{c}_desvio"] = m[c].mean(), m[c].std(ddof=0)
            linhas.append(linha)
            oof_store[(conjunto, nome)] = (y, r["oof_media"], df)
            params_store[(conjunto, nome)] = r["melhores_params"]
            h = holdout_eval(nome, est, grid, X, y, grp, num, cat)
            holds.append({"conjunto": conjunto, "modelo": nome, "n_treino": h["n_treino"], "n_teste": h["n_teste"],
                          "ufs_teste": ",".join(h["ufs_teste"]), **{f"teste_{k}": v for k, v in h["metricas_teste"].items()}})
            oof_store[(conjunto, nome, "holdout")] = h
            log(f"   {nome:22s} AUC={linha['roc_auc_media']:.3f}±{linha['roc_auc_desvio']:.3f} "
                f"({time.time() - t0:.0f}s)")
        if conjunto == "temporal":
            pb = persistence_baseline(df, LIMIAR_ALFABETIZACAO)
            linhas.append({"conjunto": conjunto, "modelo": "baseline_persistencia", "n": len(df),
                           **{f"{k}_media": v for k, v in pb.items()}, **{f"{k}_desvio": 0.0 for k in pb}})
    comp = pd.DataFrame(linhas)
    comp.to_csv(REPORTS / "comparacao_modelos.csv", index=False)
    pd.DataFrame(holds).to_csv(REPORTS / "holdout_teste.csv", index=False)
    plots.fig_comparacao_modelos(comp[comp.modelo != "baseline_persistencia"].rename(
        columns={"roc_auc_media": "roc_auc_media", "roc_auc_desvio": "roc_auc_desvio"}))

    # 4) Seleção do modelo final -------------------------------------------------
    # Regra: maior ROC AUC (nested CV) entre os conjuntos temporais; se o modelo parcimonioso (só
    # a taxa anterior) estiver a menos de 0,01 do melhor, prefere-se o mais simples.
    tcomp = comp[comp.conjunto.isin(["temporal", "temporal_parcimonioso"]) & ~comp.modelo.isin(BASELINES)]
    topo = tcomp.sort_values("roc_auc_media", ascending=False).iloc[0]
    parc = tcomp[tcomp.conjunto == "temporal_parcimonioso"].sort_values("roc_auc_media", ascending=False).iloc[0]
    escolhido = parc if parc.roc_auc_media >= topo.roc_auc_media - 0.01 else topo
    conj_f, modelo_f = escolhido.conjunto, escolhido.modelo
    log(f"Modelo final: {modelo_f} no conjunto '{conj_f}' (AUC nested CV = {escolhido.roc_auc_media:.3f})")

    dft = build_temporal_dataset(gold)
    cfg_f = FEATURE_SETS[conj_f]
    numf, catf = cfg_f["numeric"], cfg_f["categorical"]
    Xf, y, grp = dft[numf + catf], dft[TARGET].values, dft["grupo"].values
    est, grid = candidates()[modelo_f]
    final, best_params, best_cv = fit_final(modelo_f, est, grid, Xf, y, grp, numf, catf)
    joblib.dump(final, MODELS / "modelo_final.joblib")

    # ponto em que a taxa anterior "empata" (P(alfabetização >= limiar) = 50%) no modelo parcimonioso
    corte = None
    if conj_f == "temporal_parcimonioso":
        prep, lr = final.named_steps["prep"], final.named_steps["model"]
        nomes = list(prep.get_feature_names_out())
        w = lr.coef_.ravel()[nomes.index("lag_taxa")]
        sc = prep.named_transformers_["num"].named_steps["scaler"]
        mu, sd = sc.mean_[0], sc.scale_[0]
        corte = float(mu - sd * lr.intercept_[0] / w)
        log(f"Taxa anterior em que P(>= limiar) = 50%: {corte:.1f}%")

    # modelo explicativo = melhor modelo do conjunto temporal completo (para interpretabilidade)
    expl = comp[(comp.conjunto == "temporal") & ~comp.modelo.isin(BASELINES)].sort_values(
        "roc_auc_media", ascending=False).iloc[0].modelo
    cfg_t = FEATURE_SETS["temporal"]
    numt, catt = cfg_t["numeric"], cfg_t["categorical"]
    Xt = dft[numt + catt]
    est_t, grid_t = candidates()[expl]
    explic, params_expl, _ = fit_final(expl, est_t, grid_t, Xt, y, grp, numt, catt)
    joblib.dump(explic, MODELS / "modelo_explicativo.joblib")

    curvas = {n: oof_store[("temporal", n)][:2] for n in ("regressao_logistica", "random_forest", "gradient_boosting")}
    curvas["logística só com taxa anterior"] = oof_store[("temporal_parcimonioso", "regressao_logistica")][:2]
    curvas["persistência (taxa 2023)"] = (y, dft["lag_taxa"].fillna(dft["lag_taxa"].median()).values)
    plots.fig_roc(curvas)
    h = oof_store[(conj_f, modelo_f, "holdout")]
    plots.fig_confusao(h["matriz_confusao"], f"Teste final ({modelo_f})\nUFs: {', '.join(h['ufs_teste'])}")

    imp = cv_permutation_importance(explic, Xt, y, grp)
    imp.to_csv(REPORTS / "importancia_permutacao.csv")
    plots.fig_importancia(imp, f"Importância por permutação — {expl} (CV agrupada por UF)")
    grupos_feat = {"desempenho anterior (todas as lag_*)": [c for c in numt if c.startswith("lag_")],
                   "região": ["regiao"], "rede": ["rede_descricao"]}
    impg = cv_group_permutation_importance(explic, Xt, y, grp, grupos_feat)
    impg.to_csv(REPORTS / "importancia_grupos.csv")
    plots.fig_importancia(impg, f"Importância por permutação em bloco — {expl}", save="13_importancia_grupos.png")
    nat = native_importance(explic)
    if nat is not None:
        nat.to_csv(REPORTS / "importancia_nativa.csv", index=False)
    shap_msg = shap_summary(explic, Xt, str(plots.IMAGES / "12_shap_summary.png"))
    log(shap_msg)

    # 5) Clusters ---------------------------------------------------------------
    M = uf_matrix(gold)
    clus, sil, k, perfil = cluster_ufs(M)
    clus.drop(columns=["pc1", "pc2"]).to_csv(REPORTS / "clusters_ufs.csv")
    perfil.to_csv(REPORTS / "clusters_perfil.csv")
    plots.fig_clusters(clus)
    log(f"Clusters: k={k}, silhouette={sil}")

    # 6) Previsão de risco para a próxima edição --------------------------------
    fc = build_forecast_frame(gold)
    fc["prob_risco"] = 1 - final.predict_proba(fc[numf + catf])[:, 1]
    fc["faixa_risco"] = pd.cut(fc.prob_risco, [-0.01, 0.25, 0.5, 1.0], labels=["baixo", "médio", "alto"])
    fc = fc.sort_values(["rede", "prob_risco"], ascending=[True, False])
    fc.to_csv(REPORTS / "risco_proxima_edicao.csv", index=False)
    plots.fig_risco(fc)
    _, oof, _ = oof_store[(conj_f, modelo_f)]
    dft[["sigla_uf", "rede", "taxa_alfabetizacao", TARGET]].assign(prob_risco_oof=1 - oof).to_csv(
        REPORTS / "risco_oof_2024.csv", index=False)

    res.update({
        "modelo_final": modelo_f, "conjunto_final": conj_f, "modelo_explicativo": expl,
        "corte_taxa_anterior_p50": corte,
        "params_final": {k_: (v if not isinstance(v, np.generic) else v.item()) for k_, v in best_params.items()},
        "params_explicativo": {k_: (v if not isinstance(v, np.generic) else v.item()) for k_, v in params_expl.items()},
        "cv_auc_final": best_cv, "silhouette": sil, "k_clusters": k, "shap": shap_msg,
        "prevalencia_temporal": float(y.mean()), "n_temporal": int(len(dft)),
    })
    (REPORTS / "resultados.json").write_text(json.dumps(res, indent=2, ensure_ascii=False, default=str))
    log("Pipeline concluída.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="1 repetição (teste rápido)")
    main(ap.parse_args().quick)
