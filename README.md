# Tech Challenge – Fase 3: Predição e Inteligência Analítica para Alfabetização no Brasil

Projeto de Machine Learning construído sobre a **camada Gold** da Fase 2 (indicador de alfabetização do
Inep/Saeb, via Base dos Dados). A pipeline cobre EDA, hipóteses estatísticas, modelagem supervisionada com
`scikit-learn`, interpretabilidade, clusters de UFs e uma estimativa de risco educacional.

> ## ⚠️ Leia primeiro: o que esta base permite (e o que não permite)
> O enunciado fala em prever se **um aluno** será alfabetizado, usando metas municipais, dados territoriais e
> socioeconômicos. A Gold disponível (`br_inep_avaliacao_alfabetizacao_uf`) é **agregada por ano × UF × rede**
> (144 linhas, 25 UFs, ciclos 2023 e 2024) e **não contém** microdados de alunos, metas municipais nem variáveis
> socioeconômicas. Portanto:
> * o alvo modelado é o **status de alfabetização da UF/rede** (taxa ≥ 50% dos alunos), **não o do aluno**;
> * "municípios em risco" foi respondido no nível de **UF/rede**;
> * nenhuma fonte externa foi integrada (ver `data/external/README.md`).
>
> A arquitetura é agnóstica à granularidade: com microdados ou dados municipais, basta trocar o construtor de dataset
> e as listas de variáveis em `src/config.py`.

## Resultados em uma página

| Pergunta | Resposta (com esta base) |
|---|---|
| Região/rede preveem alfabetização? | **Pouco.** Só com região + rede + ano, ROC AUC ≈ 0,59–0,63 (quase acaso). Kruskal-Wallis não distingue regiões (p ≈ 0,20 em 2023; 0,32 em 2024). |
| O que mais influencia? | **O desempenho anterior.** Embaralhar o bloco `lag_*` derruba a AUC em ≈ 0,37; região e rede ≈ 0. |
| Dá para prever 2024 com 2023? | **Sim, razoavelmente:** ROC AUC ≈ 0,84 (regressão logística, nested CV agrupada por UF). |
| ML supera uma regra simples? | **Não nesta amostra.** O baseline de persistência ("quem estava ≥ 50% em 2023 continua") obtém AUC ≈ 0,88. |
| Quem tem maior risco na próxima edição? | Rede pública, P(risco) ≥ 0,5: **BA, SE, RN, RS, PA, AP, AM**; atenção (0,25–0,5): **AL, TO, AC, RJ, MS**. |
| Há grupos de UFs parecidas? | 3 clusters (silhouette ≈ 0,29, estrutura fraca/moderada) que **cruzam regiões**. |

## 1. Contexto do problema
A alfabetização na idade certa é pilar do desenvolvimento educacional. O **Compromisso Nacional Criança Alfabetizada**
mobiliza União, estados e municípios para que todas as crianças estejam alfabetizadas ao fim do 2º ano; o Inep fixou
o ponto de corte de **743 pontos** na escala Saeb (Pesquisa Alfabetiza Brasil, 2023), e o **Indicador Criança
Alfabetizada** é o percentual de estudantes acima dele. Gestores precisam **antecipar riscos**, identificar redes
vulneráveis e entender o que pesa nos resultados — não só olhar o indicador do ano.

## 2. Objetivo analítico
1. Explorar a Gold, formular e testar hipóteses (EDA).
2. Construir uma pipeline de ML **sem data leakage** que preveja se uma UF/rede terá taxa ≥ 50% no ciclo seguinte.
3. Identificar os fatores de maior influência e padrões semelhantes entre UFs.
4. Estimar quais UFs/redes têm maior risco de ficar abaixo do limiar na próxima edição.

## 3. Base utilizada
`data/gold/br_inep_avaliacao_alfabetizacao_uf_gold.csv` — 144 linhas × 30 colunas.

| Item | Detalhe |
|---|---|
| Granularidade | ano (2023, 2024) × UF (25 de 27; faltam **RR** e **DF**) × rede (2 = estadual, 3 = municipal, 5 = pública) |
| Medidas | `taxa_alfabetizacao`, `media_portugues`, proporções por nível 0–8 (**só 2024**: 70 linhas nulas em 2023) |
| Derivadas na Gold | ponto de corte 743, ranking, diferença contra média das UFs, média da região, variação 2023→2024 |
| Alvo (`alfabetizacao_ok`) | 1 se `taxa_alfabetizacao ≥ 50` (limiar configurável em `src/config.py`; **proxy de meta**, pois a tabela de metas não está na Gold) |
| Prevalência | 66% classe 1 / 34% classe 0 (risco) |

Os rótulos de `rede` (2/3/5) foram **inferidos** (o 5 fica sempre entre 2 e 3); vale confirmar no dicionário oficial.

## 4. Estrutura do repositório

tech-challenge-fase3/
├── data/
│   ├── gold/            # Gold da Fase 2 (entrada)
│   ├── external/        # (vazio) enriquecimento opcional – ver README da pasta
│   └── processed/
├── notebooks/
│   ├── 01_eda.ipynb          # EDA + hipóteses (executado)
│   └── 02_modelagem.ipynb    # pipeline de ML, resultados e perguntas de negócio (executado)
├── src/
│   ├── config.py             # alvo, limiar, sementes, conjuntos de variáveis
│   ├── preprocessing/        # load.py (contrato de dados), features.py (alvo/lags), pipeline.py (sklearn)
│   ├── modeling/             # train.py (nested CV, tuning), clustering.py
│   ├── evaluation/           # metrics.py, interpret.py (permutação/SHAP), hypotheses.py
│   ├── visualization/        # plots.py
│   └── run_pipeline.py      # orquestra tudo
├── reports/                 # tabelas (.csv/.json), log e roteiro do vídeo
├── images/                  # gráficos
├── models/                  # modelos treinados (.joblib)
├── docs/decisoes_analiticas.md
├── requirements.txt
└── README.md

Semente fixa (`RANDOM_STATE = 42`). Ambiente em que os resultados foram gerados: Python 3.12, scikit-learn 1.8.0,
pandas 3.0.2, numpy 2.4.4, scipy 1.17.1. Os notebooks já vêm executados; para reexecutá-los instale `jupyter`.

## 5. Etapas de modelagem
1. **Contrato de dados** (`load.py`): colunas obrigatórias, chave única, taxa em [0, 100], chaves sem nulos.
2. **Engenharia de atributos** (`features.py`): `lag_taxa`, `lag_media`, `lag_diff_vs_media_ufs`, `lag_media_regiao`,
   `lag_ranking_pct` = valores de **2023** da mesma UF/rede para prever **2024**.
3. **Pré-processamento integrado ao modelo** (`pipeline.py`): numéricas → imputação por mediana com indicador de
   ausência + padronização; categóricas → imputação por moda + one-hot (`handle_unknown="ignore"`).
4. **Validação** (`train.py`): `StratifiedGroupKFold` agrupado por **UF**; **nested CV** (3 folds internos para
   hiperparâmetros, 5 externos) repetida com **5 sementes**; mais um **teste final** com UFs inteiras fora do treino.
5. **Interpretação** (`interpret.py`): permutação em folds não vistos (por variável e **em bloco**), coeficientes e SHAP opcional.

### Controle de data leakage
| Risco | Controle |
|---|---|
| Preditor calculado a partir do alvo (média, níveis, ranking, diff vs. média do **mesmo** ciclo) | Excluídos; só entram valores do ciclo **anterior** (`lag_*`) |
| Redes 2/3/5 da mesma UF são quase réplicas | Folds **agrupados por UF** |
| Estatísticas de imputação/escala aprendidas com dados de validação | Tudo dentro do `Pipeline`, ajustado só no treino de cada fold |
| Escolher hiperparâmetros olhando o teste | Validação aninhada; o laço externo nunca participa da escolha |
| Resultado bom demais | Experimento `vazamento_demo` mostra AUC ≈ 0,97 com `media_portugues` – prova do problema |

## 6. Escolha do algoritmo
Candidatos: **regressão logística** (regularizada), **random forest** e **gradient boosting**, com grades pequenas
(poucos dados → risco de overfitting), mais baselines `Dummy` e **persistência**. A escolha foi pela **ROC AUC da
nested CV**: a regressão logística (C = 0,1) ficou à frente (0,842 ± 0,009), com a vantagem de ser simples e interpretável.
Um modelo mínimo só com `lag_taxa` obteve 0,799 ± 0,020; como não ficou a menos de 0,01 do melhor, não foi preferido.

## 7. Métricas e resultados
Classe 1 = alfabetização ≥ 50%; classe 0 = risco. Média ± desvio entre 5 repetições (nested CV, agrupada por UF):

| Conjunto de variáveis | Modelo | ROC AUC | PR AUC | F1 (risco) | Acurácia balanceada | Brier |
|---|---|---|---|---|---|---|
| estrutural (n=144) | regressão logística | 0,587 ± 0,037 | 0,771 | 0,15 | 0,49 | 0,225 |
| estrutural | random forest | 0,595 ± 0,032 | 0,780 | 0,14 | 0,47 | 0,224 |
| estrutural | gradient boosting | 0,626 ± 0,033 | 0,797 | 0,31 | 0,52 | 0,233 |
| **temporal (n=74)** | **regressão logística** | **0,842 ± 0,009** | 0,880 | 0,38 | 0,61 | 0,168 |
| temporal | random forest | 0,813 ± 0,009 | 0,811 | 0,75 | 0,81 | 0,131 |
| temporal | gradient boosting | 0,812 ± 0,008 | 0,838 | 0,80 | 0,84 | 0,124 |
| temporal | **baseline de persistência** | **0,882** | 0,924 | 0,80 | 0,85 | — |
| temporal parcimonioso | logística só com `lag_taxa` | 0,799 ± 0,020 | 0,867 | 0,08 | 0,52 | 0,202 |
| vazamento (demonstração) | RF / GB / logística | 0,965 – 0,975 | — | — | — | — |

(Tabela completa, com desvios, em `reports/comparacao_modelos.csv`.)

Como ler:
* **AUC mede ranqueamento** (qual UF é mais arriscada). Com limiar 0,5 a logística quase nunca prevê risco (F1 de risco 0,38),
  enquanto RF/GB detectam melhor a classe 0 (F1 ≈ 0,75–0,80). A decisão de limiar é de política pública (seção 11).
* O `Dummy` com AUC < 0,5 é artefato do método (previsão constante muda por fold); leia como 0,5.
* **Teste final** (7 UFs, 21 linhas: AP, ES, MT, PA, PB, PR, SC): AUC 0,71 (logística), 0,90 (RF), 0,89 (GB). Amostra
  minúscula; usar como verificação, não como estimativa principal.

## 8. Interpretação dos resultados
* **Desempenho anterior é o único sinal relevante.** Permutação em bloco: queda de AUC ≈ 0,37 ± 0,18 para `lag_*`;
  ≈ 0,00 para região e para rede. Variáveis `lag_*` isoladas aparecem com importância baixa e instável porque são
  **quase colineares** (todas medem o mesmo nível de desempenho).
* **Região não separa bem os resultados.** Dentro do Nordeste, CE (~85%) e BA (~36%) estão em extremos do país.
* **O ML não bate a persistência** nesta amostra. O mais honesto é dizer que a taxa anterior já carrega quase toda a informação.
* **SHAP:** `shap` não estava instalado no ambiente de geração (sem internet). O código gera `images/12_shap_summary.png`
  automaticamente após `pip install shap`; **esta etapa não foi executada nesta entrega**.

## 9. Insights encontrados
1. **Persistência forte** entre ciclos (Spearman 0,85 entre as taxas públicas de 2023 e 2024).
2. **Melhora geral com um outlier:** 18 de 24 UFs melhoraram (mediana ≈ +3,5 p.p.; Wilcoxon p ≈ 0,004), mas o **RS caiu ≈ 19 p.p.**
   (63% → 45%). AM também cruzou para baixo de 50%; MS e TO saíram da zona de risco.
3. **Desigualdade estável:** amplitude entre UFs ≈ 53 p.p. (2023) e ≈ 49 p.p. (2024); 8 UFs abaixo de 50% nos dois anos.
4. **Rede estadual × municipal:** sem diferença sistemática detectada (p ≈ 0,11) nem diferença de volatilidade (p ≈ 0,37).
5. **Clusters** (K-Means, k = 3 por silhouette): *Alto* (CE, PR, RO, RS, SC), *Intermediário* (11 UFs) e *Baixo* (AL, AP, BA, MS, MT, PB, RN, SE, TO).
   Atenção: o RS está no cluster "Alto" pelo nível de 2023 (já caiu); o AC não tem 2023 (valor imputado).
6. **Risco para a próxima edição** (rede pública, P(taxa < 50%)): ver `reports/risco_proxima_edicao.csv` e `images/11_risco_proxima_edicao.png`.

## 10. Aplicação prática para políticas públicas
* **Alerta precoce, não ranking punitivo.** A faixa *alta* (P ≥ 0,5) tem precisão ≈ 87% e recall ≈ 52% nas previsões fora da amostra
  de 2024; incluindo a faixa *média* (P ≥ 0,25), o recall sobe para ≈ 84% com precisão ≈ 47%. Escolher o limiar é decisão de gestão:
  concentrar apoio em quem tem risco claro ou acompanhar mais UFs.
* **Priorizar por trajetória**, não por região: como a região explica pouco, programas regionais genéricos tendem a acertar mal.
* **Investigar os choques** (RS, PR-estadual, PI-estadual): variações de ±17–26 p.p. precisam ser checadas quanto à amostragem do Saeb antes de virarem decisão.
* **Acoplar às metas oficiais:** trocar o limiar de 50% pela meta de cada UF/município transforma a saída em "probabilidade de não cumprir a meta".

## 11. Limitações
* **Granularidade agregada e amostra pequena** (74 linhas no modelo temporal, 25 UFs, 2 ciclos): resultados instáveis e intervalos largos.
* **Não é previsão de alunos nem de municípios.**
* **Sem fontes externas** (socioeconomia, infraestrutura, financiamento) — onde estaria o maior ganho potencial.
* **O modelo não antecipa choques** (RS recebeu P(risco) ≈ 0,21 nas previsões fora da amostra); TO foi sinalizado como risco e ficou ≥ 50%.
* O limiar de 50% é um **proxy**; RR e DF não constam da base; os rótulos de `rede` são inferidos.
* Variações entre ciclos podem refletir **amostragem do Saeb** (hipótese não testada aqui).
* A previsão para a "próxima edição" aplica ao ciclo 2024→seguinte um modelo ajustado só em 2023→2024.
* SHAP não executado nesta entrega.

## 12. Possíveis evoluções
1. Incorporar **microdados do Saeb** (nível do aluno) e **metas municipais**; reexecutar com o mesmo pipeline.
2. Enriquecer com IBGE/PNAD, Censo Escolar, FUNDEB, Atlas do Desenvolvimento Humano e Cadastro Único (ver `data/external/README.md`).
3. Mais ciclos históricos para validação temporal real (treinar em t, testar em t+1) e intervalos de confiança.
4. Calibração de probabilidades e modelos hierárquicos (UF/rede) com incerteza explícita.
5. Rodar SHAP e análise de equidade; acompanhamento contínuo (monitoramento de *drift*).
