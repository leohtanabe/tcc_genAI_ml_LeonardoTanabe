# tcc-genai-ml

Código do TCC **"Aplicação de Machine Learning para avaliação do impacto da IA Generativa e das práticas de estudo no desempenho acadêmico: uma abordagem sob a perspectiva da eficiência operacional"**.

- Autor: Leonardo Hideki Tanabe
- Curso: Engenharia de Produção, UNESP Guaratinguetá
- Orientador: Prof. Dr. Blaha Gregory Correia dos Santos Goussain

Estudo de caso metodológico (prova de conceito) de um pipeline preditivo e interpretável sobre uma base secundária do Kaggle (Nabi, 2024). O alvo é o ganho líquido de desempenho (ΔGPA = Post − Pre), binarizado pela mediana (Baixo Ganho / Alto Ganho).

## Dados

A base não é versionada aqui. Veja [`data/README.md`](data/README.md).

## Estrutura

Os arquivos em `cells/` são as células do notebook do Colab, na ordem de execução. Rode-as em uma única sessão, pois cada uma usa variáveis criadas pelas anteriores.

| Código | Arquivo | Conteúdo |
|---|---|---|
| 1 e 2 | `cells/01_02_ambiente_e_dados.py` | Imports e carga dos dados |
| 3A, 3B e 4 | `cells/03_04_engenharia_e_alvo.py` | Alvo V1 (preliminar), alvo V2 (ΔGPA), winsorização, encodings, estatísticas descritivas |
| 5 a 8 | `cells/05_08_experimento_preliminar_v1.py` | Experimento preliminar (alvo = GPA pós-semestre) que mostra a inércia acadêmica |
| 9 | `cells/09_modelo_v2_lightgbm.py` | Modelo oficial V2 (LightGBM), partição 80/20, IPD, matriz de confusão |
| 10 | `cells/10_kmeans_personas.py` | K-Means, versão inicial (substituída pelos Códigos 14 a 16) |
| 11 a 13 | `cells/11_13_diagnosticos_visuais.py` | Boxplots, importância por impureza, índices compostos (EIA, REH, IVC) |
| 14 a 16 | `cells/14_16_kmeans_validacao_e_personas.py` | Cotovelo e silhueta, personas por centroides, teste χ² e V de Cramér |
| 17 e 18 | `cells/17_18_benchmark_cv_e_shap_v2.py` | Benchmark de quatro modelos, validação cruzada 5-fold, SHAP |
| 19 | `cells/19_ranking_shap_exato.py` | Ranking exato de E[\|SHAP\|] |
| 20 a 22 | `cells/20_22_figuras_complementares.py` | Figuras 4.1, 4.2 e 4.3 do notebook |
| 23 | `cells/23_radar_personas.py` | Figura 14: radar dos centroides padronizados, calculado dos dados |
| 24 | `cells/24_exportacao_resultados.py` | Compacta PNGs e CSVs gerados (executar por último) |
| legado | `legacy/` | Execuções antigas, inválidas ou superadas; não usar |

## Como executar

```bash
pip install -r requirements.txt
```

Versões de referência: Python 3.13 (Google Colab), conforme `requirements.txt`.

1. Baixe o CSV do Kaggle e salve em `data/ai_student_impact_dataset.csv`.
2. No Colab (ou Jupyter), cole as células de `cells/` na ordem dos códigos, uma por célula, e execute em sequência.
3. Parâmetros fixos: `random_state=42` em todos os splits, modelos e no K-Means; partição 80/20 estratificada; hiperparâmetros fixados manualmente, sem busca.

## Pontos de atenção

- A winsorização (1% por cauda) e o ΔGPA são calculados na base inteira antes da partição treino/teste. O IPD do modelo oficial usa média e desvio do treino.
- Precisão, Revocação e F1 são médias ponderadas pelo suporte das classes, por isso a Revocação ponderada é igual à acurácia.
- O Código 10 está superado: use os Códigos 14 a 16 para as personas.
- O radar do Código 23 é calculado dos dados. A versão antiga, com valores digitados à mão, está em `legacy/`.
- As tabelas de personas do texto do TCC devem ser geradas a partir dos Códigos 15, 16 e 23 numa única execução.
