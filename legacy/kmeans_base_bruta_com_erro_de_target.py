# LEGADO (NÃO USAR NO PIPELINE FINAL): "CÓDIGO FINAL DEFINITIVO: K-MEANS + DETECÇÃO AUTOMÁTICA DE TARGET"
# Mantido apenas como registro da execução que gerou as contagens 14.955 / 8.218 / 12.968 / 13.859,
# iguais às das Tabelas 2 e 3 do TCC.
# Por que NÃO usar:
#  1. Lê o CSV direto (sem winsorização e sem os encodings do Código 3B): resultado diferente do pipeline final.
#  2. persona_map fixo pelo número do cluster (sem rotulação por centroides).
#  3. A detecção automática do alvo escolheu 'Pre_Semester_GPA' (primeira coluna com "gpa" no nome),
#     gerando target_binario = (Pre_Semester_GPA > 0) = 1 para todos: tabela com uma só coluna,
#     χ² = 0,000, gl = 0, p = 1,0. A saída desta célula é INVÁLIDA.
# Conclusão de auditoria: as Tabelas 2 e 3 do TCC parecem vir de K-Means sobre a base BRUTA;
# o texto (χ² = 2.615,21; V = 0,2287) vem do pipeline final (base winsorizada). Regerar as tabelas
# a partir dos Códigos 15 e 16 (e 23b).

import os

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

arquivos_csv = [f for f in os.listdir('.') if f.endswith('.csv')]
if arquivos_csv:
    nome_arquivo = arquivos_csv[0]
    df = pd.read_csv(nome_arquivo)
else:
    raise FileNotFoundError('Nenhum arquivo CSV encontrado na pasta do Colab.')

cols_cluster = [
    'Weekly_GenAI_Hours', 'Traditional_Study_Hours', 'Perceived_AI_Dependency',
    'Skill_Retention_Score', 'Anxiety_Level_During_Exams', 'Prompt_Engineering_Skill',
]

mapeamento_ordinal = {
    'Beginner': 1, 'Low': 1, 'Very Low': 1,
    'Intermediate': 2, 'Medium': 2, 'Moderate': 2,
    'Advanced': 3, 'High': 3, 'Very High': 4, 'Expert': 4,
}

for col in cols_cluster:
    if col in df.columns:
        if df[col].dtype == 'object' or isinstance(df[col].iloc[0], str):
            df[col] = df[col].map(mapeamento_ordinal).fillna(df[col])
            if df[col].dtype == 'object':
                df[col] = pd.factorize(df[col])[0] + 1

X_cluster = df[cols_cluster].astype(float)
scaler_km = StandardScaler()
X_cluster_scaled = scaler_km.fit_transform(X_cluster)

kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
df['cluster_kmeans'] = kmeans.fit_predict(X_cluster_scaled)

persona_map = {
    0: 'Persona A: Mestre em Prompts',
    1: 'Persona B: Dependente Hiperconectado',
    2: 'Persona C: Tradicionalista Dedicado',
    3: 'Persona D: Desengajado em Risco',
}
df['persona'] = df['cluster_kmeans'].map(persona_map)

# Detecção automática do alvo (ERRO: seleciona Pre_Semester_GPA)
coluna_target = None
possiveis_nomes_target = ['target', 'Target', 'Delta_GPA_Binary', 'GPA_Gain_Binary',
                          'GPA_Change_Binary', 'Delta_GPA', 'GPA_Change']
for nome in possiveis_nomes_target:
    if nome in df.columns:
        coluna_target = nome
        break

if coluna_target is None:
    colunas_gpa = [c for c in df.columns
                   if 'gpa' in c.lower() or 'target' in c.lower() or 'gain' in c.lower()]
    if colunas_gpa:
        coluna_target = colunas_gpa[0]  # <- 'Pre_Semester_GPA'
        print(f"--> Coluna alvo identificada automaticamente: '{coluna_target}'")
    else:
        raise KeyError("Não foi possível encontrar a coluna 'target' ou 'GPA'.")

if df[coluna_target].nunique() > 2:
    df['target_binario'] = (df[coluna_target] > 0).astype(int)  # todos = 1 para Pre_Semester_GPA
    coluna_target_final = 'target_binario'
else:
    coluna_target_final = coluna_target

tab_cruzada = pd.crosstab(df['persona'], df[coluna_target_final],
                          margins=True, margins_name='Total Amostral')
print(tab_cruzada)

chi2, p_value, dof, expected = chi2_contingency(pd.crosstab(df['persona'], df[coluna_target_final]))
print(f'Estatística Qui-Quadrado (χ²): {chi2:.3f}')
print(f'Graus de Liberdade (dof):      {dof}')
print(f'p-valor:                      {p_value:.4e}')
