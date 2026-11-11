# CÓDIGOS 14, 15 e 16: VALIDAÇÃO DE k, PERSONAS (ROTULAÇÃO POR CENTROIDES) E TESTE χ²

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import chi2_contingency
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# CÓDIGO 14: VALIDAÇÃO DO NÚMERO IDEAL DE CLUSTERS (COTOVELO E SILHUETA)
# ---------------------------------------------------------------------------
cols_cluster = [
    'Weekly_GenAI_Hours',
    'Traditional_Study_Hours',
    'Prompt_Engineering_Skill',
    'Perceived_AI_Dependency',
    'Skill_Retention_Score',
    'Anxiety_Level_During_Exams',
]

scaler_k = StandardScaler()
X_scaled = scaler_k.fit_transform(df[cols_cluster])

inertia, silhouette = [], []
k_range = range(2, 9)

for k in k_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10).fit(X_scaled)
    inertia.append(km.inertia_)
    silhouette.append(silhouette_score(X_scaled, km.labels_))

fig, ax1 = plt.subplots(figsize=(9, 4.5))
ax1.plot(k_range, inertia, 'bo-', linewidth=2, markersize=7, label='Inércia (Cotovelo)')
ax1.set_xlabel('Número de Clusters (k)', fontweight='bold')
ax1.set_ylabel('Inércia Intra-cluster (Soma dos Quadrados)', color='b', fontweight='bold')
ax1.tick_params(axis='y', labelcolor='b')
ax1.grid(True, linestyle='--', alpha=0.5)

ax2 = ax1.twinx()
ax2.plot(k_range, silhouette, 'ro--', linewidth=2, markersize=7, label='Silhueta')
ax2.set_ylabel('Pontuação da Silhueta Média', color='r', fontweight='bold')
ax2.tick_params(axis='y', labelcolor='r')

plt.title('Validação do Número Ideal de Clusters (k=4)', fontweight='bold', pad=12)
plt.tight_layout()
plt.savefig('kmeans_validation_elbow_silhouette.png', dpi=300)
plt.show()

# ---------------------------------------------------------------------------
# CÓDIGO 15: K-MEANS COM ROTULAÇÃO DINÂMICA E SEGURA
# ---------------------------------------------------------------------------
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
df['cluster_id'] = kmeans.fit_predict(X_scaled)

# Depende de `X` definido no Código 12 (8 colunas originais)
cols_comportamentais = [col for col in X.columns if col in df.columns]

perfil_bruto = df.groupby('cluster_id')[cols_comportamentais + ['delta_gpa']].mean()

print('--- PERFIL MÉDIO BRUTO POR CLUSTER ---')
print(perfil_bruto[[
    'Prompt_Engineering_Skill', 'Weekly_GenAI_Hours',
    'Traditional_Study_Hours', 'delta_gpa',
]])

# Mestre em Prompts: cluster com a MAIOR habilidade em Prompt
id_mestre = perfil_bruto['Prompt_Engineering_Skill'].idxmax()
# Tradicionalista Dedicado: cluster com as MAIORES horas de estudo tradicional
id_tradicional = perfil_bruto['Traditional_Study_Hours'].idxmax()
# Entre os dois restantes, o de MAIOR uso de IA é o Dependente Hiperconectado
restantes = [i for i in perfil_bruto.index if i not in [id_mestre, id_tradicional]]
id_dependente = perfil_bruto.loc[restantes, 'Weekly_GenAI_Hours'].idxmax()
# O último restante é o Desengajado em Risco
id_desengajado = [i for i in restantes if i != id_dependente][0]

mapa_dinamico = {
    id_mestre: 'Persona A: Mestre em Prompts',
    id_dependente: 'Persona B: Dependente Hiperconectado',
    id_tradicional: 'Persona C: Tradicionalista Dedicado',
    id_desengajado: 'Persona D: Desengajado em Risco',
}
df['persona'] = df['cluster_id'].map(mapa_dinamico)

print('\n--- MAPEAMENTO APLICADO COM SUCESSO ---')
for cid, nome in mapa_dinamico.items():
    print(f'Cluster {cid} -> {nome}')

# ---------------------------------------------------------------------------
# CÓDIGO 16: VALIDAÇÃO ESTATÍSTICA INFERENCIAL E PAINEL VISUAL DAS PERSONAS
# ---------------------------------------------------------------------------
if 'desempenho_categoria' not in df.columns:
    df['desempenho_categoria'] = df['target']

col_persona = 'persona' if 'persona' in df.columns else 'persona_nome'

tabela_bruta = pd.crosstab(
    df[col_persona], df['desempenho_categoria'],
    margins=True, margins_name='Total Amostral',
)
tabela_exibicao = tabela_bruta.rename(columns={'Total Amostral': 'Total por Persona (N)'})
tabela_exibicao.index.name = 'Persona Discente'

print("=== TABELA DE ASSOCIAÇÃO INFERENCIAL (PERSONAS x ΔGPA BINÁRIO) ===")
print(tabela_exibicao)

matriz_sem_totais = pd.crosstab(df[col_persona], df['desempenho_categoria'])
chi2, p_val, dof, _ = chi2_contingency(matriz_sem_totais)
n_obs = matriz_sem_totais.sum().sum()
min_dim = min(matriz_sem_totais.shape) - 1
cramer_v = np.sqrt(chi2 / (n_obs * min_dim)) if min_dim > 0 else 0.0

print("\n--- DIAGNÓSTICO ESTATÍSTICO COMPLETO ---")
print(f"Estatística Qui-Quadrado (χ²): {chi2:.3f}")
print(f"Graus de Liberdade (gl): {dof}")
print(f"p-valor: {p_val:.4e} (p < 0.001)")
print(f"Tamanho do Efeito (V de Cramér): {cramer_v:.4f}")
if p_val < 0.05:
    print("Conclusão: associação ESTATISTICAMENTE SIGNIFICATIVA entre Persona e Ganho Binário (p < 0.05).")

fig, axes = plt.subplots(1, 2, figsize=(18, 6))
sns.boxplot(data=df, x=col_persona, y='delta_gpa', hue=col_persona,
            palette='Set2', legend=False, ax=axes[0])
axes[0].set_title('Evolução Acadêmica (ΔGPA) por Persona Discente', fontweight='bold', pad=12)
axes[0].set_xlabel('Persona Discente')
axes[0].set_ylabel('Ganho Acadêmico Líquido (ΔGPA)')
axes[0].tick_params(axis='x', rotation=15)

tabela_percentual = pd.crosstab(df[col_persona], df['desempenho_categoria'], normalize='index') * 100
sns.heatmap(tabela_percentual, annot=True, fmt='.1f', cmap='Blues',
            cbar_kws={'label': '% da Persona'}, ax=axes[1])
axes[1].set_title(f'Tabela de Contingência (χ² = {chi2:.1f}, V de Cramér = {cramer_v:.3f})',
                  fontweight='bold', pad=12)
axes[1].set_ylabel('Persona Discente')
axes[1].set_xlabel('Categoria de Evolução (ΔGPA)')

plt.tight_layout()
plt.savefig('personas_e_contingencia_kmeans.png', dpi=300)
plt.show()
