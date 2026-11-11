# CÓDIGO 10: K-MEANS (k=4) E PERSONAS (versão inicial)
# ATENÇÃO: os nomes das personas aqui são atribuídos pelo número do cluster (persona_map fixo)
# e ficam ERRADOS para A e B. Esta célula foi SUBSTITUÍDA pelos Códigos 14 a 16, que rotulam
# as personas por centroides. Mantida só por fidelidade ao notebook original.
# Pré-requisito: df (cells/03_04_engenharia_e_alvo.py).

from scipy.stats import chi2_contingency
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import pandas as pd

# 1. Variáveis comportamentais para o Agrupamento Não Supervisionado
cols_cluster = [
    'Weekly_GenAI_Hours',
    'Traditional_Study_Hours',
    'Perceived_AI_Dependency',
    'Skill_Retention_Score',
    'Anxiety_Level_During_Exams',
    'Prompt_Engineering_Skill',
]

# 2. Normalização dos dados para o K-Means
scaler_km = StandardScaler()
X_cluster_scaled = scaler_km.fit_transform(df[cols_cluster])

# 3. Treinamento do K-Means (k=4 clusters conforme estrutura das Personas)
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
df['cluster_kmeans'] = kmeans.fit_predict(X_cluster_scaled)

# 4. Mapeamento das Personas (A, B, C, D)
persona_map = {
    0: 'Persona A: Mestre em Prompts',
    1: 'Persona B: Dependente Hiperconectado',
    2: 'Persona C: Tradicionalista Dedicado',
    3: 'Persona D: Desengajado em Risco',
}

df['persona'] = df['cluster_kmeans'].map(persona_map)

# --- TABELA DE ASSOCIAÇÃO INFERENCIAL (PERSONAS x FAIXAS BINÁRIAS DE ΔGPA) ---
tab_cruzada = pd.crosstab(
    df['persona'], df['target'], margins=True, margins_name='Total Amostral'
)

tab_cruzada.columns.name = 'Faixa Binária de ΔGPA'
tab_crosstab_exibicao = tab_cruzada.rename(
    columns={'Total Amostral': 'Total por Persona (N)'}
)

print("=== ESTATÍSTICAS DE ASSOCIAÇÃO INFERENCIAL ===")
print(tab_crosstab_exibicao)  # no Colab: display(...)

# Teste Qui-Quadrado de Independência (χ²)
chi2, p, dof, ex = chi2_contingency(pd.crosstab(df['persona'], df['target']))
print(f"\nTeste Qui-Quadrado (χ²): {chi2:.3f}")
print(f"p-valor: {p:.4e}")
