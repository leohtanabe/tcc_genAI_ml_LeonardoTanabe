# CÓDIGOS 11, 12 e 13: DIAGNÓSTICOS VISUAIS DO V2

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.ensemble import GradientBoostingClassifier

# ---------------------------------------------------------------------------
# CÓDIGO 11: COMPARATIVO DE HÁBITOS DE ESTUDO POR FAIXA BINÁRIA DE ΔGPA
# ---------------------------------------------------------------------------
if 'desempenho_categoria' not in df.columns:
    df['desempenho_categoria'] = df['target']

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

sns.boxplot(data=df, x='desempenho_categoria', y='Traditional_Study_Hours',
            ax=axes[0], palette='Blues')
axes[0].set_title('Estudo Tradicional por Faixa Binária de Ganho', fontweight='bold', pad=12)
axes[0].set_xlabel('Categoria de Evolução (ΔGPA)')
axes[0].set_ylabel('Horas Semanais Tradicionais')

sns.boxplot(data=df, x='desempenho_categoria', y='Weekly_GenAI_Hours',
            ax=axes[1], palette='Oranges')
axes[1].set_title('Uso de IA Generativa por Faixa Binária de Ganho', fontweight='bold', pad=12)
axes[1].set_xlabel('Categoria de Evolução (ΔGPA)')
axes[1].set_ylabel('Horas Semanais de GenAI')

plt.tight_layout()
plt.savefig('comparativo_habitos_estudo.png', dpi=300)
plt.show()

# ---------------------------------------------------------------------------
# CÓDIGO 12: ANÁLISE AVANÇADA DA EVOLUÇÃO ACADÊMICA
# ---------------------------------------------------------------------------
candidatas = [
    'Traditional_Study_Hours', 'Weekly_GenAI_Hours', 'Anxiety_Level_During_Exams',
    'Prompt_Skill_Score', 'Pre_Semester_GPA', 'ai_efficiency_index',
    'hybrid_study_ratio', 'cognitive_vulnerability_index',
]

if 'features' in locals():
    features_validas = [c for c in features if c in df.columns]
else:
    features_validas = []
if not features_validas:
    features_validas = [c for c in candidatas if c in df.columns]

if 'target' not in df.columns:
    df['target'] = (df['delta_gpa'] > df['delta_gpa'].median()).astype(int)

X = df[features_validas]
y = df['target']
gbc = GradientBoostingClassifier(random_state=42)
gbc.fit(X, y)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

importancias = (
    pd.Series(gbc.feature_importances_, index=features_validas)
    .sort_values(ascending=False)
    .head(8)
)
sns.barplot(x=importancias.values, y=importancias.index, ax=axes[0], palette='viridis')
axes[0].set_title('Top 8 Fatores para Ganho de Nota (GBC Binário)', fontweight='bold', pad=12)
axes[0].set_xlabel('Peso de Importância no Modelo')
axes[0].set_ylabel('')

colunas_corr = ['delta_gpa', 'Weekly_GenAI_Hours', 'Traditional_Study_Hours',
                'Anxiety_Level_During_Exams']
colunas_corr_validas = [c for c in colunas_corr if c in df.columns]
matriz_corr = df[colunas_corr_validas].corr()
sns.heatmap(matriz_corr, annot=True, cmap='coolwarm', fmt='.2f', ax=axes[1], cbar=False)
axes[1].set_title('Correlação c/ Evolução Acadêmica (ΔGPA)', fontweight='bold', pad=12)

prompt_col = (
    'Prompt_Engineering_Skill' if 'Prompt_Engineering_Skill' in df.columns
    else ('Prompt_Skill_Score' if 'Prompt_Skill_Score' in df.columns else None)
)
if prompt_col:
    sns.barplot(data=df, x=prompt_col, y='delta_gpa', ax=axes[2], palette='magma')
    axes[2].set_title('Evolução de Nota por Nível de Prompt', fontweight='bold', pad=12)
    axes[2].set_xlabel('Habilidade de Prompt')
    axes[2].set_ylabel('Ganho Médio (ΔGPA)')

plt.tight_layout()
plt.savefig('analise_avancada_delta_gpa.png', dpi=300, bbox_inches='tight')
plt.show()


# ---------------------------------------------------------------------------
# CÓDIGO 13: ENGENHARIA DE VARIÁVEIS COMPOSTAS E DIAGNÓSTICO DE INTERAÇÕES
# ---------------------------------------------------------------------------
df['ai_efficiency_index'] = df['Prompt_Engineering_Skill'] / (df['Weekly_GenAI_Hours'] + 1)
df['hybrid_study_ratio'] = df['Traditional_Study_Hours'] / (df['Weekly_GenAI_Hours'] + 1)
df['cognitive_vulnerability_index'] = df['Perceived_AI_Dependency'] * df['Anxiety_Level_During_Exams']

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

sns.boxplot(data=df, x='desempenho_categoria', y='ai_efficiency_index',
            ax=axes[0], palette='Greens')
axes[0].set_title('Eficiência de IA por Faixa de Ganho', fontweight='bold', pad=12)
axes[0].set_ylabel('Eficiência (Prompt Skill / Horas IA)')
axes[0].set_xlabel('Categoria de Evolução (ΔGPA)')

sns.boxplot(data=df, x='desempenho_categoria', y='hybrid_study_ratio',
            ax=axes[1], palette='Purples')
axes[1].set_title('Rácio de Estudo Híbrido por Faixa de Ganho', fontweight='bold', pad=12)
axes[1].set_ylabel('Proporção (Estudo Tradicional / Horas IA)')
axes[1].set_xlabel('Categoria de Evolução (ΔGPA)')

vars_correlacao = ['delta_gpa', 'ai_efficiency_index', 'hybrid_study_ratio',
                   'cognitive_vulnerability_index', 'Skill_Retention_Score']
matriz_corr_comp = df[vars_correlacao].corr()
sns.heatmap(matriz_corr_comp, annot=True, fmt='.2f', cmap='coolwarm', ax=axes[2], cbar=False)
axes[2].set_title('Correlação de Variáveis Compostas', fontweight='bold', pad=12)
axes[2].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('variaveis_compostas_correlacao.png', dpi=300)
plt.show()
