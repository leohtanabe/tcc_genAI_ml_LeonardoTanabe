# CÓDIGOS 20, 21 e 22: FIGURAS 4.1, 4.2 e 4.3 do notebook

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# ---------------------------------------------------------------------------
# CÓDIGO 20: FIGURA 4.1 (DISTRIBUIÇÃO DO ΔGPA E EQUILÍBRIO ENTRE CLASSES)
# ---------------------------------------------------------------------------
plt.figure(figsize=(8, 4))
sns.histplot(data=df, x='delta_gpa', hue='target', kde=True, palette='Set2', bins=30)
plt.title('Distribuição do ΔGPA e Equilíbrio entre Classes de Ganho',
          fontsize=11, fontweight='bold', pad=12)
plt.xlabel('Variação do Desempenho Acadêmico (ΔGPA)')
plt.ylabel('Frequência de Estudantes')
plt.tight_layout()
plt.savefig('distribuicao_target.png', dpi=300, bbox_inches='tight')
plt.show()

# ---------------------------------------------------------------------------
# CÓDIGO 21: FIGURA 4.2 (BOXPLOTS PAINEL DUPLO: EIA E REH)
# ---------------------------------------------------------------------------
prompt_col = (
    'Prompt_Engineering_Skill' if 'Prompt_Engineering_Skill' in df.columns
    else 'Prompt_Skill_Score'
)
dep_col = (
    'Perceived_AI_Dependency' if 'Perceived_AI_Dependency' in df.columns
    else 'AI_Dependency'
)

df['ai_efficiency_index'] = df[prompt_col] / (df['Weekly_GenAI_Hours'] + 1)
df['hybrid_study_ratio'] = df['Traditional_Study_Hours'] / (df['Weekly_GenAI_Hours'] + 1)

if dep_col in df.columns and 'Anxiety_Level_During_Exams' in df.columns:
    df['cognitive_vulnerability_index'] = df[dep_col] * df['Anxiety_Level_During_Exams']

cat_col = (
    'desempenho_categoria' if 'desempenho_categoria' in df.columns
    else ('target' if 'target' in df.columns else None)
)
if not cat_col:
    df['desempenho_categoria'] = (
        df['delta_gpa'] > df['delta_gpa'].median()
    ).map({True: 'Alto Ganho', False: 'Baixo Ganho'})
    cat_col = 'desempenho_categoria'

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
sns.boxplot(data=df, x=cat_col, y='ai_efficiency_index', ax=axes[0], palette='Greens')
axes[0].set_title('Eficiência de IA por Faixa de Ganho', fontweight='bold', pad=12)
axes[0].set_ylabel('Eficiência (Prompt Skill / Horas IA)')
axes[0].set_xlabel('Categoria de Evolução (ΔGPA)')

sns.boxplot(data=df, x=cat_col, y='hybrid_study_ratio', ax=axes[1], palette='Purples')
axes[1].set_title('Rácio de Estudo Híbrido por Faixa de Ganho', fontweight='bold', pad=12)
axes[1].set_ylabel('Proporção (Estudo Tradicional / Horas IA)')
axes[1].set_xlabel('Categoria de Evolução (ΔGPA)')

plt.tight_layout()
plt.savefig('figura_4_2_boxplots_compostos.png', dpi=300, bbox_inches='tight')
plt.show()

# ---------------------------------------------------------------------------
# CÓDIGO 22: FIGURA 4.3 (MATRIZ DE CORRELAÇÃO DE VARIÁVEIS COMPOSTAS)
# ---------------------------------------------------------------------------
plt.figure(figsize=(7, 5))
vars_correlacao = ['delta_gpa', 'ai_efficiency_index', 'hybrid_study_ratio',
                   'cognitive_vulnerability_index', 'Skill_Retention_Score']
vars_validas = [c for c in vars_correlacao if c in df.columns]
matriz_corr_comp = df[vars_validas].corr()

sns.heatmap(matriz_corr_comp, annot=True, fmt='.2f', cmap='coolwarm',
            cbar=True, vmin=-0.5, vmax=1.0)
plt.title('Correlação de Variáveis Compostas', fontweight='bold', pad=12)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('figura_4_3_matriz_correlacao.png', dpi=300, bbox_inches='tight')
plt.show()
