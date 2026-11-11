# CÓDIGOS 3A, 3B e 4

import pandas as pd
from scipy.stats.mstats import winsorize

# ----------------------------------------------------------------------------
# CÓDIGO 3A: EXPERIMENTO PRELIMINAR - TARGET V1 (POST-SEMESTER GPA)
# ----------------------------------------------------------------------------
df_v1 = df_raw.copy()

# 1. Variável-Alvo V1 em Mediana/2 Níveis (Baixo, Alto Desempenho)
df_v1['desempenho_categoria'] = pd.qcut(
    df_v1['Post_Semester_GPA'],
    q=2,
    labels=['Baixo Desempenho', 'Alto Desempenho'],
)

# 2. Tratamento de Outliers via Winsorização (1% nas extremidades)
cont_cols = [
    'Weekly_GenAI_Hours',
    'Traditional_Study_Hours',
    'Pre_Semester_GPA',
    'Post_Semester_GPA',
]
for col in cont_cols:
    if col in df_v1.columns:
        df_v1[col] = winsorize(df_v1[col], limits=[0.01, 0.01])

# 3. Encodings Ordinais
ordinal_map = {
    'Year_of_Study': {
        'Freshman': 1,
        'Sophomore': 2,
        'Junior': 3,
        'Senior': 4,
        'Graduate': 5,
    },
    'Prompt_Engineering_Skill': {
        'Beginner': 1,
        'Intermediate': 2,
        'Advanced': 3,
    },
    'Burnout_Risk_Level': {'Low': 1, 'Medium': 2, 'High': 3},
}
for col, mapper in ordinal_map.items():
    if col in df_v1.columns and df_v1[col].dtype == 'object':
        df_v1[col] = df_v1[col].map(mapper)

if 'Paid_Subscription' in df_v1.columns:
    df_v1['Paid_Subscription'] = df_v1['Paid_Subscription'].astype(int)

# 4. One-Hot Encoding para Categóricas Nominais
df_v1_encoded = pd.get_dummies(
    df_v1,
    columns=[
        c
        for c in ['Major_Category', 'Primary_Use_Case', 'Institutional_Policy']
        if c in df_v1.columns
    ],
    drop_first=True,
)

# 5. Definição de Atributos e Alvo
ignore_cols_v1 = ['Student_ID', 'Post_Semester_GPA', 'desempenho_categoria']
X_features_v1 = [c for c in df_v1_encoded.columns if c not in ignore_cols_v1]

X_v1 = df_v1_encoded[X_features_v1]
y_v1 = df_v1_encoded['desempenho_categoria']

# 6. Diagnóstico do Vício de Inércia Acadêmica
corr_pre_post = df_v1['Pre_Semester_GPA'].corr(df_v1['Post_Semester_GPA'])

print("=== [ABORDAGEM V1: PREVISÃO DIRETA DO POST_SEMESTER_GPA (BINÁRIO)] ===")
print(f"Total de Atributos Preditivos (Iniciais): {X_v1.shape[1]}")
print("\nDistribuição Absoluta da Variável-Alvo V1 (N):")
print(y_v1.value_counts())
print("\nDistribuição Percentual da Variável-Alvo V1 (%):")
print((y_v1.value_counts(normalize=True) * 100).round(2))
print(
    f"\nCorrelação entre Pre_Semester_GPA e Post_Semester_GPA: {corr_pre_post:.4f}"
)

# ----------------------------------------------------------------------------
# CÓDIGO 3B: MODELAGEM DEFINITIVA DO TCC - TARGET V2 (ΔGPA BINÁRIO)
# ----------------------------------------------------------------------------
df = df_raw.copy()

# 1. Variável-Alvo Oficial do TCC: Ganho Líquido de Desempenho (ΔGPA Binário)
df['delta_gpa'] = df['Post_Semester_GPA'] - df['Pre_Semester_GPA']

df['target'] = pd.qcut(df['delta_gpa'], q=2, labels=['Baixo Ganho', 'Alto Ganho'])

# 2. Tratamento de Outliers via Winsorização (1% nas extremidades)
cont_cols = [
    'Weekly_GenAI_Hours',
    'Traditional_Study_Hours',
    'Pre_Semester_GPA',
    'Post_Semester_GPA',
    'delta_gpa',
]
for col in cont_cols:
    if col in df.columns:
        df[col] = winsorize(df[col], limits=[0.01, 0.01])

# 3. Encodings Ordinais (mesmo mapa do 3A)
for col, mapper in ordinal_map.items():
    if col in df.columns and df[col].dtype == 'object':
        df[col] = df[col].map(mapper)

if 'Paid_Subscription' in df.columns:
    df['Paid_Subscription'] = df['Paid_Subscription'].astype(int)

# 4. One-Hot Encoding para Categóricas Nominais
df_encoded = pd.get_dummies(
    df,
    columns=[
        c
        for c in ['Major_Category', 'Primary_Use_Case', 'Institutional_Policy']
        if c in df.columns
    ],
    drop_first=True,
)

# 5. Definição de Atributos e Alvo Oficial
ignore_cols = ['Student_ID', 'Post_Semester_GPA', 'delta_gpa', 'target']
X_features = [c for c in df_encoded.columns if c not in ignore_cols]

X = df_encoded[X_features]
y = df['target']

print("=== [ABORDAGEM V2: GANHO LÍQUIDO ACADÊMICO (ΔGPA BINÁRIO) - OFICIAL DO TCC] ===")
print(f"Total de Atributos Preditivos (Iniciais): {X.shape[1]}")
print("\nDistribuição Absoluta da Variável-Alvo Oficial (N):")
print(y.value_counts())
print("\nDistribuição Percentual da Variável-Alvo Oficial (%):")
print((y.value_counts(normalize=True) * 100).round(2))
print(f"\nVariável ΔGPA isolada em 2 grupos (50/50). Média de ganho: {df['delta_gpa'].mean():.4f}")

# ----------------------------------------------------------------------------
# CÓDIGO 4: ESTATÍSTICAS DESCRITIVAS DAS VARIÁVEIS CHAVE (TABELA 4.1 do notebook = Tabela 1 do TCC)
# ----------------------------------------------------------------------------
vars_chave = [
    'Pre_Semester_GPA',
    'Post_Semester_GPA',
    'delta_gpa',
    'Traditional_Study_Hours',
    'Weekly_GenAI_Hours',
    'Anxiety_Level_During_Exams',
]

desc_table = df[vars_chave].describe().T[['mean', 'std', 'min', '50%', 'max']]
desc_table.columns = ['Média', 'Desvio Padrão', 'Mínimo', 'Mediana', 'Máximo']

print("=== TABELA 4.1: ESTATÍSTICAS DESCRITIVAS DAS VARIÁVEIS CHAVE ===")
print(desc_table.round(3))
