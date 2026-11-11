# CÓDIGO 9: MODELO OFICIAL V2, LightGBM, alvo ΔGPA binário

import lightgbm as lgb
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

# 1. Recuperação da base já tratada no CÓDIGO 3B
df_v2 = df_encoded.copy()
df_v2['desempenho_categoria'] = df['target']

# 2. Partição 80/20 e Cálculo do Índice de Procrastinação Digital Pós-Split
train_df, test_df = train_test_split(
    df_v2,
    test_size=0.20,
    random_state=42,
    stratify=df_v2['desempenho_categoria'],
)
train_df = train_df.copy()
test_df = test_df.copy()

dep_m, dep_s = (
    train_df['Perceived_AI_Dependency'].mean(),
    train_df['Perceived_AI_Dependency'].std(),
)
trad_m, trad_s = (
    train_df['Traditional_Study_Hours'].mean(),
    train_df['Traditional_Study_Hours'].std(),
)

train_df['digital_procrastination_index'] = (
    (train_df['Perceived_AI_Dependency'] - dep_m) / dep_s
) - ((train_df['Traditional_Study_Hours'] - trad_m) / trad_s)
test_df['digital_procrastination_index'] = (
    (test_df['Perceived_AI_Dependency'] - dep_m) / dep_s
) - ((test_df['Traditional_Study_Hours'] - trad_m) / trad_s)

# 3. Isolamento de Atributos (Elimina Pre_GPA e Post_GPA para evitar Vício de Inércia)
ignore = [
    'Student_ID',
    'Pre_Semester_GPA',
    'Post_Semester_GPA',
    'delta_gpa',
    'desempenho_categoria',
    'target',
]
features = [c for c in train_df.columns if c not in ignore]

X_tr, y_tr = train_df[features], train_df['desempenho_categoria']
X_te, y_te = test_df[features], test_df['desempenho_categoria']

# 4. Treinamento do Modelo Campeão (LightGBM Binário)
lgbm = lgb.LGBMClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    random_state=42,
    verbose=-1,
)
lgbm.fit(X_tr, y_tr)
y_pred = lgbm.predict(X_te)

# 5. Visualização da Matriz de Confusão V2 (2x2)
labels_binarios = ['Baixo Ganho', 'Alto Ganho']
cm = confusion_matrix(y_te, y_pred, labels=labels_binarios)

plt.figure(figsize=(6, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=labels_binarios,
    yticklabels=labels_binarios,
)
plt.title(
    'Matriz de Confusão V2 - LightGBM (ΔGPA Binário)',
    fontsize=11,
    fontweight='bold',
    pad=12,
)
plt.ylabel('Classe Real (Observada)')
plt.xlabel('Classe Preditiva (Modelo)')
plt.tight_layout()
plt.savefig('matriz_confusao_delta_gpa.png', dpi=300)
plt.show()

# 6. Relatório de Classificação V2
print("=== RELATÓRIO DE CLASSIFICAÇÃO COMPLETO - LIGHTGBM (ΔGPA BINÁRIO) ===")
print(classification_report(y_te, y_pred))

