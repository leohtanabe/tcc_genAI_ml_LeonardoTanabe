# CÓDIGOS 5 a 8: EXPERIMENTO PRELIMINAR V1

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

# ---------------------------------------------------------------------------
# CÓDIGO 5: TREINAMENTO E BENCHMARK PRELIMINAR - TARGET V1
# ---------------------------------------------------------------------------
X_train_v1, X_test_v1, y_train_v1, y_test_v1 = train_test_split(
    X_v1, y_v1, test_size=0.20, random_state=42, stratify=y_v1
)

# Padronização ajustada SOMENTE no treino
scaler_v1 = StandardScaler()
X_train_v1_scaled = scaler_v1.fit_transform(X_train_v1)
X_test_v1_scaled = scaler_v1.transform(X_test_v1)

modelos_v1 = {
    'Árvore de Decisão': DecisionTreeClassifier(max_depth=8, random_state=42),
    'Random Forest': RandomForestClassifier(
        n_estimators=100, max_depth=12, random_state=42, n_jobs=-1
    ),
    'XGBoost': xgb.XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        eval_metric='logloss',
    ),
    'LightGBM': lgb.LGBMClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        verbose=-1,
    ),
}

modelos_treinados_v1 = {}
resultados_v1 = []

# Codificação binária: 0 = Baixo, 1 = Alto (ordem dos rótulos do qcut)
y_test_v1_num = y_test_v1.cat.codes

for nome, model in modelos_v1.items():
    model.fit(X_train_v1_scaled, y_train_v1.cat.codes)
    y_pred = model.predict(X_test_v1_scaled)
    y_proba = model.predict_proba(X_test_v1_scaled)[:, 1]  # classe positiva = Alto

    acc = accuracy_score(y_test_v1_num, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_test_v1_num, y_pred, average='weighted'
    )
    auc = roc_auc_score(y_test_v1_num, y_proba)

    resultados_v1.append({
        'Modelo': nome, 'Acurácia': acc, 'Precisão': prec,
        'Recall': rec, 'F1-Score': f1, 'ROC-AUC': auc,
    })
    modelos_treinados_v1[nome] = model

df_res_v1 = pd.DataFrame(resultados_v1)
print("=== DEMONSTRAÇÃO V1: BENCHMARK PRELIMINAR BINÁRIO (TARGET: POST_SEMESTER_GPA) ===")
print(df_res_v1.round(4))

# ---------------------------------------------------------------------------
# CÓDIGO 6: MATRIZES DE CONFUSÃO - V1
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(10, 8))
class_labels = ['Baixo', 'Alto']
for (nome, model), ax in zip(modelos_treinados_v1.items(), axes.flatten()):
    y_pred = model.predict(X_test_v1_scaled)
    cm = confusion_matrix(y_test_v1_num, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=class_labels, yticklabels=class_labels)
    ax.set_title(f'Matriz V1 - {nome}', fontweight='bold')
    ax.set_xlabel('Classe Preditiva')
    ax.set_ylabel('Classe Real')
plt.tight_layout()
plt.show()

# ---------------------------------------------------------------------------
# CÓDIGO 7: SHAP - V1 (XGBoost)
# ---------------------------------------------------------------------------
modelo_escolhido = modelos_treinados_v1['XGBoost']
X_test_v1_df = pd.DataFrame(X_test_v1_scaled, columns=X_v1.columns)

explainer = shap.TreeExplainer(modelo_escolhido)
shap_values = explainer.shap_values(X_test_v1_df)

plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, X_test_v1_df, plot_type="bar", show=False)
plt.title("Importância Relativa das Variáveis (SHAP Values - XGBoost V1 Binário)",
          fontsize=14, fontweight='bold', pad=15)
plt.tight_layout()
plt.savefig("shap_bar_importance_v1.png", dpi=300, bbox_inches='tight')
plt.show()

plt.figure(figsize=(10, 6))
if isinstance(shap_values, list):
    shap_vals_class0 = shap_values[0]
elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
    shap_vals_class0 = shap_values[:, :, 0]
else:
    shap_vals_class0 = -shap_values  # binário 2D: impacto na classe 0 = inverso da classe 1

shap.summary_plot(shap_vals_class0, X_test_v1_df, show=False)
plt.title("Impacto das Variáveis na Classe de 'Baixo Desempenho' (SHAP Summary V1)",
          fontsize=14, fontweight='bold', pad=15)
plt.tight_layout()
plt.savefig("shap_summary_baixo_desempenho_v1.png", dpi=300, bbox_inches='tight')
plt.show()

plt.figure(figsize=(10, 6))
shap.dependence_plot('Pre_Semester_GPA', shap_vals_class0, X_test_v1_df, show=False)
plt.title("Evidência de Viés V1: Pre_Semester_GPA Ditando o Resultado Final",
          fontsize=12, fontweight='bold', pad=15)
plt.tight_layout()
plt.savefig("shap_dependence_pre_gpa_v1.png", dpi=300, bbox_inches='tight')
plt.show()

# ---------------------------------------------------------------------------
# CÓDIGO 8: PERFIL MÉDIO POR CLASSE (V1)
# ---------------------------------------------------------------------------
if 'digital_procrastination_index' not in df_v1.columns:
    dep_m, dep_s = (df_v1['Perceived_AI_Dependency'].mean(),
                    df_v1['Perceived_AI_Dependency'].std())
    trad_m, trad_s = (df_v1['Traditional_Study_Hours'].mean(),
                      df_v1['Traditional_Study_Hours'].std())
    df_v1['digital_procrastination_index'] = (
        (df_v1['Perceived_AI_Dependency'] - dep_m) / dep_s
    ) - ((df_v1['Traditional_Study_Hours'] - trad_m) / trad_s)

colunas_analise = [
    'Pre_Semester_GPA', 'Post_Semester_GPA', 'digital_procrastination_index',
    'Weekly_GenAI_Hours', 'Traditional_Study_Hours', 'Perceived_AI_Dependency',
    'Skill_Retention_Score', 'Anxiety_Level_During_Exams', 'Prompt_Engineering_Skill',
]
colunas_validas = [c for c in colunas_analise if c in df_v1.columns]

df_diagnostico_v1 = (
    df_v1.groupby('desempenho_categoria', observed=False)[colunas_validas].mean().T
)
print("=== [DEMONSTRAÇÃO V1: PERFIL MÉDIO POR FAIXA DE DESEMPENHO (BINÁRIO)] ===")
print(df_diagnostico_v1.round(3))
df_diagnostico_v1.to_csv("tabela_perfil_estudantes_v1.csv")
