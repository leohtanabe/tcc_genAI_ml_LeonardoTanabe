# CÓDIGOS 17 e 18: BENCHMARK V2 COM VALIDAÇÃO CRUZADA E SHAP

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.tree import DecisionTreeClassifier

# ---------------------------------------------------------------------------
# CÓDIGO 17: BENCHMARK MULTIMODELO E VALIDAÇÃO CRUZADA (5-FOLD CV BINÁRIO)
# ---------------------------------------------------------------------------
label_map = {'Baixo Ganho': 0, 'Alto Ganho': 1}
y_tr_num = y_tr.map(label_map)
y_te_num = y_te.map(label_map)

modelos_v2 = {
    'Árvore de Decisão': DecisionTreeClassifier(max_depth=8, random_state=42),
    'Random Forest': RandomForestClassifier(
        n_estimators=100, max_depth=12, random_state=42, n_jobs=-1
    ),
    'XGBoost': xgb.XGBClassifier(
        n_estimators=100, max_depth=6, learning_rate=0.1,
        random_state=42, eval_metric='logloss',
    ),
    'LightGBM': lgb.LGBMClassifier(
        n_estimators=100, max_depth=6, learning_rate=0.1,
        random_state=42, verbose=-1,
    ),
}

resultados_v2 = []
modelos_v2_treinados = {}

for nome, model in modelos_v2.items():
    model.fit(X_tr, y_tr_num)
    y_pred = model.predict(X_te)
    y_proba = model.predict_proba(X_te)[:, 1]  # classe 1 = Alto Ganho

    acc = accuracy_score(y_te_num, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_te_num, y_pred, average='weighted'
    )
    auc = roc_auc_score(y_te_num, y_proba)

    resultados_v2.append({
        'Modelo': nome,
        'Acurácia (ΔGPA Binário)': acc,
        'Precisão': prec,
        'Recall': rec,
        'F1-Score': f1,
        'ROC-AUC': auc,
    })
    modelos_v2_treinados[nome] = model

df_res_v2 = pd.DataFrame(resultados_v2)
print("=== TABELA 4.4: BENCHMARK COMPARATIVO DE MODELOS (HOLDOUT 80/20 BINÁRIO) ===")
print(df_res_v2.round(4))
df_res_v2.to_csv("benchmark_modelos_v2_delta_gpa.csv", index=False)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scoring = ['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted', 'roc_auc']
results_cv = []

for nome, model in modelos_v2.items():
    cv_out = cross_validate(model, X_tr, y_tr_num, cv=skf, scoring=scoring, n_jobs=-1)
    results_cv.append({
        'Modelo': nome,
        'Acurácia CV (Média ± DP)': f"{cv_out['test_accuracy'].mean():.4f} ± {cv_out['test_accuracy'].std():.4f}",
        'F1-Score CV (Média ± DP)': f"{cv_out['test_f1_weighted'].mean():.4f} ± {cv_out['test_f1_weighted'].std():.4f}",
        'ROC-AUC CV (Média ± DP)': f"{cv_out['test_roc_auc'].mean():.4f} ± {cv_out['test_roc_auc'].std():.4f}",
    })

print("\n=== VALIDAÇÃO CRUZADA ESTRATIFICADA (5-FOLD CV BINÁRIO) ===")
print(pd.DataFrame(results_cv))

# ---------------------------------------------------------------------------
# CÓDIGO 18: EXPLICABILIDADE SHAP (SUMMARY E DEPENDENCE PLOTS BINÁRIO)
# ---------------------------------------------------------------------------
# Amostra de 500 observações do teste, usada nos gráficos
X_te_shap = X_te.sample(n=min(500, len(X_te)), random_state=42)

modelo_v2_lgbm = modelos_v2_treinados['LightGBM']
explainer_v2 = shap.TreeExplainer(modelo_v2_lgbm)
shap_values_v2 = explainer_v2.shap_values(X_te_shap)

plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values_v2, X_te_shap, feature_names=features,
                  plot_type="bar", show=False)
plt.title("Importância Relativa das Variáveis no Ganho Acadêmico (SHAP - LightGBM ΔGPA Binário)",
          fontsize=13, fontweight='bold', pad=15)
plt.tight_layout()
plt.savefig("shap_bar_v2_delta_gpa.png", dpi=300, bbox_inches='tight')
plt.show()

# Tratamento de dimensão (binário; classe 1 = 'Alto Ganho')
if isinstance(shap_values_v2, list):
    shap_target = shap_values_v2[1] if len(shap_values_v2) > 1 else shap_values_v2[0]
elif isinstance(shap_values_v2, np.ndarray) and shap_values_v2.ndim == 3:
    shap_target = shap_values_v2[:, :, 1]
else:
    shap_target = shap_values_v2

features_dep = ['Prompt_Engineering_Skill', 'Traditional_Study_Hours', 'Weekly_GenAI_Hours']

for feat in features_dep:
    if feat in X_te_shap.columns:
        plt.figure(figsize=(8, 4.5))
        shap.dependence_plot(feat, shap_target, X_te_shap, show=False)
        plt.title(f"Efeito Marginal (SHAP Dependence): {feat}", fontsize=11, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f"shap_dep_{feat.lower()}.png", dpi=300)
        plt.show()
