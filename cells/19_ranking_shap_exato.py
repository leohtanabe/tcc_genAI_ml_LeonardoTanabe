# CÓDIGO 19: RANKING EXATO DE E[|SHAP|]

import numpy as np
import pandas as pd
import shap

explainer = shap.TreeExplainer(lgbm)
shap_values = explainer(X_te)

if len(shap_values.values.shape) == 3:
    vals = shap_values.values[:, :, 1]
else:
    vals = shap_values.values

mean_abs_shap = np.abs(vals).mean(axis=0)

df_shap_ranking = pd.DataFrame(
    {"Variavel": X_te.columns, "E_SHAP_Absoluto": mean_abs_shap}
).sort_values(by="E_SHAP_Absoluto", ascending=False)

df_shap_ranking["E_SHAP_Texto_TCC"] = df_shap_ranking["E_SHAP_Absoluto"].round(2)

print("=== RANKING REAL E EXATO DE IMPORTÂNCIA SHAP (E[|SHAP|]) ===")
print(df_shap_ranking.to_string(index=False))
df_shap_ranking.to_csv("ranking_shap_exato_v2.csv", index=False)
