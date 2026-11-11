# CÓDIGO 19 (transcrito do Colab, célula sem título): RANKING EXATO DE E[|SHAP|]
# Pré-requisitos: lgbm, X_te (cells/09_modelo_v2_lightgbm.py)
# Pontos verificados:
#  - Calculado sobre TODO o teste (10.000 linhas), não sobre a amostra de 500 do Código 18.
#  - Usa o `lgbm` do Código 9 (rótulos em texto; classes_ = ['Alto Ganho', 'Baixo Ganho']).
#    Para E[|SHAP|] o sinal não importa, então o ranking é idêntico para qualquer classe.
#  - Os valores batem com o texto do TCC (0,59 / 0,45 / 0,25 / 0,22; IPD 0,07; dependência 0,02).

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
df_shap_ranking.to_csv("ranking_shap_exato_v2.csv", index=False)  # linha adicionada na versão do repositório
