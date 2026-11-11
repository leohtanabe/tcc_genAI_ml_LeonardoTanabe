# CÓDIGO 1 e 2 (transcrito das imagens do Colab; versão reprodutível fora do Colab)
# Original: !pip install xgboost lightgbm imbalanced-learn shap matplotlib seaborn scikit-learn pandas numpy -q
# Observação: imblearn/SMOTE foi importado no original, mas o texto do TCC informa que NÃO foi aplicado.

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats.mstats import winsorize
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Original (Colab): uploaded = files.upload(); df_raw = pd.read_csv(nome_arquivo)
# Versão reprodutível: arquivo baixado do Kaggle e colocado em data/
ARQUIVO = "data/ai_student_impact_dataset.csv"
df_raw = pd.read_csv(ARQUIVO)

print(f"Dataset carregado: {df_raw.shape[0]} linhas e {df_raw.shape[1]} colunas.")
print(df_raw.head())
