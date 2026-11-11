# CÓDIGO 1 e 2

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats.mstats import winsorize
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

ARQUIVO = "data/ai_student_impact_dataset.csv"
df_raw = pd.read_csv(ARQUIVO)

print(f"Dataset carregado: {df_raw.shape[0]} linhas e {df_raw.shape[1]} colunas.")
print(df_raw.head())
