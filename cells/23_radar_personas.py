# CÓDIGO 23: FIGURA 14 (RADAR DOS CENTROIDES PADRONIZADOS POR PERSONA)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 1. Variáveis do K-Means (as mesmas do Código 14) e rótulos dos eixos
eixos = {
    'Traditional_Study_Hours': 'Estudo Tradicional\n(Horas)',
    'Weekly_GenAI_Hours': 'Uso de IA\n(Horas)',
    'Prompt_Engineering_Skill': 'Habilidade em\nPrompts',
    'Perceived_AI_Dependency': 'Dependência\nPercebida de IA',
    'Skill_Retention_Score': 'Retenção de\nHabilidades',
    'Anxiety_Level_During_Exams': 'Ansiedade em\nExames',
}
colunas = list(eixos)

ordem = [
    'Persona A: Mestre em Prompts',
    'Persona B: Dependente Hiperconectado',
    'Persona C: Tradicionalista Dedicado',
    'Persona D: Desengajado em Risco',
]
cores = ['#1f77b4', '#8c564b', '#2ca02c', '#d62728']

# 2. Centroides padronizados: média do grupo em z-score da base inteira
#    (ddof=0, igual ao StandardScaler usado no K-Means)
z = (df[colunas] - df[colunas].mean()) / df[colunas].std(ddof=0)
z['persona'] = df['persona']
centroides = z.groupby('persona')[colunas].mean().loc[ordem]

print('=== CENTROIDES PADRONIZADOS (z-score) POR PERSONA ===')
print(centroides.rename(columns=lambda c: eixos[c].replace('\n', ' ')).round(2).T)
centroides.round(4).to_csv('centroides_z_por_persona.csv')

# 3. Radar
N = len(colunas)
angulos = [n / float(N) * 2 * np.pi for n in range(N)]
angulos += angulos[:1]

lim = np.ceil(np.abs(centroides.values).max() * 4) / 4   # limite arredondado a 0,25
passo = 0.5 if lim <= 2 else 1.0
ticks = np.arange(-lim, lim + 1e-9, passo)

fig = plt.figure(figsize=(9, 9), dpi=300)
ax = plt.subplot(111, polar=True)
ax.set_theta_offset(np.pi / 2)
ax.set_theta_direction(-1)

plt.xticks(angulos[:-1], [eixos[c] for c in colunas],
           size=10, fontweight='bold', color='#222222')
ax.set_rlabel_position(0)
plt.yticks(ticks, [f'{t:+.1f}'.replace('+0.0', '0.0 (Média)') for t in ticks],
           color='grey', size=8)
plt.ylim(-lim, lim)

for nome, cor in zip(ordem, cores):
    valores = centroides.loc[nome].tolist()
    valores += valores[:1]
    ax.plot(angulos, valores, linewidth=2, label=nome, color=cor)
    ax.fill(angulos, valores, cor, alpha=0.15)

plt.title('Perfil dos Centróides Médios Padronizados (Z-Score) por Persona',
          size=12, fontweight='bold', pad=25)
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.08), ncol=2,
           frameon=True, fontsize=9)

plt.tight_layout()
plt.savefig('figura_14_perfil_personas_radar.png', dpi=300, bbox_inches='tight')
plt.show()

# Download no Colab
try:
    from google.colab import files
    files.download('figura_14_perfil_personas_radar.png')
except Exception:
    pass
