# CÓDIGO 23 (transcrito do Colab, célula sem título): RADAR DAS PERSONAS (FIGURA 14 DO TCC)
# *** ATENÇÃO: OS VALORES DOS CENTROIDES ABAIXO ESTÃO DIGITADOS À MÃO (NÃO VÊM DOS DADOS) ***
# Esta célula NÃO calcula nada a partir de df. Os z-scores foram escritos manualmente.
# Inconsistências detectadas:
#  - Persona D tem ansiedade +1,20 (maior) no radar, mas a Tabela 2 do TCC dá a ela a MENOR
#    ansiedade (3,78).
#  - Persona A tem estudo tradicional +0,45 no radar, mas sua média (11,43 h, Código 15) fica
#    praticamente na média global (≈11,2 h, média ponderada dos clusters), ou seja, z perto de 0.
#  - O radar usa 5 eixos (inclui "Eficiência de IA") e não os 6 atributos do K-Means.
# Use cells/23b_radar_personas_calculado.py para regerar a Figura 14 a partir dos dados.

import matplotlib.pyplot as plt
import numpy as np

categories = [
    'Estudo Tradicional\n(Horas)',
    'Uso de IA\n(Horas)',
    'Habilidade em\nPrompts',
    'Ansiedade em\nExames',
    'Eficiência\nde IA',
]
N = len(categories)

angles = [n / float(N) * 2 * np.pi for n in range(N)]
angles += angles[:1]

# Centróides "médios padronizados (Z-Score)": VALORES FIXOS DIGITADOS
persona_a = [0.45, 0.50, 1.25, -0.30, 1.15]
persona_b = [-0.85, 1.35, 0.20, 0.40, -0.75]
persona_c = [1.10, -0.60, -0.40, -0.80, 0.10]
persona_d = [-0.90, -0.70, -0.95, 1.20, -0.85]

persona_a += persona_a[:1]
persona_b += persona_b[:1]
persona_c += persona_c[:1]
persona_d += persona_d[:1]

plt.figure(figsize=(8, 8), dpi=300)
ax = plt.subplot(111, polar=True)
ax.set_theta_offset(np.pi / 2)
ax.set_theta_direction(-1)

plt.xticks(angles[:-1], categories, size=10, fontweight='bold', color='#222222')
ax.set_rlabel_position(0)
plt.yticks([-1.0, -0.5, 0.0, 0.5, 1.0], ["-1.0", "-0.5", "0.0 (Média)", "+0.5", "+1.0"],
           color="grey", size=8)
plt.ylim(-1.5, 1.5)

ax.plot(angles, persona_a, linewidth=2, label='Persona A: Mestre em Prompts', color='#1f77b4')
ax.fill(angles, persona_a, '#1f77b4', alpha=0.15)
ax.plot(angles, persona_b, linewidth=2, label='Persona B: Dependente Hiperconectado', color='#8c564b')
ax.fill(angles, persona_b, '#8c564b', alpha=0.15)
ax.plot(angles, persona_c, linewidth=2, label='Persona C: Tradicionalista Dedicado', color='#2ca02c')
ax.fill(angles, persona_c, '#2ca02c', alpha=0.15)
ax.plot(angles, persona_d, linewidth=2, label='Persona D: Desengajado em Risco', color='#d62728')
ax.fill(angles, persona_d, '#d62728', alpha=0.15)

plt.title('Perfil dos Centróides Médios Padronizados (Z-Score) por Persona\n',
          size=12, fontweight='bold', pad=25)
plt.legend(loc='upper right', bbox_to_anchor=(1.35, 1.1), frameon=True, fontsize=9)

plt.tight_layout()
nome_arquivo = 'figura_14_perfil_personas_radar.png'
plt.savefig(nome_arquivo, dpi=300, bbox_inches='tight')
plt.show()
# Original (Colab): from google.colab import files; files.download(nome_arquivo)
