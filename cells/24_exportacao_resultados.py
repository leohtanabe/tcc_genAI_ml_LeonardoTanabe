# CÓDIGO 24 (transcrito do Colab, seção 9 "Exportação Unificada"): COMPACTAÇÃO DE RESULTADOS
# No notebook original esta célula aparece ANTES dos Códigos 19 a 23; deve ser executada por último.
# Versão do repositório: não depende do Colab (o download só é tentado se google.colab existir).

import os
import zipfile

arquivos_para_baixar = [f for f in os.listdir('.') if f.endswith('.png') or f.endswith('.csv')]
zip_filename = 'resultados_tcc_graficos_e_tabelas.zip'

print("--- COMPACTANDO ARQUIVOS GERADOS ---")
with zipfile.ZipFile(zip_filename, 'w') as zipf:
    for arq in arquivos_para_baixar:
        zipf.write(arq)
        print(f"Adicionado ao ZIP: {arq}")

try:
    from google.colab import files  # só existe no Colab
    files.download(zip_filename)
    print("Download do arquivo ZIP iniciado com sucesso.")
except Exception:
    print(f"O arquivo '{zip_filename}' foi salvo na pasta atual.")
