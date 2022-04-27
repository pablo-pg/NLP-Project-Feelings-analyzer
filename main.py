# Proyecto sobre Procesamiento de Lenguaje Natural para Inteligencia Artificial Avanzada
# Autor: Pablo Pérez González (alu0101318318@ull.edu.es)

import pandas as pd
df = pd.read_excel(r"COV_train.xlsx", sheet_name=0, dtype=str)
df2 = pd.DataFrame(df, columns= ['Message'])
print(df["Emotion"])