import pandas as pd

datos = {"calorias": [420, 380, 390, 430], "duracion": [50, 40, 45, 55]}

indexes = [f"dia {i+1}" for i in range(len(datos['calorias']))]

df = pd.DataFrame(datos, index=indexes)

print(df)
print(df.loc["dia 1"])
