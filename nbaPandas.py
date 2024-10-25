import pandas as pd

datos = pd.read_csv('nba.csv', index_col="Name")

jugador = datos.loc["Amir Johnson"]

print(jugador)
