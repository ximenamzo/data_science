import csv
import pandas as pd
from matplotlib import pyplot as plt

# df = dataframe
df = pd.read_csv('paises.csv')
# df = df.head(15)

df_baja = df.sort_values(by='Med.  Age', ascending=True).head(15)
df_alta = df.sort_values(by='Med.  Age', ascending=False).head(15)

new_df = df_alta
# print(df.head())
# print(df.info())

x = new_df['Country (or dependency)']
y = new_df['Med.  Age']

plt.figure(figsize=(10, 6))

bars = plt.bar(x, y, color="pink")

plt.xlabel('Paises')
plt.ylabel('Edad Promedio')
plt.title('Edad promedio mas baja')

plt.xticks(rotation=45, ha='right', fontsize=10)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# para q se vea el valor total
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval, f'{yval:,.0f}', ha='center', va='bottom', fontsize=10)

plt.tight_layout()  # para que no se encimen los nombres
plt.show()