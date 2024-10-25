import pandas as pd
from matplotlib import pyplot as plt

# df = dataframe
df = pd.read_csv('paises.csv')
df = df.head(15)

print(df.head())
print(df.info())

x = df['Country (or dependency)']
# y = df['Population  (2024)']
y = df['Med.  Age']

plt.figure(figsize=(10, 6))

bars = plt.bar(x, y, color="pink")

plt.xlabel('Paises')
plt.ylabel('Poblacion')
plt.title('15 paises mas poblados')

plt.legend(['Paises por poblacion'])

plt.xticks(rotation=45, ha='right', fontsize=10)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# valor total
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval, f'{yval:,.0f}', ha='center', va='bottom', fontsize=10)


plt.tight_layout()  # ajuste de margenes para evitar que se corten las etiquetas
plt.show()