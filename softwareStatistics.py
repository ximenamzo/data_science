import pandas as pd
from matplotlib import pyplot as plt

df = pd.read_csv('Data-Science-Jobs.csv')

df_baja = df.sort_values(by='Salary', ascending=True).head(15)
df_alta = df.sort_values(by='Salary', ascending=False).head(15)

new_df = df_baja

x = new_df['Job Title']
y = new_df['Salary']

plt.figure(figsize=(10, 6))

bars = plt.bar(x, y, color="blue")

plt.xlabel('Job Title')
plt.ylabel('Salary (yearly)')
plt.title('Best salary')

plt.xticks(rotation=45, ha='right', fontsize=10)
plt.grid(axis='y', linestyle='--', alpha=0.7)

for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval, f'{yval:,.0f}', ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.show()