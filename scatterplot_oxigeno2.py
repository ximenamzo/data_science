# Ximena Manzo Castrejón 7D
# Introducción a Data Science FIE UdeC
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression

pd.set_option('display.float_format', lambda x: '%.2f' % x)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)

df = pd.read_csv(r"oxigeno.csv")
df.head()
df.describe().T
df.isnull().any()
df_ = df[['reduccion_solidos','reduccion_demanda_oxigenos']]
df_.head()

df_.hist()
plt.show()

plt.scatter(df_['reduccion_solidos'], df_['reduccion_demanda_oxigenos'], color='black')
plt.xlabel("reduccion_solidos")
plt.ylabel("reduccion_demanda_oxigenos")
plt.show()

## Aquí inicia el código de regresión lineal
X = df_[['reduccion_solidos']]
y = df_[['reduccion_demanda_oxigenos']]

reg_model = LinearRegression().fit(X, y)

reg_model.intercept_[0] + reg_model.coef_[0][0]*26.10

g = sns.regplot(x=X, y=y, scatter_kws={'color': 'b', 's':9}, ci=False, color='r')
g.set_title(f'Model Equation: Oxigeno = {round(reg_model.intercept_[0], 2)}' 
            f' + Reduccion solidos *{round(reg_model.coef_[0][0], 2)}')
g.set_ylabel('reduccion_demanda_oxigenos')
g.set_xlabel('reduccion_solidos')
plt.show()



