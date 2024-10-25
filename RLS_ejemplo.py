import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split, cross_val_score

pd.set_option('display.float_format', lambda x: '%.2f' % x)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)

df = pd.read_csv(r"Fuel_Consumption_Ratings.csv")
df.head()
df.describe().T
df.isnull().any()
df_ = df[['Engine Size','Cylinders','Fuel Consumption', 'CO2 Emissions']]
df_.head()

df_.hist()
plt.show()

plt.scatter(df_['Fuel Consumption'], df_['CO2 Emissions'], color='red')
plt.xlabel("Fuel Consumption")
plt.ylabel("CO2 Emissions")
plt.show()

plt.scatter(df_['Engine Size'], df_['CO2 Emissions'], color='blue')
plt.xlabel("Engine Size")
plt.ylabel("CO2 Emissions")
plt.show()

plt.scatter(df_['Cylinders'], df_['CO2 Emissions'], color='black')
plt.xlabel("Cylinders")
plt.ylabel("CO2 Emissions")
plt.show()

## Aquí inicia el código de regresión lineal
X = df_[['Fuel Consumption']]
y = df_[['CO2 Emissions']]

reg_model = LinearRegression().fit(X, y)

reg_model.intercept_[0] + reg_model.coef_[0][0]*26.10

g = sns.regplot(x=X, y=y, scatter_kws={'color': 'b', 's':9}, ci=False, color='r')
g.set_title(f'Model Equation: CO2 Emission = {round(reg_model.intercept_[0], 2)}' 
            f' + Fuel*{round(reg_model.coef_[0][0], 2)}')
g.set_ylabel('CO2 Emissions')
g.set_xlabel('Fuel Consumption')
plt.show()



