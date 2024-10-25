import pandas as pd

df = pd.read_csv('cleaning_dataFormat.csv')
#df.dropna(inplace=True) # Borra todas las filas que tienen Nan
#df["Calories"].fillna(130, inplace=True)
#media = df["Calories"].mean()
moda = df["Calories"].mode()[0]
df["Calories"].fillna(moda, inplace = True)
print(df.to_string())

#new_df = df.dropna(inplace=True)
#print(new_df.to_string())
