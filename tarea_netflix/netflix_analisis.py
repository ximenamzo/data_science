# Ximena Manzo Castrejón - Int Data Science - IS FIE UDC

import pandas as pd

# Cargar la base de datos de Netflix como DataFrame
df = pd.read_csv('netflix_titles.csv')

# 1. Obtener los títulos de las columnas y salvarlos en una lista llamada "columnas"
columnas = list(df.columns)
print("\n1. Columnas del DataFrame:", columnas)
print("Total de filas: ", len(df), "\n")

# 2. Imprimir (print) el tipo de dato de cada columna
print("2. Tipos de datos de cada columna:\n", df.dtypes, "\n")
# Segun la documentación:
## object = str or mixed
## int64 = int

# 3. Encontrar e imprimir el total de valores perdidos por columna
# https://saturncloud.io/blog/how-to-count-the-number-of-missingnan-values-in-each-row-in-python-pandas/
valores_perdidos = df.isnull().sum()
print("3. Valores perdidos por columna:\n", valores_perdidos, "\n")

# 4. De las columnas con valores perdidos, identificar cuáles contienen datos de solo cadenas y cuáles son mixtas
columnas_con_perdidos = valores_perdidos[valores_perdidos > 0].index

solo_cadenas = []
mixtas = []

for columna in columnas_con_perdidos:
    if df[columna].apply(lambda x: isinstance(x, str)).all():
        solo_cadenas.append(columna)
    else:
        mixtas.append(columna)


print("\n4. Columnas con valores perdidos que son solo cadenas:", solo_cadenas)
print("4.2. Columnas con valores perdidos mixtas:", mixtas, "\n")


# 5. Para las columnas de tipo cadena, reemplazar los NaN con la cadena "dato no disponible"
print("5. Identificación de los de solo cadenas y de las mixtas...")
for columna in solo_cadenas:
    df[columna].fillna("dato no disponible", inplace=True)


# 6. Limpiar las cadenas de caracteres en blanco al inicio o final de las mismas en las columnas de solo cadenas
print("6. Limpieza...")
for columna in solo_cadenas:
    df[columna] = df[columna].apply(lambda x: x.strip() if isinstance(x, str) else x)


# 7. Sustituir valores incorrectos en la columna "rating"
print("7. Sustitución de valores incorrectos...")
rating_validos = ['PG-13','PG','TV-MA','TV-PG','TV-14','TV-Y','R','TV-G','TV-Y7','G','NC-17','NR','','TV-Y7-FV']
df['rating'] = df['rating'].apply(lambda x: x if x in rating_validos else 'NR')


# 8. Sustituir países incorrectos en la columna "country" (xq ya no existen jej)
print("8. Sustitución... \n")
df['country'] = df['country'].replace({'East Germany':'Germany', 'West Germany':'Germany', 'Soviet Union':'Russia'})


# yo quiero todas las columnas y solo 15 filas
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 15)

# Imprimir los primeros registros del DataFrame para revisar las correcciones
print("Primeros registros del DataFrame ya corregido: \n", df.head(15), "\n")

# para guardar el DataFrame corregido en archivo:
# df.to_csv('netflix_titles_corregido.csv', index=False)
