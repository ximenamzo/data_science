import pandas as pd

# Cargar la base de datos de Netflix en un DataFrame
df = pd.read_csv('netflix_titles.csv')

# 1. Obtén los títulos de las columnas y guárdalos en una lista llamada columnas
columnas = list(df.columns)
print("Columnas del DataFrame:", columnas)

# 2. Imprime el tipo de dato de cada columna
print("\nTipos de datos de cada columna:")
print(df.dtypes)

# 3. Encuentra e imprime el total de valores perdidos por columna
valores_perdidos = df.isnull().sum()
print("\nValores perdidos por columna:")
print(valores_perdidos)

# 4. De las columnas con valores perdidos, identifica cuáles contienen datos de solo cadenas y cuáles son mixtas
columnas_con_perdidos = valores_perdidos[valores_perdidos > 0].index

solo_cadenas = []
mixtas = []

for columna in columnas_con_perdidos:
    # Verificar si la columna es de solo cadenas (objeto) o mixta (objeto pero contiene números también)
    if df[columna].apply(lambda x: isinstance(x, str)).all():
        solo_cadenas.append(columna)
    else:
        mixtas.append(columna)

print("\nColumnas con valores perdidos que son solo cadenas:", solo_cadenas)
print("Columnas con valores perdidos mixtas:", mixtas)

# 5. Para las columnas de tipo cadena, reemplaza los NaN con la cadena "dato no disponible"
for columna in solo_cadenas:
    df[columna].fillna("dato no disponible", inplace=True)

# 6. Limpia las cadenas de caracteres en blanco al inicio o final de las mismas en las columnas de solo cadenas
for columna in solo_cadenas:
    df[columna] = df[columna].apply(lambda x: x.strip() if isinstance(x, str) else x)

# 7. Sustituir valores incorrectos en la columna "rating"
rating_validos = ['PG-13', 'PG', 'TV-MA', 'TV-PG', 'TV-14', 'TV-Y', 'R', 'TV-G',
                  'TV-Y7', 'G', 'NC-17', 'NR', '', 'TV-Y7-FV']

df['rating'] = df['rating'].apply(lambda x: x if x in rating_validos else 'NR')

# 8. Sustituir países incorrectos en la columna "country"
df['country'] = df['country'].replace({
    'East Germany': 'Germany',
    'West Germany': 'Germany',
    'Soviet Union': 'Russia'
})

# Imprimir los primeros registros del DataFrame para revisar las correcciones
print("\nPrimeros registros del DataFrame corregido:")
print(df.head())

# Si quieres guardar el DataFrame corregido en un archivo CSV:
# df.to_csv('netflix_titles_corregido.csv', index=False)
