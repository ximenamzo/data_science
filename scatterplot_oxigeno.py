import pandas as pd
from matplotlib import pyplot as plt

# Lee los datos desde el archivo CSV
data = pd.read_csv('oxigeno.csv')

# Asigna las columnas a las variables correspondientes
reduccion_solidos = data['reduccion_solidos']
reduccion_demanda_oxigenos = data['reduccion_demanda_oxigenos']

# Crea la gráfica de dispersión
plt.scatter(reduccion_solidos, reduccion_demanda_oxigenos)

# Añade anotaciones si es necesario (opcional)
for i, (x, y) in enumerate(zip(reduccion_solidos, reduccion_demanda_oxigenos)):
    plt.annotate(f'({x}, {y})', xy=(x, y), xytext=(5, -5), textcoords='offset points')

# Añade título y etiquetas
plt.title("Reducción de Sólidos vs. Reducción de Demanda de Oxígeno")
plt.xlabel("Reducción de Sólidos")
plt.ylabel("Reducción de Demanda de Oxígeno")

# Muestra la gráfica
plt.show()
