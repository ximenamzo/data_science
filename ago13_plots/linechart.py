from matplotlib import pyplot as plt

years = [1950, 1960, 1970, 1980, 1990, 2000, 2010]  # eje x
pib = [300.2, 543.3, 1075.9, 2862.5, 5979.6, 10289.7, 14958.3]  # eje y

plt.plot(years, pib, color='blue', marker='o', linestyle='solid')
plt.title('Line chart del Producto Interno Bruto')
plt.ylabel('Billones de dólares')
plt.xlabel('Década')
plt.show()
