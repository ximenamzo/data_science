import csv

with open('stock_prices_1.txt', 'r') as f:
    reader = csv.reader(f, delimiter=',')
    for row in reader:
        fecha = row[0]
        simbolo = row[1]
        precio = row[2]
        print(fecha + " " + simbolo + " " + precio)
