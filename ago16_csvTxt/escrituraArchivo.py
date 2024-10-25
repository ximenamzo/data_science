import csv

precios = {'AAPL': 90.91, 'MSFT': 41.68, 'FB': 64.5}

with open('stock_prices_3.txt', 'w', newline='') as f:
    writer = csv.writer(f, delimiter=',')
    for accion, precio in precios.items():
        writer.writerow([accion, precio])


