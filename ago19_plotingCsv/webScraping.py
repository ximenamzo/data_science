import csv
from io import StringIO
import requests
from bs4 import BeautifulSoup
import pandas as pd

url = "https://www.worldometers.info/world-population/population-by-country/"

req = requests.get(url)

soup = BeautifulSoup(req.text, features="lxml")

data = soup.find_all("table")[0]

df_population = pd.read_html(StringIO(str(data)))[0]

df_population.head()

export_csv = df_population.to_csv(r"paises.csv", index=None, header=True)
