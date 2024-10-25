import requests
from bs4 import BeautifulSoup

try:
    url = ("https://www.oreilly.com/library/view/designing-machine-learning/9781098107956/"
            "?_gl=1*anh74g*_ga*NDg0NTI0NDMxLjE3MjM4NDQ0MzA.*_ga_092EL089CH*"
            "MTcyMzg0NDQyOS4xLjEuMTcyMzg0NDUxMC42MC4wLjA.&quot;")

    # Hacer la solicitud y parsear los datos
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')  # 'html5lib' por 'html.parser'

    # Revisa el contenido de la página
    # print(soup.prettify())

    tds = soup.find_all('td', class_='thumbtext')
    print(f"Se encontraron: {tds}")

except Exception as e:
    print(f"Error: {e}")
