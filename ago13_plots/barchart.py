from matplotlib import pyplot as plt

peliculas = ["Annie Hall", "Ben-Hur", "Casablanca", "Gandhi", "West Side Story"]
oscars = [5, 11, 3, 8, 10]

xs = [i + 0.1 for i, _ in enumerate(peliculas)]
plt.bar(xs, oscars)
plt.ylabel("Oscars ganados")
plt.xlabel("Películas")
plt.title("Peliculas más premiadas")
plt.xticks([i + 0.5 for i, _ in enumerate(peliculas)], peliculas)
plt.show()
