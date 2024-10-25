from matplotlib import pyplot as plt
from sklearn import datasets
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier

# Cargar data set de las flores iris
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Crear el clasificador de arboles
clf = DecisionTreeClassifier(random_state=1234)
model = clf.fit(X, y)

# Imprimir el arbol creado
text_representation = tree.export_text(clf)
print(text_representation)

fig = plt.figure(figsize=(25,20))
_ = tree.plot_tree(clf, feature_names=iris.feature_names, class_names=iris.target_names, filled=True)
fig.savefig("Arbol_de_Decision.png")
