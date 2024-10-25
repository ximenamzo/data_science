import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import centroid
from sklearn.cluster import KMeans

df = pd.read_csv(r"analisis.csv")
#print(dataframe.head(50))
#print(dataframe.describe())

#porcat = dataframe[dataframe['categoria'] == 9][['usuario', 'categoria']]
#print(porcat)


X = np.array(df[["op","ex","ag"]])
y = np.array(df['categoria'])
print(X.shape)

plt.rcParams['figure.figsize'] = (16, 9)
plt.style.use('ggplot')
fig = plt.figure()

ax = fig.add_subplot(111, projection='3d')

colores = ['blue', 'red', 'green', 'cyan', 'yellow', 'orange',
           'black', 'pink', 'brown', 'purple']

asignar=[]

for row in y:
    asignar.append(colores[row])

ax.set_xlabel('Op')
ax.set_ylabel('Ex')
ax.set_zlabel('Ag')
ax.scatter(X[:,0], X[:,1], X[:,2], c=df['categoria'], s=60)
#plt.show()

kmeans = KMeans(n_clusters=9).fit(X)
centroids = kmeans.cluster_centers_
#print(centroids)

ax.scatter(centroids[:,0], centroids[:,1], centroids[:,2], c='black', s=300, marker='o')
plt.show()


