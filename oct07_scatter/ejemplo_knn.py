import numpy as np
import pandas as pd
from requests.packages import target
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

fruits = pd.read_csv('fruit_data_with_colours.csv')
print(fruits.head(10))

target_fruits_name = dict(zip(fruits.fruit_label.unique(), fruits.fruit_name.unique()))

print(target_fruits_name)

X = fruits[['mass', 'width', 'height']]
y = fruits['fruit_label']

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

fruit_prediction = knn.predict([[101,101,101]])
print(target_fruits_name[fruit_prediction[0]])
