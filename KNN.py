import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
import pandas as pd

users =pd.read_csv('/content/telecom_users (1).csv', delimiter = ';')

# sélection des features pour prédire
features = ['mariee', 'retraite', 'a_charge', 'facture_mensuelle', 'telephone', 'plusieurs_numeros', 'internet', 'total_factures', 'contrat', 'facture_par_mail', 'client_depuis_mois']
X = users[features].values
# sélection de la feature à prédire
y = users[['sortie_client']].values.flatten()

from sklearn.model_selection import train_test_split
# On découpe notre jeu d'entraînement en ne prenant que 10% de notre jeu complet
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)

from sklearn import neighbors

# application du knn
knn =  neighbors.KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train,y_train)

test_user = pd.read_csv('/content/telecom_users_eval (2).csv', delimiter = ';')

# sélection des features du jeu d'évaluation
X_test2 = test_user[features].values

# application du modèle sur le jeu d'évaluation
knn = neighbors.KNeighborsClassifier(10)
knn.fit(X_train, y_train)

# On récupère les prédictions sur les données test
predicted = knn.predict(X_test2)

predicted