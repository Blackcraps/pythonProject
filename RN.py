#Imports
import numpy as np
import pandas as pd
from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix

#Réseau de neurones

#Ouverture des fichiers csv
df_base = pd.read_csv('https://raw.githubusercontent.com/Blackcraps/pythonProject/master/data/telecom_users.csv', delimiter = ';')
df_services = pd.read_csv('https://raw.githubusercontent.com/Blackcraps/pythonProject/master/data/telecom_services.csv', delimiter = ';')
df_eval = pd.read_csv('https://raw.githubusercontent.com/Blackcraps/pythonProject/master/data/telecom_users_eval.csv', delimiter = ';')

#0 -> Non / Month-to_month/H
#1 -> Oui / ADSL / One year/F
#2 -> No phone service / Fibre / Two years

#On supprime les attributs que nous n'utiliserons pas
del df_base['id_client']
del df_base['type_de_paiement']
del df_base['total_factures']
del df_eval['id_client']
del df_eval['type_de_paiement']
del df_eval['total_factures']

#X prend les données et y prend les étiquettes
X = df_base.loc[:, df_base.columns != 'sortie_client']
y = df_base[['sortie_client']]

#Retourne un tableau de rang 1 ce qui est nécessaire sinon provoque une erreur dans le code
y = np.ravel(y)

#Emsembles d'apprentissage de 80% et ensemble de tests de 20%
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2)

#Il ne faut pas utiliser de scaler dans notre cas car sinon nous ne prédirons que des 0
"""scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
"""

#On crée un réseau de neurones composés de 3 couches cachées de 10 neurones chacunes
mlp = MLPClassifier(hidden_layer_sizes=(10,10,10), activation='identity', solver='lbfgs', max_iter = 10000)

#On entraîne notre réseau de neurones sur nos données
mlp.fit(X_train, y_train)

#On affiche la matrice de confusion
predictions = mlp.predict(X_test)
print(confusion_matrix(y_test, predictions))

#Il suffit de faire la moyenne des 10 résultats obtenus pour avoir le f1 score moyen
print(cross_val_score(mlp, X_train, y_train, cv=10, scoring="f1"))

#On affiche les résultats obtenus sur la prédiction des 200 données
print(mlp.predict(df_eval))