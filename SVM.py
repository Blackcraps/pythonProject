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
from sklearn.metrics import classification_report, confusion_matrix

#SVM

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
y = df_base[['sortie_client']].values.flatten()

#Emsembles d'apprentissage de 80% et ensemble de tests de 20%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

#Scaling des datas et mise en place de SVC
clf = make_pipeline(StandardScaler(),SVC(gamma ='auto'))
clf.fit(X_train, y_train)
clf.score(X_test, y_test)

#On affiche la matrice de confusion
predictions = clf.predict(X_test)
print(confusion_matrix(y_test, predictions))

#Il suffit de faire la moyenne des 10 résultats obtenus pour avoir le f1 score moyen
print(cross_val_score(clf, X_train, y_train, cv=10, scoring="f1"))

#On affiche les résultats obtenus sur la prédiction des 200 données
print(clf.predict(df_eval))