# Import
import numpy as np
import pandas as pd

from sklearn.metrics import precision_score, accuracy_score, f1_score

from sklearn.linear_model import LogisticRegressionCV

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

#Importation des données
users = pd.read_csv('https://raw.githubusercontent.com/Blackcraps/pythonProject/master/data/telecom_users.csv', delimiter=';')
eval = pd.read_csv('https://raw.githubusercontent.com/Blackcraps/pythonProject/master/data/telecom_users_eval.csv', delimiter=';')

#Selection des features et de la feature a prédire
features = ['mariee', 'retraite', 'a_charge', 'facture_mensuelle', 'telephone', 'plusieurs_numeros', 'internet', 'contrat', 'facture_par_mail', 'client_depuis_mois']
X = users[features].values
y = users[['sortie_client']].values.flatten()

#Découpage des jeux d'entrainements et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Fonction pour calculer le recall
def recall(clf, X, y):
  cm = confusion_matrix(y, clf.predict(X))
  return cm[1][1] / (cm[1][1] + cm[1][0])

#Lancement de l'algorithme de la regression logistic avec cross validation
clf_cv = LogisticRegressionCV(cv=5, solver="lbfgs") #cv=5 (4/5 des données seront d'entrainement et 1/5 des données sont de test
clf_cv = clf_cv.fit(X_train,y_train)
y_pred = clf_cv.predict(X_test)

#Calcul du recall
rec = recall(clf_cv,X_test,y_test)
print("Recall   : ",rec)

#Calcul de l'Accuracy
acc = accuracy_score(y_test, y_pred)
print("Accuracy : ", acc)

#Affichage de la matrice de confusion
cm = confusion_matrix(y_test, y_pred)
print("Matrice de Confusion :")
print(cm)

#Calcul du F1_score
f1sc = f1_score(y_test,y_pred,average="macro")
print("Le score F1 est de : ", f1sc)

#Affichage de la prédiction sur les données d'évaluation
eval_pred = eval[features].values
pred = clf_cv.predict(eval_pred)
print(pred)