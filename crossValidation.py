# Import
import numpy as np
import pandas as pd

from sklearn.metrics import precision_score, accuracy_score, f1_score

from sklearn.linear_model import LogisticRegressionCV

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix


users = pd.read_csv('https://raw.githubusercontent.com/Blackcraps/pythonProject/master/data/telecom_users.csv', delimiter=';')
eval = pd.read_csv('https://raw.githubusercontent.com/Blackcraps/pythonProject/master/data/telecom_users_eval.csv', delimiter=';')
features = ['mariee', 'retraite', 'a_charge', 'facture_mensuelle', 'telephone', 'plusieurs_numeros', 'internet', 'total_factures', 'contrat', 'facture_par_mail', 'client_depuis_mois']

X = users[features].values #les features d'évaluation
y = users[['sortie_client']].values.flatten() #La donnee à predire
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

def recall(clf, X, y):
  cm = confusion_matrix(y, clf.predict(X))
  return cm[1][1] / (cm[1][1] + cm[1][0])

clf_cv = LogisticRegressionCV(cv=5, solver="lbfgs")
clf_cv = clf_cv.fit(X_train,y_train)
y_pred = clf_cv.predict(X_test)

rec = recall(clf_cv,X_test,y_test)
print("Recall   : ",rec)

acc = accuracy_score(y_test, y_pred)
print("Accuracy : ", acc)

cm = confusion_matrix(y_test, y_pred)
print("Matrice de Confusion :")
print(cm)

f1sc = f1_score(y_test,y_pred,average="macro")
print("Le score F1 est de : ", f1sc)

eval_pred = eval[features].values
pred = clf_cv.predict(eval_pred)
print(pred)