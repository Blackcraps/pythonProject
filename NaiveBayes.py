import pandas as pd
import numpy as np
import seaborn as sns
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from imblearn.over_sampling import SMOTE
from sklearn import metrics

telecom_users = pd.read_csv('https://raw.githubusercontent.com/Blackcraps/pythonProject/master/data/telecom_users.csv', delimiter=';').dropna()
telecom_users_eval = pd.read_csv('https://raw.githubusercontent.com/Blackcraps/pythonProject/master/data/telecom_users_eval.csv', delimiter=';').dropna()

features = ['mariee', 'retraite', 'a_charge', 'facture_mensuelle', 'telephone', 'plusieurs_numeros', 'internet', 'total_factures', 'contrat', 'facture_par_mail', 'client_depuis_mois']
X = telecom_users[features].values #variables predictives sur le users
y = telecom_users[['sortie_client']].values.flatten() #variable a predire

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) #decoupage des donnees pour le test et train

#application du naive bayes avec la matrice de confusion
from sklearn.naive_bayes import ComplementNB
clf = ComplementNB()
clf.fit(X_train, y_train)
X_test_eval = clf.predict(X_test)
confusion_matrix(y_test, X_test_eval)
print((confusion_matrix(y_test, X_test_eval)))

#over sampling car resultat de la matrice pas concluant
from imblearn.over_sampling import SMOTE
sm = SMOTE(random_state=42, ratio = 1.0)
X_train_res, y_train_res = sm.fit_sample(X_train, y_train)

clf = ComplementNB()
clf.fit(X_train_res, y_train_res)
X_test_eval = clf.predict(X_train_res)
confusion_matrix(y_train_res, X_test_eval)

#prediction sur users eval
X_test_eval = telecom_users_eval[features].values
X_pred = clf.predict(X_test_eval)
X_pred