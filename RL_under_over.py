#collab : https://colab.research.google.com/drive/1Bi1EQeOj0vvVwJcOTzxXJQ5uLbVTrrL_?usp=sharing

import pandas as pd
import numpy as np
import seaborn as sns
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from imblearn.combine import SMOTEENN
from imblearn.combine import SMOTETomek
from sklearn.model_selection import cross_val_score

from sklearn import metrics
from sklearn.metrics import f1_score

users = pd.read_csv('https://raw.githubusercontent.com/Blackcraps/pythonProject/master/data/telecom_users.csv', delimiter = ';')
eval = pd.read_csv('https://raw.githubusercontent.com/Blackcraps/pythonProject/master/data/telecom_users_eval.csv', delimiter = ';')

# selection des features pour prédire
features = ['mariee', 'retraite', 'a_charge', 'facture_mensuelle', 'telephone', 'plusieurs_numeros', 'internet', 'contrat', 'facture_par_mail', 'client_depuis_mois']
X = users[features].values
# selection de la feature à prédire
y = users[['sortie_client']].values.flatten()

# découpage des jeux d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# association de l'undersampling et oversampling
smote_tomek = SMOTETomek(random_state=42, sampling_strategy=0.9)
X_train_sm, y_train_sm = smote_tomek.fit_resample(X_train, y_train)
smote_enn = SMOTEENN(random_state=42, sampling_strategy=0.9)
X_train_res, y_train_res = smote_enn.fit_resample(X_train_sm, y_train_sm)

# application de la regression logistique
mlog = LogisticRegression().fit(X_train_res, y_train_res)
# prédiction
y_pred = mlog.predict(X_train)

# matrice de confusion
confusion_matrix(y_train, y_pred)

# score f1
f1_score(y_train, y_pred, average='macro')

# prédiction sur le jeu d'évaluation
eval_pred = eval[features].values
pred = mlog.predict(eval_pred)