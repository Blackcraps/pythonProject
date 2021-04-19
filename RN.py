# ===== IMPORT ====
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


df_base = pd.read_csv('https://raw.githubusercontent.com/Blackcraps/pythonProject/master/data/telecom_users.csv', delimiter = ';')
df_services = pd.read_csv('https://raw.githubusercontent.com/Blackcraps/pythonProject/master/data/telecom_services.csv', delimiter = ';')
df_eval = pd.read_csv('https://raw.githubusercontent.com/Blackcraps/pythonProject/master/data/telecom_users_eval.csv', delimiter = ';')
#0 -> Non / Month-to_month/H
#1 -> Oui / ADSL / One year/F
#2 -> No phone service / Fibre / Two years
print(df_eval)
del df_base['id_client']
del df_base['type_de_paiement']
del df_base['total_factures']
del df_eval['id_client']
del df_eval['type_de_paiement']
del df_eval['total_factures']
#print(df_base)

X = df_base.loc[:, df_base.columns != 'sortie_client']
y = df_base[['sortie_client']]
y = np.ravel(y)
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2)


#scaler = StandardScaler()
#scaler.fit(X_train)
#X_train = scaler.transform(X_train)
#X_test = scaler.transform(X_test)
#print(X_test)
#print(X_train)


mlp = MLPClassifier(hidden_layer_sizes=(10,10,10), max_iter = 10000)
mlp.fit(X_train, y_train)

predictions = mlp.predict(X_test)
#result = mlp.predict(df_eval)
#result = result.astype(int)
#result.to_csv('D:/Cours/Fouille_Donnees/Hackaton/resultat.csv', index=False,sep = ' ', header = None, line_terminator=' ')
#print(mlp.predict(X_test)[:200])
print(mlp.predict(df_eval))
#np.savetxt('D:/Cours/Fouille_Donnees/Hackaton/resultat.txt', result, delimiter='')

#print(confusion_matrix(y_test, predictions))

#print(cross_val_score(mlp, X_train, y_train, cv=10, scoring="f1"))