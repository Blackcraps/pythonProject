# ===== IMPORT ====
import numpy as np
import pandas as pd
from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score



df_base = pd.read_csv('https://raw.githubusercontent.com/Blackcraps/pythonProject/master/data/telecom_users.csv', delimiter = ';')
df_services = pd.read_csv('https://raw.githubusercontent.com/Blackcraps/pythonProject/master/data/telecom_services.csv', delimiter = ';')
df_eval = pd.read_csv('https://raw.githubusercontent.com/Blackcraps/pythonProject/master/data/telecom_users_eval.csv', delimiter = ';')
#0 -> Non / Month-to_month/H
#1 -> Oui / ADSL / One year/F
#2 -> No phone service / Fibre / Two years

#print(df_base)

X = df_base.loc[:, df_base.colums != 'sortie_client']
y = df_base[['sortie_client']].values.flatten()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
clf = make_pipeline(StandardScaler(), SVC(gamma  ='auto'))
clf.fit(X_train, y_train)
clf.score(X_test, y_test)

cross_val_score(clf, X_train, y_train, cv=10, scoring="f1score")