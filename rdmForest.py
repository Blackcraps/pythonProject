# Import
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics


users = pd.read_csv('https://raw.githubusercontent.com/Blackcraps/pythonProject/master/data/telecom_users.csv', delimiter=';')
eval = pd.read_csv('https://raw.githubusercontent.com/Blackcraps/pythonProject/master/data/telecom_users_eval.csv', delimiter=';')

features = ['mariee','retraite','a_charge','genre','telephone','plusieurs_numeros','internet','contrat','facture_par_mail','client_depuis_mois']
X = users[features].values
y = users[['sortie_client']].values.flatten()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

regressor = RandomForestRegressor(n_estimators=20, random_state=0)
regressor.fit(X_train, y_train)

#score = regressor.score(X_test, y_test)
#print(score)
eval_pred = users[features].values
y_pred = regressor.predict(eval_pred)

print("Erreur absolu moyenne: ", metrics.mean_absolute_error(y, y_pred))
print('Mean squared Error: ', metrics.mean_squared_error(y,y_pred))
print('Root mean squared Error: ', np.sqrt(metrics.mean_squared_error(y,y_pred)))
print('Accuracy score: ', metrics.accuracy_score(y,y_pred.astype(int)))
print('Recall score : ', metrics.recall_score(y,y_pred.astype(int)))

#print(y_pred)







