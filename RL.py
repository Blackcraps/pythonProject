import pandas as pd
import numpy as np
import seaborn as sns
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from imblearn.over_sampling import SMOTE
from sklearn import metrics
from sklearn.metrics import f1_score

services = pd.read_csv('https://raw.githubusercontent.com/Blackcraps/pythonProject/master/data/telecom_services.csv',
                       delimiter=';')
users = pd.read_csv('https://raw.githubusercontent.com/Blackcraps/pythonProject/master/data/telecom_users.csv',
                    delimiter=';')
eval = pd.read_csv('https://raw.githubusercontent.com/Blackcraps/pythonProject/master/data/telecom_users_eval.csv',
                   delimiter=';')

features = ['mariee', 'retraite', 'a_charge', 'facture_mensuelle', 'telephone', 'plusieurs_numeros', 'internet',
            'total_factures', 'contrat', 'facture_par_mail', 'client_depuis_mois']

X = users[features].values
y = users[['sortie_client']].values.flatten()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

print(X_train.shape)
print(y_train.shape)

sm = SMOTE(random_state=42, sampling_strategy=1)
X_train_sm, y_train_sm = sm.fit_resample(X_train, y_train)

mlog = LogisticRegression().fit(X_train_sm, y_train_sm)

y_pred = mlog.predict(X_train_sm)

confusion_matrix(y_train_sm, y_pred)

f1_score(y_train_sm, y_pred, average='macro')

eval_pred = eval[features].values
pred = mlog.predict(eval_pred)

df = pd.DataFrame(X_train_sm, columns=['mariee', 'retraite', 'a_charge', 'facture_mensuelle', 'telephone', 'plusieurs_numeros', 'internet',
            'total_factures', 'contrat', 'facture_par_mail', 'client_depuis_mois'])


print(pred)

print(pd.DataFrame(np.concatenate([mlog.intercept_.reshape(-1, 1), mlog.coef_], axis=1), index=["coef"], columns=["constante"] + list(df.columns)).T)
