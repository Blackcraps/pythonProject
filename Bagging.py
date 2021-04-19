import pandas as pd
import numpy as np
import seaborn as sns
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_auc_score
from imblearn.over_sampling import SMOTE
from sklearn import metrics
from sklearn.metrics import f1_score
from sklearn.tree import DecisionTreeClassifier, _tree

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


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.001)
df = pd.DataFrame(X_train, columns=['mariee', 'retraite', 'a_charge', 'facture_mensuelle', 'telephone', 'plusieurs_numeros', 'internet',
            'total_factures', 'contrat', 'facture_par_mail', 'client_depuis_mois'])
columns = df.columns

sm = SMOTE(random_state=42, sampling_strategy=1)
X_train_sm, y_train_sm = sm.fit_resample(X_train, y_train)

nn = 0


def tree_to_code(tree, feature_names):
    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]

    # print ("def tree({}):" .format(", " .join(feature_names)))

    def recurse(node, depth):
        indent = "    " * depth

        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            table = 'X_train'
            name = table + "['" + feature_name[node] + "']"

            threshold = tree_.threshold[node]
            print("{}if {} <= {}:".format(indent, name, threshold))
            recurse(tree_.children_left[node], depth + 1)
            print("{}else:  # if {} > {}".format(indent, name, threshold))
            recurse(tree_.children_right[node], depth + 1)
        else:
            def increment():
                global nn
                nn = nn + 1

            increment()
            print("{}return 'Node_{}'".format(indent, nn))

    recurse(0, 1)

dt = DecisionTreeClassifier(criterion='gini', min_samples_split=200,min_samples_leaf=100, max_depth=3)
dt.fit(X_train_sm, y_train_sm)
y_pred3 = dt.predict(X_test)
print('Misclassified samples: %d' % (y_test != y_pred3).sum())
print(metrics.classification_report(y_test, y_pred3))
print (roc_auc_score(y_test, y_pred3))

tree_to_code(dt, columns)

def tree(X_train):
    if X_train['contrat'] <= 0.9999941885471344:
        if X_train['internet'] <= 1.0035915970802307:
            if X_train['client_depuis_mois'] <= 4.997718811035156:
                return 'Node_1'
            else:  # if X_train['client_depuis_mois'] > 4.997718811035156
                return 'Node_2'
        else:  # if X_train['internet'] > 1.0035915970802307
            if X_train['client_depuis_mois'] <= 15.998895645141602:
                return 'Node_3'
            else:  # if X_train['client_depuis_mois'] > 15.998895645141602
                return 'Node_4'
    else:  # if X_train['contrat'] > 0.9999941885471344
        if X_train['facture_mensuelle'] <= 93.67499923706055:
            if X_train['contrat'] <= 1.9838807582855225:
                return 'Node_5'
            else:  # if X_train['contrat'] > 1.9838807582855225
                return 'Node_6'
        else:  # if X_train['facture_mensuelle'] > 93.67499923706055
            if X_train['contrat'] <= 1.999673306941986:
                return 'Node_7'
            else:  # if X_train['contrat'] > 1.999673306941986
                return 'Node_8'

"""
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


print(pred)"""
