# Import
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import precision_score, accuracy_score, f1_score

from sklearn.tree import DecisionTreeClassifier, export_graphviz

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix



#Importation des données
users = pd.read_csv('https://raw.githubusercontent.com/Blackcraps/pythonProject/master/data/telecom_users.csv', delimiter=';')
eval = pd.read_csv('https://raw.githubusercontent.com/Blackcraps/pythonProject/master/data/telecom_users_eval.csv', delimiter=';')

#Selection des features et de la feature a prédire
features = ['mariee','retraite','a_charge','facture_mensuelle','genre','telephone','plusieurs_numeros','internet','contrat','facture_par_mail','client_depuis_mois']
X = users[features].values
y = users[['sortie_client']].values.flatten()

#Découpage des jeux d'entrainements et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Lancement de l'algo de l'arbre de décision
clf = DecisionTreeClassifier(random_state=42)
clf = clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

def recall(clf, X, y):
  cm = confusion_matrix(y, clf.predict(X))
  return cm[1][1] / (cm[1][1] + cm[1][0])

#Calcul du Recall
recall1 = recall(clf, X_test, y_test) # Recall obtenu de la matrice de confusion
print("Recall   : ", recall1)

#Calcul de l'Accuracy
acc = accuracy_score(y_test,y_pred)
print("Accuracy : ", acc)

#Calcul du F1_score
print("F1_score : ", f1_score(y_test, y_pred, average="macro"))

#Affichage de la prédiction sur les données d'évaluation
eval_pred = eval[features].values
pred = clf.predict(eval_pred)
print(pred)

#Permet de calculer Alpha
path = clf.cost_complexity_pruning_path(X_train, y_train)
ccp_alphas = path.ccp_alphas
impurities = path.impurities
print(len(ccp_alphas))

#Créer un graphique nous montrant l'évolution du nombre de noeuds en fonction de la valeur de Alpha
clfs = []
for alpha in ccp_alphas:
  clf=DecisionTreeClassifier(random_state=0, ccp_alpha=alpha)
  clf.fit(X_train,y_train)
  clfs.append(clf)

node_counts = [clf.tree_.node_count for clf in clfs]
depth = [clf.tree_.max_depth for clf in clfs]
fig, ax = plt.subplots(2, 1, figsize=(12, 8))
ax[0].plot(ccp_alphas, node_counts)
ax[0].set_xlabel("alpha")
ax[0].set_ylabel("number of nodes")
ax[0].set_title("Number of nodes vs alpha")
ax[1].plot(ccp_alphas, depth)
ax[1].set_xlabel("alpha")
ax[1].set_ylabel("depth of tree")
ax[1].set_title("Depth vs alpha")
fig.tight_layout()

#Affichage d'un graphique nous montrant le recall en fonction de la valeur de Alpha
train_scores = [recall(clf, X_train, y_train) for clf in clfs]
test_scores = [recall(clf, X_test, y_test) for clf in clfs]
fig, ax = plt.subplots(figsize=(10,8))
ax.set_xlabel("alpha")
ax.set_ylabel("Recall")
ax.set_title("Recall vs alpha for training and testing sets")
ax.plot(ccp_alphas, train_scores, marker='x', label="train", drawstyle="steps-post", linestyle='None')
ax.plot(ccp_alphas, test_scores, marker='x', label="test", drawstyle="steps-post", linestyle='None')
plt.axhline(y=0.87, color='r', linestyle='--')
ax.legend()
plt.show()

#Affiche la meilleure valeur de Alpha et le meilleur recall obtenu
best_alpha = ccp_alphas[np.argmax(test_scores)]
best_score = test_scores[np.argmax(test_scores)]
best_tree = clfs[np.argmax(test_scores)]
print(best_alpha, best_score)
