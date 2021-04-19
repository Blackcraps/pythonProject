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

services = pd.read_csv('telecom_services.csv', delimiter = ';')
users = pd.read_csv('telecom_users.csv', delimiter = ';')
eval = pd.read_csv('telecom_users_eval.csv', delimiter = ';')

