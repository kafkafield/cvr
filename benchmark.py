import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, log_loss
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

classifiers = [
    KNeighborsClassifier(3),
    # SVC(probability=True),
    DecisionTreeClassifier(),
    RandomForestClassifier(),
        AdaBoostClassifier(),
    GradientBoostingClassifier(),
    GaussianNB(),
    LinearDiscriminantAnalysis(),
    QuadraticDiscriminantAnalysis(),
    LogisticRegression()
    XGBClassifier()]

log_cols = ["Classifier", "Loss"]
log      = pd.DataFrame(columns=log_cols)

sss = StratifiedShuffleSplit(n_splits=2, test_size=0.1, random_state=0)

X = np.load('X_train')
y = np.load('Y_train')

acc_dict = {}
count = 0

for train_index, test_index in sss.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        print count
        count += 1

        for clf in classifiers:
                name = clf.__class__.__name__
                print name
                clf.fit(X_train, y_train)
                train_predictions = clf.predict_proba(X_test)
                acc = log_loss(y_test, train_predictions)
                if name in acc_dict:
                        acc_dict[name] += acc
                else:
                        acc_dict[name] = acc

for clf in acc_dict:
        acc_dict[clf] = acc_dict[clf] / 2.0
        log_entry = pd.DataFrame([[clf, acc_dict[clf]]], columns=log_cols)
        log = log.append(log_entry)\

print acc_dict

plt.xlabel('Accuracy')
plt.title('Classifier Accuracy')

sns.set_color_codes("muted")
sns.barplot(x='Accuracy', y='Classifier', data=log, color="b")
