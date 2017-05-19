# coding: utf-8
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

from utils import *

# Get data
train = pd.read_csv('data/train_5_19_2.csv')
# Add some features in X_test
test=pd.read_csv('data/testbeta.csv')
test=get_clickDay(test)
test=get_clickTimeInDay(test)
test=get_province(test)
test=get_clickHistory(test)
test=get_app_click_cnt(test)
test.to_csv("data/test_19.csv", index=False)
train2=get_app_click_cnt(test)
train2.to_csv("data/train_19.csv", index=False)

# Drop some features
X_t=train2.drop(['label','userID','hometown','clickDay','conversionTime'],axis=1)
test=test.rename(columns={'appCategory':'appcategory'})
test=test.rename(columns={'camgaignID':'campaignID'})
test.to_csv("data/test_19.csv", index=False)
X_t=X_t.drop(['clickTime'], axis=1)
X_test=get_same_dim(X_t, test)
y_t=train2['label']
X=X_t.values
X_test=X_test.values
y=y_t.values

# train model
clf=XGBClassifier()
clf.fit(X,y)
prob=clf.predict_proba(X_test)
prob=prob[0::,1]
save_result(prob)
