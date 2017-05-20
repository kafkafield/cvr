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

# prob vector -> submission.csv
def save_result(prob):
    length = len(prob)
    instanceID = np.arange(length) + 1
    submit = pd.DataFrame({'instanceID':instanceID,'prob':prob})
    submit.to_csv('submission.csv', index = False)


def predict_with_loss(clf, X_test, y_test):
    y = clf.predict_proba(X_test)
    print log_loss(y_test, y)
    return y[0::,1]


def get_clickTimeInDay(data):
    data['clickTimeInDay'] = np.floor(data['clickTime'] % 10000)
    return data


def get_clickDay(data):
    data['clickDay'] = np.floor(data['clickTime'] / 10000)
    return data


def get_provinceOfHometown(data):
    data['provinceOfHometown'] = np.floor(data['hometown'] / 100)
    return data


def get_provinceOfResidence(data):
    data['provinceOfResidence'] = np.floor(data['residence'] / 100)
    return data


def get_app_click_cnt(data):
    artiUser = pd.read_csv('data/artiUser.csv')
    data = data.join(artiUser.set_index('userID'), on = 'userID')
    data = data.rename(columns = {'clickHistory':'clickOfUserCnt'})
    data = data.rename(columns = {'appNewlyCnt':'appUserNewCnt'})
    data = data.rename(columns = {'appHistoryCnt':'appUserPastCnt'})
    data['appUserPastCnt'] = data['appUserPastCnt'].fillna(0)
    data['appUserNewCnt'] = data['appUserNewCnt'].fillna(0)
    data['clickOfUserCnt'] = data['clickOfUserCnt'].fillna(0)
    return data


def get_clickOfDayCnt_train(train):
    clickdaycnt = pd.read_csv('data/clickDayCnt.csv')
    train = train.join(clickdaycnt, on = 'clickDay')
    return train


def get_clickOfDayCnt_test(test):
    ctest = {'clickDay':[31], 'clickOfDayCnt':[test.shape[0]]} 
    clickDayCntTest = pd.DataFrame(ctest)
    test = test.join(clickDayCntTest.set_index('clickDay'), on = 'clickDay')
    return test


def get_same_dim(train, test):
    index = train.columns
    return test[index]
