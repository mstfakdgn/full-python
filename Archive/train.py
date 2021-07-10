# -*- coding: utf-8 -*-
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
# import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import roc_auc_score, roc_curve
from scipy.stats import ttest_ind, wilcoxon, shapiro, mannwhitneyu, f_oneway
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
import sklearn.tree as tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler
from sksurv.svm import FastKernelSurvivalSVM, FastSurvivalSVM, HingeLossSurvivalSVM, NaiveSurvivalSVM
from sksurv.util import Surv
from sklearn.model_selection import train_test_split
from sklearn import feature_extraction, feature_selection
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_squared_log_error, median_absolute_error
from functions import memory_management
from pickle import dump


"""
    train.py dosyası algoritmaları train etmek için hazırlanmıştır.
    2 parçadan oluşmaktadır: Bilinen Algoritmalar & Survival Algoritmaları
    1. Bilinen Algoritmalar:
        İlk satırlarda yer alan kısım train verisini okuma, memory optimizasyonunu yapma ve giriş ve çıkışları ayırmadır.
        Aşağıda devam eden bölümlerde algoritmaları eğitme kısmı yer almaktadır.
    2. Survival Algorithms:
        İlk satırlarda yer alan kısım train verisini okuma, memory optimizasyonunu yapma ve giriş ve çıkışları ayırmadır.
        Veriyi düzenleme kısmı Bilinen Algoritmalara göre biraz farklıdır.
        Aşağıda devam eden bölümlerde algoritmaları eğitme kısmı yer almaktadır.
"""



################################################################################################################
##### Bilinen Algoritmalar
##############
train = pd.read_csv('improve_training_CustomerChurn-17205389.csv', sep=',')
print(len(train.columns))
train = memory_management(train, continuous_columns=train.columns)
X_train = train.loc[:, train.columns != 'churn']
y_train = train['churn']

##############
##### LOGREG
##############
# param_grid = {'C': np.arange(0.0001, 0.01, 0.0001),
#              'penalty': ['l1', 'l2']}
# logreg = LogisticRegression()
# logreg_cv = GridSearchCV(logreg, param_grid, cv=5)
# logreg_cv.fit(X_train, y_train)
# print(logreg_cv.best_params_)
# print("best score", logreg_cv.best_score_)
# logreg = LogisticRegression(C=logreg_cv.best_params_['C'], penalty=logreg_cv.best_params_['penalty'])
# logreg.fit(X_train, y_train)
# logreg_acc_score = round(logreg.score(X_train, y_train) * 100, 2)
# print("Logreg Score: ", logreg.score(X_train, y_train))
# print("***Logistic Regression***")
# print("Accuracy Score:", logreg_acc_score)
# dump(logreg, open('logreg.model', 'wb'))


##############
##### SVM
##############
# param_grid = {'C': np.arange(0.0001, 0.01, 0.0001),
#              'gamma': [1e-3, 1e-4]}
# svm = SVC()
# svm_cv = GridSearchCV(svm, param_grid, cv=5)
# svm_cv.fit(X_train, y_train)
# print(svm_cv.best_params_)
# print("best score", svm_cv.best_score_)
# svm = SVC(C=svm_cv.best_params_['C'], gamma=svm_cv.best_params_['gamma'], probability=True)
# a=svm.fit(X_train, y_train)
# svm_acc_score = round(svm.score(X_train, y_train) * 100, 2)
# print("***SVM***")
# print("Accuracy Score:", svm_acc_score)
# print("SVM Score: ", svm.score(X_train, y_train))
# dump(svm, open('svm.model', 'wb'))


##############
##### Decision Tree
##############
# param_grid = {'max_depth': np.arange(1, 100)}
# decision_tree = DecisionTreeClassifier()
# decision_tree_cv = GridSearchCV(decision_tree, param_grid, cv=5)
# decision_tree_cv.fit(X_train, y_train)
# print(decision_tree_cv.best_params_)
# print("best score", decision_tree_cv.best_score_)
# decision_tree = DecisionTreeClassifier(max_depth=decision_tree_cv.best_params_['max_depth'])
# decision_tree.fit(X_train, y_train)
# decision_tree_acc_score = round(decision_tree.score(X_train, y_train) * 100, 2)
# print("***Decision Tree***")
# # Mean accuracy score
# print("Decision tree score:", decision_tree.score(X_train, y_train))
# print("Accuracy Score:", decision_tree_acc_score)
# # Viewer: https://dreampuf.github.io/GraphvizOnline/
# # tree.export_graphviz(decision_tree, out_file='dot_data_decision_tree.dot', feature_names=[u'{}'.format(c) for c in train[train.loc[:, train.columns != 'churn']]], class_names=[str(x) for x in decision_tree.classes_], rounded=True, special_characters=True)
# dump(decision_tree, open('decisiontree.model', 'wb'))


##############
##### Random Forest
##############
# param_grid = {'n_estimators': np.arange(1, 100)}
# random_forest = RandomForestClassifier()
# random_forest_cv = GridSearchCV(random_forest, param_grid, cv=5)
# random_forest_cv.fit(X_train, y_train)
# print(random_forest_cv.best_params_)
# print("best score", random_forest_cv.best_score_)
# random_forest = RandomForestClassifier(n_estimators=random_forest_cv.best_params_['n_estimators'], max_depth=6, max_features="sqrt")
# random_forest.fit(X_train, y_train)
# random_forest.score(X_train, y_train)
# random_forest_acc_score = round(random_forest.score(X_train, y_train) * 100, 2)
# print("***Random Forest***")
# print("Accuracy Score:", random_forest_acc_score)
# print("Random Forest Score:", random_forest.score(X_train, y_train))
# tree.export_graphviz(random_forest.estimators_[0], out_file='dot_data_random_forest.dot', feature_names=[u'{}'.format(c) for c in train[set(X_train.columns)-{'churn'}]], rounded=True, special_characters=True)
# dump(random_forest, open('randomforest.model', 'wb'))


##############
##### KNN
##############
# param_grid = {'n_neighbors': np.arange(5, 50)}
# knn = KNeighborsClassifier()
# knn_cv = GridSearchCV(knn, param_grid, cv=5)
# knn_cv.fit(X_train, y_train)
# print(knn_cv.best_params_)
# print("best score", knn_cv.best_score_)
# knn = KNeighborsClassifier(n_neighbors=knn_cv.best_params_['n_neighbors'])
# knn.fit(X_train, y_train)
# knn_acc_score = round(knn.score(X_train, y_train) * 100, 2)
# print("***K Nearest Neighbors***")
# print("Accuracy Score:", knn_acc_score)
# print("KNN Score:", knn.score(X_train, y_train))
# dump(knn, open('knn.model', 'wb'))


################################################################################################################
##### Survival Algoritmaları
##############
train = pd.read_csv('improve_training_CustomerChurn-17205389.csv', sep=',')
print(len(train.columns))
train = memory_management(train, continuous_columns=train.columns)
X_train = train.drop(['churn', 'lifeTime'], axis=1)
y_train = Surv.from_dataframe(event='churn', time='lifeTime', data=train)


#############
#### SVM
#############
param_grid = {'alpha': 2. ** np.arange(-10, 11, 4)}
svm = FastSurvivalSVM()
svm_cv = GridSearchCV(svm, param_grid, cv=5)
svm_cv.fit(X_train, y_train)
print(svm_cv.best_params_)
print("best score", svm_cv.best_score_)
svm = FastSurvivalSVM(alpha=svm_cv.best_params_['alpha'], optimizer='rbtree', tol=1e-6)
svm.fit(X_train, y_train)
svm_acc_score = round(svm.score(X_train, y_train) * 100, 2)
print("***Survival SVM***")
print("Accuracy Score:", svm_acc_score)
print("Survival SVM Score:", svm.score(X_train, y_train))
y_pred = svm.predict(X_train)
roc_score = roc_auc_score(y_train, y_pred)
print(roc_score)
dump(svm, open('svm_surv.model', 'wb'))



##############
##### SVM 2
##############
param_grid = {'alpha': 2. ** np.arange(-10, 11, 4)}
svm = FastKernelSurvivalSVM()
svm_cv = GridSearchCV(svm, param_grid, cv=5)
svm_cv.fit(X_train, y_train)
print(svm_cv.best_params_)
print("best score", svm_cv.best_score_)
svm = FastKernelSurvivalSVM(alpha=svm_cv.best_params_['alpha'], optimizer='rbtree', tol=1e-2)
svm.fit(X_train, y_train)
svm_acc_score = round(svm.score(X_train, y_train) * 100, 2)
print("***Survival SVM***")
print("Accuracy Score:", svm_acc_score)
print("Survival SVM Score:", svm.score(X_train, y_train))
y_pred = svm.predict(X_train)
roc_score = roc_auc_score(y_train, y_pred)
print(roc_score)
dump(svm, open('svm_surv2.model', 'wb'))
# kenan.kurt@tesodev.com -> 5428224923
# alpkan.cicek@tesodev.com -> 5433370090

##############
##### Hinge SVM
##############
# param_grid = {'alpha': 2. ** np.arange(-10, 11, 4)}
# svm = HingeLossSurvivalSVM()
# svm_cv = GridSearchCV(svm, param_grid, cv=5)
# svm_cv.fit(X_train, y_train)
# print(svm_cv.best_params_)
# print("best score", svm_cv.best_score_)
# svm = HingeLossSurvivalSVM(alpha=svm_cv.best_params_['alpha'])
# svm.fit(X_train, y_train)
# svm_acc_score = round(svm.score(X_train, y_train) * 100, 2)
# print("***Naive SVM***")
# print("Accuracy Score:", svm_acc_score)
# print("Naive SVM Score:", svm.score(X_train, y_train))
# dump(svm, open('naivesvm_surv.model', 'wb'))

