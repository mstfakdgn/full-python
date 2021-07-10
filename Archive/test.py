# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
import warnings

from sksurv.metrics import cumulative_dynamic_auc

warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, classification_report
from sksurv.util import Surv
from functions import memory_management, pred_to_binary
from pickle import load


"""
    test.py dosyası algoritmaları test etmek için hazırlanmıştır.
    2 parçadan oluşmaktadır: Bilinen Algoritmalar & Survival Algoritmaları
    1. Bilinen Algoritmalar:
        İlk satırlarda yer alan kısım train verisini okuma, memory optimizasyonunu yapma ve giriş ve çıkışları ayırmadır.
        Aşağıda devam eden bölümlerde algoritmaları test kısmı ve sonuçları görüntüleme yer almaktadır.
    2. Survival Algorithms:
        İlk satırlarda yer alan kısım train verisini okuma, memory optimizasyonunu yapma ve giriş ve çıkışları ayırmadır.
        Veriyi düzenleme kısmı Bilinen Algoritmalara göre biraz farklıdır.
        Aşağıda devam eden bölümlerde algoritmaları test kısmı ve sonuçları görüntüleme yer almaktadır.
"""


################################################################################################################
##### Regular Algorithms
##############
# test = pd.read_csv('improve_test_CustomerChurn-17205389.csv', sep=',')
# print(len(test.columns))
# test = memory_management(test, continuous_columns=test.columns)
# X_test = test.loc[:, test.columns != 'churn']
# y_test = test['churn'].values


##############
##### LOGREG
##############
# logreg = load(open('logreg.model', 'rb'))
# y_pred = logreg.predict(X_test)
# # # print(y_pred)
# logreg_acc_score = round(logreg.score(X_test, y_test) * 100, 2)
# print("***LOGREG***")
# print("Accuracy Score:", logreg_acc_score)
# # print("Confusion Matrix:")
# # print(confusion_matrix(y_test, y_pred))
# # print("Classification Report:")
# # print(classification_report(y_test, y_pred))
# y_pred_prob = logreg.predict_proba(X_test)
# print("ROC_AUC Score:")
# roc_score = roc_auc_score(y_test, y_pred)
# print(roc_score)
# # print(logreg.coef_) # Denklemde featureların katsayıları
# # print(logreg.intercept_) # Sabit (intercept, a.k.a. bias) (a*x+b b sabit)
# # print(logreg.n_iter_) # İterasyon sayısı
# # print("MSE", np.mean((y_pred-y_test)**2))
# # print("RMSE", np.sqrt(np.mean((y_pred-y_test)**2)))
# fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob[:, 1])
# plt.figure()
# plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % roc_score)
# plt.plot([0, 1], [0, 1],'r--')
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('Roc Curve')
# plt.legend(loc="lower right")
# plt.savefig('logreg.png')


##############
##### SVM
##############
# svm = load(open('svm.model', 'rb'))
# y_pred = svm.predict(X_test)
# # print("Confusion Matrix:")
# # print(confusion_matrix(y_test, y_pred))
# svm_acc_score = round(svm.score(X_test, y_test) * 100, 2)
# print("***SVM***")
# print("Accuracy Score:", svm_acc_score)
# # print("SVM Score:", svm.score(X_test, y_test))
# # print("Classification Report:")
# # print(classification_report(y_test, y_pred))
# y_pred_prob = svm.predict_proba(X_test)[:, 1]
# print("ROC_AUC Score:")
# roc_score = roc_auc_score(y_test, y_pred)
# print(roc_score)
# # print(svm.support_vectors_)
# # print(svm.n_support_) # Her sınıf için support vektör sayısı
# # print(svm.intercept_) # Karar fonksiyonu sabiti
# # print(svm.fit_status_) # Doğru fit ettiyse 0, aksi halde 1 olur.
# # print(np.mean((y_pred-y_test)**2))
# fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
# plt.figure()
# plt.plot(fpr, tpr, label='SVM (area = %0.2f)' % roc_score)
# plt.plot([0, 1], [0, 1],'r--')
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('Roc Curve')
# plt.legend(loc="lower right")
# plt.savefig('svm.png')


##############
##### Decision Tree
##############
# decision_tree = load(open('decisiontree.model', 'rb'))
# y_pred = decision_tree.predict(X_test)
# decision_tree_acc_score = round(decision_tree.score(X_test, y_test) * 100, 2)
# print("***Decision Tree***")
# # # Mean accuracy score
# # print("Decision tree score:", decision_tree.score(X_test, y_test))
# print("Accuracy Score:", decision_tree_acc_score)
# # print("Confusion Matrix:")
# # print(confusion_matrix(y_test, y_pred))
# # print("Classification Report:")
# # print(classification_report(y_test, y_pred))
# y_pred_prob = decision_tree.predict_proba(X_test)[:,1]
# roc_score = roc_auc_score(y_test, y_pred)
# print("ROC_AUC Score:")
# print(roc_score)
# # print(decision_tree.classes_) # Sınıf isimleri
# # print(decision_tree.feature_importances_)# Return the feature importances.
# # print(decision_tree.max_features_) # The inferred value of max_features.
# # print(decision_tree.n_classes_) # Sınıf sayısı
# # print(decision_tree.n_features_) # Feature sayısı
# # print(decision_tree.n_outputs_) # Fit edildiğindeki çıkış sayısı (titanic'te 2 (0 ve 1))
# # print(decision_tree.tree_) # Ağaç objesini return eder.
# fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
# plt.figure()
# plt.plot(fpr, tpr, label='Decision Tree (area = %0.2f)' % roc_score)
# plt.plot([0, 1], [0, 1],'r--')
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('Roc Curve')
# plt.legend(loc="lower right")
# plt.savefig('decisiontree.png')
# # Viewer: https://dreampuf.github.io/GraphvizOnline/


##############
##### Random Forest
##############
# random_forest = load(open('randomforest.model', 'rb'))
# y_pred = random_forest.predict(X_test)
# # random_forest.score(X_test, y_test)
# random_forest_acc_score = round(random_forest.score(X_test, y_test) * 100, 2)
# print("***Random Forest***")
# print("Accuracy Score:", random_forest_acc_score)
# # # Mean accuracy score
# print("Random Forest Score:", random_forest.score(X_test, y_test))
# print("Confusion Matrix:")
# print(confusion_matrix(y_test, y_pred))
# print("Classification Report:")
# print(classification_report(y_test, y_pred))
# y_pred_prob = random_forest.predict_proba(X_test)[:, 1]
# roc_score = roc_auc_score(y_test, y_pred)
# print("ROC_AUC Score:")
# print(roc_score)
# # # print(random_forest.estimators_) # Fit eden tahmin ediciler topluluğu
# # print(random_forest.classes_) # Sınıf isimleri
# # print(random_forest.n_classes_) # Sınıf sayısı
# # print(random_forest.n_features_) # Feature sayısı
# # print(random_forest.n_outputs_) # Fit edildiğindeki çıkış sayısı (titanic'te 2 (0 ve 1))
# # print(random_forest.feature_importances_) # Feature Katsayıları. Yüksek=önemli
# fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
# plt.figure()
# plt.plot(fpr, tpr, label='Random Forest (area = %0.2f)' % roc_score)
# plt.plot([0, 1], [0, 1],'r--')
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('Roc Curve')
# plt.legend(loc="lower right")
# plt.savefig('randomforest.png')
# # tree.export_graphviz(random_forest.estimators_[0], out_file='dot_data_random_forest.dot', feature_names=[u'{}'.format(c) for c in ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'male', 'Q', 'S']], rounded=True, special_characters=True)


##############
##### KNN
##############
# knn = load(open('knn.model', 'rb'))
# y_pred = knn.predict(X_test)
# knn_acc_score = round(knn.score(X_test, y_test) * 100, 2)
# print("***K Nearest Neighbors***")
# print("Accuracy Score:", knn_acc_score)
# # print("KNN Score:", knn.score(X_test, y_test))
# # print("Confusion Matrix:")
# # print(confusion_matrix(y_test, y_pred))
# # print("Classification Report:")
# # print(classification_report(y_test, y_pred))
# y_pred_prob =knn.predict_proba(X_test)[:, 1]
# # print(y_pred)
# # print(knn.classes_)
# print("ROC_AUC Score:")
# roc_score = roc_auc_score(y_test, y_pred)
# print(roc_score)
# # print(knn.classes_) # Sınıflar
# # print(knn.n_neighbors) # Komşuluk sayısı
# # print(np.mean((y_pred-y_test)**2))
# fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
# plt.figure()
# plt.plot(fpr, tpr, label='KNN (area = %0.2f)' % roc_score)
# plt.plot([0, 1], [0, 1],'r--')
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('Roc Curve')
# plt.legend(loc="lower right")
# plt.savefig('knn.png')


################################################################################################################
##### Survival Algorithms
##############
test = pd.read_csv('improve_test_CustomerChurn-17205389.csv', sep=',')
# print(len(test.columns))
a = test.memory_usage().sum()
test = memory_management(test, continuous_columns=test.columns)
b = test.memory_usage().sum()
# print(a)
# print(b)
# print(((a-b)/a)*100)
X_test = test.drop(['churn', 'lifeTime'], axis=1)
y_test = Surv.from_dataframe(event='churn', time='lifeTime', data=test)
churn = test['churn']
train = pd.read_csv('improve_training_CustomerChurn-17205389.csv', sep=',')
# print(len(train.columns))
train = memory_management(train, continuous_columns=train.columns)
X_train = train.drop(['churn', 'lifeTime'], axis=1)
y_train = Surv.from_dataframe(event='churn', time='lifeTime', data=train)

##############
##### SVM
##############
svm = load(open('svm_surv.model', 'rb'))
y_pred = svm.predict(X_test)
print(y_pred)
print(y_pred)
y_pred_binary = pred_to_binary(y_pred)
knn_acc_score = round(svm.score(X_test, y_test) * 100, 2)
print("***SVM***")
print("Accuracy Score:", knn_acc_score)
print("SVM Score:", svm.score(X_test, y_test))
va_auc, va_mean_auc = cumulative_dynamic_auc(y_test, y_test, y_pred, np.arange(6, 50, 50.0/340))

plt.bar(y_test['lifeTime'], va_auc, width=0.7)

plt.axhline(va_mean_auc, linestyle="--")
plt.xlabel("days from enrollment")
plt.ylabel("time-dependent AUC")
plt.grid(True)
# plt.show()
plt.savefig('result.png')
print("AUC:", cumulative_dynamic_auc(y_train, y_test, y_pred, len(y_test)))
print("ROC_AUC Score:")
roc_score = roc_auc_score(churn, y_pred)
print(roc_score)
print("Confusion Matrix:")
print(confusion_matrix(churn, y_pred_binary))
print("Classification Report:")
print(classification_report(churn, y_pred_binary))
fpr, tpr, thresholds = roc_curve(churn, y_pred_binary)
plt.figure()
plt.plot(fpr, tpr, label='SVM (area = %0.2f)' % roc_score)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Roc Curve')
plt.legend(loc="lower right")
plt.show()


##############
##### SVM2
##############
# svm = load(open('svm_surv2.model', 'rb'))
# y_pred = svm.predict(X_test)
# print(y_pred)
# knn_acc_score = round(svm.score(X_test, y_test) * 100, 2)
# print("***SVM***")
# print("Accuracy Score:", knn_acc_score)
# print("SVM Score:", svm.score(X_test, y_test))
# y_pred_binary = list(map(pred_to_binary, churn))
# print("ROC_AUC Score:")
# roc_score = roc_auc_score(churn, y_pred_binary)
# print(roc_score)
# print("Confusion Matrix:")
# print(confusion_matrix(churn, y_pred_binary))
# print("Classification Report:")
# print(classification_report(churn, y_pred_binary))
# fpr, tpr, thresholds = roc_curve(churn, y_pred_binary)
# plt.figure()
# plt.plot(fpr, tpr, label='SVM (area = %0.2f)' % roc_score)
# plt.plot([0, 1], [0, 1],'r--')
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('Roc Curve')
# plt.legend(loc="lower right")
# plt.show()


##############
##### Hinge SVM
##############
# svm = load(open('naivesvm_surv.model', 'rb'))
# y_pred = svm.predict(X_test)
# print(y_pred)
# knn_acc_score = round(svm.score(X_test, y_test) * 100, 2)
# print("***SVM***")
# print("Accuracy Score:", knn_acc_score)
# print("SVM Score:", svm.score(X_test, y_test))
# y_pred_binary = list(map(pred_to_binary, churn))
# print("ROC_AUC Score:")
# roc_score = roc_auc_score(churn, y_pred_binary)
# print(roc_score)
# print("Confusion Matrix:")
# print(confusion_matrix(churn, y_pred_binary))
# print("Classification Report:")
# print(classification_report(churn, y_pred_binary))
# fpr, tpr, thresholds = roc_curve(churn, y_pred_binary)
# plt.figure()
# plt.plot(fpr, tpr, label='SVM (area = %0.2f)' % roc_score)
# plt.plot([0, 1], [0, 1],'r--')
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('Roc Curve')
# plt.legend(loc="lower right")
# plt.show()


# pca = PCA(.8)
# print(pca.fit(X_test))
# print(pca.n_components_, pca.n_features_)
# X_test = pca.transform(X_test)
# print(X_test)