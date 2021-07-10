# -*- coding: utf-8 -*-
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.axes._axes import _log as matplotlib_axes_logger
matplotlib_axes_logger.setLevel('ERROR')
import seaborn as sns
# import statsmodels.formula.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics, feature_extraction, feature_selection
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from functions import memory_management, bool_to_binary



# df = pd.read_csv('adjustedData_CustomerChurn-17205389.csv', sep=',', index_col=0)
# df['churn'].replace(['true'], 1, inplace=True)
# df['churn'].replace(['false'], 0, inplace=True)
#
# df = df.sample(frac=1)
# print(df)
#
# train_df, test_df = train_test_split(df, test_size=0.3)
#
# train_df.to_csv('training_CustomerChurn-17205389.csv', sep=',', index=False)
# test_df.to_csv('test_CustomerChurn-17205389.csv', sep=',', index=False)
# df.to_csv('shuffled_CustomerChurn-17205389.csv', sep=',', index=False)
#
# ###########
# ###########
train_df = pd.read_csv('training_CustomerChurn-17205389.csv', sep=',')
test_df = pd.read_csv('test_CustomerChurn-17205389.csv', sep=',')
full_df = pd.read_csv('shuffled_CustomerChurn-17205389.csv', sep=',')
categorical_columns = train_df[['income', 'regionType','marriageStatus','children','smartPhone',
                          'creditRating','homeOwner','creditCard','churn']].columns

continuous_columns = train_df[['age','numHandsets','handsetAge','currentHandsetPrice','avgBill','avgMins',
                         'avgrecurringCharge','avgOverBundleMins','avgRoamCalls','callMinutesChangePct',
                         'billAmountChangePct','avgReceivedMins','avgOutCalls','avgInCalls','peakOffPeakRatio',
                         'peakOffPeakRatioChangePct','avgDroppedCalls','lifeTime','lastMonthCustomerCareCalls',
                         'numRetentionCalls','numRetentionOffersAccepted','newFrequentNumbers']].columns

test_df = memory_management(test_df, categorical_columns=categorical_columns, continuous_columns=continuous_columns)
train_df = memory_management(train_df, categorical_columns=categorical_columns, continuous_columns=continuous_columns)
full_df = memory_management(full_df, categorical_columns=categorical_columns, continuous_columns=continuous_columns)


#
#
# ##############
sns.set(style="white")
f, ax = plt.subplots(figsize=(20, 20))
#
#
# ##############
corr = train_df[continuous_columns].corr()

mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

cmap = sns.diverging_palette(225, 0, as_cmap=True)

sns.heatmap(corr, annot=True, mask=mask, cmap=cmap, vmax=1, vmin=-1,
            square=True, xticklabels=True, yticklabels=True,
            linewidths=0.5, cbar_kws={"shrink": 0.5}, ax=ax)
plt.yticks(rotation=0)
plt.xticks(rotation=90)
plt.show()
print(train_df.shape)
print(train_df.isna().sum())
#
#
# ##############
print(train_df.groupby('numRetentionCalls')['numRetentionCalls'].apply(lambda x: x.count()))
print(train_df.groupby('numRetentionOffersAccepted')['numRetentionOffersAccepted'].apply(lambda x: x.count()))

# ##############
pp = PdfPages('continuous_boxScatterCharts_17205389.pdf')
flierprops = dict(marker='o', markerfacecolor='green', markersize=6,
                  linestyle='none')
for i in continuous_columns:
    bp = train_df.boxplot(column=[i], by=['churn'], flierprops=flierprops, figsize=(10,7))
    pp.savefig(f.get_figure())

for c in continuous_columns:
    train_df.plot(kind='scatter', x=c , y='churn', label="%.3f" % train_df[[c, 'churn']].corr().values[0, 1])
    pp.savefig(f.get_figure())
pp.close()
#
#
# ##############
print(train_df.groupby('churn').describe()['handsetAge'].T)
print(train_df.groupby('churn').describe()['currentHandsetPrice'].T)
print(train_df.groupby('currentHandsetPrice')['currentHandsetPrice'].apply(lambda x: x.count()))
print(train_df.groupby('churn').describe()['avgOverBundleMins'].T)
print(train_df.groupby('churn').describe()['billAmountChangePct'].T)
#
#
# ##############
# print(full_df.groupby(by='income').count())
# chart_columns = categorical_columns.drop(['churn'],1)
# pp = PdfPages('categorical_stackedBarCharts_17205389.pdf')
# for i in chart_columns:
#     feature = pd.unique(train_df[i].ravel())
#     train_df['percent'] = 0
#
#     for p in feature:
#         count = 1 / train_df[train_df[i] == p].count()['churn']
#         index_list = train_df[train_df[i] == p].index.tolist()
#         for j in index_list:
#             train_df.loc[j, 'percent'] = count * 100
#
#     group = train_df[['percent',i,'churn']].groupby([i,'churn']).sum()
#
#     my_plot = group.unstack().plot(kind='bar', stacked=True, title= "Churn for " + i, figsize=(15,7))
#
#     green_key = mpatches.Patch(color='orange', label='Churned')
#     blue_key = mpatches.Patch(color='blue', label='Didn\'t churn')
#     my_plot.legend(handles=[green_key, blue_key], frameon = True)
#
#     my_plot.set_xlabel(i)
#     my_plot.set_ylabel("% Churn")
#     my_plot.set_ylim([0,100])
#     pp.savefig(f.get_figure())
# pp.close()
#
#
# ##############
# print(train_df.groupby('churn')['income'].describe())
# print(train_df.groupby('income')['income'].apply(lambda x: x.count()))
# print(train_df.groupby('regionType')['regionType'].apply(lambda x: x.count()))
# print(train_df.isnull().sum()['regionType'])
# print(train_df.groupby('creditCard')['creditCard'].apply(lambda x: x.count()))
#
#
# ##############
# lm_train = sm.ols(formula="churn ~ avgOverBundleMins + handsetAge + income + smartPhone", data=train_df).fit()
# print(lm_train.params)
# print(lm_train.summary())
#
# ##############
# predict_df = lm_train.predict(train_df)
# print(predict_df.head(20))
#
# ChurnClass = (predict_df  > 0.5) * 1
# df_ChurnClass = pd.DataFrame({'ChurnClass': ChurnClass})
# print(df_ChurnClass.head(20))
#
# train_y = train_df.churn
# train_predictions = df_ChurnClass
# print("Accuracy: ", metrics.accuracy_score(train_y, train_predictions))
# print("Confusion matrix:\n ", metrics.confusion_matrix(train_y, train_predictions))
# print("Classification report:\n ", metrics.classification_report(train_y, train_predictions))
# mae = abs(train_y - predict_df).mean()
# print("\nMean Absolute Error:\n", mae)
# mse = ((train_y - predict_df)** 2).mean()
# print("\nMean Squared Error:\n", mse)
#
# ##############
# ###### In [35 - 39]
# test_predict_df = lm_train.predict(test_df)
#
# TestChurnClass = (test_predict_df  > 0.5) * 1
# df_TestChurnClass = pd.DataFrame({'TestChurnClass': TestChurnClass})
# y_test = test_df.churn
# predictions_test = df_TestChurnClass
# print("Accuracy: ", metrics.accuracy_score(y_test, predictions_test))
# print("Confusion matrix:\n ", metrics.confusion_matrix(y_test, predictions_test))
# print("Classification report:\n ", metrics.classification_report(y_test, predictions_test))
# mae = abs(y_test - test_predict_df).mean()
# print("\nMean Absolute Error:\n", mae)
# mse = ((y_test - test_predict_df)** 2).mean()
# print("\nMean Squared Error:\n", mse)
#
# ###############
# ##### In [40-42]
# split_full_df = np.split(full_df, 5, axis=0)
# crossval_test1 = split_full_df[0]
# crossval_test2 = split_full_df[1]
# crossval_test3 = split_full_df[2]
# crossval_test4 = split_full_df[3]
# crossval_test5 = split_full_df[4]
# crossval_train1 = pd.concat([split_full_df[1],split_full_df[2],split_full_df[3],split_full_df[4]], axis=0, join='outer', join_axes=None, ignore_index=False,
#           keys=None, levels=None, names=None, verify_integrity=False,
#           copy=True)
# crossval_train2 = pd.concat([split_full_df[0],split_full_df[2],split_full_df[3],split_full_df[4]], axis=0, join='outer', join_axes=None, ignore_index=False,
#           keys=None, levels=None, names=None, verify_integrity=False,
#           copy=True)
# crossval_train3 = pd.concat([split_full_df[1],split_full_df[0],split_full_df[3],split_full_df[4]], axis=0, join='outer', join_axes=None, ignore_index=False,
#           keys=None, levels=None, names=None, verify_integrity=False,
#           copy=True)
# crossval_train4 = pd.concat([split_full_df[1],split_full_df[2],split_full_df[0],split_full_df[4]], axis=0, join='outer', join_axes=None, ignore_index=False,
#           keys=None, levels=None, names=None, verify_integrity=False,
#           copy=True)
# crossval_train5 = pd.concat([split_full_df[1],split_full_df[2],split_full_df[3],split_full_df[0]], axis=0, join='outer', join_axes=None, ignore_index=False,
#           keys=None, levels=None, names=None, verify_integrity=False,
#           copy=True)
#
# crossval_train_list = [crossval_train1, crossval_train2, crossval_train3, crossval_train4, crossval_train5]
# crossval_test_list = [crossval_test1, crossval_test2, crossval_test3, crossval_test4, crossval_test5]
#
# for i in range (0,5):
#     crossval_train = crossval_train_list[i]
#     crossval_test = crossval_test_list[i]
#     crossval_lm = sm.ols(formula="churn ~ avgOverBundleMins + handsetAge + income + smartPhone", data=crossval_train).fit()
#     test_predict = crossval_lm.predict(crossval_test)
#     test_predict_class = (test_predict>0.5) *1.0
#     df_test_predict_class = pd.DataFrame({'test_predict_class': test_predict_class})
#     y = crossval_test['churn']
#     print("Test", i+1, "Classification Measurements\n")
#     print("Accuracy: ", metrics.accuracy_score(y,df_test_predict_class))
#     print("Confusion matrix: \n", metrics.confusion_matrix(y, df_test_predict_class))
#     print("Classification report:\n ", metrics.classification_report(y, df_test_predict_class))
#     mse = ((crossval_test.churn - test_predict)** 2).mean()
#     print("\nMean Squared Error:\n", mse)
#     mae = abs(crossval_test.churn - test_predict).mean()
#     print("\nMean Absolute Error:\n", mae, "\n")
#
#
# ################
# ####### In[43-47]
# intercept = pd.DataFrame({'Intercept': np.ones(700)})
#
# train_X = pd.concat([intercept, train_df[['avgOverBundleMins', 'handsetAge', 'income', 'smartPhone']]], axis=1)
# train_y = train_df.churn
#
# logreg = LogisticRegression().fit(train_X, train_y)
# print("Coeficients: \n", logreg.coef_)
# print(logreg.predict_proba(train_X[:100]))
# log_train_predictions = logreg.predict(train_X)
# print("Predictions: ", log_train_predictions[:100])
# print("Accuracy: ", metrics.accuracy_score(train_y, log_train_predictions))
# print("Confusion matrix: \n", metrics.confusion_matrix(train_y, log_train_predictions))
# print("Classification report:\n ", metrics.classification_report(train_y, log_train_predictions))
#
#
# ################
# intercept = pd.DataFrame({'Intercept': np.ones(300)})
#
# test_X = pd.concat([intercept, test_df[['avgOverBundleMins', 'handsetAge', 'income', 'smartPhone']]], axis=1)
# test_y = test_df.churn
#
# logreg = LogisticRegression().fit(test_X, test_y)
# print("Coeficients: \n", logreg.coef_)
# test_predictions = logreg.predict(test_X)
# print("Predictions (class): \n", test_predictions)
# print("Accuracy: ", metrics.accuracy_score(test_y, test_predictions))
# print("Confusion matrix: \n", metrics.confusion_matrix(test_y, test_predictions))
# print("Classification report:\n ", metrics.classification_report(test_y, test_predictions))
#
#
# ################
# ###### In[51-55]
# # crossval_intercept = pd.DataFrame({'Intercept': np.ones(1000)})
# # crossval_X = pd.concat([crossval_intercept, full_df[['avgOverBundleMins', 'handsetAge', 'income', 'smartPhone']]], axis=1)
# # crossval_y = full_df.churn
# #
# # precision_scores = cross_val_score(LogisticRegression(), crossval_X, crossval_y, scoring='precision', cv=5)
# # print(precision_scores)
# # print(precision_scores.mean())
# # precision_recall = cross_val_score(LogisticRegression(), crossval_X, crossval_y, scoring='recall', cv=5)
# # print(precision_recall)
# # print(precision_recall.mean())
# # f1_scores = cross_val_score(LogisticRegression(), crossval_X, crossval_y, scoring='f1', cv=5)
# # print(f1_scores)
# # print(f1_scores.mean())
# # accuracy_scores = cross_val_score(LogisticRegression(), crossval_X, crossval_y, scoring='accuracy', cv=5)
# # print(accuracy_scores)
# # print(accuracy_scores.mean())
#
#
# ################
# ##### In[57]
# print(full_df.groupby('churn')['churn'].apply(lambda x: x.count()))
# full_predict_df = lm_train.predict(full_df)
# FullChurnClass = (full_predict_df > 0.5) * 1
# df_FullChurnClass = pd.DataFrame({'FullChurnClass': FullChurnClass})
# y_full = full_df.churn
# print("Accuracy: ", metrics.accuracy_score(y_full, df_FullChurnClass))
#
# ################
print(full_df.isnull().sum())
full_df['age']=full_df['age'].replace(np.nan, 0)
temp_df = full_df[full_df['age'] == 0]
temp_df = temp_df[temp_df['children'] == True]
print(temp_df[['age', 'children']])

temp_df = temp_df[temp_df['homeOwner'] == True]
print(temp_df.head(5)[['age', 'homeOwner']])
print(full_df['age'].median())
full_df['age'] = full_df['age'].replace(0, full_df['age'].median())

print(full_df[full_df['currentHandsetPrice'] == 0]['currentHandsetPrice'].count())
print(full_df[full_df['currentHandsetPrice'] == 0][full_df['avgMins'] == 0][['currentHandsetPrice', 'avgBill', 'avgMins']])
full_df = full_df.drop('currentHandsetPrice', axis=1)

full_df.groupby('regionType')['regionType'].apply(lambda x: x.count())
full_df = full_df.drop('regionType', axis=1)

full_df = full_df.drop('marriageStatus', axis=1)

full_df = full_df.drop('customer', axis=1)
print(full_df)


feature = pd.unique(train_df['creditRating'].ravel())
train_df['percent'] = 0

for p in feature:
    count = 1 / train_df[train_df['creditRating'] == p].count()['churn']
    index_list = train_df[train_df['creditRating'] == p].index.tolist()
    for j in index_list:
        train_df.loc[j, 'percent'] = count * 100

group = train_df[['percent','creditRating','churn']].groupby(['creditRating','churn']).sum()

my_plot = group.unstack().plot(kind='bar', stacked=True, title= "Churn for " + 'creditRating', figsize=(15,7))

green_key = mpatches.Patch(color='orange', label='Churned')
blue_key = mpatches.Patch(color='blue', label='Didn\'t churn')
my_plot.legend(handles=[green_key, blue_key], frameon = True)

my_plot.set_xlabel('creditRating')
my_plot.set_ylabel("% Churn")
my_plot.set_ylim([0,100])
plt.savefig('creditRating.png')


################
full_df = pd.concat([full_df, pd.get_dummies(full_df['creditRating'], prefix='CR', prefix_sep='_')], axis=1)
full_df.drop('creditRating', 1, inplace=True)

full_df = pd.concat([full_df, pd.get_dummies(full_df['income'], prefix='Inc', prefix_sep='_')], axis=1)
full_df.drop('income', 1, inplace=True)
print(full_df)

remaining_categorical_columns = full_df[['children','smartPhone','homeOwner','creditCard']].columns
full_df = bool_to_binary(full_df, continuous_columns=remaining_categorical_columns)

cont_columns = full_df[['age','numHandsets','handsetAge','avgBill','avgMins',
                         'avgrecurringCharge','avgOverBundleMins','avgRoamCalls','callMinutesChangePct',
                         'billAmountChangePct','avgReceivedMins','avgOutCalls','avgInCalls','peakOffPeakRatio',
                         'peakOffPeakRatioChangePct','avgDroppedCalls','lifeTime','lastMonthCustomerCareCalls',
                         'numRetentionCalls','numRetentionOffersAccepted','newFrequentNumbers']].columns

scaler = MinMaxScaler()
full_df[cont_columns] = scaler.fit_transform(full_df[cont_columns])
full_df = memory_management(full_df, continuous_columns=cont_columns)
full_df.to_csv('postFeatureAdjustments_CustomerChurn-17205389.csv', sep=',', index=False)
train1, test1 = train_test_split(full_df, test_size=0.3)
train1.to_csv('improve_training_CustomerChurn-17205389.csv', sep=',', index=False)
test1.to_csv('improve_test_CustomerChurn-17205389.csv', sep=',', index=False)

#
#
# #############
# #############
# ##### In[77-85]
# improved_df = pd.read_csv('postFeatureAdjustments_CustomerChurn-17205389.csv', sep=',')
# improved_df = memory_management(improved_df, continuous_columns=improved_df.columns)
# train1_df = pd.read_csv('improve_training_CustomerChurn-17205389.csv', sep=',')
# train1_df = memory_management(train1_df, continuous_columns=train1_df.columns)
# lm_train1 = sm.ols(formula="churn ~ avgOverBundleMins + handsetAge + smartPhone + Inc_0 + Inc_1 + Inc_2 + Inc_3 + Inc_4 + Inc_5 + Inc_6 + Inc_7 + Inc_8 + Inc_9", data=train1_df).fit()
#
# train1_predict_df = lm_train1.predict(train1_df)
# train1_ChurnClass = (train1_predict_df  > 0.5) * 1
# df_train1_ChurnClass = pd.DataFrame({'ChurnClass': train1_ChurnClass})
# train1_y = train1_df.churn
# train1_predictions = df_train1_ChurnClass
#
# print("Accuracy: ", metrics.accuracy_score(train1_y, train1_predictions))
# print("Confusion matrix:\n ", metrics.confusion_matrix(train1_y, train1_predictions))
# print("Classification report:\n ", metrics.classification_report(train1_y, train1_predictions))
# # Print the Mean Absolute Error of the model on the training set
# mae = abs(train1_y - train1_predict_df).mean()
# print("\nMean Absolute Error:\n", mae)
# # Print the Mean Squared Error of the model on the training set
# mse = ((train1_y - train1_predict_df)** 2).mean()
# print("\nMean Squared Error:\n", mse)
#
# test1_df = pd.read_csv('improve_test_CustomerChurn-17205389.csv', sep=',')
# test1_df = memory_management(test1_df, continuous_columns=test1_df.columns)
#
# test1_predict_df = lm_train1.predict(test1_df)
# test1_ChurnClass = (test1_predict_df  > 0.5) * 1
# df_test1_ChurnClass = pd.DataFrame({'ChurnClass': test1_ChurnClass})
# test1_y = test1_df.churn
# test1_predictions = df_test1_ChurnClass
#
# print("Accuracy: ", metrics.accuracy_score(test1_y, test1_predictions))
# print("Confusion matrix:\n ", metrics.confusion_matrix(test1_y, test1_predictions))
# print("Classification report:\n ", metrics.classification_report(test1_y, test1_predictions))
#
# mae = abs(test1_y - test1_predict_df).mean()
# print("\nMean Absolute Error:\n", mae)
# mse = ((test1_y - test1_predict_df)** 2).mean()
# print("\nMean Squared Error:\n", mse)
#
# split_improve_df = np.split(improved_df, 5, axis=0)
# crossvalX_test1 = split_improve_df[0]
# crossvalX_test2 = split_improve_df[1]
# crossvalX_test3 = split_improve_df[2]
# crossvalX_test4 = split_improve_df[3]
# crossvalX_test5 = split_improve_df[4]
# crossvalX_train1 = pd.concat([split_improve_df[1],split_improve_df[2],split_improve_df[3],split_improve_df[4]], axis=0, join='outer', join_axes=None, ignore_index=False,
#           keys=None, levels=None, names=None, verify_integrity=False,
#           copy=True)
# crossvalX_train2 = pd.concat([split_improve_df[0],split_improve_df[2],split_improve_df[3],split_improve_df[4]], axis=0, join='outer', join_axes=None, ignore_index=False,
#           keys=None, levels=None, names=None, verify_integrity=False,
#           copy=True)
# crossvalX_train3 = pd.concat([split_improve_df[1],split_improve_df[0],split_improve_df[3],split_improve_df[4]], axis=0, join='outer', join_axes=None, ignore_index=False,
#           keys=None, levels=None, names=None, verify_integrity=False,
#           copy=True)
# crossvalX_train4 = pd.concat([split_improve_df[1],split_improve_df[2],split_improve_df[0],split_improve_df[4]], axis=0, join='outer', join_axes=None, ignore_index=False,
#           keys=None, levels=None, names=None, verify_integrity=False,
#           copy=True)
# crossvalX_train5 = pd.concat([split_improve_df[1],split_improve_df[2],split_improve_df[3],split_improve_df[0]], axis=0, join='outer', join_axes=None, ignore_index=False,
#           keys=None, levels=None, names=None, verify_integrity=False,
#           copy=True)
#
# crossvalX_train_list = [crossvalX_train1, crossvalX_train2, crossvalX_train3, crossvalX_train4, crossvalX_train5]
# crossvalX_test_list = [crossvalX_test1, crossvalX_test2, crossvalX_test3, crossvalX_test4, crossvalX_test5]
#
# avg_accuracy = 0
# for i in range (0,5):
#     crossvalX_train = crossvalX_train_list[i]
#     crossvalX_test = crossvalX_test_list[i]
#     crossvalX_lm = sm.ols(formula="churn ~ avgOverBundleMins + handsetAge + smartPhone + Inc_0 + Inc_1 + Inc_2 + Inc_3 + Inc_4 + Inc_5 + Inc_6 + Inc_7 + Inc_8 + Inc_9", data=crossvalX_train).fit()
#     testX_predict = crossvalX_lm.predict(crossvalX_test)
#     testX_predict_class = (testX_predict>0.5) *1.0
#     df_testX_predict_class = pd.DataFrame({'testX_predict_class': testX_predict_class})
#     y = crossvalX_test['churn']
#     print("Test", i+1, "Classification Measurements\n")
#     print("Accuracy: ", metrics.accuracy_score(y,df_testX_predict_class))
#     avg_accuracy += metrics.accuracy_score(y,df_testX_predict_class)
#     print("Confusion matrix: \n", metrics.confusion_matrix(y, df_testX_predict_class))
#     print("Classification report:\n ", metrics.classification_report(y, df_testX_predict_class))
#     mse = ((crossvalX_test.churn - testX_predict)** 2).mean()
#     print("\nMean Squared Error:\n", mse)
#     mae = abs(crossvalX_test.churn - testX_predict).mean()
#     print("\nMean Absolute Error:\n", mae, "\n")
# print("Avg accuracy: ", avg_accuracy/5)
