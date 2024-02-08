from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score,roc_curve,auc, recall_score, classification_report, confusion_matrix
import pandas as pd
from sklearn import metrics
import matplotlib.pyplot as plt
from imblearn.metrics import specificity_score


df_pred=pd.read_csv("fuse_probability_all_features (1).csv",usecols=[5])  #pred label

df_true=pd.read_csv("fuse_probability_all_features (1).csv",usecols=[4])  #true label

#confusion_matrix = metrics.confusion_matrix(df_true, df_pred)

#cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = [False, True])

# cm_display.plot()
#
#
# fpr, tpr, thresholds = metrics.roc_curve(df_true, df_pred, pos_label=1)
# print(metrics.auc(fpr, tpr))
print(f1_score(df_true, df_pred))
print(precision_score(df_true, df_pred))
print(recall_score(df_true, df_pred))
print(accuracy_score(df_true, df_pred))
# auc = metrics.roc_auc_score(df_true, df_pred)
# print(auc)
print("kkkk")
print(specificity_score(df_true,df_pred))
#


import statistics

# creating a simple data - set
sample = [0.841, 0.832, 0.829,0.849]

# Prints standard deviation
# xbar is set to default value of 1
print("Standard Deviation of sample is % s "
      % (statistics.stdev(sample)))


