#!/usr/bin/env python
# coding: utf-8

# # importing libraries

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# Reading the data

# In[2]:


# Read the data
df=pd.read_csv('C:/Users/user/Downloads/Fraud.csv')
# Shape the data
df.shape


# In[3]:


# Get head of the data
df.head(100)


# In[4]:


df.info()


# In[5]:


df.isna().sum()


# we have observed no null values.

# In[6]:


obj=df.select_dtypes(include="object").columns
print(obj)


# THERE ARE 3 ATTRIBUTES WITH Object Datatype. THUS WE NEED TO LABEL ENCODE THEM IN ORDER TO CHECK MULTICOLINEARITY.

# In[7]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

for feat in obj:
    df[feat] = le.fit_transform(df[feat].astype(str))

print (df.info())


# In[8]:


# Import library for VIF (VARIANCE INFLATION FACTOR)

from statsmodels.stats.outliers_influence import variance_inflation_factor

def calc_vif(df):

    # Calculating VIF
    vif = pd.DataFrame()
    vif["variables"] = df.columns
    vif["VIF"] = [variance_inflation_factor(df.values, i) for i in range(df.shape[1])]

    return(vif)

calc_vif(df)


# We can see that oldbalanceOrg and newbalanceOrig have too high VIF thus they are highly correlated. Similarly oldbalanceDest and newbalanceDest. Also nameDest is connected to nameOrig.
# 
# Thus combine these pairs of collinear attributes and drop the individual ones.

# In[9]:


df['orignal_amount'] = df.apply(lambda x: x['oldbalanceOrg'] - x['newbalanceOrig'],axis=1)
df['dest_amount'] = df.apply(lambda x: x['oldbalanceDest'] - x['newbalanceDest'],axis=1)
df['Path'] = df.apply(lambda x: x['nameOrig'] + x['nameDest'],axis=1)


#Dropping columns
df = df.drop(['oldbalanceOrg','newbalanceOrig','oldbalanceDest','newbalanceDest','step','nameOrig','nameDest'],axis=1)

calc_vif(df)


# In[10]:


corr=df.corr()

plt.figure(figsize=(10,6))
sns.heatmap(corr,annot=True)


# visualisation of analysis

# In[11]:


nfraud = len(df[df.isFraud == 0])
fraud = len(df[df.isFraud == 1])
nfraud_percent = (nfraud / (fraud + nfraud)) * 100
fraud_percent = (fraud / (fraud + nfraud)) * 100

print("Number of authorized transactions: ", nfraud)
print("Number of Fraud transactions: ", fraud)
print("Percentage of authorizd transactions: {:.4f} %".format(nfraud_percent))
print("Percentage of Fraud transactions: {:.4f} %".format(fraud_percent))


# plotting the authorized and fraud transaction

# In[12]:


plt.figure(figsize=(5,10))
labels = ["authorized", "Fraud"]
count_classes = df.value_counts(df['isFraud'], sort= True)
count_classes.plot(kind = "bar", rot = 0)
plt.title("Visualization of Labels")
plt.ylabel("Count")
plt.xticks(range(2), labels)
plt.show()


# These results prove that this is a highly unbalanced data as Percentage of authorized transactions= 99.87 % and Percentage of Fraud transactions= 0.13 %. SO DECISION TREES AND RANDOM FORESTS ARE GOOD METHODS FOR IMBALANCED DATA.we will try random forest here.

# # model building and scaling the data

# In[13]:


from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import itertools
from collections import Counter
import sklearn.metrics as metrics
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay


# In[14]:


# Perform Scaling
scaler = StandardScaler()
df["NormalizedAmount"] = scaler.fit_transform(df["amount"].values.reshape(-1, 1))
df.drop(["amount"], inplace= True, axis= 1)

Y = df["isFraud"]
X = df.drop(["isFraud"], axis= 1)


# # train test split

# In[15]:


# Split the data
(X_train, X_test, Y_train, Y_test) = train_test_split(X, Y, test_size= 0.3, random_state= 42)

print("Shape of X_train: ", X_train.shape)
print("Shape of X_test: ", X_test.shape)


# # model training

# In[ ]:


# RANDOM FOREST

random_forest = RandomForestClassifier(n_estimators= 100)
random_forest.fit(X_train, Y_train)

Y_pred_rf = random_forest.predict(X_test)
random_forest_score = random_forest.score(X_test, Y_test) * 100


# In[ ]:


#EVALUATION
# Print scores of our classifier

print("Random Forest Score: ", random_forest_score)


# In[ ]:


# key terms of Confusion Matrix - RF

print("TP,FP,TN,FN - Random Forest")
tn, fp, fn, tp = confusion_matrix(Y_test, Y_pred_rf).ravel()
print(f'True Positives: {tp}')
print(f'False Positives: {fp}')
print(f'True Negatives: {tn}')
print(f'False Negatives: {fn}')


# In[ ]:


# confusion matrix - DT

confusion_matrix_dt = confusion_matrix(Y_test, Y_pred_dt.round())
print("Confusion Matrix - Decision Tree")
print(confusion_matrix_dt,)

print("----------------------------------------------------------------------------------------")

# confusion matrix - RF

confusion_matrix_rf = confusion_matrix(Y_test, Y_pred_rf.round())
print("Confusion Matrix - Random Forest")
print(confusion_matrix_rf)


# In[ ]:


# classification report - RF

classification_report_rf = classification_report(Y_test, Y_pred_rf)
print("Classification Report - Random Forest")
print(classification_report_rf)


# In[ ]:


# visualising confusion matrix - RF
disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix_rf)
disp.plot()
plt.title('Confusion Matrix - RF')
plt.show()


# In[ ]:


# AUC ROC - RF
# calculate the fpr and tpr for all thresholds of the classification

fpr, tpr, threshold = metrics.roc_curve(Y_test, Y_pred_rf)
roc_auc = metrics.auc(fpr, tpr)

plt.title('ROC - RF')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


# # conclusion

# In a fraud detection model, Precision is highly important because rather than predicting normal transactions correctly we want Fraud transactions to be predicted correctly and authorized to be left off.If either of the 2 reasons are not fulfiiled we may catch the innocent and leave the culprit.
# This is also one of the reason why Random Forest and Decision Tree are used unstead of other algorithms and random forest is better than decision a decision tree because it is build of many decision trees, thus has better accuracy and precision.
# 
# 

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




