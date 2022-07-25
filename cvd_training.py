# -*- coding: utf-8 -*-
"""
Created on Mon Jul 25 09:01:35 2022

@author: eshan
"""
import os
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import scipy.stats as ss
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.impute import KNNImputer
from sklearn.svm import SVC

#%% Functions
def cramers_corrected_stat(matrix):
    """ calculate Cramers V statistic for categorial-categorial association.
        uses correction from Bergsma and Wicher, 
        Journal of the Korean Statistical Society 42 (2013): 323-328
    """
    chi2 = ss.chi2_contingency(matrix)[0]
    n = matrix.sum()
    phi2 = chi2/n
    r,k = matrix.shape
    phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))    
    rcorr = r - ((r-1)**2)/(n-1)
    kcorr = k - ((k-1)**2)/(n-1)
    return np.sqrt(phi2corr / min( (kcorr-1), (rcorr-1)))

#%% Constants
CSV_PATH = os.path.join(os.getcwd(),'dataset','heart.csv')
MODEL_PATH = os.path.join(os.getcwd(), 'model', 'model.pkl')

#%% Data Loading
df = pd.read_csv(CSV_PATH)

#%% Data Inspection

df.info()
df.describe().T
df.isna().sum() # no NaNs

# Null = 0 for thall column
(df['thall'] == 0).sum() # 2 null values present

# Null = 4 for caa column
(df['caa'] == 4).sum() # 5 null values present

df.duplicated().sum() # 1 duplicate

cat_col = ['sex','exng','caa','cp','fbs','restecg','output', 'thall','slp']
con_col = list(df.drop(labels=cat_col, axis=1).columns)

for i in cat_col:
    plt.figure()
    sns.countplot(df[i])
    plt.show()
    
for i in con_col:
    plt.figure()
    sns.distplot(df[i])
    plt.show()

#outliers in chol & trtbps, but within acceptable range

#%% Data Cleaning

# Drop duplicates
df = df.drop_duplicates()

# Replace 'null' values with NaN
def nan_replace(column_name,value):
    df[column_name] = df[column_name].replace(value, np.nan)
    print((df[column_name] == value).sum())

nan_replace('thall',0)
nan_replace('caa',4)

df.isna().sum() # 4 NaN in caa, 2 NaN in thall

# Fill in missing values via imputation
knn_imputer = KNNImputer()
df_knn = knn_imputer.fit_transform(df)
df_knn = pd.DataFrame(df_knn, index=None)
df_knn.columns = df.columns

df_knn.isna().sum() # no NaNs
df_knn.info()
df_knn.describe().T

#Floats present in imputed data - change to integer
df_knn['thall'] = np.floor(df_knn['thall']).astype('int')
df_knn['caa'] = np.floor(df_knn['caa']).astype('int')

#%% Feature Selection

# Target: output (categorical)
y = df['output']

# Categorical vs Categorical - Cramer's V
features = []

for i in cat_col:
    matrix = pd.crosstab(df_knn[i],y).to_numpy()
    if cramers_corrected_stat(matrix) > 0.5:
        features.append(i)
        print(i)
        print(cramers_corrected_stat(matrix))
        
# Continuous vs Categorical - Logistic Regression
for i in con_col:
    lr=LogisticRegression()
    lr.fit(np.expand_dims(df_knn[i],axis=-1),y)
    if lr.score(np.expand_dims(df_knn[i],axis=-1),y) > 0.6:
        features.append(i)
        print(i)
        print(lr.score(np.expand_dims(df_knn[i],axis=-1),y))

features.remove('output') #remove target from features list

# Features to use in model dev: 'cp', 'thall', 'age', 'thalachh', 'oldpeak'

#%% Data Preprocessing

X = df_knn.loc[:,features]
y = df_knn['output']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, 
                                                    random_state=123)

#%% Model Development (Pipeline)

#Logistic Regression
pipeline_mms_lr = Pipeline([
                            ('Min_Max_Scaler', MinMaxScaler()),
                            ('Logistic_Classifier', 
                             LogisticRegression(solver='liblinear'))
                            ])

pipeline_ss_lr = Pipeline([
                            ('Standard_Scaler', StandardScaler()),
                            ('Logistic_Classifier', 
                             LogisticRegression(solver='liblinear'))
                            ])

#Decision Tree
pipeline_mms_dt = Pipeline([
                            ('Min_Max_Scaler', MinMaxScaler()),
                            ('DT', DecisionTreeClassifier())
                            ])

pipeline_ss_dt = Pipeline([
                            ('Standard_Scaler', StandardScaler()),
                            ('DT', DecisionTreeClassifier())
                            ])

#KNN
pipeline_mms_knn = Pipeline([
                            ('Min_Max_Scaler', MinMaxScaler()),
                            ('KNN', KNeighborsClassifier())
                            ])

pipeline_ss_knn = Pipeline([
                            ('Standard_Scaler', StandardScaler()),
                            ('KNN', KNeighborsClassifier())
                            ])

#Random forest
pipeline_mms_rf = Pipeline([
                            ('Min_Max_Scaler', MinMaxScaler()),
                            ('RF', RandomForestClassifier())
                            ])

pipeline_ss_rf = Pipeline([
                            ('Standard_Scaler', StandardScaler()),
                            ('RF', RandomForestClassifier())
                            ])

#SVC
pipeline_mms_svc = Pipeline([
                            ('Min_Max_Scaler', MinMaxScaler()),
                            ('SVC', SVC())
                            ])

pipeline_ss_svc = Pipeline([
                            ('Standard_Scaler', StandardScaler()),
                            ('SVC', SVC())
                            ])

#Gradient Boosting
pipeline_mms_gb = Pipeline([
                            ('Min_Max_Scaler', MinMaxScaler()),
                            ('GB', GradientBoostingClassifier())
                            ])

pipeline_ss_gb = Pipeline([
                            ('Standard_Scaler', StandardScaler()),
                            ('GB', GradientBoostingClassifier())
                            ])

pipelines = [pipeline_mms_lr, pipeline_ss_lr, pipeline_mms_dt, pipeline_ss_dt, 
             pipeline_mms_knn, pipeline_ss_knn, pipeline_mms_rf, 
             pipeline_ss_rf, pipeline_mms_svc, pipeline_ss_svc, 
             pipeline_mms_gb, pipeline_ss_gb]

for pipe in pipelines:
    pipe.fit(X_train, y_train)

best_accuracy = 0
pipe_score=[]

for i, pipe in enumerate(pipelines):
    print(pipe.score(X_test,y_test))
    if pipe.score(X_test, y_test) > best_accuracy:
        best_accuracy = pipe.score(X_test, y_test)
        best_pipeline = pipe
        pipe_score.append(pipe.score(X_test,y_test))

print('The best scaler and classifier for data is {} with accuracy of {}'.
      format(best_pipeline.steps, best_accuracy))

print(pipelines[np.argmax(pipe_score)])
print(pipe_score[np.argmax(pipe_score)])

# Best scaler: MinMaxScaler
# Best classifier: LogisticRegression
# Accuracy of 0.79

#%% Hyperparameter tuning

# GridSearch cross validation

pipeline_mms_lr = Pipeline([
                            ('Min_Max_Scaler', MinMaxScaler()),
                            ('LR', LogisticRegression(solver='liblinear'))
                            ])

grid_param = {'LR__warm_start': [False,True],
               'LR__C': np.arange(1,2,0.1),
               'LR__fit_intercept': [True,False],
               'LR__intercept_scaling': np.arange(1,2,0.1)
               } #hyperparameters

grid_search = GridSearchCV(pipeline_mms_lr, param_grid=grid_param, cv=5, 
                           verbose=1, n_jobs=-1)

model = grid_search.fit(X_train, y_train)

print(model.best_index_)
print(model.best_params_)
print(model.best_score_)

#2000 fits; best score: 0.80

best_model = model.best_estimator_

#%% Model Evaluation

best_pipe = pipelines[np.argmax(pipe_score)]
y_pred = best_pipe.predict(X_test)

print(classification_report(y_test,y_pred))

conmx = confusion_matrix(y_test,y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=conmx)
disp.plot(cmap=plt.cm.Blues)
plt.show()

#%% Save Model

with open(MODEL_PATH,'wb') as file:
    pickle.dump(best_model,file)
