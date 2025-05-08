import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
import matplotlib
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import mlflow
import mlflow.sklearn
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
import pickle
from mlflow.models import infer_signature

mlflow.set_experiment('Experiment 1')
mlflow.set_tracking_uri('https://dagshub.com/RajeshB-0699/mlops_fullstack.mlflow')


import dagshub
dagshub.init(repo_owner='RajeshB-0699', repo_name='mlops_fullstack', mlflow=True)

data = pd.read_csv('https://raw.githubusercontent.com/RajeshB-0699/datasets_raw/refs/heads/main/water_potability.csv')

train_data, test_data = train_test_split(data, test_size = 0.2, random_state=42)

def fill_missing_with_median(df):
  for column in df.columns:
    if df[column].isnull().any():
      median_value = df[column].median()
      df[column].fillna(median_value, inplace = True)
    return df
  
train_processed_data = fill_missing_with_median(train_data)
test_processed_data = fill_missing_with_median(test_data)

X_train = train_processed_data.drop(columns = ['Potability'], axis =1)
X_test = test_processed_data.drop(columns = ['Potability'], axis = 1)
y_train = train_processed_data['Potability']
y_test = test_processed_data['Potability']



import mlflow
with mlflow.start_run():
  n_estimators = 1000
  clf = RandomForestClassifier(n_estimators = n_estimators)
  clf.fit(X_train, y_train)

  pickle.dump(clf, open('model.pkl', 'wb'))

  y_pred = clf.predict(X_test)

  accuracy = accuracy_score(y_pred, y_test)
  precision = precision_score(y_pred, y_test)
  f1 = f1_score(y_pred, y_test)
  recall = recall_score(y_pred, y_test)

  mlflow.log_param('n_estimators', n_estimators)

  mlflow.log_metric('accuracy score', accuracy)
  mlflow.log_metric('precision score', precision)
  mlflow.log_metric('f1 score', accuracy)
  mlflow.log_metric('accuracy score', accuracy)
  mlflow.log_metric('recall_score', recall)

  cm = confusion_matrix(y_test, y_pred)
  plt.figure(figsize=(5,5))
  sns.heatmap(cm, annot = True)
  plt.xlabel('Prediction')
  plt.ylabel('Actual')
  plt.title('Confusion matrix')
  plt.savefig('confusion_matrix.png')

  mlflow.log_artifact('confusion_matrix.png')

  mlflow.log_artifact(__file__)
  sign = infer_signature(model_input=X_test, model_output=clf.predict(X_test))
  mlflow.sklearn.log_model(clf, "RandomForestClassifier", signature = sign)

  mlflow.set_tag("author","Rajesh B")
  mlflow.set_tag("model", "RandomForestClassifier_n_est")