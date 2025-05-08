import mlflow

mlflow.set_tracking_uri('https://dagshub.com/RajeshB-0699/mlops_fullstack.mlflow')

import dagshub
dagshub.init(repo_owner='RajeshB-0699', repo_name='mlops_fullstack', mlflow=True)

import mlflow
with mlflow.start_run():
  mlflow.log_param('parameter name', 'value')
  mlflow.log_metric('metric name', 1)