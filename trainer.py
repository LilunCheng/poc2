import mlflow
from mlflow import log_metric, log_param, log_artifacts
import os
def train():
  print("Tracking URI is " + str(os.environ['MLFLOW_ENDPOINT']))
  mlflow.set_tracking_uri(os.environ['MLFLOW_ENDPOINT'])
  expnm = mlflow.create_experiment('lilun_training_experiment')
  with mlflow.start_run(experiment_name=expnm) as run:
    mlflow.log_param("batch_size", 300)
    mlflow.log_metric("loss", 0.001)

def trainer():
  print("Start training ......")
  train()
