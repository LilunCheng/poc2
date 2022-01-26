import mlflow
from mlflow import log_metric, log_param, log_artifacts
import os
def train():
	with mlflow.start_run(run_name="Dummy") as run:
		print("Start MLFlow Tracking ...")

	with mlflow.start_run(run_name="Dummy") as run:
		mlflow.log_param("batch_size", 300)
		mlflow.log_metric("loss", 0.001)

def trainer():
  print("Start training ......")
  train()
