def train():
  import mlflow
  import sys,os,os.path  
  from mlflow import log_metric, log_param, log_artifacts
  
  mlflow.set_tracking_uri(os.environ['MLFLOW_ENDPOINT'])
  with mlflow.start_run(run_name="Lilun_Model_run") as run:
    mlflow.log_param("batch_size", 300)
    mlflow.log_metric("loss", 0.001)
    
def trainer():
  print("Start training ......")
  train()
