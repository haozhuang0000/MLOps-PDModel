# MLOps-PDModel

## MLOps Deployment
```
docker compose up -d --build
```

## Prerequisites

1. MINIO SERVER
2. MLFLOW SERVER


## Python Env

```
conda create -n mlops python=3.11
conda activate mlops
pip install -r requirements.txt
```

## Test your code - MLOps

```python
## training pipeline
python src/mlops/pipelines/training_pipeline.py

## inference pipeline
python src/mlops/pipelines/inference_pipeline.py
```

## Test your code - PD Evaluation

Code is under src/modeldev

- monthly: src/modeldev/model/lgbm_classifier.py
- yearly: src/modeldev/model/lgbm_classifier_yearly.py
- visualization: src/modeldev/data_visualization/plot.py

## Configuration

### .env File Setup
Create a .env file in the root directory of the project and configure the following variables:

In my case, I need to read files from Windows Files System, if you aren't, you can modify the code for all load data part, and IGNORE FILEIP, FILEIP_USERNAME, FILEIP_PASSWORD  
```
FILEIP=
FILEIP_USERNAME=
FILEIP_PASSWORD=

MYSQL_USER=
MYSQL_PASS=
MySQL_HOST=
```
