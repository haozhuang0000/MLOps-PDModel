# MLOps-PDModel

## Python Env

```
conda create -n airflow python=3.11
conda activate airflow
pip install "apache-airflow==2.4.0" "apache-airflow-providers-docker<3.8.0" "pydantic~=2.7.1"
```

```
conda create -n mlops python=3.11
conda activate mlops
pip install -r requirements.txt
```

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

### Environment Variable Setup on Local Machine

#### Linux
```bash
nano ~/.bashrc

# << MINIO >>
export MLFLOW_S3_ENDPOINT_URL=http://<YOUR_SERVER_ADDRESS>:9000
export AWS_ACCESS_KEY_ID=<YOUR_MINIO_ROOT_USER>
export AWS_SECRET_ACCESS_KEY=<YOUR_MINIO_ROOT_PASSWORD>

# << Airflow >>
export AIRFLOW_HOME=<YOUR_PATH_TO_AIRFLOW_HOME>
export ZENML_LOCAL_STORES_PATH=<YOUR_PATH_TO_ZENML>
export ZENML_CONFIG_PATH=<YOUR_PATH_TO_ZENML>

source ~/.bashrc
```

#### Windows

- Press Win + R, type sysdm.cpl, and press Enter.

- Go to the Advanced tab and click Environment Variables.

- Under User variables or System variables, click New... to add the following:

```
MLFLOW_S3_ENDPOINT_URL=http://<YOUR_SERVER_ADDRESS>:9000
AWS_ACCESS_KEY_ID=<YOUR_MINIO_ROOT_USER>
AWS_SECRET_ACCESS_KEY=<YOUR_MINIO_ROOT_PASSWORD>

AIRFLOW_HOME=<YOUR_PATH_TO_AIRFLOW_HOME>
ZENML_LOCAL_STORES_PATH=<YOUR_PATH_TO_ZENML>
ZENML_CONFIG_PATH=<YOUR_PATH_TO_ZENML>
```


## Prerequisites

### 1. MySQL Database

### 2. Docker
- Linux: https://docs.docker.com/desktop/setup/install/linux/
- Windows: https://docs.docker.com/desktop/setup/install/windows-install/

### 3. Minio setup steps- artifact storage
Add the line to ~/.bashrc

```
cd <PATH_TO_MINIO>
wget https://dl.min.io/server/minio/release/linux-amd64/minio
chmod +x <PATH_TO_MINIO>
MINIO_ROOT_USER=<YOUR_MINIO_ROOT_USER> MINIO_ROOT_PASSWORD=<YOUR_MINIO_ROOT_PASSWORD> ./minio server /data/zhuanghao/mlops/minio_data --console-address "10.230.252.6:9001"
```

1. Log into http://<YOUR_SERVER_ADDRESS>:9000
2. Create a bucket <BUCKET_NAME>

### 4. Mlflow setup steps - artifact tracker

```
conda activate mlops
mlflow server --host 0.0.0.0 --port 8885 --artifacts-destination s3://<BUCKET_NAME>
```

### 5. Airflow setup steps - automation

```bash
airflow db init
airflow users create \
  --username <AIRFLOW_USER> \
  --firstname <AIRFLOW_USER_FIRST> \
  --lastname <AIRFLOW_USER_LAST> \
  --role Admin \
  --email admin@example.com \
  --password <AIRFLOW_PASSWORD>
airflow webserver
airflow scheduler
```
Note:

Set timezone in Airflow if needed

If you want Airflow UI and scheduling to reflect Asia/Singapore, set this in airflow.cfg:

Example: 
```
default_timezone = Asia/Singapore
```
### 6. Slack setup steps - alerter
1. Go to: https://slack.com/ and create <YOUR_WORKSPACE>
2. Go to: https://api.slack.com/apps
2. Create New App -> From a manifest
3. OAuth & Permissions -> Scopes -> Give permission for:
   - channels:history
   - channels:read
   - chat:write
   - groups:read
   - im:read
   - mpim:read
   - calls:read
4. OAuth & Permissions -> OAuth Tokens -> Install to <YOUR_WORKSPACE>
5. Back to your Slack channel 
   - Create a channel -> Edit setting -> Integrations -> Add apps
6. <YOUR_SLACK_BOT_TOKEN> can be find here: OAuth & Permissions -> OAuth Tokens
## ZenML setup steps

```bash
conda activate airflow
airflow config get-value core DAGS_FOLDER ## Obtain <AIRFLOW_DAG_DIRECTORY>
```

```bash
conda activate mlops

zenml init
zenml login --local --ip-address <YOUR_IP_ADDRESS>

zenml integration install mlflow -y

## Registering experiment tracker
zenml experiment-tracker register mlflow_tracker --flavor=mlflow --tracking_uri=http://<YOUR_IP_ADDRESS>:8885/ --tracking_username=MY_USERNAME --tracking_password=MY_PASSWORD

## Registering model deployer
zenml model-deployer register mlflow --flavor=mlflow

## Registering s3 store
zenml secret create s3_secret --aws_access_key_id=<YOUR_MINIO_ROOT_USER>   --aws_secret_access_key=<YOUR_MINIO_ROOT_PASSWORD>
zenml artifact-store register minio_store -f s3 --path='s3://<BUCKET_NAME>' --authentication_secret=s3_secret --client_kwargs='{"endpoint_url": "http://<YOUR_IP_ADDRESS>:9000", "region_name": "us-east-1"}'

## Registering airflow orchestrator
zenml integration install airflow
zenml orchestrator register mlops_airflow \
    --flavor=airflow \
    --local=True 
zenml orchestrator update --dag_output_dir=<AIRFLOW_DAG_DIRECTORY>
## Registring local orchestrator
zenml orchestrator register local_orchestrator --flavor=local

## Registering slack
zenml integration install slack -y
zenml secret create slack_token --oauth_token=<YOUR_SLACK_BOT_TOKEN>
zenml alerter register slack_alerter \
    --flavor=slack \
    --slack_token={{slack_token.oauth_token}} \
    --slack_channel_id=<YOUR_SLACK_CHANNEL_ID>
    
zenml stack register mlflow_stack \
     -o mlops_airflow\
     -a minio_store\
     -e mlflow_tracker\
     -al slack_alerter
     --set
```

## Usage

1. Train a model
```
conda activate mlops
zenml stack set mlflow_stack
zenml stack update mlflow_stack -o local_orchestrator

cd /path/MLOps-PDModel
python src/mlops/pipelines/training_pipeline.py 
```

2. Schedule daily prediction
```
conda activate mlops
zenml stack set mlflow_stack
zenml stack update mlflow_stack -o mlops_airflow

cd /path/MLOps-PDModel
python src/mlops/airflow_inference.py
```

## Acknowledgments
- This project uses:
  - [ZenML](https://www.zenml.io/)
  - [LightGBM](https://github.com/microsoft/LightGBM)
  - [MLflow](https://mlflow.org/)
  - [Airflow](https://airflow.apache.org/)
  - [Evidently AI](https://github.com/evidentlyai/evidently)
  - [MINIO](https://min.io/)
  - [Slack](https://slack.com/)
