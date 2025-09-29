# MLOps-PDModel

A comprehensive MLOps system for Probability of Default (PD) and Probability of Exit (POE) prediction using both statistical and machine learning models. This project implements an end-to-end pipeline with Apache Airflow orchestration, MLflow model tracking, and a Streamlit dashboard for monitoring and visualization.

## 🏗️ Architecture Overview

This system consists of several key components:

- **Training Pipeline**: Automated model training with data preprocessing, feature engineering, and model evaluation
- **Inference Pipeline**: Real-time prediction service for daily PD/POE scoring
- **Airflow DAGs**: Automated scheduling and orchestration of prediction tasks
- **Streamlit Dashboard**: Interactive web interface for monitoring predictions and model performance
- **Database Integration**: MySQL storage for predictions, ground truth data, and model metrics

## 🚀 Quick Start

### Prerequisites

1. **MINIO SERVER** - Object storage for MLflow artifacts
2. **MLFLOW SERVER** - Model registry and experiment tracking
3. **MySQL Database** - Data storage and retrieval

### MLOps Deployment

Deploy the complete system using Docker Compose:

```bash
docker compose up -d --build
```

This will start:
- Apache Airflow (scheduler, webserver, triggerer)
- PostgreSQL database for Airflow metadata
- Your MLOps services

Access Airflow UI at: `http://localhost:8088`

### Python Environment Setup

```bash
conda create -n mlops python=3.11
conda activate mlops
pip install -r requirements.txt
```

## 📁 Project Structure

```
src/
├── dags/                          # Airflow DAG definitions
│   └── prediction_dag.py          # Daily prediction scheduling
├── mlops/                         # Core MLOps components
│   ├── pipelines/                 # Training and inference pipelines
│   │   ├── training_pipeline.py   # Model training workflow
│   │   ├── inference_pipeline.py  # Daily prediction service
│   │   └── deployment_pipeline.py # Model deployment workflow
│   ├── steps/                     # Pipeline steps for training
│   ├── steps_deployment/          # Pipeline steps for inference
│   ├── model/                     # Model implementations
│   ├── data_loading/              # Data ingestion modules
│   ├── data_cleaning/             # Data preprocessing
│   ├── evaluation/                # Model evaluation metrics
│   ├── database/                  # Database connections
│   └── configs/                   # Configuration management
├── modeldev/                      # Model development and research
│   ├── model/                     # LightGBM classifier implementations
│   │   ├── lgbm_classifier.py     # Monthly PD model
│   │   └── lgbm_classifier_yearly.py # Yearly PD model
│   ├── data_preprocessing/        # Data preprocessing utilities
│   └── data_visualization/        # Plotting and SHAP analysis
└── streamlit_app.py              # Interactive dashboard
```

## 🎯 Core Functionalities

### 1. Training Pipeline

Execute the complete model training workflow:

```bash
python src/mlops/pipelines/training_pipeline.py
```

**Pipeline Steps:**
- Data loading and validation
- Data cleaning and preprocessing
- Train/validation/test splitting with time series considerations
- LightGBM model training with hyperparameter optimization
- Model evaluation (AUC, Precision, Recall, F1-Score)
- Model registration in MLflow
- Training metrics storage in MySQL
- Evidently AI model monitoring reports

### 2. Inference Pipeline

Run daily predictions:

```bash
python src/mlops/pipelines/inference_pipeline.py
```

**Pipeline Steps:**
- Wait for daily input files
- Load and validate input data
- Load registered model from MLflow
- Generate PD/POE predictions
- Store results in MySQL database
- Send alerts via Slack integration

### 3. Streamlit Dashboard

Launch the interactive monitoring dashboard:

```bash
streamlit run src/streamlit_app.py
```

**Dashboard Features:**
- **Individual Company Analysis**: Historical PD/POE trends, prediction comparisons
- **Model Performance Monitoring**: AUC scores, PR-AUC comparisons between statistical and ML models
- **Country-wise Metrics**: Performance breakdown by economic regions
- **Real-time Data**: Live MySQL database integration

## 🔄 Automated Scheduling

The system uses Apache Airflow for automated daily predictions:

- **DAG**: `daily_prediction_service`
- **Schedule**: Daily at 10:00 AM (Singapore timezone)
- **Location**: `src/dags/prediction_dag.py:18`

## 📊 Model Development

### Statistical vs ML Models

**Statistical Model (CRIPD)**:
- Traditional credit risk modeling approach
- Monthly and yearly horizon predictions
- Established baseline for comparison

**Machine Learning Model (LightGBM)**:
- Advanced gradient boosting implementation
- Hyperparameter optimization with Optuna
- Enhanced feature engineering and cross-validation

### Model Evaluation Components

The system includes comprehensive evaluation:
- **AUC-ROC**: Area under receiver operating curve
- **PR-AUC**: Precision-recall area under curve
- **Accuracy, Precision, Recall, F1-Score**: Classification metrics
- **Evidently AI**: Model drift and data quality monitoring

## ⚙️ Configuration

### Environment Variables

Create a `.env` file in the root directory:

```bash
# File System Access (if reading from Windows file system)
FILEIP=your_file_server_ip
FILEIP_USERNAME=your_username
FILEIP_PASSWORD=your_password

# MySQL Database
MYSQL_USER=your_mysql_user
MYSQL_PASS=your_mysql_password
MYSQL_HOST=your_mysql_host
MYSQL_PORT=3306
MYSQL_DB=mlops_pd

# MLflow Configuration
MLFLOW_TRACKING_URI=your_mlflow_server
MLFLOW_S3_ENDPOINT_URL=your_minio_endpoint

# Airflow Configuration
_AIRFLOW_WWW_USER_USERNAME=admin
_AIRFLOW_WWW_USER_PASSWORD=admin
```

### Database Schema

The system expects the following MySQL tables:
- `mlops_pd.cripd_daily`: Statistical model predictions
- `mlops_pd.mlpd_daily_dev`: ML model predictions
- `mlops_pd.pd_ground_truth`: Actual default events
- `mlops_pd.model`: Model metadata and versions
- `mlops_pd.metrics`: Model performance metrics

## 🧪 Testing and Development

### Model Development Testing

Test individual model components:

```bash
# Monthly model evaluation
python src/modeldev/model/lgbm_classifier.py

# Yearly model evaluation
python src/modeldev/model/lgbm_classifier_yearly.py

# Data visualization and SHAP analysis
python src/modeldev/data_visualization/plot.py
```

### Pipeline Testing

Validate pipeline components individually:

```bash
# Test data loading
python -c "from src.mlops.steps.load_data import load_data; print('Data loading OK')"

# Test model training
python -c "from src.mlops.steps.train_model import train_model; print('Model training OK')"
```

## 🔧 Troubleshooting

### Common Issues

1. **Database Connection**: Verify MySQL credentials and network connectivity
2. **MLflow Integration**: Ensure MLflow server is running and accessible
3. **File Access**: Check file system permissions for data loading
4. **Airflow DAGs**: Verify DAG syntax and dependencies

### Logs and Monitoring

- **Airflow Logs**: `./airflow_logs/`
- **Application Logs**: `./logs/`
- **Model Artifacts**: MLflow tracking server
- **Database Metrics**: Streamlit dashboard monitoring page

## 📈 Performance Monitoring

The system provides comprehensive monitoring through:

1. **Real-time Dashboards**: Streamlit interface with live data
2. **Model Drift Detection**: Evidently AI integration
3. **Performance Metrics**: Automated AUC/PR-AUC tracking
4. **Alerting**: Slack notifications for prediction pipeline status

## 🤝 Contributing

1. Follow the existing code structure and naming conventions
2. Add comprehensive unit tests for new features
3. Update documentation for any configuration changes
4. Ensure compatibility with the existing MLflow and Airflow setup
