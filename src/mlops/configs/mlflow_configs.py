import os
import mlflow
from pathlib import Path
from src.mlops.logger.utils.logger import Log
from dotenv import load_dotenv
import json
from datetime import datetime
from typing import Dict, Any, Optional


class MLflowConfig:
    """Global MLflow configuration management with S3/Minio support"""

    _instance = None
    _is_configured = False
    load_dotenv()

    def __new__(cls, **kwargs):
        if cls._instance is None:
            cls._instance = super(MLflowConfig, cls).__new__(cls)
        return cls._instance

    def __init__(self, experiment_name):
        if not hasattr(self, 'logger'):
            self.logger = Log("MLflowConfig").getlog()
        self._tracking_uri = None
        self._experiment_name = experiment_name
        self._artifacts_destination = None
        self._current_version = None
        self._version_file = Path("mlflow_version.json")

    def setup_mlflow(self):
        """Setup MLflow with S3/Minio configuration from environment variables"""

        if self._is_configured:
            self.logger.info("MLflow already configured")
            return

        try:
            # Get configuration from environment
            tracking_uri = os.getenv('MLFLOW_TRACKING_URI', 'http://127.0.0.1:8885')
            experiment_name = self._experiment_name
            s3_endpoint_url = os.getenv('MLFLOW_S3_ENDPOINT_URL')
            aws_access_key_id = os.getenv('AWS_ACCESS_KEY_ID')
            aws_secret_access_key = os.getenv('AWS_SECRET_ACCESS_KEY')
            artifacts_destination = os.getenv('MLFLOW_ARTIFACTS_DESTINATION', 's3://forecast')

            # Set environment variables for S3/Minio access
            if aws_access_key_id:
                os.environ['AWS_ACCESS_KEY_ID'] = aws_access_key_id

            if aws_secret_access_key:
                os.environ['AWS_SECRET_ACCESS_KEY'] = aws_secret_access_key

            # Set S3 endpoint URL for Minio
            os.environ['MLFLOW_S3_ENDPOINT_URL'] = s3_endpoint_url

            # Set MLflow tracking URI
            mlflow.set_tracking_uri(tracking_uri)
            self._tracking_uri = tracking_uri
            self.logger.info(f"MLflow tracking URI set to: {tracking_uri}")

            # Set experiment
            mlflow.set_experiment(experiment_name)
            self._experiment_name = experiment_name
            self.logger.info(f"MLflow experiment set to: {experiment_name}")

            self._artifacts_destination = artifacts_destination
            self._is_configured = True

            # Load current version
            self._load_version()

            self.logger.info("MLflow configuration completed successfully")

        except Exception as e:
            self.logger.error(f"MLflow setup failed: {e}")
            raise

    def get_tracking_uri(self):
        """Get current tracking URI"""
        return self._tracking_uri

    def get_experiment_name(self):
        """Get current experiment name"""
        return self._experiment_name

    def is_configured(self):
        """Check if MLflow is properly configured"""
        return self._is_configured

    def _load_version(self):
        """Load current version from file"""
        if self._version_file.exists():
            with open(self._version_file, 'r') as f:
                version_data = json.load(f)
                self._current_version = version_data.get('current_version', 'v1.0')
                self.logger.info(f"Loaded version: {self._current_version}")
        else:
            self._current_version = 'v1.0'
            self._save_version()

    def _save_version(self):
        """Save current version to file"""
        version_data = {
            'current_version': self._current_version,
            'last_updated': datetime.now().isoformat(),
            'experiment': self._experiment_name
        }
        with open(self._version_file, 'w') as f:
            json.dump(version_data, f, indent=2)

    def increment_version(self, version_type: str = 'minor'):
        """
        Increment version number

        Args:
            version_type: 'major' or 'minor'
        """
        if not self._current_version:
            self._current_version = 'v1.0'

        # Parse current version
        version_parts = self._current_version.lstrip('v').split('.')
        major = int(version_parts[0])
        minor = int(version_parts[1]) if len(version_parts) > 1 else 0

        if version_type == 'major':
            major += 1
            minor = 0
        else:  # minor
            minor += 1

        self._current_version = f"v{major}.{minor}"
        self._save_version()
        self.logger.info(f"Version incremented to: {self._current_version}")

        return self._current_version

    def set_version(self, version: str):
        """Manually set version"""
        if not version.startswith('v'):
            version = f"v{version}"

        self._current_version = version
        self._save_version()
        self.logger.info(f"Version set to: {self._current_version}")

        return self._current_version

    def get_current_version(self):
        """Get current version"""
        return self._current_version or 'v1.0'

    def get_mlflow_alias(self):
        """Get MLflow-compatible alias (replaces dots with underscores)"""
        return self.get_current_version().replace('.', '_')