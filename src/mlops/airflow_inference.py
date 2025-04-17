from datetime import datetime, timedelta

from zenml.pipelines import Schedule
from src.mlops.pipelines import prediction_service

scheduled_pipeline = prediction_service.with_options(
    schedule=Schedule(
        start_time=datetime.now() - timedelta(hours=1),  # start in the past
        end_time=datetime.now() + timedelta(hours=1),
        interval_second=timedelta(minutes=15),  # run every 15 minutes
        catchup=False,
    )
)
scheduled_pipeline(econ=2, model_name="LGBClassifier_Multiclass_CN")