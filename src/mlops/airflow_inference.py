# from datetime import datetime, timedelta, time
#
# # from zenml.pipelines import Schedule
# from src.mlops.pipelines import prediction_service
#
# now = datetime.now()
# two_pm_today = datetime.combine(now.date(), time(14, 0))
# start_time = two_pm_today if now < two_pm_today else two_pm_today + timedelta(days=1)
#
# task_date = datetime.today().strftime('%Y-%m-%d')
# scheduled_pipeline = prediction_service.with_options(
#     schedule=Schedule(
#         start_time=datetime.now() - timedelta(hours=1),
#         interval_second=timedelta(days=1),  # once a day
#         catchup=False
#     )
# )
# scheduled_pipeline(econ=2, model_name="LGBClassifier_Multiclass_CN", task_date=task_date)
#
# # 🔧 manual trigger to test now
# prediction_service(econ=2, model_name="LGBClassifier_Multiclass_CN", task_date=task_date)