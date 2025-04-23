from zenml.config import ResourceSettings
from zenml import step
from zenml.client import Client
experiment_tracker = Client().active_stack.experiment_tracker

@step(experiment_tracker=experiment_tracker.name, enable_cache=False, settings={"resources": ResourceSettings(cpu_count=5, gpu_count=4, memory="24GB")})
def post_statement(alert_signal: str, step: str) -> None:
    """
    Post a status message to Slack based on the pipeline step and input.
    """
    slack_client = Client().active_stack.alerter

    if alert_signal == 'Fail':
        slack_client.post(f"Job failed at step: {step}")
        raise RuntimeError(f"[post_statement] Pipeline terminated due to failure at step: {step}")
    elif alert_signal == 'Success':
        messages = {
            "wait_for_files": f"Job successfully completed: {step}",
            "load_data": f"Job successfully completed: {step}",
            "predict": f"Job successfully completed: {step}",
            "save_mysql": "All jobs completed successfully. Output saved to MySQL.",
        }

        message = messages.get(step)
        if message:
            slack_client.post(message)