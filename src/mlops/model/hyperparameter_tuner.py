import optuna

class HyperparameterTuner:

    """
    Class for performing hyperparameter tuning. It uses Model strategy to perform tuning.
    """

    def __init__(self, x_train, y_train, x_test, y_test, mlflow_model_name, mlflow_model_run_id):
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.mlflow_model_name = mlflow_model_name
        self.mlflow_model_run_id = mlflow_model_run_id

    def optimize(self, n_trials=1):

        model, model_uri = ModelLoader.load_model(self.mlflow_model_name, self.mlflow_model_run_id)
        if "LGB" in mlflow_model_name:
            mlflow.lightgbm.autolog()
        elif "RF" in mlflow_model_name:
            mlflow.sklearn.autolog()

        with mlflow.start_run(run_id=mlflow_model_run_id, nested=True):
            study = optuna.create_study(direction="maximize")
            study.optimize(lambda trial: self.model.optimize(trial, self.x_train, self.y_train, self.x_test, self.y_test), n_trials=n_trials)

        mlflow.register_model(model_uri=model_uri, name=mlflow_model_name)
        return study.best_trial.params