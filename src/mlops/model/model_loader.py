import mlflow

class ModelLoader(object):

    @staticmethod
    def load_model(model_name, run_id):

        if 'LGB' in model_name:
            # Load the logged model using the run ID
            # model_uri = f"runs:/{run_id}/{model_name}"
            model_uri = f"runs:/{run_id}/model"
            model = mlflow.lightgbm.load_model(model_uri)
        elif 'RF' in model_name:
            # model_uri = f"runs:/{run_id}/{model_name}"
            model_uri = f"runs:/{run_id}/model"
            model = mlflow.sklearn.load_model(model_uri)

        return model, model_uri