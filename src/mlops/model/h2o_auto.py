from pyexpat import features

from src.mlops.abstractions import ModelABC
from typing import Union, Tuple, List
import pandas as pd

import h2o
from h2o.automl import H2OAutoML

import mlflow

class H2OAuto(ModelABC):

    h2o.init(nthreads=-1)
    def train(self, X_train: pd.DataFrame,
                    y_train: pd.Series,
                    id_bb_unique: str,
                    y: str) -> Union[H2OAutoML, str]:

        """
        Trains a model using the given training data.

        Parameters:
        - X_train (pd.DataFrame): Feature matrix for training.
        - y_train (pd.Series): Target variable corresponding to X_train.
        - id_bb_unique (str): Unique identifier for tracking the training instance.
        - y (str): Column name representing the target variable.

        Returns:
        - model(): MODEL
        - model_name (str): name of the model
        - model_run_id (str): mlflow run id of the model
        """

        df_train_h20 = pd.concat([X_train, y_train], ignore_index=False, axis=1)
        drop_cols = self.industry_info_cols[-3:]
        categorical_var = self.industry_info_cols[:-3]

        df_train_h20 = df_train_h20.drop(columns=drop_cols)
        df_train_h20 = h2o.H2OFrame(df_train_h20)

        features = self.x_cols_to_process

        for var in categorical_var:
            df_train_h20[var] = df_train_h20[var].asfactor()

        aml = H2OAutoML(
            max_models=20,
            seed=1,
            # exclude_algos=["DeepLearning"],  # Optionally exclude algorithms
            sort_metric="RMSE"  # Metric to sort models
        )


        with mlflow.start_run(run_name=f"{self.__class__.__name__}_{id_bb_unique}_{y}",  nested=True) as run:
            aml.train(x=features, y=y, training_frame=df_train_h20)
            best_model = aml.leader

            mlflow.log_param('max_models', 20)
            # Log the model to MLflow
            model_name = f"{self.__class__.__name__}_{id_bb_unique}_{y}"
            model_run_id = run.info.run_id
            mlflow.h2o.log_model(best_model, artifact_path='model')

        model = best_model
        return model, model_name, model_run_id