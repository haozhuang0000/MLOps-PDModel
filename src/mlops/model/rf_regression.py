from src.mlops.abstractions import ModelABC
from typing import Union, Tuple, List
import pandas as pd
import numpy as np
from tqdm import tqdm

from sklearn.base import RegressorMixin
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV

import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature

class RFRegression(ModelABC):

    def train(self,
              X_train: pd.DataFrame,
              y_train: pd.Series,
              id_bb_unique: str,
              y: str) -> Union[RegressorMixin, str]:

        rf = RandomForestRegressor(
            n_estimators=100,
            max_depth=None,
            min_samples_split=2,
            max_features=None,
            random_state=42
        )

        # Define the parameter grid to search over
        param_grid = {
            'n_estimators': [100, 200, 500],
            'max_depth': [10, 20, 30, None],
            'min_samples_split': [2, 5, 10, 15],
            'max_features': ['sqrt', 'log2', None],
        }

        cv = min(5, len(X_train))
        # Enable autologging
        mlflow.sklearn.autolog()
        with mlflow.start_run(run_name=f"{self.__class__.__name__}_{id_bb_unique}_{y}",  nested=True) as run:
            if cv >= 2:
                grid_search = RandomizedSearchCV(
                    estimator=rf,
                    param_distributions=param_grid,
                    n_iter=20,
                    cv=cv,
                    n_jobs=-1,
                    verbose=False,
                    random_state=42,
                    scoring='neg_root_mean_squared_error',
                )
                # Fit the model using grid search on training data
                grid_search.fit(X_train, y_train)

                # Get the best parameters and log them
                best_params = grid_search.best_params_
                mlflow.log_params(best_params)

                # Log metrics from cross-validation
                best_score = grid_search.best_score_
                mlflow.log_metric("best_neg_rmse", best_score)

                # Use the best estimator from grid search
                model = grid_search.best_estimator_
            else:
                # Not enough data for cross-validation; fit the model directly
                rf.fit(X_train, y_train)
                model = rf  # Assign the directly fitted model as the best model

                # Log default parameters
                default_params = {
                    "n_estimators": 100,
                    "max_depth": None,
                    "min_samples_split": 2,
                    "max_features": None
                }
                mlflow.log_params(default_params)

            # Log the model to MLflow
            model_name = f"{self.__class__.__name__}_{id_bb_unique}_{y}"
            # mlflow.sklearn.log_model(sk_model=model,
            #                          artifact_path=model_name)

            # model_uri = f"runs:/{run.info.run_id}/{model_name}"
            model_run_id = run.info.run_id
            # mlflow.register_model(model_uri=model_uri, name=model_name)
            print(f"Model for company: {id_bb_unique}, "
                  f"registered with model name **{model_name}**")

        return model, model_name, model_run_id