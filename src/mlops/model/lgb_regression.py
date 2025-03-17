from src.mlops.abstractions import ModelABC
from typing import Union, Tuple, List
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV
import lightgbm as lgb
from lightgbm import Booster
import mlflow
import mlflow.lightgbm
from mlflow.models import infer_signature

class LGBRegression(ModelABC):

    def train(self, X_train: pd.DataFrame,
                    y_train: pd.Series,
                    id_bb_unique: str,
                    y: str) -> Union[Booster, str]:

        """
        Trains a model using the given training data.

        Parameters:
        - X_train (pd.DataFrame): Feature matrix for training.
        - y_train (pd.Series): Target variable corresponding to X_train.
        - id_bb_unique (str): Unique identifier for tracking the training instance.
        - y (str): Column name representing the target variable.

        Returns:
        - model(Booster): MODEL
        - model_name (str): name of the model
        - model_run_id (str): mlflow run id of the model
        """
        # Adjusted parameters for regression
        params = {
            'boosting_type': 'gbdt',
            'objective': 'regression',
            'class_weight': None,
            'min_split_gain': 0.0,
            'min_child_weight': 0.001,
            'min_child_samples': 7,
            'subsample': 1.0,
            'subsample_freq': 0,
            'colsample_bytree': 1.0,
            'reg_alpha': 0.0,
            'reg_lambda': 0.0,
            'random_state': None,
            'n_jobs': -1,
            'verbose': -1,
            'device': 'gpu'
        }

        # Create parameters to search
        # grid_params = {
        #     'learning_rate': [0.01, 0.05, 0.1, 0.2],
        #     'n_estimators': [100, 500, 1000],
        #     'num_leaves': [8, 16, 45],
        #     'feature_fraction': [0.6, 0.7, 0.8, 0.9, 1.0],
        #     'max_depth': [-1, 5, 10, 20],
        # }

        grid_params = {
            'learning_rate': [0.01, 0.05, 0.1],
            'n_estimators': [100, 500],
            'num_leaves': [8, 16, 32],
            'feature_fraction': [0.7, 0.8, 0.9],
            'max_depth': [5, 10],
        }


        # Create the regressor
        mod = lgb.LGBMRegressor(**params)
        # Enable autologging
        mlflow.lightgbm.autolog()
        # Adjust cv based on the size of X_train
        cv = min(5, len(X_train))  # Ensure cv is not greater than the number of samples
        with mlflow.start_run(run_name=f"{self.__class__.__name__}_{id_bb_unique}_{y}",  nested=True) as run:
            if cv >= 2:
                grid_search = RandomizedSearchCV(
                    estimator=mod,
                    param_distributions=grid_params,
                    n_iter=20,
                    cv=cv,
                    n_jobs=-1,
                    verbose=False,
                    random_state=42,
                    scoring='neg_root_mean_squared_error',
                )
                # Fit the model using grid search on training_pipeline data
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
                mod.fit(X_train, y_train)
                model = mod  # Assign the directly fitted model as the best model
                mlflow.log_params(params)

            # Log the model to MLflow
            model_name = f"{self.__class__.__name__}_{id_bb_unique}_{y}"
            # mlflow.lightgbm.log_model(lgb_model=model,
            #                          artifact_path=model_name)
            # model_uri = f"runs:/{run.info.run_id}/{model_name}"
            model_run_id = run.info.run_id
            # mlflow.register_model(model_uri=model_uri, name=model_name)
            # print(f"Model for company: {id_bb_unique}, "
            #       f"registered with model name **{model_name}**")

        return model, model_name, model_run_id