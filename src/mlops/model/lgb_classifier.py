from src.mlops.abstractions import ModelABC
from src.mlops.model.model_utils import mlflow_insample_metrics_log
from src.mlops.configs import LGBMTrainingConfig as lgbm_cfg
from itertools import product
from joblib import Parallel, delayed
import GPUtil
import random
from typing import Union, Tuple, List, Dict
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score, recall_score,
    make_scorer, log_loss
)
from sklearn.utils.class_weight import compute_class_weight
import lightgbm as lgb
from lightgbm import LGBMClassifier
from lightgbm.callback import early_stopping
import mlflow
import mlflow.lightgbm
from mlflow.models import infer_signature

import os
os.environ['CUDA_VISIBLE_DEVICES'] = "1"
class LGBClassifier(ModelABC):

    def __init__(self, cali_group: str):

        self.cali_group = cali_group
        # ==================== <START> Model configuration ====================
        self.start_cv = lgbm_cfg.CROSS_VALIDATION
        self.param_grid = lgbm_cfg.LGBM_PARAMS_GRID
        self.additional_params = lgbm_cfg.LGBM_PARAMS
        # ==================== <END> Model configuration ====================
    def evaluate_params(self,
                        combo: Tuple,
                        param_keys: List[str],
                        train_val_splits: List[Dict[str, pd.DataFrame]],
                        features: List[str]):
        params = dict(zip(param_keys, combo))
        params.update(self.additional_params)

        loss_list = []

        for split in train_val_splits:
            train_df = split['train']
            val_df = split['val']

            X_train = train_df[features]
            y_train = train_df['Y']
            X_val = val_df[features]
            y_val = val_df['Y']

            classes = np.array([0, 1, 2])
            class_weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
            sample_weights = y_train.map(dict(zip(classes, class_weights)))
            sample_weights = sample_weights.values
            model = LGBMClassifier(**params)
            model.fit(
                X_train, y_train,
                sample_weight=sample_weights,
                eval_metric='multi_logloss',
            )

            preds_proba = model.predict_proba(X_val)
            loss = log_loss(y_val, preds_proba)
            print("round loss")
            loss_list.append(loss)

        avg_loss = np.mean(loss_list)
        return avg_loss, params, model

    def hyperParamTuning_parallel(self,
                                  train_val_splits: List[Dict[str, pd.DataFrame]],
                                  features: List[str],
                                  num_gpus: int=-1,
                                  n_iter: int=-1):

        param_grid = self.param_grid
        param_keys = list(param_grid.keys())
        param_combinations = list(product(*param_grid.values()))

        ## random search cv
        if n_iter > 0:
            param_combinations = random.sample(param_combinations, max(2, n_iter))

        results = Parallel(n_jobs=num_gpus)(
            delayed(self.evaluate_params)(combo, param_keys, train_val_splits, features)
            for i, combo in enumerate(param_combinations)
        )
        print(results)
        # Find best
        best_result = min(results, key=lambda x: x[0])
        return best_result  # avg_loss, best_params, best_model

    def train(self,
              train_val_splits: List[Dict[str, pd.DataFrame]],
              train_df: pd.DataFrame
              ):

        X_train = train_df[self.x_cols_to_process_cn]
        y_train = train_df['Y']
        classes = np.array([0, 1, 2])
        class_weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
        class_weights_dict = dict(zip(classes, class_weights))
        sample_weights = y_train.map(class_weights_dict)
        sample_weights = sample_weights.values
        # ==================== <START> Enable autologging ====================
        mlflow.lightgbm.autolog()
        # ==================== <END> Enable autologging ====================


        # ==================== <START> Model configuration ====================
        # valid_data = lgb.Dataset(X_valid, label=y_valid, reference=train_data)
        # multiclass_roc_auc = functools.partial(roc_auc_score, multi_class='ovr')
        # roc_auc_scorer = make_scorer(multiclass_roc_auc, needs_proba=True)

        #
        # grid_params = {
        #     'n_estimators': [100],
        #     'num_leaves': [31, 50, 100],
        #     'learning_rate': [0.01, 0.05, 0.1]
        # }
        # ==================== <END> Model configuration ====================

        # cv = min(2, len(X_train))  # Ensure cv is not greater than the number of samples

        # ===== Start MLflow Run & Train with GridSearchCV (No Validation Data Used Yet) =====
        with mlflow.start_run(run_name=f"{self.__class__.__name__}_Multiclass_{self.cali_group}", nested=True) as run:
            if self.start_cv:
                best_results = self.hyperParamTuning_parallel(train_val_splits,
                                                              self.x_cols_to_process_cn,
                                                              n_iter=lgbm_cfg.N_ITERS)
                best_params = best_results[1]

                final_model = LGBMClassifier(**best_params,
                                             sample_weight=sample_weights,
                                             eval_metric='multi_logloss')

                X_train = train_df[self.x_cols_to_process_cn]
                y_train = train_df['Y']
                final_model.fit(X_train, y_train)
                mlflow_insample_metrics_log(final_model, X_train, y_train, method='classifier')

            else:
                # # Not enough data for cross-validation; fit the model directly
                # mod.fit(X_train, y_train)
                # model = mod  # Assign the directly fitted model as the best model
                # mlflow.log_params(params)
                # params = {
                #     'objective': 'multiclass',
                #     'num_class': y_train.nunique(),
                #     'boosting_type': 'gbdt',
                #     'device': 'gpu',
                #     'metric': 'multi_logloss',
                #     'gpu_use_dp': False,
                #     'max_bin': 63
                # }

                # ==================== <START> Prepare data for training without cross validation ====================
                # Prepare the dataset for LightGBM
                train_data = lgb.Dataset(X_train, label=y_train, weight=sample_weights)
                # ==================== <END> Prepare data for training without cross validation ====================

                # ==================== <START> Model configuration ====================
                # Set LightGBM parameters
                params = {
                    'objective': 'multiclass',
                    'num_class': y_train.nunique(),
                    'max_bin': 63,
                    'boosting_type': 'gbdt',
                    'device': 'gpu',
                    'metric': 'multi_logloss',
                    'num_leaves': 31,
                    'learning_rate': 0.02,
                    'bagging_fraction': 1.0,
                    'bagging_freq': 5,
                    'gpu_platform_id': 0,
                    'gpu_device_id': 2
                }
                # ==================== <END> Model configuration ====================
                final_model = LGBMClassifier(**params)
                final_model.fit(
                    X_train, y_train,
                    sample_weight=sample_weights,
                    eval_metric='multi_logloss',
                )

                mlflow_insample_metrics_log(final_model, X_train, y_train, method='classifier')

                # ## Logging in sample roc_auc
                # # === In-sample predictions ===
                # y_proba = final_model.predict(X_train)
                # y_pred = np.argmax(y_proba, axis=1)
                # # y_proba = final_model.predict_proba(X_train)
                #
                # # === Compute metrics - ovr ===
                # average = 'macro'
                # in_sample_roc_auc = roc_auc_score(y_train, y_proba, multi_class='ovr')
                # in_sample_f1 = f1_score(y_train, y_pred, average=average)
                # in_sample_precision = precision_score(y_train, y_pred, average=average)
                # in_sample_recall = recall_score(y_train, y_pred, average=average)
                #
                # df = pd.DataFrame({"y": y_train, "y_pred": y_proba[:, 1]})
                # df = df[df.y <= 1]
                # df = df[df.y_pred > 0]
                # auc = roc_auc_score(df['y'], df['y_pred'])
                # in_sample_ar = 2 * auc - 1
                #
                # # === Log metrics - ovr ===
                # mlflow.log_metric("insample_ovr_roc_auc", in_sample_roc_auc)
                # mlflow.log_metric(f"insample_ovr_f1_{average}", in_sample_f1)
                # mlflow.log_metric(f"insample_ovr_precision_{average}", in_sample_precision)
                # mlflow.log_metric(f"insample_ovr_recall_{average}", in_sample_recall)
                # mlflow.log_metric("insample_ovr_ar", in_sample_ar)
                #
                # # === Compute metrics - ovo ===
                # eval_01 = evaluate_ovo(y_train, y_pred, y_proba, 0, 1)
                # eval_02 = evaluate_ovo(y_train, y_pred, y_proba, 0, 2)
                #
                # for classvs, value in eval_01.items():
                #     for metric, metric_value in value.items():
                #         print(f'{classvs}_{metric}', metric_value)
                #         mlflow.log_metric(f'insample_{classvs}{metric}', metric_value)
                #
                # for classvs, value in eval_02.items():
                #     for metric, metric_value in value.items():
                #         print(f'{classvs}_{metric}', metric_value)
                #         mlflow.log_metric(f'insample_{classvs}{metric}', metric_value)



            # Evaluate model on validation set
            # y_valid_prob = final_model.predict(X_valid)

            # # Convert probabilities to binary labels (Threshold = 0.5)
            # y_valid_pred = (y_valid_prob > 0.5).astype(int)
            #
            # valid_accuracy = accuracy_score(y_valid, y_valid_pred)
            # valid_auc = roc_auc_score(y_valid, y_valid_pred)

            # # Log validation metrics
            # mlflow.log_metric("valid_accuracy", valid_accuracy)
            # mlflow.log_metric("valid_auc", valid_auc)

            signature = infer_signature(X_train.head(100), final_model.predict(X_train.head(100)))
            mlflow.lightgbm.log_model(final_model, "model", signature=signature)
            # Log the model to MLflow
            model_name = f"{self.__class__.__name__}"
            model_run_id = run.info.run_id

        return final_model, model_name, model_run_id