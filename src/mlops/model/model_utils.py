from src.mlops.evaluation import evaluate_ovo
from src.mlops.logger.utils.logger import Log
import os
from typing import Literal, List, Tuple, Dict
import mlflow
import pandas as pd
import numpy as np
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score, recall_score,
    make_scorer
)
from sklearn.utils.class_weight import compute_class_weight
from lightgbm import LGBMClassifier
import numpy as np
from sklearn.metrics import log_loss

def mlflow_insample_metrics_log(
        final_model,
        X_train,
        y_train,
        method: Literal['lgb_train', 'classifier'] = 'classifier'
):
    ## Logging in sample roc_auc
    # === In-sample predictions ===
    if method == 'classifier':
        y_proba = final_model.predict_proba(X_train)
    else:
        y_proba = final_model.predict(X_train)
    y_pred = np.argmax(y_proba, axis=1)
    # y_proba = final_model.predict_proba(X_train)

    # === Compute metrics - ovr ===
    average = 'macro'
    in_sample_roc_auc = roc_auc_score(y_train, y_proba, multi_class='ovr')
    in_sample_f1 = f1_score(y_train, y_pred, average=average)
    in_sample_precision = precision_score(y_train, y_pred, average=average)
    in_sample_recall = recall_score(y_train, y_pred, average=average)

    df = pd.DataFrame({"y": y_train, "y_pred": y_proba[:, 1]})
    df = df[df.y <= 1]
    df = df[df.y_pred > 0]
    auc = roc_auc_score(df['y'], df['y_pred'])
    in_sample_ar = 2 * auc - 1

    # === Log metrics - ovr ===
    mlflow.log_metric("insample_ovr_auc", in_sample_roc_auc)
    mlflow.log_metric(f"insample_ovr_f1_{average}", in_sample_f1)
    mlflow.log_metric(f"insample_ovr_precision_{average}", in_sample_precision)
    mlflow.log_metric(f"insample_ovr_recall_{average}", in_sample_recall)
    mlflow.log_metric("insample_class0v1_ar", in_sample_ar)

    # === Compute metrics - ovo ===
    eval_01 = evaluate_ovo(y_train, y_pred, y_proba, 0, 1)
    eval_02 = evaluate_ovo(y_train, y_pred, y_proba, 0, 2)

    for classvs, value in eval_01.items():
        for metric, metric_value in value.items():
            print(f'{classvs}_{metric}', metric_value)
            mlflow.log_metric(f'insample_{classvs}_{metric}', metric_value)

    for classvs, value in eval_02.items():
        for metric, metric_value in value.items():
            print(f'{classvs}_{metric}', metric_value)
            mlflow.log_metric(f'insample_{classvs}_{metric}', metric_value)

def evaluate_params(combo: Tuple,
                        param_keys: List[str],
                        train_val_splits: List[Dict[str, pd.DataFrame]],
                        features: List[str],
                        additional_params: dict):
        logger = Log(f"{os.path.basename(__file__)}").getlog()
        params = dict(zip(param_keys, combo))
        logger.info(f"Evaluating with params: {params}")
        params.update(additional_params)

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
            print(f"round loss: {loss}")
            loss_list.append(loss)

        avg_loss = np.mean(loss_list)
        return avg_loss, params, model
