from src.mlops.evaluation import evaluate_ovo
from typing import Literal
import mlflow
import pandas as pd
import numpy as np
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score, recall_score,
    make_scorer
)

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


