from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
import numpy as np

def evaluate_ovo(y_true, y_pred, y_proba, class_a, class_b):
    # Filter samples where label is class_a or class_b
    mask = np.isin(y_true, [class_a, class_b])
    y_true_binary = y_true[mask]
    y_pred_binary = y_pred[mask]
    y_proba_binary = y_proba[mask, class_b]  # Prob of predicting class_b
    #
    # # Relabel: class_a -> 0, class_b -> 1
    y_true_binary = (y_true_binary == class_b).astype(int)
    y_pred_binary = (y_pred_binary == class_b).astype(int)

    f1 = f1_score(y_true_binary, y_pred_binary)
    precision = precision_score(y_true_binary, y_pred_binary)
    recall = recall_score(y_true_binary, y_pred_binary)
    auc = roc_auc_score(y_true_binary, y_proba_binary)
    ar = 2 * auc - 1

    return {
        f"class{class_a}v{class_b}": {
            'f1': f1,
            'precision': precision,
            'recall': recall,
            'auc': auc,
            # 'ar': ar
        }
    }

def  evaluate_outsample_ovo(metrics, y_true, y_pred,
                           y_proba, class_a, class_b):
    # Filter samples where label is class_a or class_b
    mask = np.isin(y_true, [class_a, class_b])
    y_true_binary = y_true[mask]
    y_pred_binary = y_pred[mask]
    y_proba_binary = y_proba[mask, class_b]  # Prob of predicting class_b

    # # Relabel: class_a -> 0, class_b -> 1
    y_true_binary = (y_true_binary == class_b).astype(int)
    y_pred_binary = (y_pred_binary == class_b).astype(int)

    if metrics == 'f1_score':
        value = f1_score(y_true_binary, y_pred_binary)
    if metrics == 'precision_score':
        value = precision_score(y_true_binary, y_pred_binary)
    if metrics == 'recall_score':
        value = recall_score(y_true_binary, y_pred_binary)
    if metrics == 'roc_auc_score':
        value = roc_auc_score(y_true_binary, y_proba_binary)
    if metrics == 'accuracy_ratio':
        auc = roc_auc_score(y_true_binary, y_proba_binary)
        value = 2 * auc - 1

    return value





