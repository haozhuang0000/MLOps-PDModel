import os
import GPUtil
available_gpus = GPUtil.getAvailable(order='memory', limit=1)
if available_gpus:
    gpu_id = available_gpus[0]
else:
    gpu_id = 1

class LGBMTrainingConfig:

    N_ITERS = 2
    CROSS_VALIDATION = True
    LGBM_PARAMS_GRID = {
        'learning_rate': [0.01, 0.1],
        'num_leaves': [31, 64],
        'max_depth': [5, 10],
        'n_estimators': [100, 200]
    }
    LGBM_PARAMS = {
        'objective': 'multiclass',
        'num_class': 3,
        'random_state': 42,
        'device': 'gpu',
        'gpu_platform_id': 0,
        'gpu_device_id': gpu_id,
        'boosting_type': 'gbdt',
        'verbosity': 1,
    }
